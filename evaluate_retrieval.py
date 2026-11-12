
"""Evaluate BM25, dense retrieval, and hybrid retrieval using Recall@3."""

import json
from pathlib import Path

from bayyina.bm25 import BM25Index
from bayyina.chunking import chunk_text
from bayyina.embedder import HashingEmbedder
from bayyina.hybrid import HybridRetriever


ROOT = Path(__file__).resolve().parent
CONTRACTS_DIR = ROOT / "data" / "contracts"
QUERIES_FILE = ROOT / "data" / "queries.jsonl"

TOP_K = 3
CANDIDATES = 50


def load_queries():
    """Load evaluation queries from the JSONL file."""
    queries = []

    with QUERIES_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                queries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} "
                    f"of {QUERIES_FILE}"
                ) from exc

    return queries


def load_chunks():
    """Load contracts and split them into offset-preserving chunks."""
    chunks = []

    for path in sorted(CONTRACTS_DIR.glob("*.txt")):
        doc_id = path.stem
        document = path.read_text(encoding="utf-8")

        document_chunks = chunk_text(
            doc_id,
            document,
            max_chars=500,
            overlap=50,
        )

        chunks.extend(document_chunks)

    return chunks


def is_relevant(hit, relevant_ranges):
    """Check whether a retrieved chunk overlaps a relevant source span."""
    return any(
        hit.doc_id == relevant["doc_id"]
        and hit.start < relevant["end"]
        and hit.end > relevant["start"]
        for relevant in relevant_ranges
    )


def evaluate_rankings(queries, rankings, method_name):
    """Calculate query-level Recall@3 and list missed queries."""
    missed = []
    successful = 0

    for query_data in queries:
        qid = query_data["qid"]
        relevant_ranges = query_data["relevant"]
        top_results = rankings.get(qid, [])[:TOP_K]

        found = any(
            is_relevant(hit, relevant_ranges)
            for hit in top_results
        )

        if found:
            successful += 1
        else:
            missed.append(qid)

    recall = successful / len(queries) if queries else 0.0

    return {
        "method": method_name,
        "recall": recall,
        "successful": successful,
        "missed": missed,
    }


def print_results(results, total_queries):
    """Display a summary of all retrieval methods."""
    print("\n" + "=" * 70)
    print("RETRIEVAL EVALUATION - RECALL@3")
    print("=" * 70)

    print(
        f"{'Retriever':<25}"
        f"{'Recall@3':<15}"
        f"{'Successful':<15}"
        f"{'Missed':<10}"
    )

    print("-" * 70)

    for result in results:
        print(
            f"{result['method']:<25}"
            f"{result['recall']:<15.4f}"
            f"{result['successful']}/{total_queries:<13}"
            f"{len(result['missed']):<10}"
        )

    print("\nMISSED QUERIES")
    print("-" * 70)

    for result in results:
        missed = result["missed"]
        missed_text = ", ".join(missed) if missed else "None"

        print(f"{result['method']}: {missed_text}")


def print_hit_details(hit, relevant_ranges, rank, score=None):
    """Print a retrieved chunk and its relevance information."""
    relevant = is_relevant(hit, relevant_ranges)

    # BM25 and Dense rankings contain Chunk objects.
    # Hybrid rankings contain Hit objects with a score.
    if score is None:
        score = getattr(hit, "score", None)

    score_text = f"{score:.4f}" if score is not None else "N/A"

    print(
        f"\nRank {rank}: {hit.chunk_id} | "
        f"Score: {score_text} | "
        f"Offsets: {hit.start}-{hit.end} | "
        f"Relevant: {relevant}"
    )
    print(hit.text[:250])


def debug_query(
    target,
    rankings,
    chunks,
    bm25,
    bm25_lookup,
    hybrid,
):
    """Inspect a query's relevant passages and retrieval rankings."""
    relevant_ranges = target["relevant"]
    qid = target["qid"]

    print("\n" + "=" * 70)
    print(f"DEBUG QUERY: {qid}")
    print("=" * 70)

    print("\nQuery:")
    print(target["query"])

    print("\nRelevant contract passages:")

    for index, relevant in enumerate(relevant_ranges, start=1):
        document_path = (
            CONTRACTS_DIR / f"{relevant['doc_id']}.txt"
        )

        document = document_path.read_text(encoding="utf-8")

        print(
            f"\nRelevant range {index}: "
            f"{relevant['doc_id']} "
            f"[{relevant['start']}-{relevant['end']}]"
        )
        print(document[relevant["start"]:relevant["end"]])

    # Compare the top results from all three retrieval methods.
    for method_name, method_rankings in rankings.items():
        print(f"\n{method_name} - Top {TOP_K}:")

        for rank, hit in enumerate(
            method_rankings[qid],
            start=1,
        ):
            print_hit_details(hit, relevant_ranges, rank)

    # Inspect every chunk overlapping a relevant passage.
    print("\n" + "=" * 70)
    print("CHUNKS OVERLAPPING RELEVANT PASSAGES")
    print("=" * 70)

    for chunk in chunks:
        if is_relevant(chunk, relevant_ranges):
            print(
                f"\n{chunk.chunk_id} | "
                f"Offsets: {chunk.start}-{chunk.end}"
            )
            print(chunk.text)

    # Inspect BM25's top 10 results.
    print("\n" + "=" * 70)
    print("BM25 TOP 10")
    print("=" * 70)

    for rank, (chunk_id, score) in enumerate(
        bm25.search(target["query"], top_k=10),
        start=1,
    ):
        chunk = bm25_lookup[chunk_id]

        print_hit_details(
            chunk,
            relevant_ranges,
            rank,
            score=score,
        )

    # Inspect Hybrid's top 10 results.
    print("\n" + "=" * 70)
    print("HYBRID TOP 10")
    print("=" * 70)

    for rank, hit in enumerate(
        hybrid.search(
            target["query"],
            top_k=10,
            candidates=CANDIDATES,
        ),
        start=1,
    ):
        print_hit_details(hit, relevant_ranges, rank)


def main():
    """Run the complete retrieval evaluation."""
    if not CONTRACTS_DIR.exists():
        raise FileNotFoundError(
            f"Contracts directory not found: {CONTRACTS_DIR}"
        )

    if not QUERIES_FILE.exists():
        raise FileNotFoundError(
            f"Queries file not found: {QUERIES_FILE}"
        )

    queries = load_queries()
    chunks = load_chunks()

    if not queries:
        raise ValueError("No evaluation queries were found.")

    if not chunks:
        raise ValueError("No contract chunks were found.")

    print(f"Loaded {len(queries)} queries.")
    print(f"Loaded {len(chunks)} chunks from contracts.")

    # Build the BM25 index.
    bm25 = BM25Index()
    bm25.add(chunks)

    # Build the embedding model and hybrid retriever.
    embedder = HashingEmbedder()
    hybrid = HybridRetriever(chunks, embedder)

    # Precompute embeddings for dense-only retrieval.
    chunk_embeddings = embedder.embed(
        [chunk.text for chunk in chunks]
    )

    # Build a lookup table for BM25 results.
    bm25_lookup = {
        chunk.chunk_id: chunk for chunk in chunks
    }

    rankings = {
        "BM25 only": {},
        "Dense only": {},
        "Hybrid (RRF)": {},
    }

    for query_data in queries:
        qid = query_data["qid"]
        query = query_data["query"]

        # 1. BM25-only ranking.
        bm25_results = bm25.search(
            query,
            top_k=TOP_K,
        )

        rankings["BM25 only"][qid] = [
            bm25_lookup[chunk_id]
            for chunk_id, _score in bm25_results
        ]

        # 2. Dense-only ranking.
        query_embedding = embedder.embed([query])[0]
        similarities = chunk_embeddings @ query_embedding

        dense_indices = sorted(
            range(len(chunks)),
            key=lambda i: (
                -float(similarities[i]),
                chunks[i].chunk_id,
            ),
        )[:TOP_K]

        rankings["Dense only"][qid] = [
            chunks[i] for i in dense_indices
        ]

        # 3. Hybrid ranking.
        rankings["Hybrid (RRF)"][qid] = hybrid.search(
            query,
            top_k=TOP_K,
            candidates=CANDIDATES,
        )

    # Evaluate all three retrieval methods.
    results = []

    for method_name, method_rankings in rankings.items():
        result = evaluate_rankings(
            queries,
            method_rankings,
            method_name,
        )
        results.append(result)

    print_results(results, len(queries))

    # Debug q12 when it exists in the evaluation dataset.
    target = next(
        (
            query_data
            for query_data in queries
            if query_data["qid"] == "q12"
        ),
        None,
    )

    if target is not None:
        debug_query(
            target,
            rankings,
            chunks,
            bm25,
            bm25_lookup,
            hybrid,
        )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
