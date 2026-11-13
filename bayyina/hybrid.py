
"""Reciprocal rank fusion and the hybrid retriever."""
from __future__ import annotations

import numpy as np

from bayyina.bm25 import BM25Index
from bayyina.types import Chunk, Embedder, Hit, Reranker


def rrf(
    rankings: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Combine ranked lists using Reciprocal Rank Fusion."""

    scores: dict[str, float] = {}

    for ranking in rankings:
        seen_in_ranking = set()

        for rank, chunk_id in enumerate(ranking, start=1):
            if chunk_id in seen_in_ranking:
                continue

            seen_in_ranking.add(chunk_id)

            scores[chunk_id] = (
                scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            )

    return sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0]),
    )


class HybridRetriever:
    """Retrieve chunks using BM25, dense search, and optional reranking."""

    def __init__(
        self,
        chunks: list[Chunk],
        embedder: Embedder,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        reranker: Reranker | None = None,
    ):
        self.chunks = chunks
        self.embedder = embedder
        self.reranker = reranker

        # Build the BM25 index from the original chunks.
        self.bm25 = BM25Index(k1=k1, b=b)
        self.bm25.add(chunks)

        # Embed the original chunk texts once.
        chunk_texts = [chunk.text for chunk in chunks]

        if chunk_texts:
            self.chunk_embeddings = np.asarray(
                embedder.embed(chunk_texts)
            )
        else:
            self.chunk_embeddings = np.empty((0, 0))

        # Keep original chunks for constructing accurate hits.
        self.chunk_lookup = {
            chunk.chunk_id: chunk for chunk in chunks
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidates: int = 50,
    ) -> list[Hit]:
        """Return the highest-ranked chunks for a query."""

        if not query.strip() or not self.chunks:
            return []

        if top_k <= 0 or candidates <= 0:
            return []

        # 1. Retrieve candidates using BM25.
        bm25_results = self.bm25.search(
            query,
            top_k=candidates,
        )

        bm25_ranking = [
            chunk_id for chunk_id, _ in bm25_results
        ]

        # 2. Retrieve candidates using dense similarity.
        query_embedding = np.asarray(
            self.embedder.embed([query])
        )[0]

        similarities = self.chunk_embeddings @ query_embedding

        dense_indices = sorted(
            range(len(self.chunks)),
            key=lambda i: (
                -float(similarities[i]),
                self.chunks[i].chunk_id,
            ),
        )[:candidates]

        dense_ranking = [
            self.chunks[i].chunk_id
            for i in dense_indices
        ]

        # 3. Fuse the two rankings using RRF.
        fused_results = rrf(
            [bm25_ranking, dense_ranking]
        )[:candidates]

        # 4. Build hits using the original chunk data.
        hits = []

        for chunk_id, score in fused_results:
            chunk = self.chunk_lookup[chunk_id]

            hits.append(
                Hit(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    start=chunk.start,
                    end=chunk.end,
                    text=chunk.text,
                    score=score,
                )
            )

        # 5. Optionally rerank the fused candidates.
        if self.reranker is not None:
            reranker_scores = self.reranker(query, hits)

            if len(reranker_scores) != len(hits):
                raise ValueError(
                    "Reranker must return one score for each hit"
                )

            hits = [
                Hit(
                    chunk_id=hit.chunk_id,
                    doc_id=hit.doc_id,
                    start=hit.start,
                    end=hit.end,
                    text=hit.text,
                    score=float(score),
                )
                for hit, score in zip(hits, reranker_scores)
            ]

            hits.sort(
                key=lambda hit: (-hit.score, hit.chunk_id)
            )

        return hits[:top_k]
