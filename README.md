# Bayyina — Find the Clause, Cite the Source

Arabic-first hybrid search over contracts: BM25 (implemented from scratch) + hashing-based dense retrieval + Reciprocal Rank Fusion,
with offset-preserving chunking and citation verification.

Built as Week 4 of the DSC Egypt AI & Deep Tech Internship 2026 (training project, synthetic contracts).

## What it does

Ask a question in Arabic or English, get the top 3 passages, each with a pointer into the original document.

```text
Query: متى يجوز إنهاء عقد التوريد بإخطار كتابي؟

[software_ar_07:3185-3616] "يوما جاز لأي من الطرفين إنهاء العقد بإخطار كتابي. المادة العاشرة: إنهاء العقد ..."
[supply_ar_01:2791-3221]   "يوما جاز لأي من الطرفين إنهاء العقد بإخطار كتابي. المادة التاسعة: إنهاء العقد ..."
[employ_ar_03:2462-2889]   "باستخدامها بالقدر اللازم للانتفاع بتلك الأعمال. المادة الثامنة: إنهاء العقد ..."
```

The contract the question is about (`supply_ar_01`) ranks second: several contracts share near-identical termination clauses.

## Pipeline

normalize (Arabic) → chunk the original text (500 chars, overlap 50, offsets kept) → BM25 + dense (3-gram hashing) → RRF → citation check

- Chunking runs on the original text and normalization happens at search time, so every hit keeps true offsets.
- RRF fuses ranks, not scores, since BM25 and embedding scores are on different scales.
- `verify_citation` checks that a quote exists inside the cited span of the source document.

## Evaluation (8 contracts, 183 chunks, 12 public queries)

| Retriever | Recall@3 |
|---|---|
| BM25 only | 0.9167 (11/12) |
| Dense only (HashingEmbedder) | 0.9167 (11/12) |
| Hybrid (RRF) | 0.9167 (11/12) |

Evaluation results are reported from the project's public evaluation set; see [EVAL.md](EVAL.md) for detailed rankings and failure analysis.

All three miss q12 ("governing law and competent court for the agency agreement"): several contracts contain near-identical
governing-law clauses, and the correct chunk ranks 4th in BM25 and 8th in hybrid.

## Run

```bash
uv sync
uv run pytest
uv run python -m bayyina.demo
uv run python evaluate_retrieval.py
```

## Limitations

- The hashing embedder rewards shared spelling, not meaning.
- 12 queries is a small evaluation set.
- No reranker or real multilingual embedding model tested yet.

## AI usage

See [AI_USAGE.md](AI_USAGE.md).