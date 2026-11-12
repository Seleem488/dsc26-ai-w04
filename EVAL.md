# Retrieval Evaluation

## Setup

The evaluation uses all 8 contracts in `data/contracts/`, with:

- `chunk_text(..., max_chars=500, overlap=50)`
- `HashingEmbedder()`
- The 12 public queries in `data/queries.jsonl`

A retrieved chunk is considered relevant if its span overlaps one of the query's relevant ranges in the same document.

## Results

| Retriever | Recall@3 | Successful Queries | Missed Queries |
|---|---:|---:|---|
| BM25 only | 0.9167 | 11/12 | q12 |
| Dense only (HashingEmbedder) | 0.9167 | 11/12 | q12 |
| Hybrid (RRF) | 0.9167 | 11/12 | q12 |

## Analysis

The only missed query for all three retrieval methods was q12:

> governing law and competent court for the agency agreement

The relevant passage is in `agency_bi_08`, covering the governing law and jurisdiction clause.

Several contracts contain nearly identical governing-law clauses. This makes it difficult for lexical retrieval and the character n-gram-based `HashingEmbedder` to distinguish the correct document based on clause text alone.

The relevant agency-agreement chunk (`agency_bi_08#13`) ranked fourth in BM25 and eighth in the hybrid top 10. Dense retrieval also failed to rank a relevant agency chunk in its top three.

The hybrid retriever combines BM25 and dense rankings using Reciprocal Rank Fusion (RRF). However, combining the two rankings did not improve Recall@3 over either method individually on this evaluation set.

## Conclusion

All three retrieval methods achieved a Recall@3 of 0.9167 (11 out of 12 queries).

The results show that BM25 and dense retrieval perform equally on this small public evaluation set. The hybrid approach did not improve the measured Recall@3, mainly because the similar governing-law clauses across different contracts make document-level distinction difficult.

Potential improvements include adding document-title context, testing a real multilingual embedding model, or integrating a reranker. These approaches should be evaluated on a larger query set before being included in the final implementation.