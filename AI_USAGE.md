# AI Usage

## Tools Used and Their Purpose

- **ChatGPT:** Used as a coding assistant to understand assignment requirements, discuss algorithms, generate implementation ideas, review code, and troubleshoot issues.
- **VS Code:** Used as the main development environment to write, edit, and maintain the Python implementation.
- **Python:** Used to implement and run the retrieval pipeline and evaluation scripts.
- **uv and pytest:** Used to manage dependencies and run automated tests.
- **PowerShell:** Used to execute commands, test scenarios, and inspect retrieval results.

I actively participated in implementing the project by reviewing AI-generated suggestions, modifying the code, integrating modules, running tests, and making technical decisions. AI-generated code was not accepted without review and testing.

## Reviewing AI-Assisted Code

I reviewed the AI-assisted chunking and BM25 implementations against the assignment requirements.

For chunking, I tested:
- Empty and whitespace-only documents.
- Long unbroken tokens.
- Maximum chunk length and overlap limits.
- Exact source-offset preservation.
- Coverage of non-whitespace characters without gaps.
- Chunk boundaries and deterministic IDs.

For BM25, I checked:
- The required inverse document frequency formula.
- Term frequency and document frequency calculations.
- Document-length normalization.
- Query-term handling.
- Deterministic score ordering and top-k limits.

I did not identify a confirmed specification-breaking bug in the final chunking or BM25 implementations during these checks, so I have not claimed to fix an issue that I could not reproduce.

During citation formatting review, however, I found an issue in the AI-assisted truncation logic: it reserved a character for the ellipsis before selecting the quote, instead of first taking the allowed maximum length and then cutting back to the last complete word. I corrected the logic and reran the test suite.

## Retrieval Evaluation

The evaluation used all 8 contracts in `data/contracts/`, with:
- `chunk_text(max_chars=500, overlap=50)`
- `HashingEmbedder()`
- The 12 public queries in `data/queries.jsonl`

A retrieved chunk was considered relevant when its span overlapped a relevant range for the same document.

### Results

| Retriever | Recall@3 | Successful Queries | Missed Queries |
|---|---:|---:|---|
| BM25 only | 0.9167 | 11/12 | q12 |
| Dense only (HashingEmbedder) | 0.9167 | 11/12 | q12 |
| Hybrid (RRF) | 0.9167 | 11/12 | q12 |

### Analysis

The only missed query was q12:

> governing law and competent court for the agency agreement

The relevant passage is in `agency_bi_08`. Several contracts contain very similar governing-law clauses, making it difficult for lexical retrieval and the character n-gram-based `HashingEmbedder` to distinguish the correct document using clause text alone.

The relevant agency-agreement chunk ranked fourth in BM25 and eighth in the hybrid top 10. Dense retrieval also failed to rank a relevant agency chunk in its top three.

The hybrid retriever combines BM25 and dense rankings using Reciprocal Rank Fusion (RRF). However, it did not improve Recall@3 over either method individually on this evaluation set because both retrieval methods struggled with the same query.

### Additional Experiment

I also tested adding document-title context to retrieval. In that separate experiment, Hybrid Recall@3 reached 1.0000 (12/12), and the relevant agency-agreement chunk ranked second.

This experiment is not part of the final implementation or the results reported above. It suggests that document-level context may help distinguish similar clauses across contracts.

Other possible improvements include testing a real multilingual embedding model and integrating a reranker, then evaluating them on a larger query set.