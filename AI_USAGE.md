# AI Usage

## Tools Used and Their Purpose

- **ChatGPT:** Used as a coding assistant to understand assignment requirements, discuss algorithms, generate implementation ideas, review code, and troubleshoot issues.
- **VS Code:** Used as the main development environment to write, edit, integrate, and maintain the Python implementation across project modules.
- **Python:** Used to implement and run the retrieval pipeline and evaluation scripts.
- **uv and pytest:** Used to manage dependencies and run automated tests throughout development.
- **PowerShell:** Used to execute commands, test different scenarios, and inspect retrieval results.

I actively participated in implementing the project by writing and modifying Python code, integrating the different modules, testing the implementation, and making technical decisions. AI-generated suggestions were treated as assistance rather than final solutions, and I reviewed and tested the code before using it.

## Reviewing AI-Assisted Chunking and BM25 Code

I reviewed the AI-assisted chunking and BM25 implementations against the assignment requirements instead of relying only on whether the code executed successfully.

For chunking, I tested empty and whitespace-only documents, long unbroken tokens, chunk boundaries, overlap limits, and exact source-offset preservation. I also checked that chunks respect the maximum character limit and that all non-whitespace characters remain covered without gaps.

For BM25, I checked the implementation against the required inverse document frequency formula, term frequency calculation, document frequency counting, query-term handling, document-length normalization, and deterministic result ordering.

I did not identify a confirmed specification-breaking bug in the final chunking or BM25 implementations during these checks. Therefore, I have not claimed to fix an unverified issue or invented a bug report. The implementations were reviewed against the stated requirements and tested with the available test suite.

## Retrieval Evaluation

The evaluation used all 8 contracts in `data/contracts/`, with `chunk_text(..., max_chars=500, overlap=50)`, `HashingEmbedder()`, and the 12 queries in `data/queries.jsonl`.

A retrieved chunk was considered relevant if its span overlapped one of the query's relevant ranges in the same document.

### Results

| Retriever | Recall@3 | Queries missed |
|---|---:|---|
| BM25 only | 0.9167 (11/12) | q12 |
| Dense only (HashingEmbedder) | 0.9167 (11/12) | q12 |
| Hybrid (RRF) | 0.9167 (11/12) | q12 |

### Analysis

The only missed query for all three retrieval methods was q12, which asks about the governing law and competent court in the agency agreement.

Several contracts contain nearly identical governing-law clauses. This makes it difficult for lexical retrieval and the character n-gram-based `HashingEmbedder` to distinguish the correct document based on clause text alone.

The hybrid retriever combines BM25 and dense rankings using Reciprocal Rank Fusion (RRF). However, on this evaluation set, combining the two rankings did not improve Recall@3 over either method individually.

The relevant agency-agreement chunk ranked fourth in BM25 and eighth in the hybrid top 10. This indicates that similar clauses across different contracts remain difficult to distinguish using the current retrieval configuration.

### Additional Experiment

I also experimented with adding document-title context to retrieval. In that separate experiment, Hybrid Recall@3 reached 1.0000 (12/12), and the relevant agency-agreement chunk ranked second.

However, document-title context is not included in the final implementation evaluated in the results above. This experiment suggests that document-level context could help distinguish similar clauses across contracts and is worth investigating further.

Other possible improvements include testing a real multilingual embedding model or adding a reranker, followed by evaluation on a larger query set.