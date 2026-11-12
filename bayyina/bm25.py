
"""BM25 implementation from scratch, without search libraries."""
from __future__ import annotations

from collections import Counter
from math import log

from bayyina.normalize import tokenize
from bayyina.types import Chunk


class BM25Index:
    """A simple BM25 index for text chunks."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        # Each entry stores: (chunk_id, term_frequencies, document_length).
        self.documents: list[tuple[str, Counter, int]] = []

        # Number of documents containing each term.
        self.doc_freq: Counter = Counter()

        # Total number of tokens across all indexed documents.
        self.total_length = 0

    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks to the BM25 index."""

        for chunk in chunks:
            tokens = tokenize(chunk.text)
            term_freq = Counter(tokens)
            doc_length = len(tokens)

            self.documents.append(
                (chunk.chunk_id, term_freq, doc_length)
            )

            # Count each term once per document.
            for term in term_freq:
                self.doc_freq[term] += 1

            self.total_length += doc_length

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Return matching chunks ordered by BM25 score."""

        if not query.strip() or not self.documents or top_k <= 0:
            return []

        # Query terms are unique to avoid counting repetitions twice.
        query_terms = set(tokenize(query))

        if not query_terms:
            return []

        n_documents = len(self.documents)
        avgdl = self.total_length / n_documents

        if avgdl == 0:
            return []

        results: list[tuple[str, float]] = []

        for chunk_id, term_freq, doc_length in self.documents:
            score = 0.0

            for term in query_terms:
                tf = term_freq.get(term, 0)

                if tf == 0:
                    continue

                df = self.doc_freq[term]

                # Inverse Document Frequency.
                idf = log(
                    1 + (n_documents - df + 0.5) / (df + 0.5)
                )

                # BM25 term-frequency component.
                numerator = tf * (self.k1 + 1)

                denominator = (
                    tf
                    + self.k1
                    * (
                        1 - self.b
                        + self.b * doc_length / avgdl
                    )
                )

                score += idf * numerator / denominator

            if score > 0:
                results.append((chunk_id, score))

        # Higher scores first; break ties by chunk ID.
        results.sort(key=lambda item: (-item[1], item[0]))

        return results[:top_k]
