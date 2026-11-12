
"""Offset-preserving chunker."""
from __future__ import annotations

from bayyina.types import Chunk


def chunk_text(
    doc_id: str,
    text: str,
    max_chars: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """Split text into chunks with offsets into the original document."""

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0")

    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must satisfy 0 <= overlap < max_chars")

    n = len(text)

    if not text or text.isspace():
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0

    # Find the first non-whitespace character.
    while start < n and text[start].isspace():
        start += 1

    while start < n:
        # Skip whitespace at the beginning of a chunk.
        while start < n and text[start].isspace():
            start += 1

        if start >= n:
            break

        remaining = n - start

        # If the remaining text fits, include it entirely.
        if remaining <= max_chars:
            raw_end = n

        else:
            paragraph_cuts = []
            sentence_cuts = []
            whitespace_cuts = []

            min_cut = max(1, max_chars // 2)

            # Find possible cut positions within the allowed window.
            for cut in range(min_cut, max_chars + 1):
                # Paragraph boundary.
                if (
                    cut >= 2
                    and text[start + cut - 2:start + cut] == "\n\n"
                ):
                    paragraph_cuts.append(cut)

                # Sentence boundary.
                if text[start + cut - 1] in ".!?؟۔\n":
                    sentence_cuts.append(cut)

                # Whitespace boundary.
                if text[start + cut - 1].isspace():
                    whitespace_cuts.append(cut)

            # Apply boundary priority.
            if paragraph_cuts:
                cut = max(paragraph_cuts)
            elif sentence_cuts:
                cut = max(sentence_cuts)
            elif whitespace_cuts:
                cut = max(whitespace_cuts)
            else:
                cut = max_chars

            raw_end = start + cut

        # Remove trailing whitespace.
        end = raw_end

        while end > start and text[end - 1].isspace():
            end -= 1

        if end <= start:
            break

        chunk_content = text[start:end]

        # Check whether the chunk ends in the middle of a word.
        ended_mid_word = (
            end < n
            and not text[end - 1].isspace()
            and not text[end].isspace()
        )

        chunks.append(
            Chunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}#{index}",
                start=start,
                end=end,
                text=chunk_content,
            )
        )

        index += 1

        if end >= n:
            break

        # Calculate the next chunk's starting position.
        next_start = max(start + 1, end - overlap)

        # Avoid leading whitespace.
        while next_start < n and text[next_start].isspace():
            next_start += 1

        # A chunk can start mid-word only if the previous
        # chunk ended mid-word.
        if (
            next_start < n
            and next_start > 0
            and not text[next_start - 1].isspace()
            and not ended_mid_word
        ):
            # Move to the end of the current word.
            while next_start < n and not text[next_start - 1].isspace():
                next_start += 1

            # Skip whitespace after the word.
            while next_start < n and text[next_start].isspace():
                next_start += 1

        start = next_start

    return chunks
