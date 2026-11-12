
"""Citation verification and formatting."""
from __future__ import annotations

from bayyina.types import Hit


def verify_citation(
    quote: str,
    document: str,
    start: int,
    end: int,
) -> bool:
    """Verify that a quote exists within the specified document span."""

    # Validate the source offsets.
    if not (0 <= start < end <= len(document)):
        return False

    # Reject empty or whitespace-only quotes.
    if not quote or not quote.strip():
        return False

    # Extract the original document span.
    document_slice = document[start:end]

    # Normalize whitespace before comparison.
    normalized_quote = " ".join(quote.split())
    normalized_slice = " ".join(document_slice.split())

    return normalized_quote in normalized_slice


def format_citation(hit: Hit, max_quote: int = 160) -> str:
    """Format a retrieval hit as a compact source citation."""

    # Collapse repeated whitespace and newlines.
    quote = " ".join(hit.text.split())

    if len(quote) > max_quote:
        if max_quote <= 0:
            quote = "…"
        else:
            # Take the first max_quote characters.
            quote = quote[:max_quote]

            # Avoid cutting a word in the middle.
            last_space = quote.rfind(" ")
            if last_space > 0:
                quote = quote[:last_space]

            # Append the truncation marker.
            quote += "…"

    return f'[{hit.doc_id}:{hit.start}-{hit.end}] "{quote}"'
