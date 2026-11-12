
"""Arabic-aware normalisation and tokenisation."""

from __future__ import annotations

import re


def normalize_ar(text: str) -> str:
    """Normalize Arabic text."""
    # Remove Arabic diacritics and superscript alef
    text = re.sub(r"[\u064B-\u0652\u0670]", "", text)

    # Remove tatweel
    text = text.replace("\u0640", "")

    # Normalize Arabic letters
    translation = str.maketrans({
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ة": "ه",
    })
    text = text.translate(translation)

    # Convert Arabic-Indic and Persian digits to ASCII
    digits = {}
    for i in range(10):
        digits[ord(chr(0x0660 + i))] = str(i)
        digits[ord(chr(0x06F0 + i))] = str(i)

    text = text.translate(digits)

    # Lowercase and normalize whitespace
    text = re.sub(r"\s+", " ", text.lower()).strip()

    return text


def tokenize(text: str) -> list[str]:
    """Normalize text, then extract Unicode word tokens."""

    normalized = normalize_ar(text)
    return re.findall(r"\w+", normalized, flags=re.UNICODE)
