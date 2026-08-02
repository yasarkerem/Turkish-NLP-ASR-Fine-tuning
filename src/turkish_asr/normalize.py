"""Turkish-aware text normalization for ASR scoring.

The tricky part is the Turkish i family. Turkish has four distinct letters:

    dotted:   i (U+0069)  /  İ (U+0130)
    dotless:  ı (U+0131)  /  I (U+0049)

so ASCII casefolding is wrong in both directions: ``"İ".lower()`` in Python
yields ``"i" + U+0307`` (combining dot above), and ``"I".lower()`` yields
``"i"`` where Turkish orthography requires ``"ı"``. Both mistakes silently
change WER/CER, so the mapping is done explicitly before any generic
lowercasing.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

COMBINING_DOT_ABOVE = "̇"

TURKISH_SPECIFIC_CHARS = "çğıöşüÇĞİÖŞÜ"

# Apostrophe-like characters are deleted rather than turned into spaces:
# Turkish attaches suffixes to proper nouns with an apostrophe ("Ankara'da"),
# and the spoken form has no boundary there.
_APOSTROPHES = "'’ʼ‘`´"

_WHITESPACE_RE = re.compile(r"\s+")


def turkish_lower(text: str) -> str:
    """Lowercase with Turkish dotted/dotless i handling.

    İ -> i and I -> ı are applied before the generic ``str.lower`` so the
    default Unicode mapping never sees the two upper-case i forms.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("İ", "i").replace("I", "ı")
    text = text.lower()
    # Repair "i + combining dot above" sequences that appear when upstream
    # data was lowercased without Turkish rules (e.g. "İ".lower() artifacts).
    text = text.replace("i" + COMBINING_DOT_ABOVE, "i")
    return text


def contains_turkish_specific(text: str) -> bool:
    """True if the text uses any letter unique to the Turkish alphabet."""
    return any(ch in TURKISH_SPECIFIC_CHARS for ch in text)


@dataclass(frozen=True)
class TurkishNormalizer:
    """Configurable normalizer applied to references and hypotheses alike.

    Every option defaults to on; scoring code reports metrics both with the
    normalizer and on untouched strings, never only one of them.
    """

    lowercase: bool = True
    strip_punctuation: bool = True
    collapse_whitespace: bool = True

    def __call__(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        if self.lowercase:
            text = turkish_lower(text)
        if self.strip_punctuation:
            text = self._strip_punctuation(text)
        if self.collapse_whitespace:
            text = _WHITESPACE_RE.sub(" ", text).strip()
        return text

    @staticmethod
    def _strip_punctuation(text: str) -> str:
        out = []
        for ch in text:
            if ch in _APOSTROPHES:
                continue
            cat = unicodedata.category(ch)
            # All punctuation (P*) and symbols (S*) become spaces; digits and
            # letters pass through untouched.
            if cat.startswith("P") or cat.startswith("S"):
                out.append(" ")
            else:
                out.append(ch)
        return "".join(out)
