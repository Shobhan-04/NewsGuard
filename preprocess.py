"""Text construction and cleaning shared by training and inference.

``build_text`` joins title and body; ``clean_text`` is plugged into the
``TfidfVectorizer`` as its ``preprocessor``. Because the cleaning lives *inside*
the saved scikit-learn Pipeline, training and serving cannot drift apart.

Leakage precautions
-------------------
Some corpora (ISOT in particular) leak the label through source artefacts, e.g.
real articles starting with a "CITY (Reuters) -" dateline. ``clean_text``
removes datelines, the word "Reuters", URLs, handles and a few other source
markers. This is a heuristic: stylistic differences between sources remain, and
the README discusses what that means for the reported metrics.
"""
from __future__ import annotations

import html
import re
import unicodedata

_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TAG_RE = re.compile(r"<[^>]+>")
# "WASHINGTON (Reuters) - ", "SAN FRANCISCO/WASHINGTON (Reuters) -", "(Reuters) -"
_DATELINE_RE = re.compile(
    r"^[ \t]*(?:[A-Z][A-Za-z.'’\- ]{0,40}(?:/[A-Z][A-Za-z.'’\- ]{0,40})*"
    r"(?:,[ \t]*[A-Za-z. ]{2,30})?)?[ \t]*\((?:Reuters|REUTERS)\)[ \t]*[-–—:]*[ \t]*",
    re.MULTILINE,
)
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PIC_TWITTER_RE = re.compile(r"pic\.twitter\.com/\S+", re.IGNORECASE)
_HANDLE_RE = re.compile(r"(?<!\w)@\w+")
_SOURCE_MARKERS_RE = re.compile(
    r"\breuters\b|21st century wire|\bfeatured image\b|\bgetty images?\b|\bimage via\b",
    re.IGNORECASE,
)
_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def build_text(title: str | None = None, text: str | None = None) -> str:
    """Combine title and body into one string (either part may be missing).

    Raises ``TypeError`` for non-string, non-None arguments.
    """
    parts: list[str] = []
    for name, value in (("title", title), ("text", text)):
        if value is None:
            continue
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string or None, got {type(value).__name__}")
        value = value.strip()
        if value:
            parts.append(value)
    return "\n\n".join(parts)


def clean_text(text: str | None) -> str:
    """Normalise one document. Output is lower-case with collapsed whitespace."""
    if text is None:
        return ""
    if not isinstance(text, str):
        raise TypeError(f"text must be a string or None, got {type(text).__name__}")
    s = unicodedata.normalize("NFKC", text)
    s = html.unescape(s)
    s = _CTRL_RE.sub(" ", s)
    s = _TAG_RE.sub(" ", s)
    s = _DATELINE_RE.sub("", s)
    s = _PIC_TWITTER_RE.sub(" ", s)
    s = _URL_RE.sub(" ", s)
    s = _EMAIL_RE.sub(" ", s)
    s = _HANDLE_RE.sub(" ", s)
    s = _SOURCE_MARKERS_RE.sub(" ", s)
    s = s.lower()
    return _WS_RE.sub(" ", s).strip()


def prepare_text(title: str | None = None, text: str | None = None) -> str:
    """``clean_text(build_text(...))`` - what the model effectively sees."""
    return clean_text(build_text(title, text))


def near_duplicate_key(cleaned: str, n_chars: int = 300) -> str:
    """Key for near-duplicate detection: first ``n_chars`` alphanumerics."""
    return _NON_ALNUM_RE.sub("", cleaned)[:n_chars]
