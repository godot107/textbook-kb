"""Derive a clean display title from a source filename.

The `title` stored at ingest comes from each PDF's embedded metadata, which in
practice is unreliable: often "Untitled", duplicated ("X : X"), or identical
across dozens of unrelated files (a notes folder). The source *filename* is the
trustworthy identifier, so for display we reconstruct a readable title from it,
stripping the cruft that libgen / Anna's Archive / z-lib leave behind.
"""
import os
import re

# Junk fragments commonly tacked onto pirated-PDF filenames.
_CRUFT = re.compile(
    r"""(
        \blibgen(\.lc|\.li|\.rs)?\b
      | z-?lib(\.org)?
      | \banna’?s?\s+archive\b
      | \(z-lib\.org\)
      | -\s*libgen\.\w+
    )""",
    re.IGNORECASE | re.VERBOSE,
)
_HEXBLOB = re.compile(r"\b[0-9a-f]{16,}\b", re.IGNORECASE)  # md5/sha download ids
_MULTISPACE = re.compile(r"\s{2,}")
_DASH_RUNS = re.compile(r"(\s*[-–—_]\s*){2,}")
_DEDUP_TITLE = re.compile(r"^(.*?)(?:\s*:\s*\1)+$", re.IGNORECASE)  # "X : X" -> "X"
_JUNK_TITLES = {"", "untitled", "title", "microsoft word", "pdf", "document"}


def _collapse_repeat(s: str) -> str:
    m = _DEDUP_TITLE.match(s.strip())
    return m.group(1).strip() if m else s


def from_filename(source: str) -> str:
    """Best-effort readable title from a relative source path."""
    stem = os.path.splitext(os.path.basename(source))[0]
    stem = _HEXBLOB.sub("", stem)
    stem = _CRUFT.sub("", stem)
    # Many filenames are "Author(s) - Title-Publisher (year)"; keep the richest
    # middle segment if the dash-split makes the title obvious, else keep whole.
    stem = stem.replace("_", " ")
    stem = _DASH_RUNS.sub(" - ", stem)
    stem = _MULTISPACE.sub(" ", stem).strip(" -–—:")
    return stem or os.path.basename(source)


def clean_title(source: str, raw_title: str = "") -> str:
    """Prefer a trustworthy title: use embedded metadata only when it looks real,
    otherwise reconstruct from the filename."""
    rt = _collapse_repeat((raw_title or "").strip())
    if rt.lower() in _JUNK_TITLES or len(rt) < 4 or _HEXBLOB.search(rt):
        return from_filename(source)
    return _MULTISPACE.sub(" ", rt)
