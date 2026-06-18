"""PDF text extraction with light textbook-aware cleanup (PyMuPDF)."""
import os
import re

import fitz  # PyMuPDF

_WS = re.compile(r"[ \t]+")
_MULTINL = re.compile(r"\n{3,}")
_HYPHEN = re.compile(r"(\w)-\n(\w)")  # join words split across a line break


def _clean(text):
    text = _HYPHEN.sub(r"\1\2", text)
    text = _WS.sub(" ", text)
    text = _MULTINL.sub("\n\n", text)
    return text.strip()


def extract_pdf(path):
    """Return (pages, meta).

    pages: list of {"page": int (1-based), "text": str}
    meta:  {"title": str, "n_pages": int}
    """
    doc = fitz.open(path)
    try:
        n_pages = doc.page_count
        meta_title = (doc.metadata or {}).get("title") or ""
        title = meta_title.strip() or os.path.splitext(os.path.basename(path))[0]
        pages = []
        for i, page in enumerate(doc):
            cleaned = _clean(page.get_text("text"))
            if cleaned:
                pages.append({"page": i + 1, "text": cleaned})
    finally:
        doc.close()
    return pages, {"title": title, "n_pages": n_pages}
