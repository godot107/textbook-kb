"""PDF/EPUB text extraction with light textbook-aware cleanup (PyMuPDF).

MuPDF opens EPUBs natively and lays them out into pages, so both formats share
one code path. EPUB "pages" are reflowed at MuPDF's default layout (not print
pages), and `toc.get_toc` uses the same layout, so page refs stay consistent.
"""
import os
import re

import fitz  # PyMuPDF

BOOK_EXTS = (".pdf", ".epub")

# EPUB stylesheets routinely trip MuPDF's CSS parser; the warnings are harmless
# noise on stderr, so silence them.
fitz.TOOLS.mupdf_display_errors(False)

_WS = re.compile(r"[ \t]+")
_MULTINL = re.compile(r"\n{3,}")
_HYPHEN = re.compile(r"(\w)-\n(\w)")  # join words split across a line break


def _clean(text):
    text = _HYPHEN.sub(r"\1\2", text)
    text = _WS.sub(" ", text)
    text = _MULTINL.sub("\n\n", text)
    return text.strip()


def extract_book(path):
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
