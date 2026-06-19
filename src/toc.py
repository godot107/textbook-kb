"""Table-of-contents extraction from a PDF's outline (PyMuPDF).

Useful for chapter/section-level navigation — e.g. "summarize chapter 2" — which
flat semantic search serves poorly. Also lets us map a hit's page to its
enclosing section for richer, metadata-aware context.
"""
import os

import fitz


def _abs_path(cfg, source):
    """Resolve a stored relative `source` back to its PDF on disk."""
    p = source if os.path.isabs(source) else os.path.join(cfg.source_dir, source)
    if not os.path.exists(p):
        raise FileNotFoundError(f"PDF not found for source '{source}' (looked at {p})")
    return p


def get_toc(cfg, source):
    """Return the outline as a list of {level, title, page} (1-based pages)."""
    doc = fitz.open(_abs_path(cfg, source))
    try:
        return [{"level": lvl, "title": title.strip(), "page": page}
                for lvl, title, page in doc.get_toc(simple=True)]
    finally:
        doc.close()


def format_toc(entries, source):
    if not entries:
        return f"No embedded table of contents in {source}."
    lines = [f"Table of contents — {source}", ""]
    for e in entries:
        indent = "  " * max(0, e["level"] - 1)
        lines.append(f"{indent}{e['title']}  ·  p.{e['page']}")
    return "\n".join(lines)


def page_to_section(entries, page):
    """Nearest outline heading at/above `page` — the section that contains it."""
    section = None
    for e in entries:
        if e["page"] <= page:
            section = e
        else:
            break
    return section
