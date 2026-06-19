#!/usr/bin/env python3
"""MCP server exposing the textbook-kb to Claude Code (stdio).

Tools:
  search_textbooks    — semantic search (over-fetch -> floor -> rerank -> MMR).
  list_sources        — what books are in the store (+ chunk counts).
  get_toc             — a book's table of contents (for chapter-level questions).
  expand_context      — neighbouring chunks around a hit (small-to-big retrieval).
  data_quality_report — duplicates / junk titles / suspect extraction.
  collection_info     — name, vector count, store path.

The embedder, Chroma collection, and reranker load lazily on the first call and
stay resident, so only the first query in a session pays the model-load cost. A
lightweight data-quality health check runs once on load (cached, rescans only
when stale) and logs a warning to stderr if issues are found.
"""
import os
import sys
import contextlib
from pathlib import Path
from typing import Optional

# Quiet HF / transformers chatter. stdio MCP uses STDOUT for JSON-RPC, so any
# stray print to stdout corrupts the protocol — load-time stdout is redirected
# to stderr below as belt-and-suspenders.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))  # so `from src...` works regardless of cwd

from mcp.server.fastmcp import FastMCP
from src.config import Config

mcp = FastMCP("textbook-kb")

# Resident singletons — populated once, on the first call.
_state = {"retriever": None, "cfg": None}


def _ensure_loaded() -> None:
    if _state["retriever"] is not None:
        return
    with contextlib.redirect_stdout(sys.stderr):  # protect the JSON-RPC channel
        from src.retrieve import Retriever
        from src import quality

        cfg = Config(str(ROOT / "config.yaml"))
        retr = Retriever(cfg)
        _state["cfg"] = cfg
        _state["retriever"] = retr
        if cfg.quality_check_on_load:
            try:
                quality.health_on_load(retr.coll, cfg)
            except Exception as e:  # noqa: BLE001 — never block startup
                print(f"[quality] health check skipped: {e}", file=sys.stderr)


@mcp.tool()
def search_textbooks(
    query: str,
    k: int = 5,
    source: Optional[str] = None,
    contains: Optional[str] = None,
    rerank: Optional[bool] = None,
    diversify: Optional[bool] = None,
) -> str:
    """Semantic search over Willie's data-science textbook library.

    Pipeline: pull a wide candidate set with the ANN index, drop anything below
    the relevance floor, rerank with a cross-encoder, then diversify (MMR) so
    near-duplicate passages don't crowd out the answer. Each hit cites its book,
    page range, and scores.

    Args:
        query: Natural-language question or topic.
        k: Number of passages to return (default 5).
        source: Restrict to one source file (exact relative path; see list_sources).
        contains: Require this exact substring to appear in returned chunks.
        rerank: Override cross-encoder reranking (default: config).
        diversify: Override MMR diversification (default: config).
    """
    _ensure_loaded()
    with contextlib.redirect_stdout(sys.stderr):
        from src.retrieve import format_hits
        hits = _state["retriever"].search(
            query, k=k, source=source, contains=contains,
            rerank=rerank, mmr=diversify,
        )
    return format_hits(hits)


@mcp.tool()
def list_sources(filter: Optional[str] = None, limit: int = 60) -> str:
    """List the books in the store (largest first), with chunk counts and the
    exact `source` path to use for filtering. Optional case-insensitive `filter`
    matches title or path."""
    _ensure_loaded()
    with contextlib.redirect_stdout(sys.stderr):
        from src.store import list_sources as _ls
        return _ls(_state["retriever"].coll, filter=filter, limit=limit)


@mcp.tool()
def get_toc(source: str) -> str:
    """Return a book's table of contents (from its PDF outline). Use the exact
    `source` path from list_sources. Great for chapter/section-level questions."""
    _ensure_loaded()
    with contextlib.redirect_stdout(sys.stderr):
        from src.toc import get_toc as _toc, format_toc
        try:
            return format_toc(_toc(_state["cfg"], source), source)
        except FileNotFoundError as e:
            return str(e)


@mcp.tool()
def expand_context(source: str, chunk_index: Optional[int] = None,
                   page: Optional[int] = None, window: int = 1) -> str:
    """Return the passage around a hit — chunks [i-window, i+window] from the same
    source stitched together — for fuller context than a single chunk. Give
    chunk_index (preferred) or a page number."""
    _ensure_loaded()
    with contextlib.redirect_stdout(sys.stderr):
        ctx = _state["retriever"].expand_context(
            source, chunk_index=chunk_index, page=page, window=window)
    if not ctx:
        return f"No chunks found for source={source} at the given location."
    ps, pe = ctx["page_start"], ctx["page_end"]
    pages = f"p.{ps}" if ps == pe else f"pp.{ps}-{pe}"
    return f"{ctx['title']} ({pages})  source: {ctx['source']}\n\n{ctx['text']}"


@mcp.tool()
def data_quality_report(deep: bool = False) -> str:
    """Report corpus health: duplicate books, junk/empty embedded titles, and
    (with deep=True) sources with suspect/garbled text extraction."""
    _ensure_loaded()
    with contextlib.redirect_stdout(sys.stderr):
        from src import quality
        rep = quality.audit(_state["retriever"].coll, deep=deep)
        return quality.format_report(rep)


@mcp.tool()
def collection_info() -> str:
    """Report the textbook-kb collection name, vector count, and store path."""
    _ensure_loaded()
    cfg = _state["cfg"]
    return (f"Collection '{cfg.collection_name}': "
            f"{_state['retriever'].coll.count()} chunks @ {cfg.chroma_path}")


if __name__ == "__main__":
    mcp.run()  # stdio transport
