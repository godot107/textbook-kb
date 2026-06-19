"""Whole-collection scan helpers.

Chroma's `get()` errors with "too many SQL variables" when pulling everything at
once, so we page through with limit/offset. Shared by list_sources, the
data-quality audit, and dedup so the pagination lives in exactly one place.
"""
import hashlib
from collections import Counter, defaultdict
from itertools import combinations

from src.titles import clean_title

_PAGE = 2000


def iter_metadata(coll, include=("metadatas",)):
    """Yield metadata dicts (and optionally documents) across the whole store."""
    total = coll.count()
    off = 0
    want_docs = "documents" in include
    inc = list(include)
    while off < total:
        got = coll.get(include=inc, limit=_PAGE, offset=off)
        metas = got.get("metadatas") or []
        docs = got.get("documents") or [None] * len(metas)
        for m, d in zip(metas, docs):
            yield (m, d) if want_docs else m
        off += _PAGE


def source_stats(coll):
    """Per-source rollup: {source: {chunks, raw_title, title, page_min, page_max}}."""
    stats = defaultdict(lambda: {"chunks": 0, "raw_title": "",
                                 "page_min": 10**9, "page_max": 0})
    for m in iter_metadata(coll):
        s = m.get("source", "?")
        st = stats[s]
        st["chunks"] += 1
        st["raw_title"] = st["raw_title"] or m.get("title", "")
        ps, pe = m.get("page_start"), m.get("page_end")
        if isinstance(ps, int):
            st["page_min"] = min(st["page_min"], ps)
        if isinstance(pe, int):
            st["page_max"] = max(st["page_max"], pe)
    for s, st in stats.items():
        st["title"] = clean_title(s, st["raw_title"])
        if st["page_min"] == 10**9:
            st["page_min"] = None
    return dict(stats)


def content_duplicate_groups(coll, min_overlap=0.80, boilerplate_max_sources=4):
    """Find sources that are the same book by SHARED CHUNK TEXT, not title.

    One pass hashes every chunk. A chunk hash appearing in 2..boilerplate_max
    sources is 'discriminative' (a hash in many books is boilerplate — blank
    pages, copyright lines — and is ignored so it can't link unrelated books).
    Two sources are linked when their shared-hash count over the LARGER source's
    distinct-chunk count meets `min_overlap` — near-identical in both directions.
    (Dividing by the larger size, not the smaller, stops a small file that is
    merely *contained* in a big compilation — a per-section note inside a
    full-course PDF — from being mistaken for a duplicate.) Links union to groups.

    Returns groups: list of {members:[(source,chunks)], redundant_chunks, title}.
    """
    src_hashes = defaultdict(set)
    hash_srcs = defaultdict(set)
    raw_title = {}
    for m, doc in iter_metadata(coll, include=("metadatas", "documents")):
        if not doc:
            continue
        s = m.get("source", "?")
        raw_title.setdefault(s, m.get("title", ""))
        h = hashlib.sha1(doc.encode("utf-8")).hexdigest()
        src_hashes[s].add(h)
        hash_srcs[h].add(s)

    shared = Counter()
    for srcs in hash_srcs.values():
        if 2 <= len(srcs) <= boilerplate_max_sources:
            for a, b in combinations(sorted(srcs), 2):
                shared[(a, b)] += 1

    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        parent[find(a)] = find(b)

    for (a, b), sh in shared.items():
        if sh / max(len(src_hashes[a]), len(src_hashes[b])) >= min_overlap:
            union(a, b)

    clusters = defaultdict(list)
    for s in src_hashes:
        if s in parent:
            clusters[find(s)].append(s)

    groups = []
    for members in clusters.values():
        if len(members) < 2:
            continue
        rows = sorted(((s, len(src_hashes[s])) for s in members), key=lambda r: -r[1])
        groups.append({
            "members": rows,
            "redundant_chunks": sum(c for _, c in rows) - rows[0][1],
            "title": clean_title(rows[0][0], raw_title.get(rows[0][0], "")),
        })
    groups.sort(key=lambda g: -g["redundant_chunks"])
    return groups


def list_sources(coll, filter=None, limit=60):
    """Human-readable list of books in the store, largest first."""
    stats = source_stats(coll)
    rows = sorted(stats.items(), key=lambda kv: -kv[1]["chunks"])
    if filter:
        f = filter.lower()
        rows = [r for r in rows if f in r[0].lower() or f in r[1]["title"].lower()]
    total_books, total_chunks = len(stats), sum(v["chunks"] for v in stats.values())
    head = (f"{total_books} source files, {total_chunks} chunks total"
            + (f"  (showing {min(limit, len(rows))} matching '{filter}')"
               if filter else f"  (top {min(limit, len(rows))} by size)"))
    lines = [head, ""]
    for src, st in rows[:limit]:
        pages = (f"pp.{st['page_min']}-{st['page_max']}"
                 if st["page_min"] is not None else "")
        lines.append(f"{st['chunks']:6d}  {st['title']}  [{pages}]\n"
                     f"          source={src}")
    return "\n".join(lines)
