"""Data-quality audit for the textbook store.

Two entry points:
  * audit(coll, deep)        — full report (duplicates, junk titles, suspect
                               extraction); deep mode also samples chunk text.
  * health_on_load(coll,cfg) — cheap, cached check the MCP server runs at
                               startup; rescans only when the cached report is
                               missing, stale, or the chunk count changed, and
                               logs a one-line warning if issues are found.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from src.store import content_duplicate_groups, iter_metadata, source_stats
from src.titles import _JUNK_TITLES, _collapse_repeat

REPORT_NAME = "quality_report.json"
_SAMPLE_PER_SOURCE = 8  # chunks sampled per source in deep mode


def _garble_score(text):
    """Rough 0..1 'looks broken' score targeting the two real extraction-failure
    modes seen in this corpus, while NOT punishing legitimately math-heavy text:

      * repetition — adjacent duplicate tokens ("Deep Deep Deep Belief Belief"),
        the signature of the duplicated-text PDF (e.g. the Goodfellow pre-pub).
      * non-word soup — tokens containing no letter at all ("# & % * 0 + $"),
        the signature of mojibake / wrong-glyph extraction (e.g. M340L).

    Inline equations raise the non-word ratio only mildly (numbers/operators are
    interspersed with words), so prose-with-math stays well under threshold.
    """
    toks = text.split()
    if len(toks) < 4:
        return 0.0
    rep = sum(1 for i in range(len(toks) - 1) if toks[i] == toks[i + 1]) / (len(toks) - 1)
    nonword = sum(1 for t in toks if not any(c.isalpha() for c in t)) / len(toks)
    return min(1.0, rep * 1.5 + nonword * 0.7)


def audit(coll, deep=False):
    stats = source_stats(coll)
    total_chunks = sum(s["chunks"] for s in stats.values())

    dup_groups = content_duplicate_groups(coll)  # by shared chunk text, not title
    redundant_total = sum(g["redundant_chunks"] for g in dup_groups)

    junk_titles = [
        {"source": s, "raw_title": st["raw_title"], "chunks": st["chunks"]}
        for s, st in stats.items()
        if _collapse_repeat((st["raw_title"] or "").strip()).lower() in _JUNK_TITLES
        or len((st["raw_title"] or "").strip()) < 4
    ]

    suspect = []
    if deep:
        acc = defaultdict(lambda: {"n": 0, "sum": 0.0})
        seen = defaultdict(int)
        for m, doc in iter_metadata(coll, include=("metadatas", "documents")):
            s = m.get("source", "?")
            if seen[s] >= _SAMPLE_PER_SOURCE or not doc:
                continue
            seen[s] += 1
            acc[s]["n"] += 1
            acc[s]["sum"] += _garble_score(doc)
        for s, a in acc.items():
            mean = a["sum"] / max(1, a["n"])
            if mean > 0.40:
                suspect.append({"source": s, "garble": round(mean, 3),
                                "chunks": stats[s]["chunks"]})
        suspect.sort(key=lambda x: -x["garble"])

    return {
        "generated_at": time.time(),
        "total_sources": len(stats),
        "total_chunks": total_chunks,
        "duplicate_groups": dup_groups,
        "redundant_chunks": redundant_total,
        "redundant_pct": round(100 * redundant_total / max(1, total_chunks), 2),
        "junk_titles": sorted(junk_titles, key=lambda x: -x["chunks"]),
        "suspect_extraction": suspect,
        "deep": deep,
    }


def format_report(rep):
    lines = [
        f"Data-quality report  ({rep['total_sources']} sources, "
        f"{rep['total_chunks']} chunks)",
        "",
        f"Duplicate book groups: {len(rep['duplicate_groups'])}  "
        f"(~{rep['redundant_pct']}% redundant chunks)",
    ]
    for g in rep["duplicate_groups"][:12]:
        lines.append(f"  • {g['title']}  (+{g['redundant_chunks']} redundant)")
        for src, c in g["members"]:
            lines.append(f"      {c:6d}  {src}")
    lines.append("")
    lines.append(f"Junk/empty embedded titles: {len(rep['junk_titles'])} sources "
                 f"(cosmetic — fixed at display time)")
    if rep["deep"]:
        lines.append("")
        lines.append(f"Suspect extraction (sampled): {len(rep['suspect_extraction'])}")
        for s in rep["suspect_extraction"][:12]:
            lines.append(f"  • garble={s['garble']}  {s['chunks']:6d} chunks  {s['source']}")
    elif not rep.get("suspect_extraction"):
        lines.append("(run with --deep to also scan for garbled extraction)")
    return "\n".join(lines)


# -- cached on-load health check ------------------------------------------
def health_on_load(coll, cfg):
    """Run on MCP startup. Returns the cached/fresh report dict, recomputing only
    when stale; logs a one-line warning to stderr if issues exist."""
    path = Path(cfg.chroma_path) / REPORT_NAME
    count = coll.count()
    rep = None
    if path.exists():
        try:
            rep = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            rep = None
    stale = (
        rep is None
        or rep.get("total_chunks") != count
        or (time.time() - rep.get("generated_at", 0)) > cfg.quality_max_age_days * 86400
    )
    if stale:
        rep = audit(coll, deep=False)
        try:
            path.write_text(json.dumps(rep, indent=2))
        except Exception as e:  # noqa: BLE001 — never block startup on cache write
            print(f"[quality] could not write cache: {e}", file=sys.stderr)

    issues = []
    if rep["duplicate_groups"]:
        issues.append(f"{len(rep['duplicate_groups'])} duplicate book groups "
                      f"(~{rep['redundant_pct']}% redundant)")
    if rep["junk_titles"]:
        issues.append(f"{len(rep['junk_titles'])} junk embedded titles")
    if issues:
        print(f"[quality] {'; '.join(issues)} — run `python cli.py audit` "
              f"or the data_quality_report tool", file=sys.stderr)
    return rep
