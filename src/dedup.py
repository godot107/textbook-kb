"""De-duplicate the store at the book level.

The dominant waste is the same book ingested under two/three filenames (e.g.
ESL x2 at 1619 chunks each). We detect groups of sources that resolve to the
same cleaned title, keep the largest copy, and remove the rest. Removed sources
are recorded in an exclude list so a later `ingest` does not re-add them (the
duplicate files still sit on disk under different names).
"""
import json
from pathlib import Path

from src.store import content_duplicate_groups

EXCLUDE_NAME = "excluded_sources.json"


def load_excluded(cfg):
    path = Path(cfg.chroma_path) / EXCLUDE_NAME
    if path.exists():
        try:
            return set(json.loads(path.read_text()))
        except Exception:  # noqa: BLE001
            return set()
    return set()


def _save_excluded(cfg, sources):
    path = Path(cfg.chroma_path) / EXCLUDE_NAME
    current = load_excluded(cfg)
    current.update(sources)
    path.write_text(json.dumps(sorted(current), indent=2))
    return current


def plan(coll):
    """Return (groups, remove). Groups are sources sharing >=80% chunk content
    (same book, regardless of filename/title); `remove` keeps the largest copy
    of each group and lists the rest for deletion."""
    groups = content_duplicate_groups(coll)
    remove = [src for g in groups for src, _ in g["members"][1:]]
    return groups, remove


def format_plan(groups, remove):
    if not groups:
        return "No duplicate books detected (by shared chunk content)."
    lines = [f"{len(groups)} duplicate group(s); {len(remove)} redundant source(s) "
             f"would be removed:\n"]
    for g in groups:
        keep = g["members"][0]
        lines.append(f"  KEEP   {keep[1]:6d}  {keep[0]}")
        for src, c in g["members"][1:]:
            lines.append(f"  REMOVE {c:6d}  {src}")
        lines.append("")
    return "\n".join(lines)


def apply(coll, cfg, remove):
    """Delete redundant sources from the store and add them to the exclude list."""
    deleted = 0
    for src in remove:
        coll.delete(where={"source": src})
        deleted += 1
    excluded = _save_excluded(cfg, remove)
    return {"sources_removed": deleted, "exclude_list_size": len(excluded)}
