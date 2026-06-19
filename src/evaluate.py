"""Retrieval evaluation: score the pipeline against a small gold set.

Each gold item names a query and the source (and optionally a page range) that a
good hit should come from. We report recall@k (did any top-k hit match) and MRR
(1 / rank of the first match), for two configurations:

  baseline  — dense ANN only (no rerank, no MMR)
  full      — the configured pipeline (floor + rerank + MMR)

so you can see whether reranking/diversification actually helps on your data.
Expand eval/gold.yaml over time; ~30 items gives a stable signal.
"""
import yaml

from src.retrieve import Retriever


def _matches(hit, item):
    expect = item["expect_source"]
    expect = [expect] if isinstance(expect, str) else expect
    src = hit["source"].lower()
    if not any(e.lower() in src for e in expect):
        return False
    pages = item.get("expect_pages")
    if not pages:
        return True
    lo, hi = pages
    hs, he = hit["page_start"] or 0, hit["page_end"] or 0
    return hs <= hi and he >= lo  # range overlap


def _score(retr, gold, k, **kw):
    recall, rr = 0, 0.0
    misses = []
    for item in gold:
        hits = retr.search(item["query"], k=k, **kw)
        rank = next((i for i, h in enumerate(hits, 1) if _matches(h, item)), None)
        if rank:
            recall += 1
            rr += 1.0 / rank
        else:
            misses.append(item["query"])
    n = len(gold)
    return {"recall_at_k": recall / n, "mrr": rr / n, "n": n, "misses": misses}


def run_eval(cfg, gold_path="eval/gold.yaml", k=10):
    gold = yaml.safe_load(open(gold_path))["queries"]
    retr = Retriever(cfg)  # one model load, reused across configs
    print(f"Gold set: {len(gold)} queries, k={k}\n")

    configs = [
        ("baseline",    dict(rerank=False, mmr=False)),
        ("rerank-only", dict(rerank=True,  mmr=False)),
        ("mmr-only",    dict(rerank=False, mmr=True)),
        ("full",        dict(rerank=True,  mmr=True)),
    ]
    results = {name: _score(retr, gold, k, **kw) for name, kw in configs}

    base = results["baseline"]
    print(f"{'config':12}  {'recall@k':>9}  {'MRR':>6}  {'Δrecall':>8}  {'ΔMRR':>7}")
    for name, _ in configs:
        r = results[name]
        print(f"{name:12}  {r['recall_at_k']:9.3f}  {r['mrr']:6.3f}  "
              f"{r['recall_at_k']-base['recall_at_k']:+8.3f}  "
              f"{r['mrr']-base['mrr']:+7.3f}")
    if results["full"]["misses"]:
        print("\nStill missed by full pipeline:")
        for q in results["full"]["misses"]:
            print(f"  - {q}")
    print("\nNote: recall here matches the expected SOURCE book; MMR trades some "
          "source-recall for cross-book diversity, so judge it alongside MRR.")
    return results
