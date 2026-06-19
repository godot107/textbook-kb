#!/usr/bin/env python3
"""Textbook knowledge base — BGE embeddings + ChromaDB.

Usage:
    python cli.py ingest
    python cli.py query "what is the bias-variance tradeoff?" -k 5
    python cli.py query "gradient boosting" --source "ESL.pdf" --no-rerank
    python cli.py sources [--filter "data mining"]
    python cli.py audit [--deep]
    python cli.py dedup [--apply]
    python cli.py toc "<source path>"
    python cli.py eval [--k 10]
    python cli.py info
"""
import argparse

from src.config import Config


def main():
    ap = argparse.ArgumentParser(description="Textbook KB (BGE + ChromaDB)")
    ap.add_argument("--config", default="config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest", help="Extract, chunk, embed and store all PDFs")

    q = sub.add_parser("query", help="Semantic search over the corpus")
    q.add_argument("text")
    q.add_argument("-k", type=int, default=5)
    q.add_argument("--source", help="Filter to one source file (relative path)")
    q.add_argument("--contains", help="Require this exact substring in the chunk")
    # tri-state: flag absent -> use config default; present -> force on/off
    q.add_argument("--rerank", dest="rerank", action="store_true", default=None,
                   help="Force cross-encoder rerank on (default: config)")
    q.add_argument("--no-rerank", dest="rerank", action="store_false",
                   help="Force cross-encoder rerank off")
    q.add_argument("--mmr", dest="mmr", action="store_true", default=None,
                   help="Force MMR diversification on (default: config)")
    q.add_argument("--no-mmr", dest="mmr", action="store_false",
                   help="Force MMR diversification off")

    s = sub.add_parser("sources", help="List books in the store")
    s.add_argument("--filter", help="Case-insensitive title/path filter")
    s.add_argument("--limit", type=int, default=60)

    a = sub.add_parser("audit", help="Data-quality report")
    a.add_argument("--deep", action="store_true", help="Also scan for garbled text")

    d = sub.add_parser("dedup", help="Detect (and optionally remove) duplicate books")
    d.add_argument("--apply", action="store_true", help="Actually delete redundant copies")

    t = sub.add_parser("toc", help="Show a book's table of contents")
    t.add_argument("source")

    e = sub.add_parser("eval", help="Score retrieval against the gold set")
    e.add_argument("--k", type=int, default=10)
    e.add_argument("--gold", default="eval/gold.yaml")

    sub.add_parser("info", help="Show collection stats")

    args = ap.parse_args()
    cfg = Config(args.config)

    if args.cmd == "ingest":
        from src.ingest import ingest
        ingest(cfg)

    elif args.cmd == "query":
        from src.retrieve import Retriever, format_hits
        hits = Retriever(cfg).search(
            args.text, k=args.k, source=args.source, contains=args.contains,
            rerank=args.rerank, mmr=args.mmr)
        print(format_hits(hits))

    elif args.cmd == "sources":
        from src.ingest import get_collection
        from src.store import list_sources
        _, coll = get_collection(cfg)
        print(list_sources(coll, filter=args.filter, limit=args.limit))

    elif args.cmd == "audit":
        from src.ingest import get_collection
        from src import quality
        _, coll = get_collection(cfg)
        print(quality.format_report(quality.audit(coll, deep=args.deep)))

    elif args.cmd == "dedup":
        from src.ingest import get_collection
        from src import dedup
        _, coll = get_collection(cfg)
        groups, remove = dedup.plan(coll)
        print(dedup.format_plan(groups, remove))
        if args.apply and remove:
            res = dedup.apply(coll, cfg, remove)
            print(f"\nRemoved {res['sources_removed']} source(s); "
                  f"exclude list now {res['exclude_list_size']}. "
                  f"Collection holds {coll.count()} chunks.")
        elif remove:
            print("\n(dry run — re-run with --apply to delete the REMOVE rows)")

    elif args.cmd == "toc":
        from src.toc import get_toc, format_toc
        print(format_toc(get_toc(cfg, args.source), args.source))

    elif args.cmd == "eval":
        from src.evaluate import run_eval
        run_eval(cfg, gold_path=args.gold, k=args.k)

    elif args.cmd == "info":
        from src.ingest import get_collection
        _, coll = get_collection(cfg)
        print(f"Collection '{cfg.collection_name}': {coll.count()} chunks @ {cfg.chroma_path}")


if __name__ == "__main__":
    main()
