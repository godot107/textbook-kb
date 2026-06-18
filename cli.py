#!/usr/bin/env python3
"""Textbook knowledge base — BGE embeddings + ChromaDB.

Usage:
    python cli.py ingest
    python cli.py query "what is the bias-variance tradeoff?" -k 5
    python cli.py query "gradient boosting" --source "ESL.pdf" --contains "shrinkage"
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

    sub.add_parser("info", help="Show collection stats")

    args = ap.parse_args()
    cfg = Config(args.config)

    if args.cmd == "ingest":
        from src.ingest import ingest
        ingest(cfg)
    elif args.cmd == "query":
        from src.query import query, format_results
        res = query(cfg, args.text, k=args.k, source=args.source, contains=args.contains)
        print(format_results(res))
    elif args.cmd == "info":
        from src.ingest import get_collection
        _, coll = get_collection(cfg)
        print(f"Collection '{cfg.collection_name}': {coll.count()} chunks @ {cfg.chroma_path}")


if __name__ == "__main__":
    main()
