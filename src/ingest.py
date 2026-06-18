"""Ingest pipeline: walk PDFs -> extract -> chunk -> embed -> upsert into Chroma.

Resumable: a manifest records each file's mtime+size, so re-running only
processes new or changed PDFs.
"""
import glob
import hashlib
import json
import os
from pathlib import Path

import chromadb
from tqdm import tqdm

from src.chunk import Chunker
from src.embed import Embedder
from src.extract import extract_pdf


def _fingerprint(path):
    st = os.stat(path)
    return f"{int(st.st_mtime)}:{st.st_size}"


def _doc_id(rel_path, idx):
    h = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:16]
    return f"{h}:{idx}"


def get_collection(cfg):
    client = chromadb.PersistentClient(path=cfg.chroma_path)
    coll = client.get_or_create_collection(
        name=cfg.collection_name,
        configuration={
            "hnsw": {
                "space": cfg.hnsw.get("space", "cosine"),
                "ef_construction": cfg.hnsw.get("ef_construction", 200),
                "max_neighbors": cfg.hnsw.get("max_neighbors", 32),
                "ef_search": cfg.hnsw.get("ef_search", 80),
            }
        },
    )
    return client, coll


def _upsert_batched(coll, ids, embeddings, documents, metadatas, batch=1000):
    for i in range(0, len(ids), batch):
        sl = slice(i, i + batch)
        coll.upsert(
            ids=ids[sl],
            embeddings=embeddings[sl],
            documents=documents[sl],
            metadatas=metadatas[sl],
        )


def ingest(cfg):
    _, coll = get_collection(cfg)
    embedder = Embedder(cfg.model_name, cfg.device, cfg.use_fp16, cfg.query_prefix)
    chunker = Chunker(cfg.model_name, cfg.max_tokens, cfg.overlap_tokens, cfg.min_chunk_chars)

    store = Path(cfg.chroma_path)
    store.mkdir(parents=True, exist_ok=True)
    manifest_path = store / "ingest_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    pdfs = sorted(glob.glob(os.path.join(cfg.source_dir, "**", "*.pdf"), recursive=True))
    print(f"Found {len(pdfs)} PDFs under {cfg.source_dir}  (device={embedder.device})")

    new_chunks = 0
    for pdf in tqdm(pdfs, desc="Books"):
        rel = os.path.relpath(pdf, cfg.source_dir)
        fp = _fingerprint(pdf)
        if manifest.get(rel) == fp:
            continue
        try:
            pages, meta = extract_pdf(pdf)
        except Exception as e:  # noqa: BLE001
            tqdm.write(f"[skip] {rel}: {e}")
            continue

        chunks = chunker.chunk_pages(pages)
        if chunks:
            texts = [c["text"] for c in chunks]
            embs = embedder.embed_documents(texts, batch_size=cfg.batch_size).tolist()
            ids = [_doc_id(rel, i) for i in range(len(chunks))]
            metadatas = [
                {
                    "source": rel,
                    "title": meta["title"],
                    "page_start": c["page_start"],
                    "page_end": c["page_end"],
                    "chunk_index": i,
                }
                for i, c in enumerate(chunks)
            ]
            _upsert_batched(coll, ids, embs, texts, metadatas)
            new_chunks += len(chunks)

        manifest[rel] = fp
        manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Done. Added/updated {new_chunks} chunks. Collection holds {coll.count()} total.")
