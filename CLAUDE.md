# textbook-kb

Local semantic-search knowledge base over a library of data-science textbook PDFs.
PDFs are extracted, chunked, embedded with **BAAI/bge-large-en-v1.5** (on a GTX 1660
via sentence-transformers), and stored in a **persistent ChromaDB** collection.

## Run

```bash
# one-time, from this directory, with the project venv active:
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python cli.py ingest                       # build/refresh the index (resumable)
python cli.py query "bias-variance tradeoff" -k 5
python cli.py info
```

## Layout

- `config.yaml` — all paths and tunables (model, chunking, HNSW).
- `src/extract.py` — PyMuPDF text extraction + de-hyphenation cleanup.
- `src/chunk.py` — token-aware, page-tracking chunker (480-token budget, 64 overlap).
- `src/embed.py` — BGE wrapper: fp16, normalize, **query-only instruction prefix**.
- `src/ingest.py` — orchestration, Chroma collection (cosine HNSW), resumable manifest.
- `src/query.py` — search with `where` (metadata) + `where_document` (keyword) filters.
- `cli.py` — `ingest` / `query` / `info`.

## Key decisions / constraints

- **GPU:** GTX 1660, 6 GB. bge-large fits in fp32; `use_fp16: true` adds headroom.
  No tensor cores, so fp16 saves memory but not much time. Lower `batch_size` on OOM.
- **BGE asymmetry:** queries get the `query_prefix`; stored passages do **not**.
  This is why we embed manually instead of using Chroma's built-in embedding fn.
- **Cosine space + normalized vectors** — set at collection creation; can't change later.
- **Storage:** PDFs on the HDD (slow, one-time read). Chroma store belongs on an SSD
  (`chroma_path`). The HNSW index is RAM-resident at query time, so SATA SSD ≈ NVMe here.
- **Context limit:** bge-large is 512 tokens. If chunk fragmentation hurts retrieval,
  switch `model_name` to a long-context model (bge-m3, nomic-embed-text-v1.5) and bump
  `max_tokens` — the pipeline is otherwise model-agnostic.
- **Resumability:** `chroma_path/ingest_manifest.json` tracks mtime+size per file;
  re-running `ingest` only processes new/changed PDFs (upsert keeps ids stable).
