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
python cli.py query "gradient boosting" --no-rerank   # disable cross-encoder
python cli.py sources --filter "data mining"  # list books + exact source paths
python cli.py audit [--deep]                # data-quality report (dups/titles/garble)
python cli.py dedup [--apply]               # remove duplicate books (content-based)
python cli.py toc "<source path>"           # a book's table of contents
python cli.py eval                          # baseline vs full pipeline (recall@k, MRR)
python cli.py info
```

### Query from Claude Code (MCP)

`mcp_server.py` exposes the KB to Claude Code over stdio (registered in the
workspace-root `.mcp.json`). Tools: `search_textbooks`, `list_sources`,
`get_toc`, `expand_context`, `data_quality_report`, `collection_info`. The
embedder, collection, and reranker load **once per session** (lazily, on first
call) and stay resident — no per-query cold start. A cached data-quality health
check runs on load and warns (to stderr) when the corpus has issues. Restart
`claude` and approve the server on first use; check it with `/mcp`.

## Retrieval pipeline

`search_textbooks` runs: ANN over-fetch (`candidate_k`) → cosine **relevance
floor** (`min_sim`, so it can honestly return nothing) → **cross-encoder rerank**
(opt-in; `bge-reranker-base`, degrades to dense-only if the model can't load) →
**MMR** diversification (`mmr_lambda`, stops near-duplicate passages crowding the
top-k) → top-k. All tunable under `retrieval:` / `rerank:` in `config.yaml`.

**Reranker is OFF by default.** On the current gold set it regressed MRR
(0.596→0.525), but that metric matches the *source book* and can't see
content-precision gains, so it's unproven, not disproven. Enable per call
(`--rerank`, or `rerank=True` on the tool). **TODO before defaulting it on:**
switch to `bge-reranker-v2-m3` and build a passage/relevance-level gold set
(current `eval/gold.yaml` is source-level and saturated at 0.93 recall). MMR is
on by default (metric-neutral, and it fixes the near-duplicate-passage problem
we actually observed).

## Layout

- `config.yaml` — all paths and tunables (model cache, chunking, HNSW, retrieval, quality).
- `src/extract.py` — PyMuPDF text extraction + de-hyphenation cleanup.
- `src/chunk.py` — token-aware, page-tracking chunker (480-token budget, 64 overlap).
- `src/embed.py` — BGE wrapper: fp16, normalize, **query-only instruction prefix**, `cache_folder`.
- `src/rerank.py` — lazy cross-encoder reranker with graceful fallback.
- `src/retrieve.py` — `Retriever`: over-fetch → floor → rerank → MMR; `expand_context`; `format_hits`.
- `src/store.py` — whole-collection scans: `list_sources`, `source_stats`, `content_duplicate_groups`.
- `src/titles.py` — clean display titles from filenames (embedded PDF titles are unreliable).
- `src/quality.py` — data-quality audit + cached on-load health check.
- `src/dedup.py` — content-based duplicate detection/removal + ingest exclude list.
- `src/toc.py` — PDF outline extraction + page→section mapping.
- `src/evaluate.py` — gold-set eval (recall@k, MRR); gold in `eval/gold.yaml`.
- `src/ingest.py` — orchestration, Chroma collection (cosine HNSW), resumable manifest,
  exclude-list + intra-book chunk dedup.
- `src/query.py` — legacy single-stage search (kept; CLI/MCP now use `retrieve.py`).
- `cli.py` — `ingest`/`query`/`sources`/`audit`/`dedup`/`toc`/`eval`/`info`.
- `mcp_server.py` — stdio MCP server; resident `Retriever` + on-load health check.

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
