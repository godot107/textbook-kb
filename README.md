# textbook-kb

Semantic search over your data-science textbook library, fully local.

PDFs → text (PyMuPDF) → token-aware chunks → **BGE-large** embeddings (GTX 1660) →
**ChromaDB** persistent vector store with rich metadata for filtering and citations.
Wired into **Claude Code** over MCP so an LLM can retrieve *and reason over* your
own books, with citations.

```
PDFs ─▶ extract ─▶ chunk ─▶ embed (BGE) ─▶ ChromaDB
                                              │
  Claude Code ◀─ MCP tools ◀─ search: floor ▸ rerank ▸ MMR ◀┘
```

📖 **The story:** [blog post](blog/from-textbooks-to-a-knowledge-base-i-can-reason-with.md) ·
☁️ **Going to prod:** [cloud deployment guide](docs/cloud-deployment.md)

## Setup

A project-local venv is recommended (CUDA torch shouldn't pollute the shared one):

```bash
cd projects/textbook-kb
python -m venv .venv && source .venv/bin/activate

# install the CUDA build of torch FIRST so sentence-transformers uses the GPU.
# (nvidia-smi's "CUDA Version: 13.0" is the driver's max, not a required wheel;
#  driver 580 runs any current wheel — cu128 is the newest channel offered.)
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Verify the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Configure

Edit `config.yaml`:

- `source_dir` — your PDFs (default `/media/williemaize/HDD/Textbooks`).
- `chroma_path` — the SATA SSD at `/mnt/ssd/textbook-kb` (NTFS, ~830 GB free).
  SQLite-WAL was smoke-tested OK there and the store is rebuildable, so that's fine.
- `models_dir` — model weights cache, co-located on the SSD (`/mnt/ssd/hf-cache`).
  The SSD is NTFS, so the code disables HF blob symlinks and writes real files.

## Use

```bash
python cli.py ingest                                  # build the index (resumable)
python cli.py query "what is regularization?" -k 8
python cli.py query "kernel trick" --source "ESL.pdf" # restrict to one book
python cli.py query "p-value" --contains "null hypothesis"  # require a keyword
python cli.py query "gradient descent" --no-rerank --no-mmr  # dense-only

python cli.py sources --filter "statistics"           # what's in the store
python cli.py toc "<source path>"                     # a book's table of contents
python cli.py audit --deep                            # duplicates / titles / garble
python cli.py dedup --apply                           # remove duplicate books
python cli.py eval                                    # measure retrieval quality
python cli.py info
```

## Retrieval pipeline

Search is multi-stage: ANN over-fetch → cosine relevance floor → **cross-encoder
rerank** (`bge-reranker-base`) → **MMR** diversification → top-k (all tunable in
`config.yaml`). The reranker degrades gracefully to dense-only if it can't load.
`python cli.py eval` scores baseline vs each stage on `eval/gold.yaml` (recall@k,
MRR) so changes are measured, not guessed.

## Data quality

`audit` reports duplicate books (detected by **shared chunk content**, not title),
junk embedded titles (fixed at display time via filename), and suspect/garbled
extraction. `dedup` removes redundant copies and records them in an exclude list
so re-ingest won't re-add them. A cached health check runs on MCP startup.

## Why these choices

### Storage: SSD vs M.2
For a corpus of ~tens of thousands of chunks the store is only a few hundred MB, and
Chroma loads its HNSW graph into RAM at query time. So queries are memory-bound and a
**SATA SSD performs essentially the same as NVMe** — your instinct is right. The only
cost of SATA is a marginally slower cold-start index load, irrelevant at this size.
(Your SATA SSD `sdb` isn't mounted yet — mount it and point `chroma_path` there.)

### Indexing & metadata for fast retrieval
- **HNSW (cosine).** Tuned in `config.yaml`: `ef_construction`/`max_neighbors` trade
  one-time build cost for recall; `ef_search` trades per-query latency for recall.
  These are set when the collection is **created** and can't be changed afterward —
  delete the collection to re-tune.
- **Metadata `where` filters** (`source`, `title`, `page_start/end`, `chunk_index`).
  Filtering by book/section narrows the candidate set, which is both faster and more
  precise than searching the whole corpus.
- **Keyword `where_document`** (`$contains`, and `$regex`/`$and`/`$or` if you extend it)
  gives a lightweight hybrid: combine semantic similarity with a hard keyword constraint.
- **Citations.** Every chunk stores its book title and page range, so results are
  traceable back to the source.

### Model
`bge-large-en-v1.5` is a solid default but capped at 512 tokens. If chunk fragmentation
hurts answers, swap `model_name` to a long-context retriever (`bge-m3`,
`nomic-embed-text-v1.5`) and raise `max_tokens`; nothing else changes.
