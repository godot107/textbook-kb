# textbook-kb

Semantic search over your data-science textbook library, fully local.

PDFs → text (PyMuPDF) → token-aware chunks → **BGE-large** embeddings (GTX 1660) →
**ChromaDB** persistent vector store with rich metadata for filtering and citations.

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
- `chroma_path` — set to `/media/williemaize/SSD/textbook-kb` (SATA SSD, already
  mounted, 831 GB free). It's NTFS; SQLite-WAL was smoke-tested OK there and the
  store is rebuildable from the PDFs, so that's fine.

## Use

```bash
python cli.py ingest                                  # build the index (resumable)
python cli.py query "what is regularization?" -k 8
python cli.py query "kernel trick" --source "ESL.pdf" # restrict to one book
python cli.py query "p-value" --contains "null hypothesis"  # require a keyword
python cli.py info
```

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
