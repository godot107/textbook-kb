# Cloud deployment: hosting the vector DB and a query endpoint

This project runs fully local. To make the knowledge base reachable from
anywhere (other tools, a web UI, teammates), you lift three things off the
desktop and into the cloud. Keep them separate — they scale and cost differently:

1. **The vector store** — where embeddings live and similarity search runs.
2. **The embedding model** — turns a query string into a vector (BGE, asymmetric: queries get the instruction prefix).
3. **The serving API** — an HTTP endpoint that ties them together with auth.

```
client ─▶ HTTPS ─▶ [ API service ]
                      │  embed query (BGE)         ┌─ managed vector DB ─┐
                      └─ similarity search ───────▶│  Chroma/Qdrant/etc. │
                         (+ floor/rerank/MMR)      └─────────────────────┘
```

---

## Step 1 — choose where the vectors live

You have ~78k vectors × 1024 dims × 4 bytes ≈ **320 MB** of raw vectors. That is
small. Almost any option has a free or cheap tier.

| Option | What it is | Good when |
|---|---|---|
| **Chroma Cloud** | Hosted version of what you already run | Smallest migration — same client/API |
| **Qdrant Cloud** | Fast Rust vector DB, generous free tier | You want a clean managed API + filtering |
| **Pinecone** | Fully managed, serverless | You want zero ops and don't mind vendor lock-in |
| **Weaviate Cloud** | Managed, hybrid search built in | You want native BM25 + dense hybrid |
| **pgvector** on Supabase/Neon/RDS | Postgres extension | You already run Postgres and want one system |

**Recommended for this project:** **Qdrant Cloud** (free tier fits 320 MB
easily) or **Chroma Cloud** (least code change). Both are managed, so you skip
running a database yourself.

### Self-hosting instead (full control)

Chroma also runs in server mode on any VM/container:

```bash
chroma run --host 0.0.0.0 --port 8000 --path /data/textbook-kb
# then connect with chromadb.HttpClient(host=..., port=8000)
```

Put it on a small VM (Fly.io, a $5 VPS, a Cloud Run service with a volume) behind
the API in Step 3. More control, more ops. For a portfolio piece, managed is the
better use of time.

---

## Step 2 — migrate the local store to the cloud

Read every chunk + vector out of the local Chroma collection and upsert it into
the cloud store. You already have the vectors, so **no re-embedding is needed** —
this is a copy, and it's fast.

```python
# migrate.py — local Chroma -> Qdrant Cloud (sketch)
import chromadb
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.config import Config

cfg = Config("config.yaml")
local = chromadb.PersistentClient(path=cfg.chroma_path).get_collection(cfg.collection_name)

qc = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
qc.recreate_collection("textbooks",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE))

total, off, step = local.count(), 0, 1000
while off < total:
    got = local.get(include=["embeddings", "documents", "metadatas"],
                    limit=step, offset=off)
    qc.upsert("textbooks", points=[
        PointStruct(id=off + i, vector=emb,
                    payload={"text": doc, **meta})
        for i, (emb, doc, meta) in enumerate(
            zip(got["embeddings"], got["documents"], got["metadatas"]))])
    off += step
```

(Chroma Cloud is even simpler: same `chromadb` API, swap `PersistentClient` for
`CloudClient` and upsert in batches.)

---

## Step 3 — the embedding model in the cloud

The query path still needs to embed the incoming question with **the same model**
and the **query instruction prefix** (BGE asymmetry — see `src/embed.py`). Three
ways to host it, cheapest-effort first:

- **Hosted embedding API.** Swap BGE for a hosted embedding endpoint (e.g. a
  managed `bge`/`e5` endpoint, or a provider's embedding model). No GPU to run.
  Note: you must re-embed the whole corpus with the *same* model you query with,
  so pick this **before** migrating vectors if you go this route.
- **CPU in the API container.** `bge-large` runs on CPU at ~hundreds of ms per
  query — fine for low traffic. Zero extra infra; just slower than your GPU.
- **Dedicated GPU inference** (Modal, Replicate, a GPU VM). Fastest, priciest.
  Worth it only under real load.

**Recommended:** start with CPU embedding in the API container (simple, keeps
your existing vectors), move to a GPU/hosted embedder only if latency hurts.

---

## Step 4 — the serving API (FastAPI)

A thin HTTP wrapper around the existing retrieval logic. Reuses `Retriever` so
the floor/rerank/MMR pipeline behaves exactly as it does locally.

```python
# serve/app.py — query endpoint (template)
import os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from src.config import Config
from src.retrieve import Retriever, format_hits

cfg = Config("config.yaml")          # point chroma_path at the cloud store
retr = Retriever(cfg)                # loads embedder (+ reranker) once
app = FastAPI(title="textbook-kb")
API_KEY = os.environ["KB_API_KEY"]

class Query(BaseModel):
    query: str
    k: int = 5
    rerank: bool | None = None

@app.post("/search")
def search(q: Query, x_api_key: str = Header(default="")):
    if x_api_key != API_KEY:
        raise HTTPException(401, "bad api key")
    hits = retr.search(q.query, k=q.k, rerank=q.rerank)
    return {"results": [
        {"title": h["title"], "source": h["source"],
         "page_start": h["page_start"], "page_end": h["page_end"],
         "sim": h["sim"], "text": h["text"]} for h in hits]}

@app.get("/healthz")
def healthz():
    return {"ok": True, "chunks": retr.coll.count()}
```

```dockerfile
# serve/Dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt serve/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r serve/requirements.txt
COPY . .
ENV CHROMA_REMOTE=1
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8080"]
```

`serve/requirements.txt`: `fastapi`, `uvicorn[standard]`, plus the cloud client
(`qdrant-client` or `chromadb`).

### Deploy targets

- **Render / Railway / Fly.io** — push the Dockerfile, set env vars
  (`KB_API_KEY`, the vector-DB URL + key), done. Best for a portfolio.
- **Google Cloud Run** — serverless containers, scales to zero (cheap when idle).
- **A small VM** — most control, most ops.

### Make it production-ish

- **Auth:** the `X-API-Key` header above is the minimum. Rotate the key; never commit it.
- **CORS:** add `fastapi.middleware.cors.CORSMiddleware` if a browser calls it.
- **Rate limiting:** `slowapi` or your platform's gateway.
- **Cost control:** scale-to-zero (Cloud Run) if traffic is bursty; the reranker
  is the heaviest component — leave it off (the default) unless you need it.

---

## Recommended path for this project

1. **Qdrant Cloud** free tier for the vectors (or Chroma Cloud for least change).
2. `migrate.py` to copy the existing 78k vectors up — no re-embedding.
3. **FastAPI on Render/Fly.io**, embedding on CPU to start, API-key auth.
4. Point Claude Code's MCP tool (or any client) at the public `/search` endpoint.

That turns the local project into a service you can reach anywhere and show off —
without changing the retrieval logic that already works.
