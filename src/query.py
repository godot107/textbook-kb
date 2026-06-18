"""Query the collection with optional metadata (`where`) and keyword
(`where_document`) filters, then format results with citations."""
from src.embed import Embedder
from src.ingest import get_collection


def query(cfg, text, k=5, source=None, contains=None):
    _, coll = get_collection(cfg)
    embedder = Embedder(cfg.model_name, cfg.device, cfg.use_fp16, cfg.query_prefix)
    qemb = embedder.embed_query(text).tolist()

    where = {"source": source} if source else None
    where_document = {"$contains": contains} if contains else None

    return coll.query(
        query_embeddings=[qemb],
        n_results=k,
        where=where,
        where_document=where_document,
        include=["documents", "metadatas", "distances"],
    )


def format_results(res):
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    if not docs:
        return "No results."

    out = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
        sim = 1.0 - dist  # cosine distance -> similarity
        title = meta.get("title", "?")
        ps, pe = meta.get("page_start"), meta.get("page_end")
        pages = f"p.{ps}" if ps == pe else f"pp.{ps}-{pe}"
        snippet = " ".join(doc.split())
        if len(snippet) > 400:
            snippet = snippet[:400] + "…"
        out.append(f"[{i}] {title} ({pages})  sim={sim:.3f}\n    {snippet}")
    return "\n\n".join(out)
