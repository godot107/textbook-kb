"""Retrieval pipeline and store helpers.

search() runs: ANN over-fetch -> cosine floor -> optional cross-encoder rerank
-> optional MMR diversification -> top-k. The Retriever holds the embedder,
collection, and reranker resident so the MCP server pays load costs once.
"""
import numpy as np

from src.embed import Embedder
from src.ingest import get_collection
from src.rerank import Reranker
from src.titles import clean_title


class Retriever:
    def __init__(self, cfg):
        self.cfg = cfg
        _, self.coll = get_collection(cfg)
        self.embedder = Embedder(
            cfg.model_name, cfg.device, cfg.use_fp16, cfg.query_prefix,
            cache_folder=cfg.models_dir,
        )
        self.reranker = Reranker(
            cfg.rerank_model,
            device=self.embedder.device,
            cache_folder=cfg.models_dir,
            batch_size=cfg.rerank_batch_size,
        ) if cfg.rerank_enabled else None

    # -- main search -------------------------------------------------------
    def search(self, text, k=5, source=None, contains=None,
               rerank=None, mmr=None, min_sim=None):
        cfg = self.cfg
        rerank = cfg.rerank_enabled if rerank is None else rerank
        mmr = cfg.mmr_enabled if mmr is None else mmr
        min_sim = cfg.min_sim if min_sim is None else min_sim
        cand_k = max(cfg.candidate_k, k)

        qemb = self.embedder.embed_query(text)
        where = {"source": source} if source else None
        where_document = {"$contains": contains} if contains else None
        res = self.coll.query(
            query_embeddings=[qemb.tolist()],
            n_results=cand_k,
            where=where,
            where_document=where_document,
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        hits = self._to_hits(res)
        hits = [h for h in hits if h["sim"] >= min_sim]
        if not hits:
            return []

        # Stage 2: cross-encoder rerank (sets each hit's primary score).
        if rerank and self.reranker is not None:
            scores = self.reranker.rerank(text, [h["text"] for h in hits])
            if scores is not None:
                for h, s in zip(hits, scores):
                    h["rerank"] = s
                hits.sort(key=lambda h: h["rerank"], reverse=True)

        # Stage 3: MMR diversification (counters near-duplicate passages).
        if mmr and len(hits) > k:
            hits = self._mmr(qemb, hits, k, cfg.mmr_lambda)
        else:
            hits = hits[:k]
        return hits

    def _to_hits(self, res):
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        embs = (res.get("embeddings") or [[]])[0]
        hits = []
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
            hits.append({
                "text": doc,
                "meta": meta,
                "sim": 1.0 - dist,            # cosine distance -> similarity
                "rerank": None,
                "emb": np.asarray(embs[i], dtype=np.float32) if len(embs) else None,
                "title": clean_title(meta.get("source", ""), meta.get("title", "")),
                "source": meta.get("source", "?"),
                "page_start": meta.get("page_start"),
                "page_end": meta.get("page_end"),
            })
        return hits

    @staticmethod
    def _mmr(qemb, hits, k, lam):
        """Maximal Marginal Relevance. Relevance = primary score (rerank if set,
        else cosine), min-max normalized so it mixes with the cosine diversity
        penalty on a comparable [0,1] scale."""
        if any(h["emb"] is None for h in hits):
            return hits[:k]
        rel = np.array([h["rerank"] if h["rerank"] is not None else h["sim"]
                        for h in hits], dtype=np.float32)
        lo, hi = float(rel.min()), float(rel.max())
        rel = (rel - lo) / (hi - lo) if hi > lo else np.ones_like(rel)
        D = np.vstack([h["emb"] for h in hits])
        D /= (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
        sim_mat = D @ D.T  # pairwise cosine among candidates

        selected, remaining = [], list(range(len(hits)))
        while remaining and len(selected) < k:
            if not selected:
                j = int(rel[remaining].argmax())
                selected.append(remaining.pop(j))
                continue
            best, best_score = None, -1e9
            for idx in remaining:
                penalty = max(sim_mat[idx][s] for s in selected)
                score = lam * rel[idx] - (1 - lam) * penalty
                if score > best_score:
                    best, best_score = idx, score
            selected.append(best)
            remaining.remove(best)
        return [hits[i] for i in selected]

    # -- context expansion -------------------------------------------------
    def expand_context(self, source, chunk_index=None, page=None, window=1):
        """Fetch neighbouring chunks around a hit (small-to-big retrieval).

        Give either chunk_index (preferred) or a page. Returns the stitched text
        of chunks [i-window, i+window] from the same source, in reading order.
        """
        if chunk_index is None and page is None:
            raise ValueError("expand_context needs chunk_index or page")
        if chunk_index is None:
            anchor = self.coll.get(
                where={"$and": [{"source": source},
                                {"page_start": {"$lte": page}},
                                {"page_end": {"$gte": page}}]},
                include=["metadatas"], limit=1,
            )
            metas = anchor.get("metadatas") or []
            if not metas:
                return None
            chunk_index = metas[0].get("chunk_index", 0)
        lo, hi = max(0, chunk_index - window), chunk_index + window
        got = self.coll.get(
            where={"$and": [{"source": source},
                            {"chunk_index": {"$gte": lo}},
                            {"chunk_index": {"$lte": hi}}]},
            include=["documents", "metadatas"],
        )
        rows = sorted(
            zip(got.get("documents", []), got.get("metadatas", [])),
            key=lambda r: r[1].get("chunk_index", 0),
        )
        if not rows:
            return None
        text = "\n\n".join(d for d, _ in rows)
        metas = [m for _, m in rows]
        return {
            "source": source,
            "title": clean_title(source, metas[0].get("title", "")),
            "page_start": min(m.get("page_start") for m in metas),
            "page_end": max(m.get("page_end") for m in metas),
            "chunk_index_range": [lo, hi],
            "text": text,
        }


def format_hits(hits):
    """Render search hits with citations (clean title, page range, scores)."""
    if not hits:
        return "No results above the relevance floor."
    out = []
    for i, h in enumerate(hits, 1):
        ps, pe = h["page_start"], h["page_end"]
        pages = f"p.{ps}" if ps == pe else f"pp.{ps}-{pe}"
        score = f"sim={h['sim']:.3f}"
        if h.get("rerank") is not None:
            score += f"  rerank={h['rerank']:.2f}"
        snippet = " ".join(h["text"].split())
        if len(snippet) > 400:
            snippet = snippet[:400] + "…"
        out.append(f"[{i}] {h['title']} ({pages})  {score}\n"
                   f"    source: {h['source']}\n    {snippet}")
    return "\n\n".join(out)
