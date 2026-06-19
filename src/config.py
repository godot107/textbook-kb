"""Configuration loader. No heavy imports here so the CLI stays snappy."""
import os
from pathlib import Path

import yaml


class Config:
    def __init__(self, path):
        self.path = Path(path).resolve()
        base = self.path.parent
        d = yaml.safe_load(self.path.read_text())

        def resolve(p):
            p = Path(p)
            return str(p if p.is_absolute() else (base / p).resolve())

        self.source_dir = resolve(d["source_dir"])
        self.chroma_path = resolve(d["chroma_path"])
        self.collection_name = d.get("collection_name", "textbooks")

        # Model cache. When set, both the embedder and reranker load/store weights
        # here instead of ~/.cache/huggingface. The store SSD is NTFS, so we also
        # disable HF blob symlinks process-wide (real files load fine over ntfs-3g;
        # symlinks do not). Set the env here, before any HF import downstream.
        self.models_dir = resolve(d["models_dir"]) if d.get("models_dir") else None
        if self.models_dir:
            os.environ.setdefault("HF_HOME", self.models_dir)
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

        self.model_name = d.get("model_name", "BAAI/bge-large-en-v1.5")
        self.device = d.get("device", "auto")
        self.query_prefix = d.get("query_prefix", "")
        self.batch_size = int(d.get("batch_size", 16))
        self.use_fp16 = bool(d.get("use_fp16", True))

        self.max_tokens = int(d.get("max_tokens", 480))
        self.overlap_tokens = int(d.get("overlap_tokens", 64))
        self.min_chunk_chars = int(d.get("min_chunk_chars", 64))

        self.hnsw = d.get("hnsw", {
            "space": "cosine",
            "ef_construction": 200,
            "max_neighbors": 32,
            "ef_search": 80,
        })

        r = d.get("retrieval", {})
        self.candidate_k = int(r.get("candidate_k", 40))
        self.min_sim = float(r.get("min_sim", 0.30))
        self.mmr_lambda = float(r.get("mmr_lambda", 0.6))
        self.mmr_enabled = bool(r.get("mmr_enabled", True))

        rr = d.get("rerank", {})
        self.rerank_enabled = bool(rr.get("enabled", True))
        self.rerank_model = rr.get("model_name", "BAAI/bge-reranker-base")
        self.rerank_batch_size = int(rr.get("batch_size", 32))

        q = d.get("quality", {})
        self.quality_max_age_days = float(q.get("max_age_days", 7))
        self.quality_check_on_load = bool(q.get("check_on_load", True))
