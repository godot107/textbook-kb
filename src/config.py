"""Configuration loader. No heavy imports here so the CLI stays snappy."""
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
