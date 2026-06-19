"""BGE embedding wrapper: handles device selection, fp16, normalization, and the
asymmetric query instruction prefix that BGE retrieval requires."""
import torch
from sentence_transformers import SentenceTransformer


def resolve_device(pref):
    if pref in ("cpu", "cuda"):
        return pref
    return "cuda" if torch.cuda.is_available() else "cpu"


class Embedder:
    def __init__(self, model_name, device="auto", use_fp16=True, query_prefix="",
                 cache_folder=None):
        self.device = resolve_device(device)
        self.model = SentenceTransformer(model_name, device=self.device,
                                         cache_folder=cache_folder)
        if use_fp16 and self.device == "cuda":
            self.model = self.model.half()
        self.query_prefix = query_prefix

    def embed_documents(self, texts, batch_size=16, show_progress=False):
        # No prefix on passages — that's the BGE convention.
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )

    def embed_query(self, text):
        emb = self.model.encode(
            [self.query_prefix + text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return emb[0]
