"""Cross-encoder reranker.

A bi-encoder (BGE) embeds query and passage independently, so ranking is a
coarse vector similarity. A cross-encoder scores each (query, passage) pair with
full cross-attention — much more precise, but too slow to run over the whole
corpus. So we use it as a *second stage*: pull N candidates with the fast ANN
index, then rerank only those.

Loaded lazily on first use and held resident. If the model can't be loaded
(offline, missing, OOM), `available` stays False and callers fall back to
dense-only ranking instead of failing the query.
"""
import logging
import sys

log = logging.getLogger("textbook-kb.rerank")


class Reranker:
    def __init__(self, model_name, device=None, cache_folder=None, batch_size=32,
                 max_length=512):
        self.model_name = model_name
        self.device = device
        self.cache_folder = cache_folder
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None
        self._tried = False
        self.available = False

    def _load(self):
        if self._tried:
            return
        self._tried = True
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device,
                cache_folder=self.cache_folder,
            )
            self.available = True
        except Exception as e:  # noqa: BLE001 — degrade, never crash a query
            print(f"[rerank] disabled ({type(e).__name__}: {e}); "
                  f"falling back to dense ranking", file=sys.stderr)
            self.available = False

    def rerank(self, query, passages):
        """Return reranker scores aligned with `passages`, or None if unavailable."""
        self._load()
        if not self.available or not passages:
            return None
        pairs = [(query, p) for p in passages]
        scores = self._model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )
        return [float(s) for s in scores]
