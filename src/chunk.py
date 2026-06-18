"""Token-aware, page-tracking chunker.

Packs paragraphs into chunks under a token budget (bge-large maxes at 512),
carries a token overlap between consecutive chunks, and records the page range
each chunk spans so results can cite a page.
"""
import logging
import re

from transformers import AutoTokenizer

# The tokenizer warns ("Token indices sequence length is longer than ... 512")
# whenever we encode an oversized paragraph just to measure/split it. That is
# expected here — we split such paragraphs below — so quiet the noisy logger.
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

_PARA = re.compile(r"\n\s*\n")


class Chunker:
    def __init__(self, model_name, max_tokens=480, overlap_tokens=64, min_chunk_chars=64):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chunk_chars = min_chunk_chars

    def _ntok(self, text):
        return len(self.tok.encode(text, add_special_tokens=False))

    def _split_oversized(self, text):
        """Split a paragraph that alone exceeds the token budget into windows."""
        ids = self.tok.encode(text, add_special_tokens=False)
        if len(ids) <= self.max_tokens:
            return [text]
        out = []
        for i in range(0, len(ids), self.max_tokens):
            piece = self.tok.decode(ids[i:i + self.max_tokens]).strip()
            if piece:
                out.append(piece)
        return out

    def chunk_pages(self, pages):
        units = []
        for p in pages:
            for para in _PARA.split(p["text"]):
                para = para.strip()
                if not para:
                    continue
                for piece in self._split_oversized(para):
                    units.append({"page": p["page"], "text": piece, "ntok": self._ntok(piece)})

        chunks, cur, cur_tok = [], [], 0
        for u in units:
            if cur and cur_tok + u["ntok"] > self.max_tokens:
                chunks.append(self._emit(cur))
                cur, cur_tok = self._carry_overlap(cur)
                # if the overlap leaves no room for this unit, start clean so an
                # emitted chunk can never exceed max_tokens (stays under 512)
                if cur_tok + u["ntok"] > self.max_tokens:
                    cur, cur_tok = [], 0
            cur.append(u)
            cur_tok += u["ntok"]
        if cur:
            chunks.append(self._emit(cur))

        return [c for c in chunks if len(c["text"]) >= self.min_chunk_chars]

    def _carry_overlap(self, units):
        # carry only small trailing units as overlap; never force-carry a large
        # one (that would blow past the token budget on the next chunk)
        carried, tok = [], 0
        for u in reversed(units):
            if tok + u["ntok"] > self.overlap_tokens:
                break
            carried.insert(0, u)
            tok += u["ntok"]
        return carried, tok

    @staticmethod
    def _emit(units):
        return {
            "text": "\n\n".join(u["text"] for u in units),
            "page_start": min(u["page"] for u in units),
            "page_end": max(u["page"] for u in units),
        }
