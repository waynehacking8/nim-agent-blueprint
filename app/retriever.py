"""Hybrid retrieval: dense (NIM embeddings) + keyword, min-max normalized and fused.

A NIM reranker is the intended next stage (see the `# rerank hook` in retrieve_idx) but is
NOT wired in this build — retrieval here is dense+keyword fusion only. Deliberately
small/in-memory so the blueprint is self-contained; swap the store for Milvus/pgvector in
production (one class).
"""
import math, re
from collections import Counter
from app import provider

class HybridRetriever:
    def __init__(self, passages):
        self.passages = passages
        self.vecs = provider.embed(passages)            # dense index
        self.tok = [Counter(re.findall(r"\w+", p.lower())) for p in passages]  # keyword index

    @staticmethod
    def _cos(a, b):
        dot = sum(x*y for x, y in zip(a, b))
        na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
        return dot / (na*nb + 1e-9)

    @staticmethod
    def _minmax(xs):
        """Scale a score vector to [0, 1] so dense and keyword channels are comparable
        before fusion. Without this, dense cosine (~0.3–0.8) dwarfs the keyword score
        (~0.0x) and the 'hybrid' collapses to dense-only."""
        lo, hi = min(xs), max(xs)
        rng = hi - lo
        return [(x - lo) / rng for x in xs] if rng > 1e-9 else [0.0 for _ in xs]

    def retrieve_idx(self, query, k=8, alpha=0.5):
        """Indices of the top-k passages, fused dense+keyword (for recall@k eval).
        Each channel is min-max normalized across the corpus before the alpha blend."""
        qv = provider.embed([query])[0]
        qt = Counter(re.findall(r"\w+", query.lower()))
        dense = [self._cos(qv, self.vecs[i]) for i in range(len(self.passages))]
        kw = [sum(qt[w] * self.tok[i][w] for w in qt) / (sum(self.tok[i].values()) + 1)
              for i in range(len(self.passages))]
        dn, kn = self._minmax(dense), self._minmax(kw)
        scores = [(alpha * dn[i] + (1 - alpha) * kn[i], i) for i in range(len(self.passages))]
        return [i for _, i in sorted(scores, reverse=True)[:k]]   # rerank hook (not wired) lands here

    def retrieve(self, query, k=8, alpha=0.5):
        return [self.passages[i] for i in self.retrieve_idx(query, k, alpha)]
