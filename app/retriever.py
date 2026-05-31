"""Hybrid retrieval: dense (NIM embeddings) + keyword, fused, then NIM reranked.

Deliberately small/in-memory so the blueprint is self-contained; swap the store for
Milvus/pgvector in production (one class).
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

    def retrieve_idx(self, query, k=8, alpha=0.5):
        """Indices of the top-k passages, fused dense+keyword (for recall@k eval)."""
        qv = provider.embed([query])[0]
        qt = Counter(re.findall(r"\w+", query.lower()))
        scores = []
        for i, p in enumerate(self.passages):
            dense = self._cos(qv, self.vecs[i])
            kw = sum(qt[w] * self.tok[i][w] for w in qt) / (sum(self.tok[i].values()) + 1)
            scores.append((alpha*dense + (1-alpha)*kw, i))
        return [i for _, i in sorted(scores, reverse=True)[:k]]   # reranker hook lands here

    def retrieve(self, query, k=8, alpha=0.5):
        return [self.passages[i] for i in self.retrieve_idx(query, k, alpha)]
