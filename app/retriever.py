"""Hybrid retrieval: dense (NIM embeddings) + keyword, min-max normalized and fused,
then optionally reranked by a NIM reranker.

Two stages: (1) dense+keyword fusion over-fetches `fetch_k` candidates; (2) if a NIM
reranker is available (opt in with `NIM_RERANK=1`), it reorders those candidates and we
keep the top-k. With no reranker (e.g. an Ollama-only setup) it falls back to the fusion
order, so the eval runs unchanged. Deliberately small/in-memory so the blueprint is
self-contained; swap the store for Milvus/pgvector in production (one class).
"""
import math, os, re
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

    def _fuse(self, query, alpha):
        """Min-max-normalized dense+keyword fusion -> passage indices, best-first."""
        qv = provider.embed([query])[0]
        qt = Counter(re.findall(r"\w+", query.lower()))
        dense = [self._cos(qv, self.vecs[i]) for i in range(len(self.passages))]
        kw = [sum(qt[w] * self.tok[i][w] for w in qt) / (sum(self.tok[i].values()) + 1)
              for i in range(len(self.passages))]
        dn, kn = self._minmax(dense), self._minmax(kw)
        scores = [(alpha * dn[i] + (1 - alpha) * kn[i], i) for i in range(len(self.passages))]
        return [i for _, i in sorted(scores, reverse=True)]

    def retrieve_idx(self, query, k=8, alpha=0.5):
        """Top-k passage indices: dense+keyword fusion, then NIM rerank if enabled.

        Stage 1 over-fetches `fetch_k` fusion candidates; stage 2 reranks them with a NIM
        reranker when `NIM_RERANK` is set and an endpoint is reachable, else returns the
        fusion top-k unchanged."""
        cand = self._fuse(query, alpha)
        fetch_k = max(k, int(os.getenv("RETRIEVE_FETCH_K", "20")))
        cand = cand[:fetch_k]
        if os.getenv("NIM_RERANK"):
            try:
                order = provider.rerank(query, [self.passages[i] for i in cand], top_k=k)
                return [cand[j] for j in order][:k]
            except Exception:
                pass  # no reranker reachable -> fall back to fusion order
        return cand[:k]

    def retrieve(self, query, k=8, alpha=0.5):
        return [self.passages[i] for i in self.retrieve_idx(query, k, alpha)]
