"""NIM provider layer.

One switch (NIM_MODE) selects hosted build.nvidia.com endpoints or self-hosted NIMs.
Everything downstream (retriever, agent) is written against this interface, so the same
blueprint runs on a laptop (hosted) or on the H100 box (self-hosted) unchanged.
"""
import os, httpx

HOSTED = {
    "llm":   "https://integrate.api.nvidia.com/v1",
    "embed": "https://integrate.api.nvidia.com/v1",
    "rerank":"https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking",
}
SELFHOST = {
    "llm":   os.getenv("NIM_LLM_URL",    "http://localhost:8000/v1"),
    "embed": os.getenv("NIM_EMBED_URL",  "http://localhost:8001/v1"),
    "rerank":os.getenv("NIM_RERANK_URL", "http://localhost:8002/v1/ranking"),
}

def _base():
    return HOSTED if os.getenv("NIM_MODE", "hosted") == "hosted" else SELFHOST

def _headers():
    key = os.getenv("NVIDIA_API_KEY", "")
    return {"Authorization": f"Bearer {key}"} if key else {}

def chat(messages, model=None, max_tokens=512, temperature=0.0, base_url=None):
    """`model`/`base_url` overrides let a single agent talk to more than one serving
    endpoint — e.g. a generator on one GPU and an independent judge model (different
    model family) on another. Defaults preserve the single-endpoint behavior."""
    model = model or os.getenv("NIM_LLM_MODEL", "meta/llama-3.1-70b-instruct")
    body = {"model": model, "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature}
    # Hybrid-reasoning models (e.g. Qwen3) emit <think> blocks that blow short prompt
    # budgets and corrupt structured replies. Disable for the agent's terse calls.
    # (Models whose chat template has no enable_thinking variable simply ignore it.)
    if os.getenv("NIM_DISABLE_THINKING"):
        body["chat_template_kwargs"] = {"enable_thinking": False}
    r = httpx.post(f"{base_url or _base()['llm']}/chat/completions", headers=_headers(),
                   timeout=120, json=body)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def embed(texts, model=None):
    model = model or os.getenv("NIM_EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")
    payload = {"model": model, "input": texts}
    # NVIDIA NIM embedding models require input_type ("query"/"passage"); generic
    # OpenAI-compatible endpoints (Ollama, OpenAI) reject the extra field. Opt in via env.
    itype = os.getenv("NIM_EMBED_INPUT_TYPE")
    if itype:
        payload["input_type"] = itype
    r = httpx.post(f"{_base()['embed']}/embeddings", headers=_headers(), timeout=120,
                   json=payload)
    r.raise_for_status()
    return [d["embedding"] for d in r.json()["data"]]

def rerank(query, passages, model=None, top_k=None):
    """Rerank `passages` against `query` via a NIM reranking endpoint.

    Returns passage indices (into the input list) best-first. The rerank URL in `_base()`
    is already the full endpoint, so we POST to it directly. Raises on HTTP error so the
    caller can fall back to the fusion order when no reranker is available.
    """
    model = model or os.getenv("NIM_RERANK_MODEL", "nvidia/llama-3.2-nv-rerankqa-1b-v2")
    body = {"model": model, "query": {"text": query},
            "passages": [{"text": p} for p in passages]}
    r = httpx.post(_base()["rerank"], headers=_headers(), timeout=120, json=body)
    r.raise_for_status()
    rankings = r.json()["rankings"]          # [{"index": i, "logit": s}, ...] best-first
    idx = [d["index"] for d in rankings]
    return idx[:top_k] if top_k else idx
