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

def chat(messages, model=None, max_tokens=512, temperature=0.0):
    model = model or os.getenv("NIM_LLM_MODEL", "meta/llama-3.1-70b-instruct")
    r = httpx.post(f"{_base()['llm']}/chat/completions", headers=_headers(), timeout=120,
                   json={"model": model, "messages": messages,
                         "max_tokens": max_tokens, "temperature": temperature})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def embed(texts, model=None):
    model = model or os.getenv("NIM_EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")
    r = httpx.post(f"{_base()['embed']}/embeddings", headers=_headers(), timeout=120,
                   json={"model": model, "input": texts, "input_type": "query"})
    r.raise_for_status()
    return [d["embedding"] for d in r.json()["data"]]
