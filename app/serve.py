"""FastAPI entrypoint. Loads a tiny corpus so /ask works out of the box."""
from fastapi import FastAPI
from pydantic import BaseModel
from app.retriever import HybridRetriever
from app.agent import answer

CORPUS = [
    "NVIDIA NIM packages optimized inference microservices with an OpenAI-compatible API.",
    "Tensor parallelism splits a model's weights across GPUs; all-reduce syncs activations.",
    "TensorRT-LLM compiles models to optimized engines; Triton serves them with in-flight batching.",
    "NeMo Agent Toolkit helps build and evaluate multi-step agentic workflows.",
]
app = FastAPI(title="NIM Agent Blueprint")
retriever = HybridRetriever(CORPUS)

class Q(BaseModel):
    q: str

@app.post("/ask")
def ask(body: Q):
    return answer(body.q, retriever)

if __name__ == "__main__":
    import uvicorn; uvicorn.run(app, host="0.0.0.0", port=9000)
