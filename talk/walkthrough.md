# Walkthrough — reproduce the talk's results

Hands-on companion to **"Agentic RAG That Doesn't Hallucinate."** By the end you'll have
regenerated [`../eval/report.md`](../eval/report.md) yourself and watched the
guarded-vs-unguarded ablation flip hallucination from **0% → 40%**.

Budget: ~5–10 minutes once an endpoint is up.

---

## 0. What you need

One **OpenAI-compatible** chat + embeddings endpoint. Any of these works — NIM is
OpenAI-compatible, so a local server is a faithful stand-in:

| option | LLM | embeddings | needs |
|---|---|---|---|
| **Local (no key)** | vLLM or Ollama | Ollama | a GPU, or a small model on CPU |
| **Hosted NIM** | `build.nvidia.com` | `build.nvidia.com` | a free NGC API key |

---

## 1. Clone + install

```bash
git clone https://github.com/waynehacking8/nim-agent-blueprint
cd nim-agent-blueprint
pip install -r requirements.txt
```

## 2. Point the provider at your endpoint

```bash
# Local (example: vLLM serving Qwen3-8B on :8000, Ollama embeddings on :11434)
export NIM_MODE=selfhost
export NIM_LLM_URL=http://localhost:8000/v1
export NIM_EMBED_URL=http://localhost:11434/v1

# …or hosted NIMs:
# export NIM_MODE=hosted NVIDIA_API_KEY=nvapi-...
```

> The whole point of `app/provider.py` is that nothing downstream changes when you switch.
> Hosted vs self-hosted is one env var.

## 3. Run the eval

```bash
python eval/run_eval.py        # writes eval/report.md
```

You'll get the five blocks from the talk:

1. retrieval **recall@3** (answerable)
2. answer **accuracy** (answerable)
3. **hallucination** rate — **guarded vs unguarded** on the out-of-corpus questions
4. the `validate()` gate scored as a hallucination **detector** (precision / recall / F1)
5. per-hop **latency** budget

Compare your `eval/report.md` against the committed one — same shape, your numbers.

---

## 4. See the defense actually work

The headline claim is an **ablation**. The harness already runs both conditions per
question (`guarded=True` and `guarded=False`) — look at the two hallucination columns in
the per-question table at the bottom of `eval/report.md`.

The single instruction that does the work lives in [`../app/agent.py`](../app/agent.py):

```python
GUARDED_SYS   = "Answer using ONLY the context. Cite [n]. If unsupported, say you don't know."
UNGUARDED_SYS = "You are a helpful assistant. Answer the question."   # the ablation
```

**Try it:** find an out-of-corpus question in `eval/dataset.jsonl` (e.g. one of the
B200/H200/FP4 near-misses), and run it both ways:

```python
from app.retriever import HybridRetriever
from app.agent import answer_traced
import json

corpus = [json.loads(l)["text"] for l in open("eval/corpus.jsonl")]
r = HybridRetriever(corpus)

q = "What is the FP4 tensor-core throughput of an H200?"   # not in the corpus
print("guarded  :", answer_traced(q, r, guarded=True)["answer"])
print("unguarded:", answer_traced(q, r, guarded=False)["answer"])
```

Guarded should **decline**; unguarded will often **make something up** from the H100 facts
in context. That single contrast is the whole talk.

---

## 5. Where to look next

- [`../app/agent.py`](../app/agent.py) — `plan → retrieve → generate → validate`
- [`../eval/run_eval.py`](../eval/run_eval.py) — exactly how each metric is computed (incl. `is_hallucination`)
- [`../docs/design-decisions.md`](../docs/design-decisions.md) — why guarded-by-default, why a judge gate
- [`../docs/roadmap.md`](../docs/roadmap.md) — hardening the gate, hosted-NIM run, multi-hop retrieval
