# Talk — Agentic RAG That Doesn't Hallucinate

Supplementary materials for the talk **"Agentic RAG That Doesn't Hallucinate:
Guardrails & Evaluation on the NVIDIA Stack."** Everything an attendee needs to
reproduce the results live.

| | |
|---|---|
| **Format** | ~25–30 min + Q&A · on-site or remote · English / Mandarin |
| **Level** | Intermediate — assumes basic RAG familiarity |
| **Audience** | Engineers shipping LLM apps; anyone who has to make an agent *dependable*, not just *demo-able* |
| **Artifact** | This repo — [`nim-agent-blueprint`](https://github.com/waynehacking8/nim-agent-blueprint) |

## The one-sentence thesis

You can drive an enterprise agent's hallucination rate on out-of-corpus questions from
**40% → 0%** with two cheap defenses and **no fine-tune** — and the only way you *know*
that is an evaluation that deliberately asks questions the corpus can't answer.

## What's in here

| file | what it is |
|---|---|
| [`slides.md`](slides.md) | The deck (Marp markdown). Speaker notes are in `<!-- -->` comments under each slide. |
| [`walkthrough.md`](walkthrough.md) | Hands-on: clone → run the eval → reproduce the guarded-vs-unguarded ablation yourself. |

## Build the slides to PDF / HTML

```bash
npm i -g @marp-team/marp-cli      # one-time
marp talk/slides.md -o slides.pdf      # or: --html, or: --preview
```

## Where the numbers come from

Every figure in the deck is read straight from [`../eval/report.md`](../eval/report.md),
produced by [`../eval/run_eval.py`](../eval/run_eval.py) on a self-hosted vLLM endpoint
(Qwen3-8B, an OpenAI-compatible NIM stand-in) + Ollama embeddings, on an H100. Re-run it
and the deck's claims regenerate. See [`walkthrough.md`](walkthrough.md).
