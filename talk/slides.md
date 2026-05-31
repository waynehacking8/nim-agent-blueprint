---
marp: true
theme: default
paginate: true
title: Agentic RAG That Doesn't Hallucinate
description: Guardrails & evaluation on the NVIDIA stack
---

<!--
Speaker notes live in HTML comments like this one, under each slide.
Total target: ~25–30 min. Rough pacing is noted per section.
Build: marp talk/slides.md -o slides.pdf
-->

# Agentic RAG That Doesn't Hallucinate
### Guardrails & Evaluation on the NVIDIA Stack

**Wei Cheng (Wayne) Chiu**
AI Solutions Engineer
github.com/waynehacking8

<!--
~30s. One line about me: I ship enterprise multi-agent systems from PoC to production;
this talk is the reliability half of that job — the part that decides whether a demo
survives contact with real users. Everything here is in one public repo you can run today.
-->

---

## The demo that passes, then fails in production

- Your RAG agent answers the 20 questions you tried. Ship it.
- A user asks something **your corpus doesn't cover.**
- A confident, fluent, **wrong** answer comes back.
- In an enterprise setting that's not a bug — it's a **trust incident.**

> The hard part of agents isn't getting an answer. It's knowing when to **refuse.**

<!--
~2 min. Land the framing: the failure mode that matters in enterprise isn't "wrong on a
hard in-domain question" — it's "made something up on a question it should have declined."
This is the #1 thing that blocks a PoC from going to production. Ask the room: how many of
you measure your agent's behavior on questions it CAN'T answer? Usually few hands.
-->

---

## What "hallucination" means in this talk

Not vibes — a measurable event:

> The agent **asserts** an answer (doesn't abstain) on a question the
> **retrieval corpus cannot support.**

The nasty case is the **adversarial near-miss**:

- Corpus has **H100** facts.
- I ask about **B200 / H200 / FP4.**
- The model is *tempted* to answer from the H100 facts sitting in context.

<!--
~2 min. Make the definition concrete and operational — this is what we'll score. The
near-miss is the part naive evals miss: a totally out-of-domain question ("capital of
Australia") is easy to refuse; the dangerous one looks 90% like something in the corpus.
This is why the eval set is built on purpose to include these.
-->

---

## The blueprint: plan → retrieve → generate → validate

```
 query ─▶ planner ─▶ hybrid retriever ─▶ generator ─▶ validator ─▶ answer
            │             │                 │             │
        NIM LLM    NIM embed + rerank    NIM LLM      NIM LLM (judge)
```

- **NIM-backed** — one provider switch: hosted `build.nvidia.com` ↔ self-hosted NIM/vLLM.
- **`validate()` is the differentiator** — every answer is checked for groundedness
  *before* it's returned.
- OpenTelemetry-traced, so each hop is observable.

<!--
~2 min. Walk the loop left to right. Emphasize that plan/generate/validate are all LLM
calls through the same provider layer, so the whole thing runs identically against a
hosted NIM or a self-hosted vLLM on your own H100 — that's the "OpenAI-compatible" point.
validate() is what the rest of the talk is about.
-->

---

## Two cheap defenses. No fine-tune.

**Defense 1 — guarded generator prompt** *(first line)*

> "Answer using ONLY the context. Cite [n]. If unsupported, say you don't know."

**Defense 2 — `validate()` groundedness gate** *(safety net)*

> An LLM-as-judge checks the answer against the retrieved context and
> **blocks** unsupported ones.

Both are prompt-level. Zero training. The question is: **how much do they actually buy?**

<!--
~2 min. The whole pitch is "defense in depth with two prompt-level tools." No fine-tune is
a deliberate selling point for partners who can't/won't train. But don't oversell — set up
the next slides where we MEASURE each defense honestly, including where the judge is weak.
-->

---

## You can't claim this without an eval that tries to break it

The eval set is built to be adversarial:

- **20-passage** corpus (NVIDIA inference facts)
- **16 answerable** + **10 unanswerable** questions
- unanswerable = out-of-corpus, incl. the **B200/H200/FP4 near-misses**
- two runs per question: **guarded** vs **unguarded** (ablation)

Self-hosted on H100: vLLM **Qwen3-8B** (NIM stand-in) + Ollama embeddings · k=3 · temp=0

<!--
~1.5 min. The key methodological point of the whole talk: a normal RAG eval only asks
answerable questions, so it can NEVER catch hallucination. You have to author questions
the corpus can't answer. The guarded/unguarded ablation is what lets us attribute the
improvement to a specific defense rather than to the model being good.
-->

---

## Headline result

| metric | value |
|---|---|
| retrieval recall@3 (answerable) | **94%** |
| answer accuracy (answerable) | **94%** |
| hallucination on unanswerable — **guarded** | **0%** |
| hallucination on unanswerable — **unguarded** (ablation) | **50%** |

The same model, same corpus, same questions.
The **only** difference is one instruction in the system prompt.

<!--
~2 min. This is the money slide. Pause on it. The guarded prompt alone takes hallucination
on out-of-corpus questions to zero in this eval; removing it sends the SAME model to 50%.
That's the "cheap defense, big effect" story. Then immediately complicate it — next slide —
so you're credible, not salesy.
-->

---

## The honest part: the judge gate is a *weak* second line

Score `validate()` as a hallucination detector on the **unguarded** run
(where hallucinations actually exist):

| | gate blocks | gate passes |
|---|---|---|
| hallucinated (should block) | TP = 2 | **FN = 3** |
| grounded / abstained (should pass) | FP = 2 | TN = 19 |

**precision 50% · recall 40% · F1 0.44** → residual hallucination **50% → 30%**

A single-pass LLM judge is **not** enough on its own. It's a net, not a wall.

<!--
~2.5 min. This is the slide that earns trust. Most talks would stop at "0%!" — this one
shows the judge alone catches only 40% of what it should (3 dangerous misses). The lesson:
the guarded prompt does the heavy lifting; the judge is a backstop. Stacking two weak-ish
prompt defenses is what gets you a low residual without a fine-tune. Say plainly: if you
need stronger guarantees, this is where a fine-tuned grounding classifier or a second
retrieval pass goes — that's the roadmap.
-->

---

## The agentic tax: per-hop latency budget

| plan | retrieve | generate | validate | **total** |
|---|---|---|---|---|
| 0.21s | 0.155s | 0.29s | 0.16s | **0.81s** |

- `plan` + `validate` = **two extra LLM calls** wrapping every answer.
- `retrieve` is cheap (embed + in-memory fuse).
- The reliability you saw isn't free — it's **~0.37s of extra LLM calls** per query.

<!--
~1.5 min. Be the engineer who quotes the cost. The validate hop is what buys the safety-net
recall — so reliability has a latency price, and you should be able to state it at design
time. This is the kind of "I can reason about cost/latency trade-offs" signal an SA needs.
Mention: you can drop plan or run validate async if your SLA is tight.
-->

---

## Reproduce it yourself (≈5 min)

```bash
git clone https://github.com/waynehacking8/nim-agent-blueprint
cd nim-agent-blueprint

# point at any OpenAI-compatible endpoint (hosted NIM, or local vLLM/Ollama)
export NIM_MODE=selfhost NIM_LLM_URL=http://localhost:8000/v1
python eval/run_eval.py        # regenerates eval/report.md
```

Flip `guarded=False` in the harness and watch the 0% become 50%.
Full walkthrough: **`talk/walkthrough.md`**

<!--
~1.5 min. The supplementary-repo payoff: they leave able to run it. Stress that NIM is
OpenAI-compatible, so a local vLLM or even Ollama is a faithful stand-in — nobody needs an
NGC key to follow along. Point at walkthrough.md for the step-by-step.
-->

---

## Takeaways

1. In enterprise RAG, **refusing** is a feature. Measure it.
2. Your eval must include questions the corpus **can't** answer — incl. near-misses.
3. A **guarded prompt** is the cheapest, biggest win. (50% → 0% here.)
4. An **LLM-judge gate** is a backstop, not a guarantee. Know its recall.
5. Reliability has a **latency cost** — quote it at design time.

**Repo:** github.com/waynehacking8/nim-agent-blueprint
**Me:** waynehacking8.github.io

<!--
~1 min. Recap and land. The meta-point for an SA/FDE audience: the value isn't the agent,
it's the discipline of measuring the failure mode that actually blocks production. Open for
Q&A. Likely questions: hosted vs self-host cost; does this generalize beyond 20 passages;
what about multi-hop; how to harden the judge. Answers are in docs/design-decisions.md and
docs/roadmap.md.
-->

---

## Q&A

**Where to go deeper in the repo:**

- `app/agent.py` — the `plan → retrieve → generate → validate` loop
- `eval/run_eval.py` — how every number on these slides is computed
- `eval/report.md` — the full per-question breakdown
- `docs/design-decisions.md` — why guarded-by-default, why a judge gate
- `docs/roadmap.md` — hardening the gate, hosted-NIM run, multi-hop

*Thank you.*

<!--
Backup answers:
- "Why Qwen3-8B?" — small, open, OpenAI-compatible; the point is the method, not the model.
  Swap in a real build.nvidia.com NIM by changing NIM_MODE; harness is unchanged.
- "Does 0% hold at scale?" — no claim beyond this eval; the methodology is the transferable
  part. Bigger corpus / more near-misses is exactly how you'd re-validate for a customer.
- "Isn't temp=0 cheating?" — it's the right default for a grounded QA agent; higher temp
  raises hallucination, which only strengthens the case for the guarded prompt + gate.
-->
