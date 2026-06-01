# Roadmap

## Phase 1 — Working blueprint
- [ ] provider layer (hosted + self-host) verified against live NIMs.
- [x] hybrid retrieval + NIM reranker wired into retrieve() (rerank opt-in via `NIM_RERANK`, falls back to fusion).
- [ ] plan/retrieve/generate/validate loop end-to-end; OTel traces visible.
- [ ] eval (hit-rate, groundedness, latency) -> report.md with real numbers.

## Phase 2 — Productionize
- [ ] Milvus/pgvector store; ingestion script for a partner corpus.
- [ ] NeMo Agent Toolkit variant in app/nat_variant/.
- [ ] NeMo Guardrails on the validate step.

## Phase 3 — Self-host on H100/RTX Pro 6000
- [ ] docker compose NIMs on local GPUs; latency hosted vs self-hosted.
- [ ] "hand-off to a partner" one-pager (the SA deliverable).

## Phase 4 — Eval hardening (specified)

- [ ] **Independent judge model** (different model family) for `validate()`.
  - **Question:** gate recall is 25% and the attribution is shared blind spots — the judge is the
    same model that hallucinated, so it is missing the same knowledge. If correct, swapping in a
    judge from a different model family should raise recall substantially; if recall stays ~25%,
    the attribution needs rework.
  - **Method:** `provider.chat()` already accepts `model=` (app/provider.py); `validate()` doesn't
    pass it (app/agent.py). Change validate() to
    `provider.chat([...], model=os.getenv("NIM_JUDGE_MODEL"), max_tokens=16)` — `None` keeps
    current behavior. Serve a second model from a different family (generator Qwen3-8B → judge
    Llama-3.1-8B), then:
    `NIM_JUDGE_MODEL="meta-llama/Llama-3.1-8B-Instruct" python eval/run_eval.py`.
  - **Read-out:** recall 25% → 50%+ (literature: cross-model detection well above self-detection)
    → confirms the shared-blind-spots attribution. Unchanged → re-examine the judge prompt and
    threshold. Also track precision: an independent judge may be stricter (more FPs) — the
    precision/recall trade-off is part of the result.

- [ ] **Scale the eval set** to statistical usefulness.
  - **Question:** N=26 percentages (0% / 40% / 25%) demonstrate the method, not statistics. What
    are the confidence intervals at scale, and does the guarded-prompt result hold?
  - **Method:** extend `eval/corpus.jsonl` (20 → 100+ passages) and `eval/dataset.jsonl` (26 →
    200+ questions), keeping the answerable/unanswerable ratio and the adversarial near-miss
    design. At the same time, replace the substring accuracy check (`gold in answer`) with an
    LLM-judge or semantic match (the current 100% accuracy is likely overstated). Then
    `python eval/run_eval.py`.
  - **Read-out:** report each metric with Wilson confidence intervals; if guarded hallucination
    stays <5% at scale, the "one guarded prompt" conclusion upgrades from demonstration to
    statistically supported.
