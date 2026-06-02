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

- [x] **Independent judge model** (different model family) for `validate()`. **DONE — README
  "Results at scale" / eval/report_squad.md.** Run at N=200 (SQuAD profile) for statistical
  power: self-judge recall 32% [23–41%] → cross-family (Llama-3.1-8B) recall 47% [38–57%],
  **+16 points, McNemar exact p=0.0026** (paired — same answers, only the judge differs).
  Precision trade-off observed as predicted (79%→71%). Shared-blind-spots attribution
  supported on this setup (single dataset, one 8B judge pair; unanswerable-only robustness
  check p=0.0043); the deeper honest finding is that 46/95 hallucinations escape BOTH 8B judges.
  Implementation: validate() takes judge_model/judge_url (or NIM_JUDGE_MODEL/NIM_JUDGE_URL);
  the eval harness scores both judges in one pass (NIM_XJUDGE_MODEL/NIM_XJUDGE_URL).
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

- [x] **Scale the eval set** to statistical usefulness. **DONE — eval/build_squad_eval.py +
  eval/report_squad.md.** 100 SQuAD 2.0 dev passages, 100 answerable + 100 unanswerable
  (crowdworker-written adversarial near-misses — strictly harder than the demo set's
  out-of-corpus design). All metrics carry 95% Wilson CIs; accuracy now scored by both
  substring (71%) and LLM judge (72% — the two agree, so the demo set's substring check was
  not overstating). Key scale finding: the guarded prompt that achieved 0% hallucination on
  out-of-corpus questions only achieves 49% [39–59%] on near-miss questions — guardrails must
  be evaluated on the hard case.
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

- [ ] **Attack the 46/95 shared blind spots: larger judge vs judge panel vs retrieval-grounded.**
  - **Question:** 46 of 95 hallucinations escape BOTH 8B judges (self + cross-family). Is that
    a capacity problem (8B judges are too small) or a structural one (judging from parametric
    knowledge has a ceiling no matter the size)? The answer decides where validate() should
    invest: bigger judge, more judges, or grounding.
  - **Method:** three arms scored against the same SQuAD N=200 rows (temp=0 → generations are
    identical, so all arms judge the same answers and remain McNemar-comparable):
    (a) **larger judge** — Llama-3.3-70B-Instruct (FP8/AWQ on 1–2×H100) via `NIM_XJUDGE_MODEL`;
    (b) **judge panel (PoLL, arXiv:2404.18796)** — qwen3-8b + llama-3.1-8b + gemma-2-9b,
    majority vote (reuse the existing dual-judge plumbing in eval/run_eval.py, add a third);
    (c) **retrieval-grounded judge** — pass the retrieved context into the judge prompt so it
    compares the answer against evidence instead of its own knowledge (one-line prompt change
    in validate()).
  - **Read-out:** recall on the 95 hallucinations per arm (baselines: self 32%, cross-family
    47%). The decisive comparison is (c) vs (a): if the grounded 8B judge beats the 70B
    parametric judge, the bottleneck is grounding, not capacity — directly actionable (the
    validate() step should always receive the retrieved context). Track precision per arm;
    panel and grounding may trade precision differently.
