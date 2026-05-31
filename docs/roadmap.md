# Roadmap

## Phase 1 — Working blueprint
- [ ] provider layer (hosted + self-host) verified against live NIMs.
- [ ] hybrid retrieval + NIM reranker wired into retrieve().
- [ ] plan/retrieve/generate/validate loop end-to-end; OTel traces visible.
- [ ] eval (hit-rate, groundedness, latency) -> report.md with real numbers.

## Phase 2 — Productionize
- [ ] Milvus/pgvector store; ingestion script for a partner corpus.
- [ ] NeMo Agent Toolkit variant in app/nat_variant/.
- [ ] NeMo Guardrails on the validate step.

## Phase 3 — Self-host on H100/RTX Pro 6000
- [ ] docker compose NIMs on local GPUs; latency hosted vs self-hosted.
- [ ] "hand-off to a partner" one-pager (the SA deliverable).
