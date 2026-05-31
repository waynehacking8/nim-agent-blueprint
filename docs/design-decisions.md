# Design decisions

**D1 — Provider layer first.** Hosted vs self-hosted NIM is one env var. This is the exact
pattern an SA needs: prototype on hosted endpoints, then move the partner to self-hosted on
their GPUs without rewriting the app.

**D2 — Validate step is non-optional.** The agent checks groundedness before returning. Most
RAG demos skip this; it's the part partners actually care about (and the part I can measure).

**D3 — Measure, don't assert.** The blueprint ships an eval (hit-rate, groundedness, latency)
so the README shows numbers, not adjectives. Small gold set now; expand per Phase 2.

**D4 — Framework-light core.** The agent loop is plain Python so the logic is legible; a
NeMo Agent Toolkit variant is added alongside, not as a hard dependency.
