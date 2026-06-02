# The three-arm blind-spot experiment: capacity vs grounding

Same 200 rows / same answers as every other gate (paired, exact McNemar). Ground truth this run: 95 hallucinations; **48** escape both plain 8B judges; **23** escape every previously-measured gate.

Judges: arm (a) `casperhansen/llama-3.3-70b-instruct-awq` (the open AWQ build of Llama-3.3-70B-Instruct; the official Meta repo is license-gated), arm (b) PoLL panel = qwen3-8b + llama-3.1-8b + microsoft/phi-4 (third family; gemma-2-9b is gated), arm (c) llama-3.1-8b with the question-aware grounded prompt (`validate(question=...)`).

## All gates on the same hallucinations

| gate | precision | recall | F1 | of the 48 both-8B-missed | of the 23 escape-everything |
|---|---|---|---|---|---|
| self 8B (plain) [baseline] | 79% (31/39, 95% CI 64%–89%) | 33% (31/95, 95% CI 24%–43%) | 0.46 | 0 | 0 |
| cross-family 8B (plain) [baseline] | 67% (44/66, 95% CI 55%–77%) | 46% (44/95, 95% CI 37%–56%) | 0.55 | 0 | 0 |
| MiniCheck 770M [baseline] | 74% (39/53, 95% CI 60%–84%) | 41% (39/95, 95% CI 32%–51%) | 0.53 | 14 | 0 |
| cross OR MiniCheck union [baseline] | 65% (59/91, 95% CI 55%–74%) | 62% (59/95, 95% CI 52%–71%) | 0.63 | 14 | 0 |
| (a) Llama-3.3-70B parametric | 68% (46/68, 95% CI 56%–78%) | 48% (46/95, 95% CI 39%–58%) | 0.56 | 9 | 2 |
| (b) PoLL 3-family majority | 70% (39/56, 95% CI 57%–80%) | 41% (39/95, 95% CI 32%–51%) | 0.52 | 0 | 0 |
| (b3) phi-4 alone | 60% (51/85, 95% CI 49%–70%) | 54% (51/95, 95% CI 44%–63%) | 0.57 | 13 | 3 |
| (c) question-aware grounded 8B | 64% (9/14, 95% CI 39%–84%) | 9% (9/95, 95% CI 5%–17%) | 0.17 | 2 | 1 |

## Paired tests (exact McNemar, on hallucinated answers)

| comparison | A only | B only | both | neither | p |
|---|---|---|---|---|---|
| cross 8B vs (a) 70B | 9 | 11 | 35 | 40 | 0.8238 |
| cross 8B vs (b) PoLL | 6 | 1 | 38 | 50 | 0.1250 |
| cross 8B vs (c) grounded | 37 | 2 | 7 | 49 | 0.0000 |
| (a) 70B vs (c) grounded  <- the decisive one | 40 | 3 | 6 | 46 | 0.0000 |
| (a) 70B vs (b) PoLL | 13 | 6 | 33 | 43 | 0.1671 |

**Best achievable union (grounded-8B OR MiniCheck OR 70B): recall 62% (59/95, 95% CI 52%–71%), precision 65% (59/91, 95% CI 55%–74%), F1 0.63 — residual 36/95 hallucinations escape even this.**

Chart: `python eval/make_charts.py` regenerates `blind_spot_arms.png` from the rows JSON.

