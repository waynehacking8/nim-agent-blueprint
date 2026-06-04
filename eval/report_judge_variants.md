# Judge variants - CoT judging and grounded NLI vs the plain 8B judges

Same 200 SQuAD 2.0 rows / same unguarded answers (temp=0) as `report_squad.md`; only the judge differs, so all comparisons are paired (exact McNemar).

Ground truth: 96 hallucinations in total (86 unanswerable-answered + 10 answerable substring-miss); **53** of these 96 escape BOTH plain 8B judges (the shared-blind-spot set, the same /96 set every recall denominator below uses).

## Gate metrics (hallucination detection on the unguarded run)

| gate | precision | recall | F1 | catches of the 53 shared blind spots |
|---|---|---|---|---|
| self (plain) | 79% (26/33, 95% CI 62%–89%) | 27% (26/96, 95% CI 19%–37%) | 0.40 | 0 |
| cross-family (plain) | 74% (39/53, 95% CI 60%–84%) | 41% (39/96, 95% CI 31%–51%) | 0.52 | 0 |
| self + CoT | 81% (38/47, 95% CI 67%–90%) | 40% (38/96, 95% CI 30%–50%) | 0.53 | 11 |
| cross-family + CoT | 62% (47/76, 95% CI 51%–72%) | 49% (47/96, 95% CI 39%–59%) | 0.55 | 11 |
| MiniCheck-FT5 770M (grounded NLI) | 66% (35/53, 95% CI 53%–77%) | 36% (35/96, 95% CI 28%–46%) | 0.47 | 12 |
| cross-family OR MiniCheck (union) | 63% (53/84, 95% CI 52%–73%) | 55% (53/96, 95% CI 45%–65%) | 0.59 | 12 |

## Paired tests (exact McNemar, two-sided, on hallucinated answers)

| comparison | A only | B only | both | neither | p |
|---|---|---|---|---|---|
| self plain vs self+CoT | 4 | 16 | 22 | 54 | 0.0118 |
| cross plain vs cross+CoT | 5 | 13 | 34 | 44 | 0.0963 |
| self+CoT vs cross+CoT | 6 | 15 | 32 | 43 | 0.0784 |
| cross plain vs MiniCheck | 18 | 14 | 21 | 43 | 0.5966 |
| cross+CoT vs MiniCheck | 24 | 12 | 23 | 37 | 0.0652 |

**Residual blind spots: 29 of 96 hallucinations escape EVERY gate above** (both plain judges, both CoT judges, MiniCheck, and the union). This is the floor that the next method class (a 70B judge, retrieval-aware generation, or semantic-entropy filtering) has to attack.

## Literature comparison

| claim | published | measured here |
|---|---|---|
| CoT raises self-detection recall (arXiv:2511.11087) | ~22% -> ~58% | 27% -> 40% |
| CoT on the cross-family judge | (not reported) | 41% -> 49% |
| MiniCheck 770M ~ GPT-4 level grounded verification (arXiv:2404.10774) | 77.4% bal. acc. on LLM-AGGREFACT | recall 36% / precision 66% on this set |

Charts: `python eval/make_charts.py` regenerates `judge_variants.png` from the rows JSON.

