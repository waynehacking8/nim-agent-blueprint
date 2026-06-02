# Judge variants - CoT judging and grounded NLI vs the plain 8B judges

Same 200 SQuAD 2.0 rows / same unguarded answers (temp=0) as `report_squad.md`; only the judge differs, so all comparisons are paired (exact McNemar).

Ground truth: 95 hallucinations in total (78 unanswerable-answered + 17 answerable substring-miss); **48** of these 95 escape BOTH plain 8B judges (the shared-blind-spot set, the same /95 set every recall denominator below uses).

## Gate metrics (hallucination detection on the unguarded run)

| gate | precision | recall | F1 | catches of the 48 shared blind spots |
|---|---|---|---|---|
| self (plain) | 79% (31/39, 95% CI 64%–89%) | 33% (31/95, 95% CI 24%–43%) | 0.46 | 0 |
| cross-family (plain) | 67% (44/66, 95% CI 55%–77%) | 46% (44/95, 95% CI 37%–56%) | 0.55 | 0 |
| self + CoT | 82% (42/51, 95% CI 70%–90%) | 44% (42/95, 95% CI 35%–54%) | 0.58 | 8 |
| cross-family + CoT | 66% (50/76, 95% CI 55%–75%) | 53% (50/95, 95% CI 43%–62%) | 0.58 | 10 |
| MiniCheck-FT5 770M (grounded NLI) | 74% (39/53, 95% CI 60%–84%) | 41% (39/95, 95% CI 32%–51%) | 0.53 | 14 |
| cross-family OR MiniCheck (union) | 65% (59/91, 95% CI 55%–74%) | 62% (59/95, 95% CI 52%–71%) | 0.63 | 14 |

## Paired tests (exact McNemar, two-sided, on hallucinated answers)

| comparison | A only | B only | both | neither | p |
|---|---|---|---|---|---|
| self plain vs self+CoT | 6 | 17 | 25 | 47 | 0.0347 |
| cross plain vs cross+CoT | 5 | 11 | 39 | 40 | 0.2101 |
| self+CoT vs cross+CoT | 9 | 17 | 33 | 36 | 0.1686 |
| cross plain vs MiniCheck | 20 | 15 | 24 | 36 | 0.4996 |
| cross+CoT vs MiniCheck | 25 | 14 | 25 | 31 | 0.1081 |

**Residual blind spots: 23 of 95 hallucinations escape EVERY gate above** (both plain judges, both CoT judges, MiniCheck, and the union). This is the floor that the next method class (a 70B judge, retrieval-aware generation, or semantic-entropy filtering) has to attack.

## Literature comparison

| claim | published | measured here |
|---|---|---|
| CoT raises self-detection recall (arXiv:2511.11087) | ~22% -> ~58% | 33% -> 44% |
| CoT on the cross-family judge | (not reported) | 46% -> 53% |
| MiniCheck 770M ~ GPT-4 level grounded verification (arXiv:2404.10774) | 77.4% bal. acc. on LLM-AGGREFACT | recall 41% / precision 74% on this set |

Charts: `python eval/make_charts.py` regenerates `judge_variants.png` from the rows JSON.

