# Semantic entropy (Farquhar et al., Nature 2024) on the SQuAD eval set

200 questions x 10 samples at T=1.0 from the unguarded generator; clusters = bidirectional DeBERTa-large-MNLI entailment; labels = whether the temp=0 answer was a hallucination (same rows as every other gate in this repo).

## AUROC vs published

| method | full set | answerable only | unanswerable (adversarial near-miss) | published |
|---|---|---|---|---|
| semantic entropy (Rao, logprob-weighted) | 0.614 | 0.640 | 0.504 | 0.790 |
| discrete semantic entropy (cluster counts) | 0.620 | 0.652 | 0.513 | ~0.790 |
| naive entropy (mean -logprob) | 0.630 | 0.697 | 0.387 | 0.691 |

## Reading - the published ceiling does NOT transfer to the adversarial case

On **answerable** questions the signal works as published: hallucinated answers have much higher semantic entropy than correct ones (0.97 vs 0.54) and naive entropy lands within a few points of the paper's 0.691. On the **adversarial near-miss** questions the separation collapses (0.87 vs 0.82) and every variant drops to chance.

Two mechanisms, both visible in the per-question clusters:

1. **Systematic (confident) errors**: the near-miss design makes the generator give the same wrong answer in most samples - low entropy, but hallucinated. Farquhar et al. scope semantic entropy to *confabulations* (arbitrary wrong answers) and explicitly exclude systematic errors; this eval set is built from the excluded case.
2. **Abstention phrasing diversity**: when the generator correctly abstains, the 10 samples phrase the refusal differently and NLI does not cluster refusals as mutually entailing - high entropy, but NOT hallucinated.

Connection to the judge results: the 48 hallucinations that escape both 8B judges are the same confident, systematic errors that defeat semantic entropy. Sampling-based uncertainty is not a path to recovering them; grounded verification (MiniCheck) partially is.

Chart: `python eval/make_charts.py` regenerates `semantic_entropy.png`.

