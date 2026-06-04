"""Shared statistical utilities for eval scripts."""


def auroc_mannwhitney(scores_pos, scores_neg):
    """AUROC via Mann-Whitney U statistic."""
    from itertools import product
    n_pos, n_neg = len(scores_pos), len(scores_neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    u = sum(1 for p, n in product(scores_pos, scores_neg) if p > n)
    u += 0.5 * sum(1 for p, n in product(scores_pos, scores_neg) if p == n)
    return u / (n_pos * n_neg)
