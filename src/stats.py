"""
Statistical significance testing across models.

Friedman test (omnibus) followed by the Nemenyi post-hoc test on the per-fold
macro-F1 scores, plus a critical-difference visualization. This is the standard
protocol (Demsar, 2006) for comparing multiple classifiers over multiple
datasets/folds.
"""
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare


def friedman_nemenyi(f1_matrix: pd.DataFrame):
    """
    f1_matrix: rows = folds, columns = models, values = macro-F1.
    Returns (friedman_stat, p_value, nemenyi_pvalue_dataframe, mean_ranks).
    """
    data = [f1_matrix[col].values for col in f1_matrix.columns]
    stat, p = friedmanchisquare(*data)

    # mean ranks (higher F1 -> better -> rank 1)
    ranks = f1_matrix.rank(axis=1, ascending=False)
    mean_ranks = ranks.mean(axis=0).sort_values()

    try:
        import scikit_posthocs as sp
        nemenyi = sp.posthoc_nemenyi_friedman(f1_matrix.values)
        nemenyi.index = f1_matrix.columns
        nemenyi.columns = f1_matrix.columns
    except Exception as e:            # library optional
        nemenyi = None
        print(f"[stats] Nemenyi skipped ({e}). pip install scikit-posthocs")

    return stat, p, nemenyi, mean_ranks


def critical_difference(n_models: int, n_folds: int, alpha: float = 0.05) -> float:
    """Nemenyi critical difference for a CD diagram (Demsar, 2006)."""
    # q_alpha values (Studentized range / sqrt(2)) for alpha=0.05, k = 2..20.
    q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
           8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313,
           14: 3.354, 15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517,
           20: 3.544}
    if n_models not in q05:
        # clamp to the nearest tabulated k (use the largest for big benchmarks)
        q = q05[min(max(n_models, 2), 20)]
    else:
        q = q05[n_models]
    return q * np.sqrt(n_models * (n_models + 1) / (6.0 * n_folds))
