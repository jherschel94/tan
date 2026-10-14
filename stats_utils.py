import numpy as np
from scipy.stats import t as t_dist


def nadeau_bengio_ttest(diffs, n_train, n_test):
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    if n < 2:
        return np.nan, 1.0
    d_mean = diffs.mean()
    d_var = diffs.var(ddof=1)  # sample variance, matches df=n-1 below
    if d_var == 0:
        return np.nan, 1.0 if d_mean == 0 else 0.0

    n_train_arr = np.asarray(n_train, dtype=float)
    n_test_arr = np.asarray(n_test, dtype=float)
    ratio = np.mean(n_test_arr / n_train_arr)  # mean, in case sizes vary slightly by fold

    correction = (1.0 / n) + ratio
    se = np.sqrt(correction * d_var)
    if se == 0:
        return np.nan, 1.0 if d_mean == 0 else 0.0

    t_stat = d_mean / se
    df = n - 1
    p_value = 2 * t_dist.sf(np.abs(t_stat), df)
    return t_stat, p_value


def holm_bonferroni(pvalues, alpha=0.05):
    pvalues = np.array(pvalues, dtype=float)
    n = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min((n - rank) * pvalues[idx], 1.0)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    reject = adjusted < alpha
    return adjusted, reject
