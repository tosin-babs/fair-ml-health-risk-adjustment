"""
Accuracy and fairness metrics, all survey-weighted.

Accuracy: R-squared, mean absolute error, root mean squared error, Cumming's
prediction measure, predictive ratios by decile of predicted spending, and the
share of the true top spenders a model places in its own top group.

Fairness, following the risk-adjustment literature: for a group G,
  predictive ratio  PR_G = sum_G w yhat / sum_G w y
  net compensation  NC_G = sum_G w (yhat - y) / sum_G w
A negative NC_G means the model pays less for the group than the group costs.
"""

from __future__ import annotations

import numpy as np


def _wmean(x, w):
    return float(np.sum(w * x) / np.sum(w))


def r2(y, p, w):
    ybar = _wmean(y, w)
    return float(1 - np.sum(w * (y - p) ** 2) / np.sum(w * (y - ybar) ** 2))


def mae(y, p, w):
    return _wmean(np.abs(y - p), w)


def rmse(y, p, w):
    return float(np.sqrt(_wmean((y - p) ** 2, w)))


def cpm(y, p, w):
    """Cumming's prediction measure: 1 - sum|y - p| / sum|y - ybar|."""
    ybar = _wmean(y, w)
    return float(1 - np.sum(w * np.abs(y - p)) / np.sum(w * np.abs(y - ybar)))


def weighted_quantile_groups(p, w, n=10):
    """Assign each person to one of n weighted quantile groups of p."""
    order = np.argsort(p, kind="mergesort")
    cw = np.cumsum(w[order]) / np.sum(w)
    g = np.minimum((cw * n).astype(int), n - 1)
    out = np.empty(len(p), dtype=int)
    out[order] = g
    return out


def predictive_ratios_by_decile(y, p, w, n=10):
    g = weighted_quantile_groups(p, w, n)
    return np.array([np.sum(w[g == k] * p[g == k]) / np.sum(w[g == k] * y[g == k])
                     for k in range(n)])


def top_capture(y, p, w, share):
    """Weighted share of the true top-`share` spenders in the predicted top."""
    ty = weighted_quantile_groups(y, w, 1000) >= int(1000 * (1 - share))
    tp = weighted_quantile_groups(p, w, 1000) >= int(1000 * (1 - share))
    return float(np.sum(w[ty & tp]) / np.sum(w[ty]))


def accuracy(y, p, w):
    pr = predictive_ratios_by_decile(y, p, w)
    return {"r2": r2(y, p, w), "mae": mae(y, p, w), "rmse": rmse(y, p, w),
            "cpm": cpm(y, p, w), "pr_overall": float(np.sum(w * p) / np.sum(w * y)),
            "pr_bottom_decile": float(pr[0]), "pr_top_decile": float(pr[-1]),
            "top5_capture": top_capture(y, p, w, 0.05),
            "top10_capture": top_capture(y, p, w, 0.10)}


def group_fairness(y, p, w, masks):
    rows = {}
    for name, m in masks.items():
        if m.sum() == 0:
            continue
        rows[name] = {"n": int(m.sum()),
                      "pr": float(np.sum(w[m] * p[m]) / np.sum(w[m] * y[m])),
                      "nc": _wmean(p[m] - y[m], w[m]),
                      "mean_cost": _wmean(y[m], w[m]),
                      "mae": mae(y[m], p[m], w[m])}
    return rows


def fairness_summary(groups):
    if not groups:
        return {"max_abs_nc": np.nan, "max_abs_pr_gap": np.nan}
    return {"max_abs_nc": max(abs(g["nc"]) for g in groups.values()),
            "max_abs_pr_gap": max(abs(g["pr"] - 1) for g in groups.values())}


def bootstrap_index(clusters, strata, rng, rao_wu=True):
    """One resample of PSUs within strata, as (row index, weight multiplier).

    Rao and Wu (1988): drawing n_h PSUs with replacement from a stratum of
    n_h understates the design variance by (n_h - 1)/n_h, which is a half in
    the 327 two-PSU strata here. Drawing n_h - 1 and multiplying their
    weights by n_h/(n_h - 1) removes that bias.
    """
    idx_by_cluster = {}
    for i, c_ in enumerate(clusters):
        idx_by_cluster.setdefault(c_, []).append(i)
    stratum_clusters = {}
    for c_ in idx_by_cluster:
        stratum_clusters.setdefault(strata[idx_by_cluster[c_][0]], []).append(c_)
    idx, mult = [], []
    for cs in stratum_clusters.values():
        n_h = len(cs)
        m = n_h - 1 if (rao_wu and n_h > 1) else n_h
        f = n_h / m if m else 1.0
        for j in rng.choice(n_h, size=m, replace=True):
            rows = idx_by_cluster[cs[j]]
            idx.extend(rows)
            mult.extend([f] * len(rows))
    return np.asarray(idx), np.asarray(mult)


def cluster_bootstrap(stat, y, p, w, clusters, strata, reps=500, seed=2026, rao_wu=True):
    """Bootstrap a scalar statistic over PSUs resampled within strata."""
    rng = np.random.default_rng(seed)
    clusters, strata = np.asarray(clusters), np.asarray(strata)
    draws = []
    for _ in range(reps):
        idx, mult = bootstrap_index(clusters, strata, rng, rao_wu)
        draws.append(stat(y[idx], p[idx], w[idx] * mult, idx))
    return np.asarray(draws)
