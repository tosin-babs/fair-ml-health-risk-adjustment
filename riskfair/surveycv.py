"""
Cross-validation folds that respect a clustered sampling design.

Folds are built from primary sampling units (PSUs), not persons. Within each
stratum the PSUs are shuffled and dealt across folds, so every fold draws PSUs
from every stratum and no PSU appears in both a training and a test set.
Persons in one PSU share neighbourhood, provider and interviewer effects;
splitting them across training and test data leaks that shared variation and
overstates out-of-sample accuracy.

For claims data with no survey design, pass a household, employer or provider
identifier as the cluster and a single constant as the stratum.
"""

from __future__ import annotations

import numpy as np


def survey_folds(clusters, strata, n_splits=5, repeats=3, seed=2026):
    """List of (repeat, fold, train_idx, test_idx)."""
    clusters = np.asarray(clusters)
    strata = np.asarray(strata)
    psu_stratum = {}
    for c_, s_ in zip(clusters, strata):
        psu_stratum.setdefault(c_, s_)
    by_stratum = {}
    for c_, s_ in psu_stratum.items():
        by_stratum.setdefault(s_, []).append(c_)

    out = []
    for rep in range(repeats):
        rng = np.random.default_rng(seed + rep)
        fold_of = {}
        for s_ in sorted(by_stratum, key=str):
            psus = np.array(sorted(by_stratum[s_], key=str), dtype=object)
            rng.shuffle(psus)
            offset = int(rng.integers(n_splits))
            for i, c_ in enumerate(psus):
                fold_of[c_] = (offset + i) % n_splits
        f = np.array([fold_of[c_] for c_ in clusters])
        for fold in range(n_splits):
            out.append((rep, fold, np.flatnonzero(f != fold), np.flatnonzero(f == fold)))
    return out


def check_disjoint(folds, clusters):
    """Raise if any PSU is in both the training and test data of a fold."""
    clusters = np.asarray(clusters)
    for rep, fold, tr, te in folds:
        overlap = set(clusters[tr]) & set(clusters[te])
        if overlap:
            raise AssertionError(f"repeat {rep} fold {fold}: {len(overlap)} PSUs "
                                 f"in both training and test data")
    return True


def inner_folds(clusters, k=3, seed=0):
    """k cluster-disjoint (train, validation) splits of a training set.

    Averaging a validation score over k splits selects hyperparameters far
    more stably than one split when the outcome is heavy-tailed: a handful of
    very expensive people in one validation set can otherwise decide the
    choice.
    """
    rng = np.random.default_rng(seed)
    clusters = np.asarray(clusters)
    u = np.unique(clusters)
    fold_of = dict(zip(u.tolist(), (rng.permutation(len(u)) % k).tolist()))
    f = np.array([fold_of[c] for c in clusters.tolist()])
    return [(np.flatnonzero(f != j), np.flatnonzero(f == j)) for j in range(k)]


def inner_split(clusters, frac=0.15, seed=0):
    """Hold out whole clusters from a training set, for tuning or early stopping."""
    rng = np.random.default_rng(seed)
    clusters = np.asarray(clusters)
    u = np.unique(clusters)
    val = set(rng.choice(u, size=max(1, int(len(u) * frac)), replace=False).tolist())
    m = np.array([c in val for c in clusters])
    return np.flatnonzero(~m), np.flatnonzero(m)
