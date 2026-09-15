"""
Fairness-aware estimators for risk adjustment.

Net compensation for group G is NC_G(beta) = mean_G(x' beta) - mean_G(y) =
a_G' beta - b_G, where a_G and b_G are weighted group means of the design row
and of spending. Because that is linear in beta, fair linear estimators have
exact solutions.

  FairWLS(lam)           min (1/sum w) sum w (y - X beta)^2
                               + lam * sum_G (a_G' beta - b_G)^2
                         (the net-compensation penalty of Zink and Rose 2020;
                         lam is scaled by the outcome variance to be unit-free)
  FairWLS(constrained)   the same loss subject to a_G' beta = b_G for every
                         target group, the lam -> infinity limit, solved as a
                         KKT system (the constrained regression of McGuire,
                         Zink and Rose 2021)
  StackedFairGBM(lam)    a LightGBM risk score, cross-fitted within the
                         training data, entered with the features into FairWLS

Group membership enters estimation only through a_G and b_G. It never enters
the prediction formula, so a model fitted this way pays by the same features as
an unconstrained one and needs no group information at payment time.

Why the boosted model is stacked rather than penalized inside the trees. A
net-compensation penalty added to a boosting objective has rank-one curvature
across a group. A diagonal Hessian either understates it (Newton steps
overshoot by roughly the inverse of the group's weight share and diverge) or,
if the group's curvature is spread across its members, overstates it for every
member and damps the fitting of their own residuals, which in testing made
under-compensation of small high-cost groups worse as the penalty rose. The
stacked form keeps the boosted model's accuracy in the score and puts the
fairness adjustment where it has an exact solution.
"""

from __future__ import annotations

import numpy as np

from .metrics import weighted_quantile_groups
from .surveycv import inner_split


def _design(X):
    return np.column_stack([np.ones(len(X)), X])


def _group_moments(A, y, w, masks):
    a, b = [], []
    for m in masks:
        ww = w[m]
        a.append((A[m] * ww[:, None]).sum(axis=0) / ww.sum())
        b.append(np.sum(ww * y[m]) / ww.sum())
    return np.array(a), np.array(b)


class FairWLS:
    def __init__(self, lam=0.0, constrained=False, name=None):
        self.lam, self.constrained = lam, constrained
        self.name = name or ("Constrained WLS" if constrained
                             else f"Penalized WLS (lambda={lam:g})")

    def fit(self, X, y, w, masks):
        A = _design(X)
        wn = w / w.sum()
        G = (A * wn[:, None]).T @ A
        h = (A * wn[:, None]).T @ y
        G[np.diag_indices_from(G)] += 1e-9 * np.trace(G) / len(G)
        aG, bG = _group_moments(A, y, w, masks)
        s2 = float(np.sum(wn * (y - np.sum(wn * y)) ** 2))
        if self.constrained:
            k = len(bG)
            K = np.block([[2 * G, aG.T], [aG, np.zeros((k, k))]])
            sol = np.linalg.solve(K, np.concatenate([2 * h, bG]))
            self.coef_ = sol[:A.shape[1]]
        else:
            lam = self.lam * s2 / max(np.mean(bG ** 2), np.finfo(float).tiny)
            self.coef_ = np.linalg.solve(G + lam * aG.T @ aG, h + lam * aG.T @ bG)
        return self

    def predict(self, X):
        return _design(X) @ self.coef_


def _cross_fit_scores(make_model, X, y, w, clusters, k=3, seed=0):
    """Out-of-fold scores for the training rows, from k cluster-disjoint folds."""
    rng = np.random.default_rng(seed)
    clusters = np.asarray(clusters)
    u = np.unique(clusters)
    fold_of_cluster = dict(zip(u, rng.permutation(len(u)) % k))
    f = np.array([fold_of_cluster[c] for c in clusters])
    s = np.empty(len(y))
    for j in range(k):
        tr, te = f != j, f == j
        s[te] = make_model().fit(X[tr], y[tr], w[tr], clusters=clusters[tr]).predict(X[te])
    return s


class StackedFairGBM:
    """Cross-fitted LightGBM score plus features, fitted with FairWLS.

    The score for training rows is out-of-fold within the training data, so
    the fair layer never sees a score fitted to the same person's outcome.
    Test rows are scored by a LightGBM fitted on all training rows.
    """

    def __init__(self, lam=0.0, constrained=False, objective="tweedie", k=3,
                 seed=2026):
        self.lam, self.constrained, self.objective = lam, constrained, objective
        self.k, self.seed = k, seed
        self.name = ("Stacked LightGBM, constrained" if constrained
                     else f"Stacked LightGBM (lambda={lam:g})")

    def _make(self):
        from .models import GBM
        return GBM(self.objective, seed=self.seed)

    def fit(self, X, y, w, masks, clusters=None):
        cl = np.arange(len(y)) if clusters is None else np.asarray(clusters)
        score = _cross_fit_scores(self._make, X, y, w, cl, k=self.k, seed=self.seed)
        self.gbm_ = self._make().fit(X, y, w, clusters=cl)
        self.fair_ = FairWLS(lam=self.lam, constrained=self.constrained).fit(
            np.column_stack([score, X]), y, w, masks)
        return self

    def predict(self, X):
        return self.fair_.predict(np.column_stack([self.gbm_.predict(X), X]))


def recalibrate_by_decile(p_train, y_train, w_train, p_test, n=10):
    """Group-neutral post-processing: scale each predicted-cost decile so its
    predictive ratio on the reference predictions is one. Uses no group
    information. Pass out-of-sample reference predictions, not in-sample fits."""
    g = weighted_quantile_groups(p_train, w_train, n)
    edges = [p_train[g == k].max() for k in range(n - 1)]
    factors = np.array([np.sum(w_train[g == k] * y_train[g == k])
                        / max(np.sum(w_train[g == k] * p_train[g == k]), 1e-9)
                        for k in range(n)])
    gt = np.searchsorted(edges, p_test, side="left")
    return p_test * factors[np.minimum(gt, n - 1)]
