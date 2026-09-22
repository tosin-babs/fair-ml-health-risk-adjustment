"""
Payment formulas made robust to coding and to selection.

All three keep the payment form of `models.PaymentWLS`: a base rate for each
age band and sex, non-negative increments for everything else, no intercept.

  PenalizedPaymentWLS   ridge penalty on the increments of codable flags,
                        weighted by how easy each flag is to add
  CappedPaymentWLS      upper bound on each increment: a code may not pay
                        more than its incremental cost among the people who
                        carry it in the ungamed data
  DROPaymentWLS         penalty on the worst underpayment or overpayment of
                        any subgroup of at least a given weight share, which
                        is the conditional value at risk of the residual
                        (Rockafellar and Uryasev 2000; Duchi and Namkoong 2021)

`incremental_costs` estimates the caps.
"""

from __future__ import annotations

import numpy as np

from .models import PaymentWLS


def _cvar(r, w, alpha):
    """Weighted mean of the largest `alpha` share of r, and the rows in it.

    This is the largest weighted mean of r over any subgroup whose weight
    share is at least alpha, which is what a plan that can select on anything
    observable can reach.
    """
    o = np.argsort(-r)
    cw = np.cumsum(w[o])
    n = int(np.searchsorted(cw, alpha * cw[-1], side="left")) + 1
    top = o[:n]
    return float(np.average(r[top], weights=w[top])), top


def _upper(names, caps):
    """Upper bounds from caps: NaN means none, and a zero cap is a hair above
    the lower bound, which the solvers require."""
    out = []
    for n in names:
        c = caps.get(n, np.inf)
        out.append(np.inf if not np.isfinite(c) else max(float(c), 1e-6))
    return np.array(out)


class PenalizedPaymentWLS(PaymentWLS):
    """Payment-form WLS with a ridge penalty lambda * sum_j p_j beta_j^2 on
    the columns in `pool_weights` (column -> p_j). p_j is meant to be the
    share of people for whom the code could plausibly be added, so easy codes
    are shrunk hardest. Columns that count conditions must be in the pool as
    well (every added code raises them); left free they absorb what the flags
    lose. Implemented by augmenting the least-squares system."""

    name = "Payment-form WLS, coding-penalized"

    def __init__(self, lam=0.0, pool_weights=None):
        super().__init__()
        self.lam = lam
        self.pool_weights = dict(pool_weights or {})

    def fit(self, X, y, w, clusters=None):
        from scipy.optimize import lsq_linear
        A = self._design(X)
        names = self.names_ or [str(i) for i in range(A.shape[1])]
        sw = np.sqrt(w / w.mean())
        pen = np.array([np.sqrt(self.lam * self.pool_weights.get(n, 0.0)) for n in names])
        # rows sqrt(lam p_j) e_j add lam p_j beta_j^2 to the objective; scale
        # to the weighted mean square so lam is in units of persons
        pen = pen * np.sqrt(len(y))
        Aa = np.vstack([A * sw[:, None], np.diag(pen)[pen > 0]])
        ya = np.concatenate([y * sw, np.zeros(int((pen > 0).sum()))])
        lo = np.zeros(A.shape[1])
        if self.columns_ is None:
            lo[0] = -np.inf
        res = lsq_linear(Aa, ya, bounds=(lo, np.inf), lsmr_tol="auto", max_iter=500)
        self.coef_ = res.x
        return self


class CappedPaymentWLS(PaymentWLS):
    """Payment-form WLS with an upper bound on each named coefficient
    (`caps`: column -> dollars). Cells and unnamed columns are unbounded above."""

    name = "Payment-form WLS, cost-capped"

    def __init__(self, caps=None):
        super().__init__()
        self.caps = dict(caps or {})

    def fit(self, X, y, w, clusters=None):
        from scipy.optimize import lsq_linear
        A = self._design(X)
        names = self.names_ or [str(i) for i in range(A.shape[1])]
        lo = np.zeros(A.shape[1])
        hi = _upper(names, self.caps)
        if self.columns_ is None:
            lo[0] = -np.inf
        sw = np.sqrt(w / w.mean())
        res = lsq_linear(A * sw[:, None], y * sw, bounds=(lo, hi), lsmr_tol="auto", max_iter=500)
        self.coef_ = res.x
        return self


class DROPaymentWLS(PaymentWLS):
    """Payment-form WLS with a penalty on the worst subgroup mispricing:

        mean_w (y - f)^2 + lam * (CVaR_alpha(m - f)^2 + CVaR_alpha(f - m)^2)

    where m is a `reference` expected cost for each training row (a flexible
    model, cross-fitted) and CVaR_alpha is the weighted mean over the largest
    alpha share. The two terms are the largest expected underpayment and
    overpayment of any subgroup of weight share alpha that can be assembled
    from what the reference model sees, which is what a plan that selects on
    observables can reach. The penalty is on expected, not realized, residuals:
    the largest realized residuals belong to people who had a catastrophic
    year, whom no formula predicts and no plan can pick out. Without a
    reference, y is used and the penalty acts on realized residuals.

    Fitted by L-BFGS-B with bounds, warm-started at the capped formula. The
    CVaR is written in the Rockafellar-Uryasev form with the hinge smoothed
    by a softplus of temperature tau (in units of the root mean square of
    cost). Without normalization the objective is convex; normalizing
    payments to total cost inside it makes it non-convex, so the solution is
    a local optimum from the warm start.
    """

    name = "Payment-form WLS, distributionally robust"

    def __init__(self, lam=0.0, alpha=0.10, caps=None, lam_code=0.0, pool_weights=None,
                 reference=None, max_iter=2000, normalize=True, tau=0.005):
        super().__init__()
        self.lam, self.alpha, self.max_iter = lam, alpha, max_iter
        self.normalize, self.tau = normalize, tau
        self.caps = dict(caps or {})
        self.reference = None if reference is None else np.asarray(reference, float)
        # optional coding penalty (as PenalizedPaymentWLS) in the same fit
        self.lam_code, self.pool_weights = lam_code, dict(pool_weights or {})

    def fit(self, X, y, w, clusters=None, reference=None):
        from scipy.optimize import minimize
        base = CappedPaymentWLS(self.caps).set_columns(self.columns_) if self.columns_ else \
            CappedPaymentWLS(self.caps)
        base.fit(X, y, w)
        A = base._design(X)
        self.names_ = base.names_
        names = self.names_ or [str(i) for i in range(A.shape[1])]
        wn = w / w.sum()
        scale = float(np.sqrt(np.average(y ** 2, weights=wn)))
        m = self.reference if reference is None else np.asarray(reference, float)
        m = y if m is None else m
        assert len(m) == len(y), "reference must align with the training rows"
        # optimize coefficients in units of `scale` (the root mean square of
        # y) so they are of order one, which the quasi-Newton solver needs;
        # the whole objective is then in scaled units and lam_code is
        # comparable with PenalizedPaymentWLS.lam
        As, ys, ms = A, y / scale, m / scale
        lo = np.zeros(A.shape[1])
        hi = _upper(names, self.caps) / scale
        if self.columns_ is None:
            lo[0] = -np.inf
        lam, alpha = self.lam, self.alpha
        pen = np.array([self.lam_code * self.pool_weights.get(n, 0.0) for n in names])
        S_y = float(np.sum(wn * ys))
        tau = self.tau
        # CVaR in the Rockafellar-Uryasev form, CVaR_a(e) = min_t t + E[(e - t)+] / a,
        # with the hinge smoothed by a softplus of temperature tau (in units
        # of scale) so the objective is smooth in (b, t) and the quasi-Newton
        # solver does not stall at the kinks of the hard top-alpha set.
        from scipy.special import expit

        def obj(z):
            b, t = z[:-2], z[-2:]
            u = As @ b
            if self.normalize:
                # payments are normalized to total cost inside the objective,
                # so the penalties act on the formula as it would be paid
                m_u = float(np.sum(wn * u))
                c = S_y / m_u
                dc = -(c / m_u) * (As.T @ wn)          # dc/db
            else:
                c, dc = 1.0, np.zeros_like(b)
            f = c * u
            r = ys - f
            val = float(np.sum(wn * r ** 2))
            g = -2 * wn * r                            # dL/df
            gt = np.zeros(2)
            e = ms - f                                 # expected mispricing
            for i, sign in enumerate((1.0, -1.0)):
                x = (sign * e - t[i]) / tau
                sp = tau * np.logaddexp(0.0, x)
                sig = expit(x)
                cv = t[i] + float(np.sum(wn * sp)) / alpha
                val += lam * cv ** 2
                g += lam * 2 * cv * (-sign * wn * sig / alpha)
                gt[i] = lam * 2 * cv * (1.0 - float(np.sum(wn * sig)) / alpha)
            # df/db = c As - u (dc)^T ; grad = As^T (c g) + dc * (g . u)
            grad = c * (As.T @ g) + dc * float(g @ u)
            be = c * b
            val += float(np.sum(pen * be ** 2))
            grad += 2 * c * pen * be + dc * float(2 * np.sum(pen * be * b))
            return val, np.concatenate([grad, gt])

        b0 = base.coef_ / scale
        e0 = ms - As @ b0
        t0 = np.array([_cvar(e0, wn, alpha)[0], _cvar(-e0, wn, alpha)[0]])
        bounds = list(zip(lo, hi)) + [(-np.inf, np.inf)] * 2
        x, ok, fun = np.concatenate([b0, t0]), False, np.inf
        for _ in range(4):
            res = minimize(obj, x, jac=True, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": self.max_iter, "maxfun": 4 * self.max_iter,
                                    "ftol": 1e-14, "gtol": 1e-9})
            stuck = abs(fun - float(res.fun)) <= 1e-9 * max(abs(fun), 1e-12)
            x, ok, fun = res.x, bool(res.success) or stuck, float(res.fun)
            self.message_ = str(res.message)
            if ok:
                break
        self.t_ = x[-2:] * scale
        x = x[:-2]
        coef = x * scale
        if self.normalize:
            self.factor_ = float(np.sum(w * y) / np.sum(w * (A @ coef)))
            coef = coef * self.factor_
        self.coef_ = coef
        self.converged_ = ok
        self.objective_ = fun
        return self

    def worst_groups(self, X, y, w):
        """Largest subgroup underpayment and overpayment at this alpha, in
        dollars, against y (pass the reference expected cost for the
        quantity the fit penalizes)."""
        r = y - self.predict(X)
        under, _ = _cvar(r, w, self.alpha)
        over, _ = _cvar(-r, w, self.alpha)
        return {"underpaid": under, "overpaid": over}


def incremental_costs(X, y, w, columns, pool, strata_cols=("female",), n_conditions_col=None,
                      prefix="ccsr_", max_conditions=6):
    """Incremental cost of each flag in `pool`: the weighted difference in y
    between people with and without the flag, within cells of the columns in
    `strata_cols` (age cells taken from columns starting `age_`) and a capped
    count of conditions, averaged over cells with the flag carriers' weight.

    This is what a code is worth in the ungamed data, and the cap a
    cost-capped formula puts on its coefficient. Negative values are floored
    at zero.
    """
    X = np.asarray(X, float)
    idx = {c: i for i, c in enumerate(columns)}
    age_cols = [i for i, c in enumerate(columns) if c.startswith("age_") and not c.endswith("_x_female")]
    cell = np.zeros(len(X), int)
    for k, i in enumerate(age_cols):
        cell += (k + 1) * (X[:, i] > 0)
    for c in strata_cols:
        if c in idx:
            cell = cell * 2 + (X[:, idx[c]] > 0)
    j_cnt = idx.get(n_conditions_col) if n_conditions_col else None
    out = {}
    for c in pool:
        j = idx[c]
        has = X[:, j] > 0
        cell_c = cell
        if j_cnt is not None:
            # hold the number of *other* conditions fixed, so a person with
            # the flag is compared with one who has as many other flags
            other = np.minimum(X[:, j_cnt] - X[:, j], max_conditions).astype(int)
            cell_c = cell * (max_conditions + 1) + other
        num = den = 0.0
        for g in np.unique(cell_c[has]):
            m = cell_c == g
            a, b = m & has, m & ~has
            if w[a].sum() == 0 or w[b].sum() == 0:
                continue
            diff = np.average(y[a], weights=w[a]) - np.average(y[b], weights=w[b])
            num += w[a].sum() * diff
            den += w[a].sum()
        out[c] = max(num / den, 0.0) if den > 0 else np.nan
    return out


class StackedRobustGBM:
    """Cross-fitted LightGBM score plus the features, fitted with
    DROPaymentWLS: the boosted formula with caps on the condition flags and
    the worst-subgroup penalty on the linear layer. The score enters as a
    non-negative slope.

    As in `fairness.StackedFairGBM`, the training-row scores are out of fold
    within the training data.
    """

    name = "Stacked LightGBM, capped and distributionally robust"

    def __init__(self, columns, lam=0.0, alpha=0.10, caps=None, lam_code=0.0, pool_weights=None,
                 objective="tweedie", k=3, seed=2026):
        self.columns = list(columns)
        self.lam, self.alpha, self.caps = lam, alpha, dict(caps or {})
        self.lam_code, self.pool_weights = lam_code, dict(pool_weights or {})
        self.objective, self.k, self.seed = objective, k, seed

    def _make(self):
        from .models import GBM
        return GBM(self.objective, seed=self.seed)

    def fit(self, X, y, w, clusters=None):
        from .fairness import _cross_fit_scores
        X = np.asarray(X, float)
        cl = np.arange(len(y)) if clusters is None else np.asarray(clusters)
        score = _cross_fit_scores(self._make, X, y, w, cl, k=self.k, seed=self.seed)
        self.gbm_ = self._make().fit(X, y, w, clusters=cl)
        self.layer_ = DROPaymentWLS(self.lam, self.alpha, self.caps, self.lam_code, self.pool_weights,
                                    reference=score)
        self.layer_.set_columns(["score"] + self.columns)
        self.layer_.fit(np.column_stack([score, X]), y, w)
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        return self.layer_.predict(np.column_stack([self.gbm_.predict(X), X]))

    def coefficients(self):
        return self.layer_.coefficients()
