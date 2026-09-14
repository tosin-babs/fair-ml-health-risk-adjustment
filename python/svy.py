"""
Complex-survey estimation for a stratified, single-stage cluster design.

The GHS-Panel selects enumeration areas (PSUs) within strata and interviews a
fixed number of households in each. Every point estimate in the paper is
weighted, and every standard error is a Taylor-linearised, cluster-robust
estimate that respects that design. The formulas follow Lumley (2004) /
`R survey`, so results are directly comparable with the R implementations used
elsewhere in this literature.

Nothing here is specific to Nigeria; the module is survey-agnostic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import config


# ---------------------------------------------------------------------------
# Design object
# ---------------------------------------------------------------------------
class Design:
    """A stratified single-stage cluster design with unequal weights.

    Parameters
    ----------
    data : DataFrame holding the analysis units (households or individuals).
    weights, strata, psu : column names.
    lonely : how to treat strata that contain a single PSU. "adjust" centres
        the stratum's cluster totals on the grand mean rather than on the
        stratum mean, which is R's ``survey.lonely.psu = "adjust"``. The
        alternative, "certainty", contributes zero variance.
    """

    def __init__(self, data, weights, strata, psu, lonely=config.LONELY_PSU):
        keep = data[[weights, strata, psu]].notna().all(axis=1)
        if not keep.all():
            data = data.loc[keep]
        self.data = data.reset_index(drop=True)
        self.w = self.data[weights].to_numpy(float)
        self.h = self.data[strata].to_numpy()
        self.c = self.data[psu].astype(str).to_numpy()
        self.lonely = lonely
        # A PSU label is only unique within its stratum, so combine the two.
        self.hc = np.char.add(np.asarray(self.h).astype(str), np.char.add("|", self.c))

    def __len__(self):
        return len(self.data)

    def subset(self, mask):
        """Domain estimation: keep the full design, zero the weights outside it.

        Dropping rows instead would understate the variance, because the number
        of sampled PSUs is itself random within a domain.
        """
        mask = np.asarray(mask, dtype=bool)
        new = object.__new__(Design)
        new.data = self.data
        new.w = np.where(mask, self.w, 0.0)
        new.h, new.c, new.hc, new.lonely = self.h, self.c, self.hc, self.lonely
        return new

    # -- variance of a weighted total -------------------------------------
    def _var_total(self, values):
        """Var of T = sum_i w_i * values_i under stratified cluster sampling.

        `values` may be 2-D (n x k); the k x k covariance matrix is returned.
        """
        v = np.asarray(values, float)
        one_d = v.ndim == 1
        if one_d:
            v = v[:, None]
        contrib = self.w[:, None] * v

        # Collapse to PSU totals, remembering each PSU's stratum.
        psu_idx, psu_labels = pd.factorize(self.hc)
        k = contrib.shape[1]
        psu_tot = np.zeros((len(psu_labels), k))
        np.add.at(psu_tot, psu_idx, contrib)
        psu_stratum = pd.Series(self.h).groupby(psu_idx).first().to_numpy()

        grand_mean = psu_tot.mean(axis=0)
        cov = np.zeros((k, k))
        for stratum in pd.unique(psu_stratum):
            sel = psu_stratum == stratum
            n_h = int(sel.sum())
            t = psu_tot[sel]
            if n_h > 1:
                dev = t - t.mean(axis=0)
                cov += (n_h / (n_h - 1.0)) * dev.T @ dev
            elif self.lonely == "adjust":
                dev = t - grand_mean
                cov += dev.T @ dev
            # "certainty": a single-PSU stratum contributes nothing.
        return cov[0, 0] if one_d else cov

    # -- estimators --------------------------------------------------------
    def total(self, y):
        y = np.asarray(y, float)
        return float(self.w @ y), float(np.sqrt(self._var_total(y)))

    def mean(self, y):
        """Weighted mean, with the ratio linearisation of its variance."""
        y = np.asarray(y, float)
        ok = np.isfinite(y)
        w = np.where(ok, self.w, 0.0)
        y = np.where(ok, y, 0.0)
        denom = w.sum()
        if denom <= 0:
            return np.nan, np.nan
        est = float(w @ y / denom)
        d = self.subset(ok)
        z = (y - est) / denom
        return est, float(np.sqrt(d._var_total(z)))

    def ratio(self, num, den):
        """Weighted ratio sum(w*num)/sum(w*den) with linearised variance."""
        num, den = np.asarray(num, float), np.asarray(den, float)
        ok = np.isfinite(num) & np.isfinite(den)
        d = self.subset(ok)
        num, den = np.nan_to_num(num), np.nan_to_num(den)
        D = float(d.w @ den)
        if D == 0:
            return np.nan, np.nan
        est = float(d.w @ num) / D
        z = (num - est * den) / D
        return est, float(np.sqrt(d._var_total(z)))

    def quantile(self, y, probs):
        """Weighted quantiles (no variance; used only for cut-points)."""
        y = np.asarray(y, float)
        ok = np.isfinite(y) & (self.w > 0)
        return weighted_quantile(y[ok], self.w[ok], probs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def weighted_quantile(values, weights, probs):
    """Weighted quantiles by linear interpolation of the weighted CDF."""
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    # Midpoint convention: matches Stata/R type-4 closely enough for cut-points.
    cdf = (cw - 0.5 * w) / w.sum()
    return np.interp(np.atleast_1d(probs), cdf, v)


def ci(est, se, level=0.95):
    z = stats.norm.ppf(0.5 + level / 2)
    return est - z * se, est + z * se


def estimate_by(design, y, by, statistic="mean", denom=None, level=0.95):
    """Run `statistic` over each level of `by`, returning a tidy DataFrame.

    Domains are handled by zeroing weights rather than subsetting rows, so the
    standard errors account for the design correctly.
    """
    y = np.asarray(y, float)
    groups = pd.Series(np.asarray(by)).reset_index(drop=True)
    rows = []
    for g in [x for x in pd.unique(groups.dropna())]:
        mask = (groups == g).to_numpy()
        d = design.subset(mask)
        if statistic == "mean":
            est, se = d.mean(y)
        elif statistic == "total":
            est, se = d.total(np.where(mask, y, 0.0))
        elif statistic == "ratio":
            est, se = d.ratio(y, np.asarray(denom, float))
        else:
            raise ValueError(statistic)
        lo, hi = ci(est, se, level)
        rows.append(
            {
                "group": g,
                "estimate": est,
                "se": se,
                "ci_low": lo,
                "ci_high": hi,
                "n": int(mask.sum()),
                "wtd_n": float(design.w[mask].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def diff_test(est1, se1, est2, se2):
    """Wald test for a difference between two independent-domain estimates.

    Domains from the same survey are not strictly independent, so the covariance
    term is omitted; this is the conventional approximation and is conservative
    when the domains are positively correlated.
    """
    d = est1 - est2
    se = np.sqrt(se1**2 + se2**2)
    if se == 0:
        return d, np.nan, np.nan
    z = d / se
    return d, z, 2 * (1 - stats.norm.cdf(abs(z)))


# ---------------------------------------------------------------------------
# Concentration indices
# ---------------------------------------------------------------------------
def fractional_rank(living_standard, weights):
    """Weighted fractional rank in the living-standards distribution.

    Each unit is placed at the midpoint of the weight mass it occupies, which
    is the convention in O'Donnell et al. (2008).
    """
    ls = np.asarray(living_standard, float)
    w = np.asarray(weights, float)
    order = np.argsort(ls, kind="mergesort")
    w_sorted = w[order]
    cw = np.cumsum(w_sorted)
    rank_sorted = (cw - 0.5 * w_sorted) / w.sum()
    rank = np.empty_like(rank_sorted)
    rank[order] = rank_sorted
    return rank


def concentration_index(design, outcome, living_standard, binary=True):
    """Concentration index with the Erreygers (2009) correction.

    The CI is obtained from the "convenient covariance" identity
    CI = 2 * cov_w(y, r) / mean_w(y), and its standard error from the design,
    which keeps the clustering in the inference. For a binary outcome the
    Erreygers-corrected index E = 4 * mean(y) * CI is bounded by [-1, 1]
    regardless of the mean, which the raw CI is not.
    """
    y = np.asarray(outcome, float)
    ok = np.isfinite(y)
    d = design.subset(ok)
    y = np.nan_to_num(y)
    r = fractional_rank(np.asarray(living_standard, float), d.w)

    W = d.w.sum()
    mu = d.w @ y / W
    if mu == 0:
        return {"CI": np.nan, "CI_se": np.nan, "Erreygers": np.nan,
                "Erreygers_se": np.nan, "mean": 0.0}
    r_bar = d.w @ r / W
    cov_wr = d.w @ ((y - mu) * (r - r_bar)) / W
    idx = 2.0 * cov_wr / mu

    # Linearise CI = 2*cov(y,r)/mu as a smooth function of three totals.
    z = (2.0 / mu) * ((y - mu) * (r - r_bar) - cov_wr) / W - (idx / mu) * (y - mu) / W
    se = float(np.sqrt(d._var_total(z)))

    out = {"CI": float(idx), "CI_se": se, "mean": float(mu)}
    if binary:
        # E = 4 * mu * CI = 8 * cov_w(y, r), so its linearisation is that of
        # the covariance alone; scaling the CI's standard error by 4 * mu would
        # ignore the covariance between the estimated mean and the index.
        out["Erreygers"] = 4.0 * mu * idx
        z_e = 8.0 * ((y - mu) * (r - r_bar) - cov_wr) / W
        out["Erreygers_se"] = float(np.sqrt(d._var_total(z_e)))
    else:
        out["Erreygers"] = np.nan
        out["Erreygers_se"] = np.nan
    return out


def decompose_ci(design, outcome, living_standard, covariates):
    """Wagstaff-style linear decomposition of a concentration index.

    CI(y) = sum_k (beta_k * xbar_k / ybar) * CI(x_k) + GC(residual)/ybar.
    The regression is weighted; contributions are reported as levels and as a
    share of the total index. The decomposition is descriptive, not causal.
    """
    y = np.asarray(outcome, float)
    X = pd.DataFrame(covariates).astype(float)
    ok = np.isfinite(y) & X.notna().all(axis=1).to_numpy()
    d = design.subset(ok)
    y = np.nan_to_num(y)
    Xv = X.fillna(0.0).to_numpy()

    w = d.w
    A = np.column_stack([np.ones(len(y)), Xv])
    WA = A * w[:, None]
    beta = np.linalg.lstsq(WA.T @ A, WA.T @ y, rcond=None)[0]
    resid = y - A @ beta

    ybar = w @ y / w.sum()
    total = concentration_index(design.subset(ok), y, living_standard, binary=False)["CI"]

    rows = []
    for k, name in enumerate(X.columns):
        xk = Xv[:, k]
        xbar = w @ xk / w.sum()
        ci_k = concentration_index(d, xk, living_standard, binary=False)["CI"]
        contrib = beta[k + 1] * xbar / ybar * ci_k
        rows.append(
            {"variable": name, "beta": beta[k + 1], "mean_x": xbar,
             "CI_x": ci_k, "contribution": contrib,
             "share_pct": 100 * contrib / total if total else np.nan}
        )
    r = fractional_rank(np.asarray(living_standard, float), w)
    gc_resid = 2.0 * (w @ (resid * (r - (w @ r / w.sum())))) / w.sum() / ybar
    rows.append({"variable": "Residual", "beta": np.nan, "mean_x": np.nan,
                 "CI_x": np.nan, "contribution": gc_resid,
                 "share_pct": 100 * gc_resid / total if total else np.nan})
    out = pd.DataFrame(rows)
    out.attrs["total_CI"] = total
    return out
