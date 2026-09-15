"""
RQ2: who each model under- or over-pays, and what fairness costs.

Part 1 evaluates group predictive ratios and net compensation for every model
in Table 2, from out-of-fold predictions, with intervals from a PSU bootstrap.

Part 2 traces the accuracy-fairness frontier. Penalized WLS over a lambda grid,
constrained WLS, fair LightGBM over a shorter grid, and a group-neutral
decile recalibration of LightGBM are each fitted within the same survey folds
and evaluated out of fold on R-squared and on the largest absolute net
compensation among the target groups. Non-target groups are reported too,
because a penalty on some groups can move others.

Writes Tables 3 and 3b.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import common
import config
import features
from common import fairness, metrics, registry

OOF = config.DERIVED / "oof"
BOOT_REPS = int(os.environ.get("P6_BOOT", 200))
GBM_LAMBDAS = (0.0, 1.0, 10.0, 100.0)


def group_table(y, w, cl, st, masks):
    rows = []
    for k in registry():
        f = OOF / f"{k}_{config.PRIMARY_FEATURE_SET}.npy"
        if not f.exists():
            continue
        preds = np.load(f)
        name = registry()[k]().name
        wls = np.load(OOF / f"wls_{config.PRIMARY_FEATURE_SET}.npy")
        rng = np.random.default_rng(config.SEED)
        boots = [metrics.bootstrap_index(cl, st, rng) for _ in range(BOOT_REPS)]
        for g, m in masks.items():
            def nc(p, idx, mult):
                mm = m[idx]
                return np.average(p[idx][mm] - y[idx][mm], weights=(w[idx] * mult)[mm]) if mm.any() else np.nan
            nc_b = np.array([np.mean([nc(preds[r], idx, mult) for r in range(len(preds))]) for idx, mult in boots])
            d_b = np.array([np.mean([nc(preds[r], idx, mult) - nc(wls[r], idx, mult) for r in range(len(preds))])
                            for idx, mult in boots])
            gm = {key: float(np.mean([metrics.group_fairness(y, preds[r], w, {g: m})[g][key]
                                      for r in range(len(preds))]))
                  for key in ("pr", "nc", "mean_cost", "mae")}
            gm["n"] = int(m.sum())
            rows.append({"model": name, "model_key": k, "group": g, **gm,
                         "nc_lo": float(np.nanpercentile(nc_b, 2.5)),
                         "nc_hi": float(np.nanpercentile(nc_b, 97.5)),
                         "nc_minus_wls": float(np.mean([metrics.group_fairness(y, preds[r], w, {g: m})[g]["nc"]
                                                        - metrics.group_fairness(y, wls[r], w, {g: m})[g]["nc"]
                                                        for r in range(len(preds))])),
                         "nc_minus_wls_lo": float(np.nanpercentile(d_b, 2.5)),
                         "nc_minus_wls_hi": float(np.nanpercentile(d_b, 97.5)),
                         "target_group": g in config.FAIR_TARGET_GROUPS})
    return pd.DataFrame(rows)


def frontier(X, y, w, cl, st, masks):
    """Every estimator on every repeat of the survey folds; metrics are the
    mean over repeats of the out-of-fold value, with Rao-Wu PSU bootstrap
    intervals, and the constrained estimators' in-sample against out-of-fold
    net compensation is recorded fold by fold."""
    folds = common.folds(cl, st)
    n_rep = len({r for r, *_ in folds})
    target = [g for g in config.FAIR_TARGET_GROUPS if g in masks]
    # The stacked models differ only in the fair layer, so the cross-fitted
    # LightGBM scores are computed once per fold and reused for every lambda.
    # This is the same estimator as riskfair.fairness.StackedFairGBM.
    from riskfair.fairness import _cross_fit_scores
    from riskfair.models import GBM
    lin = ([("Penalized WLS", lam, dict(lam=lam)) for lam in config.FAIR_LAMBDAS]
           + [("Constrained WLS", np.inf, dict(constrained=True))])
    stack = ([("Stacked LightGBM", lam, dict(lam=lam)) for lam in config.FAIR_LAMBDAS]
             + [("Stacked LightGBM, constrained", np.inf, dict(constrained=True))])
    preds = {(m, l): np.full((n_rep, len(y)), np.nan) for m, l, _ in lin + stack}
    make_gbm = lambda: GBM("tweedie", seed=config.SEED)
    per_fold = []
    for rep, fold, tr, te in folds:
        tmask = [masks[g][tr] for g in target]
        for m, l, kw in lin:
            est = fairness.FairWLS(**kw).fit(X[tr], y[tr], w[tr], tmask)
            preds[(m, l)][rep, te] = est.predict(X[te])
            if np.isinf(l):
                per_fold.append(_fold_gap(m, rep, fold, est.predict(X[tr]), est.predict(X[te]),
                                          y, w, masks, target, tr, te))
        s_tr = _cross_fit_scores(make_gbm, X[tr], y[tr], w[tr], cl[tr], k=3,
                                 seed=config.SEED + rep)
        s_te = make_gbm().fit(X[tr], y[tr], w[tr], clusters=cl[tr]).predict(X[te])
        Ztr = np.column_stack([s_tr, X[tr]])
        Zte = np.column_stack([s_te, X[te]])
        for m, l, kw in stack:
            est = fairness.FairWLS(**kw).fit(Ztr, y[tr], w[tr], tmask)
            preds[(m, l)][rep, te] = est.predict(Zte)
            if np.isinf(l):
                per_fold.append(_fold_gap(m, rep, fold, est.predict(Ztr), est.predict(Zte),
                                          y, w, masks, target, tr, te))
        print(f"  repeat {rep} fold {fold}: linear and stacked fair fits done", flush=True)
    pd.DataFrame(per_fold).to_csv(config.TABLES / "table3c_constraint_by_fold.csv", index=False)
    rows = []
    for m, l, _ in lin + stack:
        rows.append(_frontier_row(m, l, y, preds[(m, l)], w, cl, st, masks, target))
        print(f"  {m:<30} lambda={l:<6g} R2 {rows[-1]['r2']:.3f}  "
              f"max|NC| target ${rows[-1]['max_abs_nc_target']:,.0f} "
              f"[{rows[-1]['max_abs_nc_target_lo']:,.0f}, {rows[-1]['max_abs_nc_target_hi']:,.0f}]")

    # Group-neutral post-processing of the Tweedie LightGBM: decile factors
    # are estimated on the out-of-fold predictions of the other folds, so the
    # recalibration of a fold never uses that fold's outcomes.
    base_all = np.load(OOF / f"gbm_{config.PRIMARY_FEATURE_SET}.npy")
    p = np.empty((n_rep, len(y)))
    for rep in range(n_rep):
        fold_of = np.empty(len(y), dtype=int)
        for r_, f, _, te in folds:
            if r_ == rep:
                fold_of[te] = f
        for f in range(config.CV_FOLDS):
            te, tr = fold_of == f, fold_of != f
            p[rep, te] = fairness.recalibrate_by_decile(base_all[rep][tr], y[tr], w[tr], base_all[rep][te])
    rows.append(_frontier_row("LightGBM, decile recalibration", 0.0, y, p, w, cl, st, masks, target))
    return pd.DataFrame(rows)


def _fold_gap(method, rep, fold, p_tr, p_te, y, w, masks, target, tr, te):
    row = {"method": method, "repeat": rep, "fold": fold}
    for g in target + ["Age 65 and over"]:
        if g not in masks:
            continue
        m_tr, m_te = masks[g][tr], masks[g][te]
        row[f"in_sample::{g}"] = float(np.average(p_tr[m_tr] - y[tr][m_tr], weights=w[tr][m_tr]))
        row[f"out_of_fold::{g}"] = float(np.average(p_te[m_te] - y[te][m_te], weights=w[te][m_te]))
    return row


def _frontier_row(method, lam, y, preds, w, cl, st, masks, target):
    """Mean over repeats of each out-of-fold metric, with bootstrap intervals."""
    n_rep = preds.shape[0]
    g = [metrics.group_fairness(y, preds[r], w, masks) for r in range(n_rep)]
    row = {"method": method, "lambda": lam,
           "r2": float(np.mean([metrics.r2(y, preds[r], w) for r in range(n_rep)])),
           "cpm": float(np.mean([metrics.cpm(y, preds[r], w) for r in range(n_rep)])),
           "max_abs_nc_target": float(np.mean([max(abs(g[r][t]["nc"]) for t in target) for r in range(n_rep)])),
           "max_abs_nc_all": float(np.mean([max(abs(v["nc"]) for v in g[r].values()) for r in range(n_rep)]))}
    for name in g[0]:
        row[f"nc::{name}"] = float(np.mean([g[r][name]["nc"] for r in range(n_rep)]))
        row[f"pr::{name}"] = float(np.mean([g[r][name]["pr"] for r in range(n_rep)]))
    rng = np.random.default_rng(config.SEED)
    b_r2, b_gap = [], []
    for _ in range(BOOT_REPS):
        idx, mult = metrics.bootstrap_index(cl, st, rng)
        ww = w[idx] * mult
        b_r2.append(np.mean([metrics.r2(y[idx], preds[r][idx], ww) for r in range(n_rep)]))
        gaps = []
        for r in range(n_rep):
            gg = metrics.group_fairness(y[idx], preds[r][idx], ww, {t: masks[t][idx] for t in target})
            gaps.append(max(abs(gg[t]["nc"]) for t in target))
        b_gap.append(np.mean(gaps))
    row["r2_lo"], row["r2_hi"] = np.percentile(b_r2, 2.5), np.percentile(b_r2, 97.5)
    row["max_abs_nc_target_lo"], row["max_abs_nc_target_hi"] = np.percentile(b_gap, 2.5), np.percentile(b_gap, 97.5)
    return row


def main():
    data = features.load()
    X, y, w, cl, st, attrs = features.build(config.PRIMARY_FEATURE_SET, data=data)
    masks = features.group_masks(attrs)

    t3 = group_table(y, w, cl, st, masks)
    t3.to_csv(config.TABLES / "table3_group_fairness.csv", index=False)
    print("=== Net compensation by group, primary feature set ($ per person-year) ===")
    piv = t3.pivot_table(index="group", columns="model", values="nc")
    print(piv.round(0).to_string())

    if os.environ.get("P6_SKIP_FRONTIER") == "1" and (config.TABLES / "table3b_frontier.csv").exists():
        print("\n  frontier reused (P6_SKIP_FRONTIER=1); it does not involve the payment-form model")
    else:
        print("\n=== Accuracy-fairness frontier (out of fold) ===")
        t3b = frontier(X.to_numpy(np.float64), y, w, cl, st, masks)
        t3b.to_csv(config.TABLES / "table3b_frontier.csv", index=False)
    print("\nwrote tables 3 and 3b")


if __name__ == "__main__":
    main()
