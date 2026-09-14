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
        p = np.load(f).mean(axis=0)
        name = registry()[k]().name
        for g, m in masks.items():
            nc_b = metrics.cluster_bootstrap(
                lambda yy, pp, ww, idx, m=m: np.average(pp[m[idx]] - yy[m[idx]],
                                                        weights=ww[m[idx]])
                if m[idx].any() else np.nan,
                y, p, w, cl, st, reps=BOOT_REPS)
            gm = metrics.group_fairness(y, p, w, {g: m})[g]
            rows.append({"model": name, "model_key": k, "group": g, **gm,
                         "nc_lo": float(np.nanpercentile(nc_b, 2.5)),
                         "nc_hi": float(np.nanpercentile(nc_b, 97.5)),
                         "target_group": g in config.FAIR_TARGET_GROUPS})
    return pd.DataFrame(rows)


def frontier(X, y, w, cl, st, masks):
    folds = common.folds(cl, st, repeats=1)
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
    preds = {(m, l): np.full(len(y), np.nan) for m, l, _ in lin + stack}
    make_gbm = lambda: GBM("tweedie", seed=config.SEED)
    for _, fold, tr, te in folds:
        tmask = [masks[g][tr] for g in target]
        for m, l, kw in lin:
            preds[(m, l)][te] = (fairness.FairWLS(**kw)
                                 .fit(X[tr], y[tr], w[tr], tmask).predict(X[te]))
        s_tr = _cross_fit_scores(make_gbm, X[tr], y[tr], w[tr], cl[tr], k=3,
                                 seed=config.SEED)
        s_te = make_gbm().fit(X[tr], y[tr], w[tr], clusters=cl[tr]).predict(X[te])
        Ztr = np.column_stack([s_tr, X[tr]])
        Zte = np.column_stack([s_te, X[te]])
        for m, l, kw in stack:
            preds[(m, l)][te] = (fairness.FairWLS(**kw)
                                 .fit(Ztr, y[tr], w[tr], tmask).predict(Zte))
        print(f"  fold {fold}: linear and stacked fair fits done", flush=True)
    rows = []
    for m, l, _ in lin + stack:
        rows.append(_frontier_row(m, l, y, preds[(m, l)], w, masks, target))
        print(f"  {m:<30} lambda={l:<6g} R2 {rows[-1]['r2']:.3f}  "
              f"max|NC| target ${rows[-1]['max_abs_nc_target']:,.0f}")

    # Group-neutral post-processing of the Tweedie LightGBM: decile factors
    # are estimated on the out-of-fold predictions of the other folds, so the
    # recalibration of a fold never uses that fold's outcomes.
    base = np.load(OOF / f"gbm_{config.PRIMARY_FEATURE_SET}.npy")[0]
    fold_of = np.empty(len(y), dtype=int)
    for _, f, _, te in folds:
        fold_of[te] = f
    p = np.empty(len(y))
    for f in range(config.CV_FOLDS):
        te, tr = fold_of == f, fold_of != f
        p[te] = fairness.recalibrate_by_decile(base[tr], y[tr], w[tr], base[te])
    rows.append(_frontier_row("LightGBM, decile recalibration", 0.0, y, p, w,
                              masks, target))
    return pd.DataFrame(rows)


def _frontier_row(method, lam, y, p, w, masks, target):
    g = metrics.group_fairness(y, p, w, masks)
    row = {"method": method, "lambda": lam, "r2": metrics.r2(y, p, w),
           "cpm": metrics.cpm(y, p, w),
           "max_abs_nc_target": max(abs(g[t]["nc"]) for t in target),
           "max_abs_nc_all": max(abs(v["nc"]) for v in g.values())}
    for name, v in g.items():
        row[f"nc::{name}"] = v["nc"]
        row[f"pr::{name}"] = v["pr"]
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

    print("\n=== Accuracy-fairness frontier (out of fold) ===")
    t3b = frontier(X.to_numpy(np.float64), y, w, cl, st, masks)
    t3b.to_csv(config.TABLES / "table3b_frontier.csv", index=False)
    print("\nwrote tables 3 and 3b")


if __name__ == "__main__":
    main()
