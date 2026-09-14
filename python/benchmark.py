"""
RQ1: out-of-sample accuracy of every model, with survey-aware cross-validation.

Each model is fitted on the training PSUs of each fold and predicts the held-out
PSUs, repeated over several random fold assignments. Every person therefore
receives one out-of-fold prediction per repeat. Metrics are computed on the
out-of-fold predictions averaged over repeats; uncertainty comes from a
bootstrap over PSUs within strata applied to those predictions, and the
between-repeat spread is reported alongside.

Writes Tables 2 and 2b and data/derived/oof/*.npy.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

import common
import config
import features
from common import metrics, registry
from common import surveycv as cv

OOF = config.DERIVED / "oof"
OOF.mkdir(parents=True, exist_ok=True)

# Models run on every feature set for Table 2b; the rest on the primary set.
NESTED_MODELS = ("wls", "tweedie", "gbm")
BOOT_REPS = int(os.environ.get("P6_BOOT", 200))
ONLY = os.environ.get("P6_MODELS")          # comma list to run a subset


def run(model_key, feature_set, data, folds=None, cap=None, tag=None):
    X, y, w, cl, st, attrs = features.build(feature_set, data=data)
    Xn = X.to_numpy(np.float32)
    y_fit = np.minimum(y, cap) if cap else y
    folds = folds or common.folds(cl, st)
    reps = sorted({r for r, *_ in folds})
    preds = np.full((len(reps), len(y)), np.nan)
    params, t0 = [], time.time()
    for rep, fold, tr, te in folds:
        m = registry()[model_key]()
        m.fit(Xn[tr], y_fit[tr], w[tr], clusters=cl[tr])
        preds[rep, te] = m.predict(Xn[te])
        if hasattr(m, "params_"):
            params.append(m.params_)
    assert not np.isnan(preds).any(), "a person missed an out-of-fold prediction"
    name = tag or f"{model_key}_{feature_set}" + (f"_cap{cap}" if cap else "")
    np.save(OOF / f"{name}.npy", preds)
    return preds, y_fit, w, cl, st, attrs, params, time.time() - t0


def summarise(model_key, feature_set, preds, y, w, cl, st, seconds):
    p = preds.mean(axis=0)
    acc = metrics.accuracy(y, p, w)
    per_rep = [metrics.r2(y, preds[r], w) for r in range(len(preds))]

    def boot(fn):
        return metrics.cluster_bootstrap(lambda yy, pp, ww, _: fn(yy, pp, ww),
                                         y, p, w, cl, st, reps=BOOT_REPS)
    r2_b = boot(metrics.r2)
    cpm_b = boot(metrics.cpm)
    return {"model": registry()[model_key]().name, "model_key": model_key,
            "feature_set": feature_set, **acc,
            "r2_lo": float(np.percentile(r2_b, 2.5)),
            "r2_hi": float(np.percentile(r2_b, 97.5)),
            "cpm_lo": float(np.percentile(cpm_b, 2.5)),
            "cpm_hi": float(np.percentile(cpm_b, 97.5)),
            "r2_repeat_sd": float(np.std(per_rep, ddof=1)) if len(per_rep) > 1 else np.nan,
            "seconds": round(seconds, 1)}


def main():
    data = features.load()
    X, y, w, cl, st, attrs = features.build(config.PRIMARY_FEATURE_SET, data=data)
    folds = common.folds(cl, st)
    cv.check_disjoint(folds, cl)
    print(f"{len(y):,} persons, {len(np.unique(cl)):,} PSUs, "
          f"{config.CV_FOLDS} folds x {config.CV_REPEATS} repeats\n")

    keys = list(registry())
    if ONLY:
        keys = [k for k in keys if k in ONLY.split(",")]

    rows, tuning = [], {}
    for k in keys:
        preds, yy, ww, cc, ss, _, params, sec = run(k, config.PRIMARY_FEATURE_SET,
                                                    data, folds)
        r = summarise(k, config.PRIMARY_FEATURE_SET, preds, yy, ww, cc, ss, sec)
        rows.append(r)
        tuning[k] = params
        print(f"  {r['model']:<26} R2 {r['r2']:.3f} [{r['r2_lo']:.3f}, {r['r2_hi']:.3f}]"
              f"  CPM {r['cpm']:.3f}  PR top decile {r['pr_top_decile']:.2f}"
              f"  top-10% capture {r['top10_capture']:.2f}  ({sec:.0f}s)")
    t2 = pd.DataFrame(rows)
    t2.to_csv(config.TABLES / "table2_accuracy.csv", index=False)

    rows = []
    for k in [m for m in NESTED_MODELS if m in keys]:
        for fs in config.FEATURE_SETS:
            preds, yy, ww, cc, ss, _, _, sec = run(k, fs, data, folds)
            rows.append(summarise(k, fs, preds, yy, ww, cc, ss, sec))
            print(f"  {rows[-1]['model']:<26} {fs}  R2 {rows[-1]['r2']:.3f}  "
                  f"CPM {rows[-1]['cpm']:.3f}")
    pd.DataFrame(rows).to_csv(config.TABLES / "table2b_feature_sets.csv", index=False)

    # Calibration by decile of predicted spending, primary feature set.
    cal = []
    for k in keys:
        p = np.load(OOF / f"{k}_{config.PRIMARY_FEATURE_SET}.npy").mean(axis=0)
        g = metrics.weighted_quantile_groups(p, w, 10)
        for d in range(10):
            m = g == d
            cal.append({"model": registry()[k]().name, "decile": d + 1,
                        "mean_predicted": float(np.average(p[m], weights=w[m])),
                        "mean_observed": float(np.average(y[m], weights=w[m])),
                        "predictive_ratio": float(np.sum(w[m] * p[m]) / np.sum(w[m] * y[m]))})
    pd.DataFrame(cal).to_csv(config.TABLES / "table2c_calibration.csv", index=False)
    (config.TABLES / "tuning_chosen.json").write_text(json.dumps(tuning, indent=1,
                                                                 default=str))
    print("\nwrote tables 2, 2b, 2c")


if __name__ == "__main__":
    main()
