"""
Paired comparisons and cell calibration from the saved out-of-fold predictions.

The benchmark reports each model's R² with its own bootstrap interval, which
says nothing about whether two models differ: the same held-out persons and
the same PSU resamples underlie every model, so differences are far more
precise than the separate intervals suggest. This step resamples PSUs within
strata (Rao-Wu rescaled) and evaluates every model on the same resample,
giving paired intervals for the difference in R², Cumming's prediction
measure and mean absolute error against WLS, and for the boosting advantage
on each feature set. Every statistic is the mean over repeats of a single
fitted model's value, as in Table 2. It also reports each model's R² per
repeat against the R² of the repeat-averaged prediction, so the ensemble gain
from averaging three fold assignments is visible, and predictive ratios by
age band and sex, the cells a payment formula is built on.

Writes Tables 2d (paired differences), 2e (repeat averaging) and 2f (cell
calibration).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
import features
from common import metrics, registry

OOF = config.DERIVED / "oof"
REPS = 500


def _stats(y, preds, w):
    """Mean over repeats of R², CPM and MAE; preds has one row per repeat."""
    return np.mean([(metrics.r2(y, p, w), metrics.cpm(y, p, w), metrics.mae(y, p, w)) for p in preds], axis=0)


def paired_bootstrap(y, w, cl, st, preds, reps=REPS, seed=config.SEED):
    """For each Rao-Wu PSU resample, every model's statistics on the same rows."""
    rng = np.random.default_rng(seed)
    out = {k: np.zeros((reps, 3)) for k in preds}
    for b in range(reps):
        idx, mult = metrics.bootstrap_index(cl, st, rng)
        yy, ww = y[idx], w[idx] * mult
        for k, p in preds.items():
            out[k][b] = _stats(yy, p[:, idx], ww)
    return out


def main():
    data = features.load()
    X, y, w, cl, st, attrs = features.build(config.PRIMARY_FEATURE_SET, data=data)
    names = {k: registry()[k]().name for k in registry()}
    fs = config.PRIMARY_FEATURE_SET
    preds = {k: np.load(OOF / f"{k}_{fs}.npy") for k in names if (OOF / f"{k}_{fs}.npy").exists()}
    avg = {k: p.mean(axis=0) for k, p in preds.items()}
    print(f"{len(y):,} persons, {len(np.unique(cl)):,} PSUs in {len(np.unique(st)):,} strata; "
          f"{len(avg)} models on {fs}")

    # --- 2d: paired differences against WLS, primary feature set
    boot = paired_bootstrap(y, w, cl, st, preds)
    rows = []
    for k in preds:
        if k == "wls":
            continue
        d = boot[k] - boot["wls"]
        point = _stats(y, preds[k], w) - _stats(y, preds["wls"], w)
        rows.append({"feature_set": fs, "model": names[k], "against": "WLS",
                     "d_r2": point[0], "d_r2_lo": np.percentile(d[:, 0], 2.5), "d_r2_hi": np.percentile(d[:, 0], 97.5),
                     "p_r2_le_0": float((d[:, 0] <= 0).mean()),
                     "d_cpm": point[1], "d_cpm_lo": np.percentile(d[:, 1], 2.5), "d_cpm_hi": np.percentile(d[:, 1], 97.5),
                     "d_mae": point[2], "d_mae_lo": np.percentile(d[:, 2], 2.5), "d_mae_hi": np.percentile(d[:, 2], 97.5)})
    # boosting and the payment form against WLS on every other feature set
    for f in config.FEATURE_SETS:
        if f == fs:
            continue
        for k in ("gbm", "pwls"):
            pk, pw_ = OOF / f"{k}_{f}.npy", OOF / f"wls_{f}.npy"
            if not (pk.exists() and pw_.exists()):
                continue
            pair = {k: np.load(pk), "wls": np.load(pw_)}
            b = paired_bootstrap(y, w, cl, st, pair)
            d = b[k] - b["wls"]
            point = _stats(y, pair[k], w) - _stats(y, pair["wls"], w)
            rows.append({"feature_set": f, "model": names[k], "against": "WLS",
                         "d_r2": point[0], "d_r2_lo": np.percentile(d[:, 0], 2.5),
                         "d_r2_hi": np.percentile(d[:, 0], 97.5), "p_r2_le_0": float((d[:, 0] <= 0).mean()),
                         "d_cpm": point[1], "d_cpm_lo": np.percentile(d[:, 1], 2.5),
                         "d_cpm_hi": np.percentile(d[:, 1], 97.5),
                         "d_mae": point[2], "d_mae_lo": np.percentile(d[:, 2], 2.5),
                         "d_mae_hi": np.percentile(d[:, 2], 97.5)})
    t2d = pd.DataFrame(rows)
    t2d.to_csv(config.TABLES / "table2d_paired_differences.csv", index=False)

    # --- 2e: per-repeat R² against the averaged prediction
    rows = []
    for k, p in preds.items():
        per = [metrics.r2(y, p[r], w) for r in range(len(p))]
        rows.append({"model": names[k], "r2_single_repeat_mean": float(np.mean(per)),
                     "r2_single_repeat_min": float(np.min(per)), "r2_single_repeat_max": float(np.max(per)),
                     "r2_repeat_averaged": metrics.r2(y, avg[k], w),
                     "averaging_gain": metrics.r2(y, avg[k], w) - float(np.mean(per))})
    t2e = pd.DataFrame(rows)
    t2e.to_csv(config.TABLES / "table2e_repeat_averaging.csv", index=False)

    # --- 2f: predictive ratios by age band and sex
    age = attrs["age"].to_numpy(float)
    band = pd.cut(age, [-1, 17, 34, 44, 54, 64, 200], labels=["0-17", "18-34", "35-44", "45-54", "55-64", "65+"])
    rows = []
    for k in ("wls", "pwls", "tweedie", "gbm", "cann"):
        if k not in preds:
            continue
        for b in band.categories:
            for fem in (0, 1):
                m = (band == b) & (attrs["female"].to_numpy() == fem)
                pr = np.mean([np.sum(w[m] * p[m]) / np.sum(w[m] * y[m]) for p in preds[k]])
                rows.append({"model": names[k], "age_band": b, "sex": "female" if fem else "male",
                             "n": int(m.sum()), "mean_observed": float(np.average(y[m], weights=w[m])),
                             "predictive_ratio": float(pr)})
    t2f = pd.DataFrame(rows)
    t2f.to_csv(config.TABLES / "table2f_cell_calibration.csv", index=False)

    with pd.option_context("display.width", 200, "display.float_format", "{:,.3f}".format):
        print("\n=== Paired differences against WLS (PSU bootstrap, same resamples) ===")
        print(t2d[["feature_set", "model", "d_r2", "d_r2_lo", "d_r2_hi", "p_r2_le_0", "d_mae"]].to_string(index=False))
        print("\n=== Repeat averaging ===")
        print(t2e.to_string(index=False))
        print("\n=== Predictive ratios by age band and sex ===")
        print(t2f.pivot_table(index=["age_band", "sex"], columns="model", values="predictive_ratio").round(3).to_string())
    print("\nwrote tables 2d, 2e, 2f")


if __name__ == "__main__":
    main()
