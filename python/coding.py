"""
RQ4: how much does each model's predicted spending rise when coding
intensifies, and when use intensifies?

Coding. For a share of held-out people, condition flags they did not have are
switched on, drawn from one of four pools: in proportion to each category's
prevalence among people with any condition; from chronic, payment-relevant
categories (riskfair.coding.CHRONIC_CCSR); targeted, the five flags with the
largest coefficients in the payment-form linear model, which is what a coder
who knows the published formula adds; or own-targeted, the five flags that
raise each model's own prediction most when switched on, found on a sample of
that model's training data, which is what a coder who can query the model
adds. The own-targeted pool differs by model and fold and is written to Table
5c. The condition and body-system counts are updated to match; use and
spending are left as they were. The experiment is
run on the primary feature set (F3) and on F2, the set closest to a
deployable formula, because a model that conditions on prior use responds
differently to a code once use is held fixed.

Use. For the same share of people, year-1 spending is raised by 20% with
nothing else changed, and the rise in next-year payment per added dollar of
year-1 spending is recorded. Payment formulas exclude prior use to close this
channel, and a model that includes it opens it.

Models are fitted on the training PSUs of each fold of the first repeat and
stressed on that fold's held-out PSUs. Results are pooled across folds as
weighted totals, with the standard deviation across folds alongside.

Writes Tables 5 (coding, both feature sets), 5b (use sensitivity) and 5c
(the own-targeted flags).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import common
import config
import features
from common import coding, fairness

MODELS = {
    "WLS": "wls",
    "Payment-form WLS (non-negative)": "pwls",
    "Tweedie GLM": "tweedie",
    "Two-part": "twopart",
    "LightGBM (Tweedie)": "gbm",
    "LightGBM (squared error)": "gbm_mse",
    "CANN": "cann",
    "Constrained WLS": None,
}
USE_RISE = 0.20
OWN_PROBE = 3000      # training persons used to find each model's most lucrative flags


def fit_models(Xn, y, w, cl, masks, target, tr, cols):
    out = {}
    for mname, key in MODELS.items():
        if key is None:
            m = fairness.FairWLS(constrained=True).fit(Xn[tr], y[tr], w[tr], [masks[g][tr] for g in target])
            out[mname] = m.predict
        else:
            m = common.make(key, cols).fit(Xn[tr].astype(np.float32), y[tr], w[tr], clusters=cl[tr])
            out[mname] = (lambda A, m=m: m.predict(A.astype(np.float32)))
    return out


def targeted_pool(Xn, y, w, cl, cols, ccsr_cols):
    """The five condition flags with the largest payment-form coefficients."""
    m = common.make("pwls", cols).fit(Xn.astype(np.float32), y, w, clusters=cl)
    coef = pd.Series(np.asarray(m.coef_).ravel(), index=m.names_)
    return coef.reindex(ccsr_cols).sort_values(ascending=False).index[:5].tolist()


def run_feature_set(fs, data, folds, masks, target):
    X, y, w, cl, st, attrs = features.build(fs, data=data)
    cols = list(X.columns)
    Xn = X.to_numpy(np.float64)
    ccsr_cols = [c for c in cols if c.startswith("ccsr_")]
    has_any = Xn[:, cols.index("n_conditions")] > 0
    prev = {c: float(Xn[has_any, cols.index(c)].mean()) for c in ccsr_cols}
    chronic = [c for c in ccsr_cols if any(k in c[5:].split("+") for k in coding.CHRONIC_CCSR)]
    top = targeted_pool(Xn, y, w, cl, cols, ccsr_cols)
    pools = {"prevalence": (ccsr_cols, prev),
             "chronic": (chronic, {c: prev[c] for c in chronic}),
             "targeted": (top, {c: 1.0 for c in top})}
    print(f"{fs}: chronic pool {len(chronic)} flags ({', '.join(c[5:] for c in chronic)}); "
          f"targeted pool {', '.join(c[5:] for c in top)}", flush=True)
    use_cols = [c for c in ("log_spend_y1", "any_spend_y1") if c in cols]

    rows, use_rows, own_rows = [], [], []
    for _, fold, tr, te in folds:
        predict = fit_models(Xn, y, w, cl, masks, target, tr, cols)
        probe = np.random.default_rng(config.SEED + 100 * fold + 3).choice(tr, size=min(OWN_PROBE, len(tr)), replace=False)
        for mname, pred in predict.items():
            gains = pd.Series(coding.flag_gains(pred, Xn[probe], w[probe], cols, ccsr_cols,
                                                count_col="n_conditions", system_col="n_body_systems"))
            own = gains.sort_values(ascending=False).index[:5].tolist()
            own_rows += [{"feature_set": fs, "model": mname, "fold": fold, "rank": i + 1, "flag": c[5:],
                          "training_gain": float(gains[c])} for i, c in enumerate(own)]
            model_pools = {**pools, "own-targeted": (own, {c: 1.0 for c in own})}
            for pool, (pc, pw) in model_pools.items():
                for share in config.CODING_SHARES:
                    for k in config.CODING_ADDED:
                        r = coding.stress(pred, Xn[te], w[te], cols, pc, pw, share, k,
                                          seed=config.SEED + 100 * fold + 10 * k + int(share * 1000),
                                          count_col="n_conditions", system_col="n_body_systems")
                        r = dict(r)
                        r["codes_added_total"] = r.pop("codes_added")
                        rows.append({**r, "feature_set": fs, "model": mname, "pool": pool,
                                     "share_affected": share, "codes_added": k, "fold": fold})
            if use_cols:
                rng = np.random.default_rng(config.SEED + 100 * fold + 7)
                chosen = rng.random(len(te)) < 0.10
                A = Xn[te].copy()
                j = cols.index("log_spend_y1")
                spend0 = np.expm1(A[:, j])
                A[chosen, j] = np.log1p(spend0[chosen] * (1 + USE_RISE))
                p0, p1 = pred(Xn[te]), pred(A)
                added = float(np.sum(w[te][chosen] * spend0[chosen] * USE_RISE))
                use_rows.append({"feature_set": fs, "model": mname, "fold": fold,
                                 "payment_per_dollar_of_use": float(np.sum(w[te][chosen] * (p1 - p0)[chosen])) / added,
                                 "pct_rise_total_predicted": 100 * float(np.sum(w[te] * (p1 - p0)) / np.sum(w[te] * p0))})
        print(f"  {fs} fold {fold} done", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(use_rows), pd.DataFrame(own_rows)


def pool_folds(df):
    """Weighted totals across folds, with the across-fold spread."""
    g = df.groupby(["feature_set", "model", "pool", "share_affected", "codes_added"])
    out = g.apply(lambda d: pd.Series({
        "pct_rise_total_predicted": 100 * d["rise_total"].sum() / d["base_total"].sum(),
        "dollars_per_added_code": (np.nansum(d["dollars_per_code"] * d["affected_weight"])
                                   / d["affected_weight"].sum()),
        "dollars_per_code_fold_sd": float(np.nanstd(d["dollars_per_code"], ddof=1)) if len(d) > 1 else np.nan,
        "n_folds": int(np.isfinite(d["dollars_per_code"]).sum())}), include_groups=False)
    return out.reset_index()


def main():
    data = features.load()
    X, y, w, cl, st, attrs = features.build(config.PRIMARY_FEATURE_SET, data=data)
    masks = features.group_masks(attrs)
    target = [g for g in config.FAIR_TARGET_GROUPS if g in masks]
    folds = common.folds(cl, st, repeats=1)
    parts, use_parts, own_parts = [], [], []
    for fs in (config.PRIMARY_FEATURE_SET, "F2"):
        r, u, o = run_feature_set(fs, data, folds, masks, target)
        parts.append(r)
        own_parts.append(o)
        if len(u):
            use_parts.append(u)
    raw = pd.concat(parts, ignore_index=True)
    raw.to_csv(config.TABLES / "table5_coding_by_fold.csv", index=False)
    t5 = pool_folds(raw)
    t5.to_csv(config.TABLES / "table5_coding_sensitivity.csv", index=False)
    own = pd.concat(own_parts, ignore_index=True)
    own.to_csv(config.TABLES / "table5c_own_targeted_flags.csv", index=False)
    if use_parts:
        u = pd.concat(use_parts, ignore_index=True)
        t5b = (u.groupby(["feature_set", "model"])
                .agg(payment_per_dollar_of_use=("payment_per_dollar_of_use", "mean"),
                     fold_sd=("payment_per_dollar_of_use", "std"),
                     pct_rise_total_predicted=("pct_rise_total_predicted", "mean")).reset_index())
        t5b.to_csv(config.TABLES / "table5b_use_sensitivity.csv", index=False)
    ref = t5[(t5["share_affected"] == 0.10) & (t5["codes_added"] == 1)]
    with pd.option_context("display.width", 200, "display.float_format", "{:,.0f}".format):
        print("\n=== Dollars per added code, 10% of people gain one code ===")
        print(ref.pivot_table(index=["feature_set", "model"], columns="pool",
                              values="dollars_per_added_code").to_string())
        if use_parts:
            print("\n=== Next-year payment per added dollar of year-1 spending ===")
            print(t5b.round(3).to_string(index=False))
    print("\nwrote tables 5, 5b and 5c")


if __name__ == "__main__":
    main()
