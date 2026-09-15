"""
Robustness: does the ranking of models, and the fairness gap, survive the
choices a referee could reasonably make differently?

Each variant changes one decision (spec Section 5.7) and reruns three models
that span the design space (WLS, Tweedie GLM, Tweedie LightGBM) on one repeat
of the survey folds. A variant matters if it reverses an ordering or moves the
fairness gap materially, not merely if it moves a level.

  baseline                 primary sample and features
  outcome capped           spending capped at config.OUTCOME_CAP, fit and scored
  log outcome (WLS)        OLS on log(1 + spending) with Duan smearing
  unweighted training      weights set to one in fitting, kept in scoring
  CCSR floor 0.2% / 1%     minimum prevalence for a condition flag
  no prior spending        F2 instead of F3 (payment formulas exclude it)
  social-risk features     F4 instead of F3
  decedents excluded       people who died in year 2 dropped (the primary
                           sample keeps them with annualized spending)
  decedents not annualized kept with their partial-year spending
  3 folds / 10 folds       fold count
  per panel                each two-year panel on its own
  pandemic outcomes out    panels whose outcome year is 2020 or 2021 dropped
  age 65 and over          fitted and scored within the subset
  under 65, private        fitted and scored within the subset

Writes Table 6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import build_data
import common
import config
import features
from common import metrics
from riskfair.models import GBM, WLS, TweedieGLM

from riskfair.models import PaymentWLS  # noqa: E402

MODELS = {"WLS": WLS, "Payment-form WLS (non-negative)": PaymentWLS, "Tweedie GLM": TweedieGLM,
          "LightGBM (Tweedie)": lambda: GBM("tweedie", seed=config.SEED)}


class LogWLS(WLS):
    """OLS on log(1 + spending), retransformed and rescaled to balance.

    Duan smearing was tried and rejected: with a tail this heavy the smearing
    factor is dominated by a few residuals and the retransformed predictions
    are off by an order of magnitude. Rescaling to the training-data balance
    is the same correction risk adjustment applies to any multiplicative
    model and is stable.
    """
    name = "WLS, log outcome"

    def fit(self, X, y, w, clusters=None):
        super().fit(X, np.log1p(y), w)
        raw = np.maximum(np.expm1(super().predict(X)), 0)
        self.balance_ = float(np.sum(w * y) / max(np.sum(w * raw), 1e-9))
        return self

    def predict(self, X):
        return self.balance_ * np.maximum(np.expm1(super().predict(X)), 0)


def evaluate(label, data, feature_set="F3", subset=None, cap=None,
             unweighted=False, min_prev=None, models=None):
    X, y, w, cl, st, attrs = features.build(feature_set, min_prevalence=min_prev,
                                            data=data)
    if subset is not None:
        keep = subset(attrs).to_numpy()
        X, y, w, cl, st = X[keep], y[keep], w[keep], cl[keep], st[keep]
        attrs = attrs[keep].reset_index(drop=True)
    cols = list(X.columns)
    X = X.to_numpy(np.float32)
    y_use = np.minimum(y, cap) if cap else y
    w_fit = np.ones_like(w) if unweighted else w
    masks = features.group_masks(attrs)
    target = [g for g in config.FAIR_TARGET_GROUPS if g in masks]
    rows = []
    for mname, make in (models or MODELS).items():
        p = np.full(len(y), np.nan)
        for _, _, tr, te in common.folds(cl, st, repeats=1):
            m = make()
            if hasattr(m, "set_columns"):
                m.set_columns(cols)
            m = m.fit(X[tr], y_use[tr], w_fit[tr], clusters=cl[tr])
            p[te] = m.predict(X[te])
        acc = metrics.accuracy(y_use, p, w)
        g = metrics.group_fairness(y_use, p, w, masks)
        rows.append({"variant": label, "model": mname, "n": len(y),
                     "r2": acc["r2"], "cpm": acc["cpm"],
                     "pr_top_decile": acc["pr_top_decile"],
                     "top10_capture": acc["top10_capture"],
                     "max_abs_nc_target": max((abs(g[t]["nc"]) for t in target), default=np.nan),
                     "nc_income": g.get("Income below 200% FPL", {}).get("nc", np.nan),
                     "nc_mental_health": g.get("Mental health condition", {}).get("nc", np.nan),
                     "nc_function": g.get("Needs ADL or IADL help", {}).get("nc", np.nan)})
        print(f"  {label:<24} {mname:<20} R2 {acc['r2']:.3f}  CPM {acc['cpm']:.3f}  "
              f"max|NC| ${rows[-1]['max_abs_nc_target']:,.0f}")
    return rows


def main():
    base = features.load()
    rows = []
    rows += evaluate("baseline", base)
    rows += evaluate("outcome capped", base, cap=config.OUTCOME_CAP)
    rows += evaluate("log outcome", base, models={"WLS, log outcome": LogWLS})
    rows += evaluate("unweighted training", base, unweighted=True)
    rows += evaluate("CCSR floor 0.2%", base, min_prev=0.002)
    rows += evaluate("CCSR floor 1%", base, min_prev=0.01)
    rows += evaluate("no prior spending (F2)", base, feature_set="F2")
    rows += evaluate("social-risk features (F4)", base, feature_set="F4")

    d, c, _ = build_data.build(exclude_died=True)
    rows += evaluate("decedents excluded", (d.reset_index(drop=True), c))
    dn = base[0].copy()
    dn["y"] = dn["y_raw"]
    rows += evaluate("decedents not annualized", (dn, base[1]))
    for k in (3, 10):
        saved = config.CV_FOLDS
        config.CV_FOLDS = k
        rows += evaluate(f"{k} folds", base)
        config.CV_FOLDS = saved
    for panel in config.PANELS:
        dp, cp = base[0], base[1]
        keep = dp["panel"] == panel
        rows += evaluate(f"panel {panel} only", (dp[keep].reset_index(drop=True), cp[cp["panel"] == panel]),
                         models={k: v for k, v in MODELS.items() if k in ("WLS", "LightGBM (Tweedie)")})

    pre = [p for p, s in config.PANELS.items() if s["year1"] + 1 not in (2020, 2021)]
    dp, cp = base[0], base[1]
    keep = dp["panel"].isin(pre)
    rows += evaluate("pandemic outcomes out",
                     (dp[keep].reset_index(drop=True), cp[cp["panel"].isin(pre)]))

    rows += evaluate("age 65 and over", base, subset=lambda a: a["age"] >= 65)
    rows += evaluate("under 65, private", base,
                     subset=lambda a: (a["age"] < 65) & (a["inscov"] == 1))
    pd.DataFrame(rows).to_csv(config.TABLES / "table6_robustness.csv", index=False)
    print("\nwrote table 6")


if __name__ == "__main__":
    main()
