"""
Export the payload for the public risk-adjustment explorer.

The page answers four questions for a non-specialist: how well each payment
formula predicts spending, which groups each formula pays too little for, what
it costs in accuracy to close those gaps, and how much each formula pays out
when diagnosis coding becomes more intensive. Everything comes from the
aggregate tables the analysis wrote; no person-level data leaves the pipeline.

Writes tool/model_data.json and prints the reference values the page must
reproduce.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd

import config

OUT = config.ROOT / "tool" / "model_data.json"
T = config.TABLES
FAMILY = {"WLS": "Linear", "Payment-form WLS (non-negative)": "Payment formula", "Elastic net": "Linear",
          "Tweedie GLM": "Actuarial GLM",
          "Two-part": "Actuarial GLM", "LightGBM (Tweedie)": "Boosted trees",
          "LightGBM (squared error)": "Boosted trees", "Random forest": "Trees",
          "Neural network": "Neural network", "CANN": "Neural network",
          "Constrained WLS": "Fair linear"}


def _f(x, d=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), d)


def main():
    t1 = pd.read_csv(T / "table1_sample.csv").set_index("characteristic")["estimate"]
    acc = pd.read_csv(T / "table2_accuracy.csv")
    comp = pd.read_csv(T / "table3_group_fairness.csv")
    fr = pd.read_csv(T / "table3b_frontier.csv")
    cod = pd.read_csv(T / "table5_coding_sensitivity.csv")
    imp = pd.read_csv(T / "table4_importance.csv").head(15)

    groups = list(dict.fromkeys(comp["group"]))
    ncols = [c for c in fr.columns if c.startswith("nc::")]
    payload = {
        "meta": {
            "source": "Medical Expenditure Panel Survey, two-year longitudinal panels 23 to 27 (2018-2019 through 2022-2023)",
            "generated": date.today().isoformat(),
            "persons": int(t1["Persons"]),
            "population_millions": _f(t1["Weighted population, millions a year"], 1),
            "dollars": "2024 dollars, total spending from all payers",
            "paper": "Interpretable and Fair Machine Learning for Health-Cost Prediction and Risk Adjustment",
            "author": "Oluwatosin Dorcas Babalola",
            "repository": "https://github.com/tosin-babs/fair-ml-health-risk-adjustment",
        },
        "facts": {
            "mean_spend": _f(t1["Year-2 spending, mean $"], 0),
            "median_spend": _f(t1["Year-2 spending, median $"], 0),
            "zero_share": _f(t1["Year-2 spending of zero, %"], 1),
            "top5_share": _f(t1["Share of year-2 spending by the top 5% of spenders, %"], 1),
        },
        "accuracy": [{"model": r["model"], "family": FAMILY.get(r["model"], ""),
                      "r2": _f(r["r2"]), "r2_lo": _f(r["r2_lo"]), "r2_hi": _f(r["r2_hi"]),
                      "cpm": _f(r["cpm"]), "top10": _f(r["top10_capture"]),
                      "negative_share": _f(r.get("negative_prediction_share", np.nan), 3)}
                     for _, r in acc.iterrows()],
        "groups": groups,
        "target_groups": [g for g in config.FAIR_TARGET_GROUPS if g in groups],
        "compensation": [{"model": r["model"], "group": r["group"], "nc": _f(r["nc"], 0),
                          "lo": _f(r["nc_lo"], 0), "hi": _f(r["nc_hi"], 0),
                          "pr": _f(r["pr"], 3), "mean_cost": _f(r["mean_cost"], 0),
                          "n": int(r["n"])}
                         for _, r in comp.iterrows()],
        "frontier": [{"method": r["method"],
                      "lambda": None if not np.isfinite(r["lambda"]) else _f(r["lambda"], 2),
                      "r2": _f(r["r2"]), "r2_lo": _f(r.get("r2_lo", np.nan)), "r2_hi": _f(r.get("r2_hi", np.nan)),
                      "max_gap": _f(r["max_abs_nc_target"], 0),
                      "gap_lo": _f(r.get("max_abs_nc_target_lo", np.nan), 0),
                      "gap_hi": _f(r.get("max_abs_nc_target_hi", np.nan), 0),
                      "nc": {c[4:]: _f(r[c], 0) for c in ncols}}
                     for _, r in fr.iterrows()],
        "coding": [{"feature_set": r.get("feature_set", config.PRIMARY_FEATURE_SET),
                    "model": r["model"], "pool": r["pool"], "share": _f(r["share_affected"], 2),
                    "codes": int(r["codes_added"]), "pct_rise": _f(r["pct_rise_total_predicted"], 3),
                    "per_code": _f(r["dollars_per_added_code"], 0),
                    "per_code_sd": _f(r.get("dollars_per_code_fold_sd", np.nan), 0)}
                   for _, r in cod.iterrows()],
        "drivers": [{"rank": int(r["rank"]), "label": r["label"],
                     "shap": _f(r["mean_abs_shap"], 0), "sd": _f(r["sd_across_folds"], 0)}
                    for _, r in imp.iterrows()],
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {OUT.relative_to(config.ROOT)} ({OUT.stat().st_size / 1024:,.0f} KB)")

    print("\nReference values the page must reproduce:")
    best = acc.loc[acc["r2"].idxmax()]
    print(f"  most accurate model: {best['model']}, R2 {best['r2']:.3f}")
    for g in payload["target_groups"]:
        w = comp[(comp["model"] == "WLS") & (comp["group"] == g)]
        if len(w):
            print(f"  WLS net compensation, {g}: ${w['nc'].iloc[0]:,.0f}")
    c = cod[(cod["pool"] == "chronic") & (cod["share_affected"] == 0.10) & (cod["codes_added"] == 1)]
    for _, r in c.iterrows():
        print(f"  coding, {r.get('feature_set', '')} {r['model']}: ${r['dollars_per_added_code']:,.0f} per code")


if __name__ == "__main__":
    main()
