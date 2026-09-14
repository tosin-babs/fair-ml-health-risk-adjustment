"""
Render the manuscript's tables from the analysis CSVs.

Manuscript numbering follows reading order; the CSV each table comes from is
named in its block. Writes manuscript/tables.md.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config

OUT = config.ROOT / "manuscript" / "tables.md"
T = config.TABLES


def num(d=2):
    return lambda x: "" if pd.isna(x) else f"{x:,.{d}f}"


def dollars(d=0):
    return lambda x: "" if pd.isna(x) else (f"-${abs(x):,.{d}f}" if x < 0 else f"${x:,.{d}f}")


def pct(d=1):
    return lambda x: "" if pd.isna(x) else f"{x:,.{d}f}%"


def share(d=1):
    return lambda x: "" if pd.isna(x) else f"{100 * x:.{d}f}%"


def render(df, cols, fmts, headers=None):
    out = pd.DataFrame({c: df[c].map(f) if f else df[c].astype(str)
                        for c, f in zip(cols, fmts)})
    out.columns = headers or cols
    align = ["---" if i == 0 else "---:" for i in range(len(out.columns))]
    lines = ["| " + " | ".join(out.columns) + " |", "|" + "|".join(align) + "|"]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for _, r in out.iterrows()]
    return "\n".join(lines)


def caption(n, title, note=None):
    s = f"\n**Table {n}.** {title}\n"
    return s + (f"\n*{note}*\n" if note else "")


def _exists(name):
    return (T / name).exists()


def main():
    parts = ["# Tables\n",
             "*Generated from `output/tables/*.csv` by `python/make_tables.py`. "
             "Spending is total expenditure from all payers in 2024 dollars. All "
             "estimates are weighted by the MEPS longitudinal weight; model "
             "metrics are out of fold from cross-validation with primary sampling "
             "units kept whole.*\n"]

    if _exists("table1_sample.csv"):
        t = pd.read_csv(T / "table1_sample.csv")
        parts += [caption(1, "Sample characteristics and year-2 spending.",
                          "Weighted estimates; standard errors Taylor-linearized for "
                          "the stratified cluster design."),
                  render(t, ["characteristic", "estimate", "se", "n"],
                         [None, num(2), num(2), num(0)],
                         ["Characteristic", "Estimate", "SE", "n"])]

    t = pd.read_csv(T / "table2_accuracy.csv")
    parts += [caption(2, "Out-of-sample accuracy by model, primary feature set (F3).",
                      "Five survey folds, three repeats. Intervals from a bootstrap over "
                      "PSUs within strata applied to repeat-averaged out-of-fold "
                      "predictions. CPM is Cumming's prediction measure. Top-10% capture "
                      "is the weighted share of the true top decile of spenders placed "
                      "in the model's top decile."),
              render(t, ["model", "r2", "r2_lo", "r2_hi", "cpm", "mae",
                         "pr_bottom_decile", "pr_top_decile", "top10_capture"],
                     [None, num(3), num(3), num(3), num(3), dollars(), num(2),
                      num(2), share(0)],
                     ["Model", "R²", "95% low", "95% high", "CPM", "MAE",
                      "PR bottom decile", "PR top decile", "Top-10% capture"])]

    t = pd.read_csv(T / "table2b_feature_sets.csv")
    wide = t.pivot_table(index="model", columns="feature_set", values="r2").reset_index()
    parts += [caption(3, "Out-of-sample R² by feature set.",
                      "F1 demographics; F2 adds year-1 CCSR condition flags; F3 adds "
                      "year-1 use and spending; F4 adds income, insurance, perceived "
                      "health and functional help."),
              render(wide, ["model"] + list(config.FEATURE_SETS),
                     [None] + [num(3)] * len(config.FEATURE_SETS),
                     ["Model"] + list(config.FEATURE_SETS))]

    t = pd.read_csv(T / "table3_group_fairness.csv")
    key = ["WLS", "Tweedie GLM", "LightGBM (Tweedie)", "CANN"]
    t = t[t["model"].isin(key)].copy()
    t["cell"] = t.apply(lambda r: f"{dollars()(r['nc'])} ({dollars()(r['nc_lo'])}, "
                                  f"{dollars()(r['nc_hi'])})", axis=1)
    wide = t.pivot_table(index="group", columns="model", values="cell",
                         aggfunc="first").reindex(columns=key).reset_index()
    parts += [caption(4, "Net compensation by group: predicted minus observed "
                         "spending, $ per person-year (95% interval).",
                      "Negative values mean the model pays less for the group than "
                      "the group costs. Race and ethnicity are used for evaluation "
                      "only and never as model inputs."),
              render(wide, ["group"] + key, [None] * (len(key) + 1), ["Group"] + key)]

    t = pd.read_csv(T / "table3b_frontier.csv")
    ncols = [c for c in t.columns if c.startswith("nc::")]
    t["lambda_label"] = t["lambda"].map(lambda x: "constrained" if np.isinf(x) else f"{x:g}")
    parts += [caption(5, "The accuracy-fairness frontier.",
                      "Penalized and constrained estimators target the three groups "
                      "marked in the text; the other groups show spillover. Stacked "
                      "LightGBM enters a cross-fitted LightGBM score with the features "
                      "into the fair regression. One repeat of the survey folds."),
              render(t, ["method", "lambda_label", "r2", "cpm", "max_abs_nc_target"] + ncols,
                     [None, None, num(3), num(3), dollars()] + [dollars()] * len(ncols),
                     ["Method", "λ", "R²", "CPM", "Max |NC| target"]
                     + [c[4:] for c in ncols])]

    t = pd.read_csv(T / "table4_importance.csv").head(20)
    s = pd.read_csv(T / "table4b_stability.csv")
    parts += [caption(6, "The 20 largest drivers of the squared-error LightGBM predictions.",
                      f"Weighted mean absolute SHAP value on held-out folds. Rank "
                      f"stability across folds: mean Spearman correlation "
                      f"{s['spearman_all'].mean():.2f} over all features and "
                      f"{s['spearman_top30'].mean():.2f} over the top 30. The last "
                      f"column is the loss in held-out R² when the feature is permuted "
                      f"in the CANN."),
              render(t, ["rank", "label", "mean_abs_shap", "sd_across_folds",
                         "cann_permutation_r2_loss"],
                     [num(0), None, dollars(), dollars(), num(4)],
                     ["Rank", "Feature", "Mean |SHAP|", "SD across folds",
                      "CANN R² loss"])]

    t = pd.read_csv(T / "table5_coding_sensitivity.csv")
    main_ = t[(t["share_affected"] == 0.10)]
    wide = main_.pivot_table(index="model", columns=["pool", "codes_added"],
                             values="dollars_per_added_code")
    wide.columns = [f"{p}, {k} code{'s' if k > 1 else ''}" for p, k in wide.columns]
    wide = wide.reset_index()
    tot = main_[(main_["pool"] == "chronic") & (main_["codes_added"] == 1)].set_index("model")
    wide["pct_rise"] = wide["model"].map(tot["pct_rise_total_predicted"])
    vcols = [c for c in wide.columns if c not in ("model", "pct_rise")]
    parts += [caption(7, "Coding sensitivity: rise in predicted spending per added "
                         "condition code, 10% of people affected.",
                      "Codes the person did not have are switched on; use and spending "
                      "are unchanged. The last column is the rise in total predicted "
                      "spending when 10% of people gain one chronic code."),
              render(wide, ["model"] + vcols + ["pct_rise"],
                     [None] + [dollars()] * len(vcols) + [pct(2)],
                     ["Model"] + vcols + ["Total rise, 1 chronic code"])]

    t = pd.read_csv(T / "table6_robustness.csv")
    parts += [caption(8, "Robustness: each row changes one decision.",
                      "One repeat of the survey folds. Max |NC| is over the three "
                      "target groups."),
              render(t, ["variant", "model", "n", "r2", "cpm", "pr_top_decile",
                         "max_abs_nc_target"],
                     [None, None, num(0), num(3), num(3), num(2), dollars()],
                     ["Variant", "Model", "n", "R²", "CPM", "PR top decile",
                      "Max |NC| target"])]

    parts += ["\n\n# Appendix tables\n"]
    t = pd.read_csv(T / "table0_sample_flow.csv")
    parts += [caption("A1", "Sample flow by MEPS panel."),
              render(t, ["panel", "years", "persons_in_file", "both_years_positive_weight",
                         "died_year2", "analysis_sample", "with_any_condition"],
                     [num(0), None, num(0), num(0), num(0), num(0), num(0)],
                     ["Panel", "Years", "In file", "Both years, weight > 0",
                      "Died in year 2", "Analysis sample", "Any year-1 condition"])]

    t = pd.read_csv(T / "table2c_calibration.csv")
    wide = t.pivot_table(index="decile", columns="model", values="predictive_ratio").reset_index()
    mcols = [c for c in wide.columns if c != "decile"]
    parts += [caption("A2", "Predictive ratio by decile of predicted spending."),
              render(wide, ["decile"] + mcols, [num(0)] + [num(2)] * len(mcols),
                     ["Decile"] + mcols)]

    t = pd.read_csv(T / "table5_coding_sensitivity.csv")
    parts += [caption("A3", "Coding sensitivity, full grid."),
              render(t, ["model", "pool", "share_affected", "codes_added",
                         "pct_rise_total_predicted", "dollars_per_added_code"],
                     [None, None, share(0), num(0), pct(2), dollars()],
                     ["Model", "Pool", "Share affected", "Codes added",
                      "Total rise", "$ per code"])]

    t = pd.read_csv(T / "table4c_sense_checks.csv")
    g = t.groupby("feature")["spearman_value_vs_shap"].agg(["mean", "min", "max"]).reset_index()
    parts += [caption("A4", "Actuarial sense checks on the SHAP values.",
                      "Spearman correlation between a feature's value and its SHAP "
                      "contribution across held-out persons, by fold. Positive means "
                      "predicted spending rises with the feature."),
              render(g, ["feature", "mean", "min", "max"],
                     [None, num(3), num(3), num(3)],
                     ["Feature", "Mean", "Min across folds", "Max across folds"])]

    if (T / "tuning_chosen.json").exists():
        tune = json.loads((T / "tuning_chosen.json").read_text())
        rows = []
        for k, v in tune.items():
            if v:
                df = pd.DataFrame(v)
                for col in df.columns:
                    vals = df[col].astype(str).value_counts()
                    rows.append({"model": k, "parameter": col,
                                 "chosen (count of folds)": "; ".join(f"{a} ({b})" for a, b in vals.items())})
        if rows:
            parts += [caption("A5", "Hyperparameters chosen by inner validation, "
                                    "across the 15 outer folds."),
                      render(pd.DataFrame(rows), ["model", "parameter",
                                                  "chosen (count of folds)"],
                             [None, None, None])]

    if _exists("tableA_ccsr_features.csv"):
        t = pd.read_csv(T / "tableA_ccsr_features.csv")
        parts += [caption("A6", "Year-1 CCSR categories entering the feature sets.",
                          f"Categories held by at least {100 * config.CCSR_MIN_PREVALENCE:.1f}% "
                          "of the sample in year 1. MEPS body-system placeholder codes "
                          "(XXX000) are excluded."),
                  render(t, ["ccsr", "description", "prevalence"],
                         [None, None, share(2)], ["CCSR", "Description", "Prevalence"])]

    OUT.write_text("\n".join(parts) + "\n")
    n = sum(1 for line in OUT.read_text().splitlines() if line.startswith("**Table"))
    print(f"wrote {OUT.relative_to(config.ROOT)} with {n} tables")


if __name__ == "__main__":
    main()
