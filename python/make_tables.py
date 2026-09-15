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
                      "Five survey folds, three repeats. Each metric is the mean over repeats of a single "
                      "fitted model's out-of-fold value, which is what one deployed model achieves; the "
                      "ensemble column averages the three repeats' predictions first. Intervals from a "
                      "Rao-Wu rescaled bootstrap over PSUs within strata (200 resamples). CPM is Cumming's "
                      "prediction measure. Top-10% capture is the weighted share of the true top decile of "
                      "spenders placed in the model's top decile. The last column is the weighted share of "
                      "people with a negative predicted payment."),
              render(t, ["model", "r2", "r2_lo", "r2_hi", "r2_ensemble", "cpm", "mae",
                         "pr_bottom_decile", "pr_top_decile", "top10_capture", "negative_prediction_share"],
                     [None, num(3), num(3), num(3), num(3), num(3), dollars(), num(2),
                      num(2), share(0), share(1)],
                     ["Model", "R²", "95% low", "95% high", "R², 3-fit ensemble", "CPM", "MAE",
                      "PR bottom decile", "PR top decile", "Top-10% capture", "Negative predictions"])]

    if _exists("table2d_paired_differences.csv"):
        t = pd.read_csv(T / "table2d_paired_differences.csv")
        parts += [caption("2b", "Paired differences from WLS in out-of-sample accuracy.",
                          "Every model is evaluated on the same 500 resamples of PSUs within strata, so "
                          "the interval is for the difference itself. The last column is the share of "
                          "resamples in which the model's R² does not exceed WLS's. Negative MAE "
                          "differences mean smaller errors than WLS."),
                  render(t, ["feature_set", "model", "d_r2", "d_r2_lo", "d_r2_hi", "p_r2_le_0", "d_cpm",
                             "d_mae", "d_mae_lo", "d_mae_hi"],
                         [None, None, num(3), num(3), num(3), num(3), num(3), dollars(), dollars(), dollars()],
                         ["Features", "Model", "ΔR²", "95% low", "95% high", "Share ΔR² ≤ 0", "ΔCPM",
                          "ΔMAE", "95% low", "95% high"])]
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
    key = ["WLS", "Payment-form WLS (non-negative)", "Tweedie GLM", "LightGBM (Tweedie)", "CANN"]
    t = t[t["model"].isin(key)].copy()
    t["cell"] = t.apply(lambda r: f"{dollars()(r['nc'])} ({dollars()(r['nc_lo'])}, "
                                  f"{dollars()(r['nc_hi'])}); PR {r['pr']:.2f}", axis=1)
    wide = t.pivot_table(index="group", columns="model", values="cell",
                         aggfunc="first").reindex(columns=key).reset_index()
    parts += [caption(4, "Net compensation by group: predicted minus observed "
                         "spending, $ per person-year (95% interval), and predictive ratio.",
                      "Negative values mean the model pays less for the group than "
                      "the group costs. Means over repeats of single fits; Rao-Wu PSU "
                      "bootstrap intervals. Race and ethnicity are used for evaluation "
                      "only and never as model inputs."),
              render(wide, ["group"] + key, [None] * (len(key) + 1), ["Group"] + key)]
    d = t[t["model"] != "WLS"].copy()
    d["cell"] = d.apply(lambda r: f"{dollars()(r['nc_minus_wls'])} ({dollars()(r['nc_minus_wls_lo'])}, "
                                  f"{dollars()(r['nc_minus_wls_hi'])})", axis=1)
    wd = d.pivot_table(index="group", columns="model", values="cell", aggfunc="first").reset_index()
    dk = [k for k in key if k != "WLS" and k in wd.columns]
    parts += [caption("4b", "Difference in net compensation from WLS, by group (95% paired interval).",
                      "Same PSU resamples for both models, so the interval is for the difference itself. "
                      "Positive values mean the model pays the group more than WLS does."),
              render(wd, ["group"] + dk, [None] * (len(dk) + 1), ["Group"] + dk)]

    t = pd.read_csv(T / "table3b_frontier.csv")
    ncols = [c for c in t.columns if c.startswith("nc::")]
    t["lambda_label"] = t["lambda"].map(lambda x: "constrained" if np.isinf(x) else f"{x:g}")
    parts += [caption(5, "The accuracy-fairness frontier.",
                      "Penalized and constrained estimators target the three groups "
                      "marked in the text; the other groups show spillover. Stacked "
                      "LightGBM enters a cross-fitted LightGBM score with the features "
                      "into the fair regression. All three repeats of the survey folds; "
                      "metrics are means over repeats of out-of-fold values, with Rao-Wu "
                      "PSU bootstrap intervals for R² and the largest target-group gap."),
              render(t, ["method", "lambda_label", "r2", "r2_lo", "r2_hi", "max_abs_nc_target",
                         "max_abs_nc_target_lo", "max_abs_nc_target_hi"] + ncols,
                     [None, None, num(3), num(3), num(3), dollars(), dollars(), dollars()]
                     + [dollars()] * len(ncols),
                     ["Method", "λ", "R²", "low", "high", "Max |NC| target", "low", "high"]
                     + [c[4:] for c in ncols])]
    if _exists("table3c_constraint_by_fold.csv"):
        f = pd.read_csv(T / "table3c_constraint_by_fold.csv")
        fcols = [c for c in f.columns if c.startswith(("in_sample::", "out_of_fold::"))]
        parts += [caption("5b", "Constrained estimators fold by fold: net compensation in the training data "
                                "and out of fold.",
                          "In the training data the constraint holds exactly for the target groups. Out of fold "
                          "it does not; the pooled values in Table 5 average across these folds."),
                  render(f, ["method", "repeat", "fold"] + fcols,
                         [None, num(0), num(0)] + [dollars()] * len(fcols),
                         ["Method", "Repeat", "Fold"] + [c.replace("in_sample::", "in: ").replace("out_of_fold::", "out: ")
                                                         for c in fcols])]

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
    main_ = t[(t["share_affected"] == 0.10) & (t["codes_added"] == 1)].copy()
    main_["cell"] = main_.apply(lambda r: f"{dollars()(r['dollars_per_added_code'])} "
                                          f"(sd {dollars()(r['dollars_per_code_fold_sd'])})", axis=1)
    wide = main_.pivot_table(index=["feature_set", "model"], columns="pool", values="cell",
                             aggfunc="first").reset_index()
    pools = [p for p in ("prevalence", "chronic", "targeted", "own-targeted") if p in wide.columns]
    parts += [caption(7, "Coding sensitivity: rise in predicted spending per added condition code, "
                         "10% of people gain one code (standard deviation across folds).",
                      "Codes the person did not have are switched on; use and spending are unchanged. "
                      "Prevalence: drawn in proportion to prevalence. Chronic: from payment-relevant chronic "
                      "categories. Targeted: the five flags with the largest payment-form coefficients. "
                      "Own-targeted: the five flags that raise that model's own prediction most, found on "
                      "its training data. F2 excludes prior use and spending; F3 includes them."),
              render(wide, ["feature_set", "model"] + pools, [None] * (2 + len(pools)),
                     ["Features", "Model"] + [p.capitalize() for p in pools])]
    if _exists("table5c_own_targeted_flags.csv"):
        o = pd.read_csv(T / "table5c_own_targeted_flags.csv")
        s_ = (o.groupby(["feature_set", "model", "flag"])
               .agg(folds=("fold", "nunique"), gain=("training_gain", "mean")).reset_index()
               .sort_values(["feature_set", "model", "folds", "gain"], ascending=[True, True, False, False]))
        s_["cell"] = s_.apply(lambda r: f"{r['flag']} ({int(r['folds'])}, {dollars()(r['gain'])})", axis=1)
        top = (s_.groupby(["feature_set", "model"]).head(5).groupby(["feature_set", "model"])["cell"]
                 .agg("; ".join).reset_index())
        parts += [caption("7c", "Own-targeted pools: the flags that raise each model's prediction most.",
                          "For each fold, the five flags with the largest mean rise in that model's prediction "
                          "when switched on for 3,000 training persons. Listed are the five chosen most often, "
                          "with the number of folds (of five) and the mean training gain per code."),
                  render(top, ["feature_set", "model", "cell"], [None, None, None],
                         ["Features", "Model", "Flag (folds chosen, mean gain)"])]
    if _exists("table5b_use_sensitivity.csv"):
        u = pd.read_csv(T / "table5b_use_sensitivity.csv")
        parts += [caption("7b", "Use sensitivity: next-year payment per added dollar of year-1 spending.",
                          "Year-1 spending raised by 20% for 10% of held-out people, nothing else changed. "
                          "Only feature sets with prior spending respond."),
                  render(u, ["feature_set", "model", "payment_per_dollar_of_use", "fold_sd", "pct_rise_total_predicted"],
                         [None, None, num(3), num(3), pct(2)],
                         ["Features", "Model", "$ per $ of year-1 spending", "SD across folds", "Total rise"])]

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

    if _exists("table2f_cell_calibration.csv"):
        t = pd.read_csv(T / "table2f_cell_calibration.csv")
        wide = t.pivot_table(index=["age_band", "sex"], columns="model", values="predictive_ratio").reset_index()
        mcols = [c for c in wide.columns if c not in ("age_band", "sex")]
        parts += [caption("A2b", "Predictive ratio by age band and sex, the cells of a payment formula."),
                  render(wide, ["age_band", "sex"] + mcols, [None, None] + [num(3)] * len(mcols),
                         ["Age", "Sex"] + mcols)]

    t = pd.read_csv(T / "table5_coding_sensitivity.csv")
    parts += [caption("A3", "Coding sensitivity, full grid."),
              render(t, ["feature_set", "model", "pool", "share_affected", "codes_added",
                         "pct_rise_total_predicted", "dollars_per_added_code", "dollars_per_code_fold_sd"],
                     [None, None, None, share(0), num(0), pct(2), dollars(), dollars()],
                     ["Features", "Model", "Pool", "Share affected", "Codes added",
                      "Total rise", "$ per code", "SD across folds"])]

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
