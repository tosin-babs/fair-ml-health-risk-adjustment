"""
Check that the manuscript's headline numbers still match the analysis output.

The prose is written by hand, so a rerun that moves an estimate leaves the text
stale unless someone notices. This reads the current tables, formats each
headline figure the way the manuscript writes it, and fails if the string is
absent. It also checks that every table cited is rendered and vice versa, that
every embedded figure exists, and that the prose carries no em dashes.

Run after run_all.py and before make_manuscript.py.
"""

from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd

import config

MS = config.ROOT / "manuscript" / "Paper6_manuscript.md"
T = config.TABLES


def load():
    t1 = pd.read_csv(T / "table1_sample.csv").set_index("characteristic")["estimate"]
    acc = pd.read_csv(T / "table2_accuracy.csv").set_index("model")
    fs = pd.read_csv(T / "table2b_feature_sets.csv").set_index(["model", "feature_set"])
    g = pd.read_csv(T / "table3_group_fairness.csv").set_index(["model", "group"])
    fr = pd.read_csv(T / "table3b_frontier.csv")
    imp = pd.read_csv(T / "table4_importance.csv").set_index("feature")
    stab = pd.read_csv(T / "table4b_stability.csv")
    sense = pd.read_csv(T / "table4c_sense_checks.csv").groupby("feature")["spearman_value_vs_shap"].mean()
    cod = pd.read_csv(T / "table5_coding_sensitivity.csv")
    rob = pd.read_csv(T / "table6_robustness.csv").set_index(["variant", "model"])
    flow = pd.read_csv(T / "table0_sample_flow.csv")
    feats = pd.read_csv(T / "tableA_ccsr_features.csv")

    ADL = "Needs ADL or IADL help"
    adl_nc = g.xs(ADL, level="group")["nc"]
    chronic = cod[(cod["pool"] == "chronic") & (cod["share_affected"] == 0.10) & (cod["codes_added"] == 1)].set_index("model")["dollars_per_added_code"]
    prev = cod[(cod["pool"] == "prevalence") & (cod["share_affected"] == 0.10) & (cod["codes_added"] == 1)].set_index("model")["dollars_per_added_code"]
    pen0 = fr[(fr["method"] == "Penalized WLS") & (fr["lambda"] == 0)].iloc[0]
    cwls = fr[fr["method"] == "Constrained WLS"].iloc[0]
    st0 = fr[(fr["method"] == "Stacked LightGBM") & (fr["lambda"] == 0)].iloc[0]
    stc = fr[fr["method"] == "Stacked LightGBM, constrained"].iloc[0]
    recal = fr[fr["method"] == "LightGBM, decile recalibration"].iloc[0]
    gbm = "LightGBM (Tweedie)"

    return {
        "persons": f"{int(t1['Persons']):,} persons",
        "population": f"{t1['Weighted population, millions a year']:.1f} million",
        "mean spend": f"${t1['Year-2 spending, mean $']:,.0f}",
        "median spend": f"${t1['Year-2 spending, median $']:,.0f}",
        "zero share": f"{t1['Year-2 spending of zero, %']:.1f}%",
        "top-5% share": f"{t1['Share of year-2 spending by the top 5% of spenders, %']:.1f}%",
        "ADL share": f"{t1['Needs help with ADL or IADL, %']:.1f}%",
        "CCSR features": f"{len(feats)} categories",
        "R2 gbm": f"{acc.loc[gbm, 'r2']:.3f}",
        "R2 gbm lo": f"{acc.loc[gbm, 'r2_lo']:.3f}",
        "R2 gbm hi": f"{acc.loc[gbm, 'r2_hi']:.3f}",
        "R2 wls": f"{acc.loc['WLS', 'r2']:.3f}",
        "R2 wls lo": f"{acc.loc['WLS', 'r2_lo']:.3f}",
        "R2 wls hi": f"{acc.loc['WLS', 'r2_hi']:.3f}",
        "R2 tweedie": f"{acc.loc['Tweedie GLM', 'r2']:.3f}",
        "R2 two-part": f"{acc.loc['Two-part', 'r2']:.3f}",
        "R2 enet": f"{acc.loc['Elastic net', 'r2']:.3f}",
        "R2 mlp": f"{acc.loc['Neural network', 'r2']:.3f}",
        "R2 cann": f"{acc.loc['CANN', 'r2']:.3f}",
        "PR top decile wls": f"{acc.loc['WLS', 'pr_top_decile']:.2f}",
        "PR top decile gbm": f"{acc.loc[gbm, 'pr_top_decile']:.2f}",
        "top10 wls": f"{100 * acc.loc['WLS', 'top10_capture']:.0f}%",
        "top10 gbm": f"{100 * acc.loc[gbm, 'top10_capture']:.0f}%",
        "MAE wls": f"${acc.loc['WLS', 'mae']:,.0f}",
        "MAE gbm": f"${acc.loc[gbm, 'mae']:,.0f}",
        "F1 R2": f"{fs.loc[('WLS', 'F1'), 'r2']:.3f}",
        "F2 R2 wls": f"{fs.loc[('WLS', 'F2'), 'r2']:.3f}",
        "F2 R2 gbm": f"{fs.loc[(gbm, 'F2'), 'r2']:.3f}",
        "ADL mean cost": f"${g.loc[('WLS', ADL), 'mean_cost']:,.0f}",
        "ADL nc wls": f"${abs(g.loc[('WLS', ADL), 'nc']):,.0f}",
        "ADL nc wls lo": f"${abs(g.loc[('WLS', ADL), 'nc_hi']):,.0f}",
        "ADL nc wls hi": f"${abs(g.loc[('WLS', ADL), 'nc_lo']):,.0f}",
        "ADL nc tweedie": f"${abs(g.loc[('Tweedie GLM', ADL), 'nc']):,.0f}",
        "ADL nc gbm": f"${abs(g.loc[(gbm, ADL), 'nc']):,.0f}",
        "ADL nc min": f"${abs(adl_nc).min():,.0f}",
        "ADL nc max": f"${abs(adl_nc).max():,.0f}",
        "Hispanic nc wls": f"${abs(g.loc[('WLS', 'Hispanic'), 'nc']):,.0f}",
        "Black nc wls": f"${abs(g.loc[('WLS', 'Non-Hispanic Black'), 'nc']):,.0f}",
        "65+ nc gbm": f"${abs(g.loc[(gbm, 'Age 65 and over'), 'nc']):,.0f}",
        "65+ nc gbm mse": f"${abs(g.loc[('LightGBM (squared error)', 'Age 65 and over'), 'nc']):,.0f}",
        "frontier wls gap": f"${pen0['max_abs_nc_target']:,.0f}",
        "frontier wls floor": f"${cwls['max_abs_nc_target']:,.0f}",
        "frontier wls r2": f"{pen0['r2']:.3f}",
        "frontier cwls r2": f"{cwls['r2']:.3f}",
        "frontier stacked gap": f"${st0['max_abs_nc_target']:,.0f}",
        "frontier stacked floor": f"${stc['max_abs_nc_target']:,.0f}",
        "frontier stacked r2": f"{st0['r2']:.3f}",
        "frontier stacked c r2": f"{stc['r2']:.3f}",
        "recalibration gap": f"${recal['max_abs_nc_target']:,.0f}",
        "65+ spillover wls": f"${cwls['nc::Age 65 and over']:,.0f}",
        "65+ spillover stacked": f"${stc['nc::Age 65 and over']:,.0f}",
        "Black spillover wls": f"${abs(cwls['nc::Non-Hispanic Black']):,.0f}",
        "Black spillover stacked": f"${abs(stc['nc::Non-Hispanic Black']):,.0f}",
        "shap spend": f"${imp.loc['log_spend_y1', 'mean_abs_shap']:,.0f}",
        "shap rx": f"${imp.loc['log_rx_fills_y1', 'mean_abs_shap']:,.0f}",
        "shap conditions": f"${imp.loc['n_conditions', 'mean_abs_shap']:,.0f}",
        "shap visits": f"${imp.loc['log_office_visits_y1', 'mean_abs_shap']:,.0f}",
        "shap inpatient": f"${imp.loc['log_inpatient_y1', 'mean_abs_shap']:,.0f}",
        "shap systems": f"${imp.loc['n_body_systems', 'mean_abs_shap']:,.0f}",
        "stability all": f"{stab['spearman_all'].mean():.2f}",
        "stability top30": f"{stab['spearman_top30'].mean():.2f}",
        "sense spend": f"{sense['log_spend_y1']:.2f}",
        "sense conditions": f"{sense['n_conditions']:.2f}",
        "cann perm spend": f"{imp.loc['log_spend_y1', 'cann_permutation_r2_loss']:.3f}",
        "coding wls": f"${chronic['WLS']:,.0f}",
        "coding two-part": f"${chronic['Two-part']:,.0f}",
        "coding cwls": f"${chronic['Constrained WLS']:,.0f}",
        "coding cann": f"${chronic['CANN']:,.0f}",
        "coding tweedie": f"${chronic['Tweedie GLM']:,.0f}",
        "coding gbm mse": f"${chronic['LightGBM (squared error)']:,.0f}",
        "coding gbm": f"${chronic[gbm]:,.0f}",
        "prevalence trees lo": f"${min(prev[gbm], prev['LightGBM (squared error)']):,.0f}",
        "prevalence trees hi": f"${max(prev[gbm], prev['LightGBM (squared error)']):,.0f}",
        "prevalence linear lo": f"${min(prev['WLS'], prev['Constrained WLS'], prev['Tweedie GLM']):,.0f}",
        "prevalence linear hi": f"${max(prev['WLS'], prev['Constrained WLS'], prev['Tweedie GLM']):,.0f}",
        "robust cap gbm": f"{rob.loc[('outcome capped', gbm), 'r2']:.3f}",
        "robust F2 gap": f"${rob.loc[('no prior spending (F2)', gbm), 'max_abs_nc_target']:,.0f}",
        "robust F4 gap gbm": f"${rob.loc[('social-risk features (F4)', gbm), 'max_abs_nc_target']:,.0f}",
        "robust F4 gap wls": f"${rob.loc[('social-risk features (F4)', 'WLS'), 'max_abs_nc_target']:,.0f}",
        "robust log R2": f"{rob.loc[('log outcome', 'WLS, log outcome'), 'r2']:.3f}".replace("-", "−"),
        "robust 65+ gbm": f"{rob.loc[('age 65 and over', gbm), 'r2']:.3f}",
        "robust 65+ wls": f"{rob.loc[('age 65 and over', 'WLS'), 'r2']:.3f}",
    }


def cross_reference():
    tb = config.ROOT / "manuscript" / "tables.md"
    if not tb.exists():
        print("  tables.md not built yet; skipping the cross-reference check")
        return []
    rendered = set(re.findall(r"\*\*Table ([0-9A-Za-z]+)\.\*\*", tb.read_text()))
    cited = set()
    for m in re.finditer(r"Tables? ([0-9]+[a-f]?|A[0-9]+)(?:\s+and\s+([0-9]+[a-f]?|A[0-9]+))?",
                         MS.read_text()):
        cited.add(m.group(1))
        if m.group(2):
            cited.add(m.group(2))
    problems = [f"Table {t} is cited in the prose but not rendered" for t in sorted(cited - rendered)]
    problems += [f"Table {t} is rendered but never cited" for t in sorted(rendered - cited)]
    print(f"  {len(rendered)} tables rendered, {len(cited)} cited"
          + ("" if not problems else f"  <-- {len(problems)} mismatch(es)"))
    for p_ in problems:
        print(f"      {p_}")
    return problems


def figures():
    missing = [p for p in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", MS.read_text())
               if not (config.ROOT / p).exists()]
    for p in missing:
        print(f"      figure not found: {p}")
    return missing


def main():
    raw = MS.read_text()
    text = raw.replace("−", "-").replace("**", "")
    checks = load()
    bad = [(k, v) for k, v in checks.items() if v.replace("−", "-") not in text]
    width = max(len(k) for k in checks)
    for k, v in checks.items():
        print(f"  {'ok ' if (k, v) not in bad else 'MISSING'}  {k:<{width}}  {v}")
    xref, figs = cross_reference(), figures()
    dashes = raw.count("—")
    if dashes:
        print(f"  {dashes} em dash(es) in the prose")
    if bad or xref or figs or dashes:
        if bad:
            print(f"\n{len(bad)} headline figure(s) do not appear in {MS.name}.")
        sys.exit(1)
    print(f"\nAll {len(checks)} headline figures match the current tables.")


if __name__ == "__main__":
    main()
