"""
Check that the manuscript's numbers still match the analysis output.

The prose is written by hand, so a rerun that moves an estimate leaves the text
stale unless someone notices. This reads the current tables, formats each
quoted figure the way the manuscript writes it, and fails if the string is
absent. It also checks that every table cited is rendered and vice versa, that
every embedded figure exists, that no placeholder is left, and that the prose
carries no em dashes.

Run after run_all.py and make_tables.py, before make_manuscript.py.
"""

from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd

import config

MS = config.ROOT / "manuscript" / "Paper6_manuscript.md"
T = config.TABLES
GBM, MSE, ADL = "LightGBM (Tweedie)", "LightGBM (squared error)", "Needs ADL or IADL help"
PWLS, CWLS = "Payment-form WLS (non-negative)", "Constrained WLS"


def d0(x):
    return f"${abs(x):,.0f}"


def f3(x):
    return f"{x:.3f}"


def sample_checks():
    t1 = pd.read_csv(T / "table1_sample.csv").set_index("characteristic")["estimate"]
    flow = pd.read_csv(T / "table0_sample_flow.csv")
    feats = pd.read_csv(T / "tableA_ccsr_features.csv")
    g = pd.read_csv(T / "table3_group_fairness.csv").set_index(["model", "group"])
    return {
        "persons": f"{int(t1['Persons']):,} persons",
        "population": f"{t1['Weighted population, millions a year']:.1f} million",
        "decedents": f"{int(flow['died_year2'].sum())} people who died in year 2",
        "mean spend": f"${t1['Year-2 spending, mean $']:,.0f}",
        "median spend": f"${t1['Year-2 spending, median $']:,.0f}",
        "p99 spend": f"${t1['Year-2 spending, 99th percentile $']:,.0f}",
        "zero share": f"{t1['Year-2 spending of zero, %']:.1f}%",
        "top-5% share": f"{t1['Share of year-2 spending by the top 5% of spenders, %']:.1f}%",
        "ADL share": f"{t1['Needs help with ADL or IADL, %']:.1f}%",
        "ADL n": f"{int(g.loc[('WLS', ADL), 'n']):,} persons",
        "CCSR features": f"{len(feats)} categories",
    }


def accuracy_checks():
    acc = pd.read_csv(T / "table2_accuracy.csv").set_index("model")
    fs = pd.read_csv(T / "table2b_feature_sets.csv").set_index(["model", "feature_set"])
    pr = pd.read_csv(T / "table2d_paired_differences.csv").set_index(["feature_set", "model"])
    cell = pd.read_csv(T / "table2f_cell_calibration.csv")
    c = {}
    for m, key in [(GBM, "gbm"), ("WLS", "wls"), (PWLS, "pwls")]:
        c[f"R2 {key}"] = f3(acc.loc[m, "r2"])
        c[f"R2 {key} interval"] = f"{f3(acc.loc[m, 'r2_lo'])} to {f3(acc.loc[m, 'r2_hi'])}"
    for m in ["Tweedie GLM", "Two-part", "Elastic net", "Neural network", "CANN", "Random forest", MSE]:
        c[f"R2 {m}"] = f3(acc.loc[m, "r2"])
    c["NN ensemble"] = f"rises to {acc.loc['Neural network', 'r2_ensemble']:.3f}"
    sd = acc["r2_repeat_sd"]
    c["repeat sd max"] = f"at most {sd.drop('Neural network').max():.3f}"
    c["repeat sd NN"] = f"({sd['Neural network']:.3f})"
    c["WLS negative share"] = f"{100 * acc.loc['WLS', 'negative_prediction_share']:.1f}%"
    c["WLS bottom PR"] = f"{acc.loc['WLS', 'pr_bottom_decile']:.2f}"
    for m, key in [("WLS", "wls"), (PWLS, "pwls"), ("Tweedie GLM", "tweedie"), (GBM, "gbm")]:
        c[f"CPM {key}"] = f3(acc.loc[m, "cpm"])
    c["PR top wls"] = f"{acc.loc['WLS', 'pr_top_decile']:.2f} for WLS"
    c["PR top gbm"] = f"{acc.loc[GBM, 'pr_top_decile']:.2f} for Tweedie LightGBM"
    c["top10"] = (f"from {100 * acc.loc['WLS', 'top10_capture']:.0f}% to "
                  f"{100 * acc.loc[GBM, 'top10_capture']:.0f}%")
    c["MAE"] = f"from ${acc.loc['WLS', 'mae']:,.0f} to ${acc.loc[GBM, 'mae']:,.0f}"
    for m in [GBM, "Random forest"]:
        r = pr.loc[("F3", m)]
        c[f"paired {m}"] = f"{r['d_r2']:.3f} ({r['d_r2_lo']:.3f} to {r['d_r2_hi']:.3f})"
    for m in ["Tweedie GLM", "Neural network", PWLS]:
        r = pr.loc[("F3", m)]
        c[f"paired {m}"] = f"({r['d_r2']:.3f}, {r['d_r2_lo']:.3f} to {r['d_r2_hi']:.3f})"
    c["paired gbm_mse"] = f"{pr.loc[('F3', MSE), 'd_r2']:.3f}"
    r = pr.loc[("F2", GBM)]
    c["paired F2 gbm"] = f"{r['d_r2']:.3f} ({r['d_r2_lo']:.3f} to {r['d_r2_hi']:.3f})"
    c["no resample favors"] = "no resample favors WLS" if (pr.loc[("F3", GBM), "p_r2_le_0"] == 0) else "FAIL"
    c["F1"] = f"about {fs.loc[('WLS', 'F1'), 'r2']:.3f}"
    c["F2 wls"] = f"WLS to {fs.loc[('WLS', 'F2'), 'r2']:.3f}"
    c["F2 pwls"] = f"payment-form model to {fs.loc[(PWLS, 'F2'), 'r2']:.3f}"
    c["F2 gbm"] = f"boosting to {fs.loc[(GBM, 'F2'), 'r2']:.3f}"
    tw = fs.loc[("Tweedie GLM", "F2")]
    c["Tweedie F2"] = (f"R² of {tw['r2']:.3f} with a repeat standard deviation of {tw['r2_repeat_sd']:.3f} "
                       f"and a top-decile predictive ratio of {tw['pr_top_decile']:.2f}")
    pc = cell.pivot_table(index=["age_band", "sex"], columns="model", values="predictive_ratio")
    c["cell WLS"] = "within 0.01 of 1" if (pc["WLS"] - 1).abs().max() < 0.01 else "FAIL"
    c["cell gbm"] = f"from {pc[GBM].min():.2f}"
    c["cell gbm hi"] = f"to {pc[GBM].max():.2f}"
    glm = pd.concat([pc["Tweedie GLM"], pc["CANN"]])
    c["cell glm"] = f"from {glm.min():.2f} to {glm.max():.2f}"
    kids = pc.loc["0-17", PWLS]
    c["cell pwls kids"] = f"about {round(100 * (kids.mean() - 1), -1):.0f}% above cost"
    c["pwls excess"] = f"exceed total spending by {100 * (acc.loc[PWLS, 'pr_overall'] - 1):.0f}%"
    return c


def group_checks():
    g = pd.read_csv(T / "table3_group_fairness.csv").set_index(["model", "group"])
    adl = g.xs(ADL, level="group")
    died = g.xs("Died in year 2", level="group")
    c = {"ADL mean cost": f"${g.loc[('WLS', ADL), 'mean_cost']:,.0f}",
         "ADL wls": f"{d0(g.loc[('WLS', ADL), 'nc'])} per person-year by WLS (interval "
                    f"{d0(g.loc[('WLS', ADL), 'nc_hi'])} to {d0(g.loc[('WLS', ADL), 'nc_lo'])})",
         "ADL pwls": f"{d0(adl.loc[PWLS, 'nc'])} by the payment-form model",
         "ADL tweedie": f"{d0(adl.loc['Tweedie GLM', 'nc'])} by the Tweedie GLM",
         "ADL gbm": f"{d0(adl.loc[GBM, 'nc'])} by Tweedie LightGBM",
         "ADL cann": f"{d0(adl.loc['CANN', 'nc'])} by the CANN",
         "ADL range": f"between {d0(adl['nc'].abs().min())} and {d0(adl['nc'].abs().max())}",
         "ADL range abstract": f"by {d0(adl['nc'].abs().min())} to {d0(adl['nc'].abs().max())}"}
    for m, lab in [("Tweedie GLM", "Tweedie GLM"), (GBM, "Tweedie LightGBM")]:
        r = adl.loc[m]
        c[f"ADL paired {lab}"] = (f"{d0(r['nc_minus_wls'])} more {'than WLS (interval ' if m == 'Tweedie GLM' else '('}"
                                  f"{d0(r['nc_minus_wls_lo'])} to {d0(r['nc_minus_wls_hi'])})")
    c["ADL paired pwls"] = f"{d0(adl.loc[PWLS, 'nc_minus_wls'])} less"
    c["died range"] = f"{d0(died['nc'].abs().min())} to {d0(died['nc'].abs().max())}"
    c["died PR"] = f"{died['pr'].min():.2f} to {died['pr'].max():.2f}"
    c["hisp black wls"] = (f"{d0(g.loc[('WLS', 'Hispanic'), 'nc'])} and "
                           f"{d0(g.loc[('WLS', 'Non-Hispanic Black'), 'nc'])}")
    c["gbm race"] = f"within {d0(max(abs(g.loc[(GBM, 'Hispanic'), 'nc']), abs(g.loc[(GBM, 'Non-Hispanic Black'), 'nc'])))}"
    u = g.loc[(GBM, "Uninsured all year")]
    c["uninsured gbm"] = f"{d0(u['nc'])} (interval {d0(u['nc_lo'])} to {d0(u['nc_hi'])}), a predictive ratio of {u['pr']:.2f}"
    c["uninsured cost"] = f"mean cost of ${u['mean_cost']:,.0f}"
    c["MH rf cann"] = (f"{d0(g.loc[('Random forest', 'Mental health condition'), 'nc'])} and "
                       f"{d0(g.loc[('CANN', 'Mental health condition'), 'nc'])}")
    o = g.loc[(MSE, "Age 65 and over")]
    c["65 mse"] = f"{d0(o['nc'])} ({d0(o['nc_hi'])} to {d0(o['nc_lo'])})"
    c["65 rf"] = d0(g.loc[("Random forest", "Age 65 and over"), "nc"])
    races = [g.loc[(PWLS, k), "nc"] for k in ("Hispanic", "Non-Hispanic Black", "Non-Hispanic Asian")]
    c["pwls races"] = f"{d0(min(races))} to {d0(max(races))}"
    c["pwls income"] = f"{d0(g.loc[(PWLS, 'Income below 200% FPL'), 'nc'])} for low income"
    return c


def frontier_checks():
    fr = pd.read_csv(T / "table3b_frontier.csv")
    byf = pd.read_csv(T / "table3c_constraint_by_fold.csv")
    acc = pd.read_csv(T / "table2_accuracy.csv").set_index("model")

    def row(method, lam):
        return fr[(fr["method"] == method) & (np.isclose(fr["lambda"], lam) | (np.isinf(lam) & np.isinf(fr["lambda"])))].iloc[0]
    p0, p1, cw = row("Penalized WLS", 0), row("Penalized WLS", 1), row(CWLS, np.inf)
    s0, s1 = row("Stacked LightGBM", 0), row("Stacked LightGBM", 1)
    sc, rc = row("Stacked LightGBM, constrained", np.inf), row("LightGBM, decile recalibration", 0)
    c = {"pen0": f"{d0(p0['max_abs_nc_target'])} at λ = 0",
         "pen1": f"to {d0(p1['max_abs_nc_target'])} at λ = 1 reduces R² from {p0['r2']:.3f} to {p1['r2']:.3f}",
         "cwls": f"leaves {d0(cw['max_abs_nc_target'])} at an R² of {cw['r2']:.3f}",
         "stacked0": f"gap of {d0(s0['max_abs_nc_target'])} at an R² of {s0['r2']:.3f}",
         "stacked1": f"reaches {d0(s1['max_abs_nc_target'])} at λ = 1 with an R² of {s1['r2']:.3f}",
         "stackedc": f"{d0(sc['max_abs_nc_target'])} at {sc['r2']:.3f} when constrained",
         "stacked loss": f"a loss of {100 * (1 - sc['r2'] / s0['r2']):.0f}% of its own R²",
         "stacked vs gbm": f"of {acc.loc[GBM, 'r2'] - sc['r2']:.3f} relative to the unconstrained boosted model",
         "abstract stackedc": f"to {d0(sc['max_abs_nc_target'])} at an R² of {sc['r2']:.3f}",
         "recal": f"keeps R² at {rc['r2']:.3f} and leaves the gap at {d0(rc['max_abs_nc_target'])}",
         "pen0 interval": f"{d0(p0['max_abs_nc_target_lo'])} to {d0(p0['max_abs_nc_target_hi'])} for WLS at λ = 0",
         "cwls interval": f"{d0(cw['max_abs_nc_target_lo'])} to {d0(cw['max_abs_nc_target_hi'])} under the constraint",
         "stackedc interval": f"{d0(sc['max_abs_nc_target_lo'])} to {d0(sc['max_abs_nc_target_hi'])}, lies above",
         "cwls 65": f"overpayment of {d0(cw['nc::Age 65 and over'])} (predictive ratio {cw['pr::Age 65 and over']:.2f})",
         "cwls hisp": f"{d0(cw['nc::Hispanic'])} below cost",
         "cwls black": f"{d0(cw['nc::Non-Hispanic Black'])} below cost",
         "cwls uninsured": f"{d0(cw['nc::Uninsured all year'])} below cost, a predictive ratio of {cw['pr::Uninsured all year']:.2f}",
         "uninsured pct": f"{100 * (1 - cw['pr::Uninsured all year']):.0f}% underpayment of the uninsured",
         "stackedc spill": (f"{d0(sc['nc::Age 65 and over'])} for the elderly, and −{d0(sc['nc::Hispanic'])}, "
                            f"−{d0(sc['nc::Non-Hispanic Black'])} and −{d0(sc['nc::Uninsured all year'])}"),
         "spill range": f"moved {d0(sc['nc::Age 65 and over'])} to {d0(cw['nc::Age 65 and over'])}"}
    col = "out_of_fold::Needs ADL or IADL help"
    for m, lab in [(CWLS, "cwls"), ("Stacked LightGBM, constrained", "stacked")]:
        v = byf.loc[byf["method"] == m, col]
        c[f"fold {lab}"] = f"from −{d0(v.min())} to +{d0(v.max())}"
    return c


def driver_checks():
    imp = pd.read_csv(T / "table4_importance.csv").set_index("feature")
    stab = pd.read_csv(T / "table4b_stability.csv")
    sense = pd.read_csv(T / "table4c_sense_checks.csv").groupby("feature")["spearman_value_vs_shap"].mean()
    perm = imp["cann_permutation_r2_loss"]
    return {
        "shap": (f"(mean absolute SHAP {d0(imp.loc['log_spend_y1', 'mean_abs_shap'])}), followed by prescription fills "
                 f"({d0(imp.loc['log_rx_fills_y1', 'mean_abs_shap'])}), the condition count "
                 f"({d0(imp.loc['n_conditions', 'mean_abs_shap'])}), office visits "
                 f"({d0(imp.loc['log_office_visits_y1', 'mean_abs_shap'])}), inpatient discharges "
                 f"({d0(imp.loc['log_inpatient_y1', 'mean_abs_shap'])}) and the body-system count "
                 f"({d0(imp.loc['n_body_systems', 'mean_abs_shap'])})"),
        "diabetes": f"diabetes ({d0(imp.loc['ccsr_END002+END005', 'mean_abs_shap'])})",
        "stability": f"Spearman {stab['spearman_all'].mean():.2f} over all features, {stab['spearman_top30'].mean():.2f} over the top 30",
        "sense": (f"rank correlations {sense['log_spend_y1']:.2f}, {sense['n_conditions']:.2f} and "
                  f"{sense['n_body_systems']:.2f}"),
        "cann perm": f"costs {perm['log_spend_y1']:.3f} of R²",
        "cann perm other": f"more than {perm.drop('log_spend_y1').max():.3f}",
    }


def use_checks():
    u = pd.read_csv(T / "table5b_use_sensitivity.csv")
    u = u[u["feature_set"] == "F3"].set_index("model")
    cents = lambda m: f"{100 * u.loc[m, 'payment_per_dollar_of_use']:.0f}"
    return {
        "use gbm": f"Tweedie LightGBM pays {cents(GBM)} cents (fold standard deviation {100 * u.loc[GBM, 'fold_sd']:.0f})",
        "use cwls": f"constrained WLS {cents(CWLS)} cents",
        "use mse": f"squared-error LightGBM {cents(MSE)}",
        "use wls": f"WLS {cents('WLS')}",
        "use glm": f"Tweedie GLM and CANN {cents('Tweedie GLM')}" if cents("Tweedie GLM") == cents("CANN") else "FAIL",
        "use pwls": f"payment-form model pays {cents(PWLS)} cent",
        "use twopart": f"two-part model {cents('Two-part')}.",
    }


def robustness_checks():
    rob = pd.read_csv(T / "table6_robustness.csv").set_index(["variant", "model"])
    r2 = lambda v, m: f3(rob.loc[(v, m), "r2"])
    gap = lambda v, m: d0(rob.loc[(v, m), "max_abs_nc_target"])
    wide = rob["r2"].unstack("model")
    adv = (wide[GBM] - wide["WLS"]).drop(["no prior spending (F2)", "log outcome"], errors="ignore").dropna()
    panels = wide.loc[[v for v in wide.index if v.startswith("panel ")]]
    pgap = rob.loc[[v for v in rob.index.get_level_values(0).unique() if v.startswith("panel ")], "max_abs_nc_target"]
    return {
        "advantage range": f"between {adv.min():.3f} and {adv.max():.3f}",
        "cap": f"Tweedie LightGBM to {r2('outcome capped', GBM)}, WLS to {r2('outcome capped', 'WLS')}",
        "F2 gap": f"boosting to {gap('no prior spending (F2)', GBM)}",
        "F4 gap": f"{gap('social-risk features (F4)', 'WLS')} for WLS and {gap('social-risk features (F4)', GBM)} for boosting",
        "decedents excluded": f"(Tweedie LightGBM {r2('decedents excluded', GBM)}, WLS {r2('decedents excluded', 'WLS')})",
        "not annualized": f"({r2('decedents not annualized', GBM)} and {r2('decedents not annualized', 'WLS')})",
        "tweedie floor": f"rises from {r2('baseline', 'Tweedie GLM')} to {r2('CCSR floor 1%', 'Tweedie GLM')}",
        "log": f"R² {rob.loc[('log outcome', 'WLS, log outcome'), 'r2']:.3f}",
        "panels wls": f"WLS ranges from {panels['WLS'].min():.3f} to {panels['WLS'].max():.3f}",
        "panels gbm": f"Tweedie LightGBM from {panels[GBM].min():.3f} to {panels[GBM].max():.3f}",
        "panels gap": f"about ${np.round(pgap.min() / 500) * 500:,.0f} to ${np.round(pgap.max() / 1000) * 1000:,.0f}",
        "pandemic": f"Tweedie LightGBM {r2('pandemic outcomes out', GBM)} and WLS {r2('pandemic outcomes out', 'WLS')}",
        "65+": (f"(Tweedie LightGBM {r2('age 65 and over', GBM)}, Tweedie GLM {r2('age 65 and over', 'Tweedie GLM')}, "
                f"WLS {r2('age 65 and over', 'WLS')})"),
        "u65": f"Tweedie LightGBM reaches {r2('under 65, private', GBM)} and WLS {r2('under 65, private', 'WLS')}",
    }


def coding_checks():
    t = pd.read_csv(T / "table5_coding_sensitivity.csv")
    t = t[(t["share_affected"] == 0.10) & (t["codes_added"] == 1)]
    d = t.set_index(["feature_set", "pool", "model"])["dollars_per_added_code"]
    rise = t.set_index(["feature_set", "pool", "model"])["pct_rise_total_predicted"]
    feats = pd.read_csv(T / "tableA_ccsr_features.csv").set_index("ccsr")["prevalence"]
    targeted = ["NVS015", "MUS003", "GEN006", "NVS009", "CIR019"]
    c = {}
    c["prevalence F3"] = (f"{d0(d[('F3', 'prevalence', 'WLS')])} per code under WLS, "
                          f"{d0(d[('F3', 'prevalence', GBM)])} under Tweedie LightGBM and "
                          f"{d0(d[('F3', 'prevalence', PWLS)])} under the payment-form model")
    c["chronic F3"] = (f"a chronic code by {d0(d[('F3', 'chronic', 'WLS')])}, "
                       f"{d0(d[('F3', 'chronic', GBM)])} and {d0(d[('F3', 'chronic', PWLS)])}")
    c["chronic F2"] = (f"raises WLS by {d0(d[('F2', 'chronic', 'WLS')])}, the Tweedie GLM by "
                       f"{d0(d[('F2', 'chronic', 'Tweedie GLM')])}, Tweedie LightGBM by "
                       f"{d0(d[('F2', 'chronic', GBM)])} and the payment-form model by "
                       f"{d0(d[('F2', 'chronic', PWLS)])}")
    c["targeted prevalence"] = f"{100 * feats[targeted].min():.1f}% to {100 * feats[targeted].max():.1f}% of the sample"
    c["targeted F3"] = (f"{d0(d[('F3', 'targeted', 'WLS')])} under WLS, {d0(d[('F3', 'targeted', PWLS)])} under the "
                        f"payment-form model and {d0(d[('F3', 'targeted', CWLS)])} under constrained WLS, against "
                        f"{d0(d[('F3', 'targeted', GBM)])} under Tweedie LightGBM")
    c["targeted F2"] = (f"{d0(d[('F2', 'targeted', 'WLS')])}, {d0(d[('F2', 'targeted', PWLS)])} and "
                        f"{d0(d[('F2', 'targeted', CWLS)])} against {d0(d[('F2', 'targeted', GBM)])}")
    own = lambda fs, m: d0(d[(fs, "own-targeted", m)])
    c["own F3"] = (f"{own('F3', GBM)} under Tweedie LightGBM, {own('F3', 'Two-part')} under the two-part model, "
                   f"{own('F3', MSE)} under squared-error LightGBM, {own('F3', 'Tweedie GLM')} under the Tweedie GLM, "
                   f"{own('F3', 'CANN')} under the CANN, {own('F3', 'WLS')} under WLS, {own('F3', PWLS)} under the "
                   f"payment-form model and {own('F3', CWLS)} under constrained WLS")
    c["own F2"] = (f"{own('F2', GBM)} for Tweedie LightGBM, {own('F2', 'Tweedie GLM')} for the Tweedie GLM, "
                   f"{own('F2', 'WLS')} for WLS, {own('F2', PWLS)} for the payment-form model and "
                   f"{own('F2', CWLS)} for constrained WLS")
    r = lambda m: f"{rise[('F2', 'own-targeted', m)]:.1f}%"
    c["own F2 rise"] = (f"rises by {r(GBM)} under Tweedie LightGBM, {r('WLS')} under WLS, {r(PWLS)} under the "
                        f"payment-form model and {r(CWLS)} under constrained WLS")
    c["abstract coding"] = (f"{d0(d[('F2', 'own-targeted', PWLS)])} per added code under the payment-form model and "
                            f"{d0(d[('F2', 'own-targeted', CWLS)])} under the constrained linear formula, against "
                            f"{d0(d[('F2', 'own-targeted', GBM)])} under boosting")
    return c


def cross_reference():
    tb = config.ROOT / "manuscript" / "tables.md"
    if not tb.exists():
        print("  tables.md not built yet; skipping the cross-reference check")
        return []
    rendered = set(re.findall(r"\*\*Table ([0-9A-Za-z]+)\.\*\*", tb.read_text()))
    cited = set()
    tok = r"([0-9]+[a-f]?|A[0-9]+[a-f]?)"
    for m in re.finditer(rf"Tables? {tok}(?:(?:,\s*|\s+and\s+){tok})*", MS.read_text()):
        cited.update(re.findall(tok, m.group(0)))
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
    checks = {}
    for fn in (sample_checks, accuracy_checks, group_checks, frontier_checks, driver_checks,
               use_checks, robustness_checks, coding_checks):
        checks.update(fn())
    bad = [(k, v) for k, v in checks.items() if v.replace("−", "-") not in text]
    width = max(len(k) for k in checks)
    for k, v in checks.items():
        print(f"  {'ok ' if (k, v) not in bad else 'MISSING'}  {k:<{width}}  {v}")
    xref, figs = cross_reference(), figures()
    dashes = raw.count("—")
    left = re.findall(r"\b(?:CODING_[A-Z]+|WORDCOUNT)\b", raw)
    if dashes:
        print(f"  {dashes} em dash(es) in the prose")
    if left:
        print(f"  placeholders left: {sorted(set(left))}")
    if bad or xref or figs or dashes or left:
        if bad:
            print(f"\n{len(bad)} figure(s) do not appear in {MS.name}.")
        sys.exit(1)
    print(f"\nAll {len(checks)} figures match the current tables.")


if __name__ == "__main__":
    main()
