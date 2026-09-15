"""
The prospective dataset: year-1 characteristics, year-2 spending.

One row per person observed in both years of a MEPS two-year panel with a
positive longitudinal weight. Predictors come only from year 1; the outcome is
total expenditure from all payers in year 2, in 2024 dollars.

Writes data/derived/prospective.pkl (person level) and
data/derived/conditions_y1.pkl (person x CCSR category, year 1 only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config

PERSON_COLS = ["DUPERSID", "PANEL", "YEARIND", "DIED", "LONGWT", "VARSTR",
               "VARPSU", "AGEY1X", "SEX", "RACETHX", "REGIONY1", "POVCATY1",
               "INSCOVY1", "TOTEXPY1", "TOTEXPY2", "TOTSLFY1", "OBTOTVY1",
               "ERTOTY1", "IPDISY1", "RXTOTY1", "ENDRFMY2"]


def _neg_to_nan(s):
    """MEPS codes inapplicable, refused and don't-know as negative values."""
    s = pd.to_numeric(s, errors="coerce").astype(float)
    return s.where(s >= 0)


def load_panel(panel, spec):
    h, f = config.HEALTH_ROUND, config.FUNCTION_ROUND
    cols = PERSON_COLS + [f"RTHLTH{h}", f"MNHLTH{h}", f"ADLHLP{f}", f"IADLHP{f}"]
    d = pd.read_stata(config.MEPS / spec["file"], columns=cols,
                      convert_categoricals=False)
    y1 = spec["year1"]
    out = pd.DataFrame({
        "person_id": d["DUPERSID"].astype(str),
        "panel": panel,
        "year1": y1,
        "in_both_years": d["YEARIND"] == 1,
        "died_y2": d["DIED"] == 1,
        "weight_panel": pd.to_numeric(d["LONGWT"], errors="coerce"),
        "cluster": (f"p{panel}_" + d["VARSTR"].astype(str) + "_"
                    + d["VARPSU"].astype(str)),
        "stratum": f"p{panel}_" + d["VARSTR"].astype(str),
        "age": _neg_to_nan(d["AGEY1X"]),
        "female": (d["SEX"] == 2).astype(int),
        "race": _neg_to_nan(d["RACETHX"]),
        "region": _neg_to_nan(d["REGIONY1"]),
        "povcat": _neg_to_nan(d["POVCATY1"]),
        "inscov": _neg_to_nan(d["INSCOVY1"]),
        "health": _neg_to_nan(d[f"RTHLTH{h}"]),
        "mental_health": _neg_to_nan(d[f"MNHLTH{h}"]),
        "adl_help": (_neg_to_nan(d[f"ADLHLP{f}"]) == 1).astype(int),
        "iadl_help": (_neg_to_nan(d[f"IADLHP{f}"]) == 1).astype(int),
        "spend_y1": _neg_to_nan(d["TOTEXPY1"]) * config.CPI_TO_BASE[y1],
        "oop_y1": _neg_to_nan(d["TOTSLFY1"]) * config.CPI_TO_BASE[y1],
        "office_visits_y1": _neg_to_nan(d["OBTOTVY1"]),
        "er_visits_y1": _neg_to_nan(d["ERTOTY1"]),
        "inpatient_y1": _neg_to_nan(d["IPDISY1"]),
        "rx_fills_y1": _neg_to_nan(d["RXTOTY1"]),
        "y": _neg_to_nan(d["TOTEXPY2"]) * config.CPI_TO_BASE[y1 + 1],
        "months_y2": _neg_to_nan(d["ENDRFMY2"]).clip(1, 12),
    })
    # A person who died in year 2 was in scope for part of the year. A payer
    # pays for the months of coverage, so the outcome is annualized by the
    # months in scope, from the reference-period end month; survivors have a
    # full year.
    # The weight is multiplied by the same fraction, as CMS does when it fits
    # payment models on annualized costs: a person in scope for one month
    # counts for a twelfth of a person-year, so an annualized amount cannot
    # dominate the fit or the evaluation.
    out["y_raw"] = out["y"]
    out["exposure"] = np.where(out["died_y2"], out["months_y2"].fillna(12) / 12.0, 1.0)
    if config.ANNUALIZE_DECEDENTS:
        out["y"] = out["y"] / out["exposure"]
        out["weight_panel"] = out["weight_panel"] * out["exposure"]
    out["any_function_help"] = ((out["adl_help"] == 1)
                                | (out["iadl_help"] == 1)).astype(int)
    return out


def load_conditions(panel, spec, ids):
    """Year-1 CCSR categories per person. Year-2 conditions are never read."""
    c = pd.read_stata(config.MEPS / spec["conditions"],
                      convert_categoricals=False)
    c = c[c["DUPERSID"].astype(str).isin(ids)]
    ccsr_cols = [k for k in c.columns if k.startswith("CCSR") and k.endswith("X")]
    long = (c.melt(id_vars=["DUPERSID"], value_vars=ccsr_cols,
                   value_name="ccsr")
             .assign(ccsr=lambda x: x["ccsr"].astype(str).str.strip()))
    long = long[long["ccsr"].str.match(r"^[A-Z]{3}\d{3}$")]
    long = (long.rename(columns={"DUPERSID": "person_id"})
                .assign(person_id=lambda x: x["person_id"].astype(str),
                        panel=panel)
                [["panel", "person_id", "ccsr"]].drop_duplicates())
    return long


def build(exclude_died=None, panels=None):
    """Assemble the person and condition frames without writing them."""
    exclude = config.EXCLUDE_DIED if exclude_died is None else exclude_died
    persons, conds, audit = [], [], []
    for panel, spec in config.PANELS.items():
        if panels is not None and panel not in panels:
            continue
        p = load_panel(panel, spec)
        n0 = len(p)
        p = p[p["in_both_years"] & (p["weight_panel"] > 0)]
        n_both = len(p)
        died = int(p["died_y2"].sum())
        if exclude:
            p = p[~p["died_y2"]]
        p = p[p["y"].notna() & p["age"].notna()]
        c = load_conditions(panel, spec, set(p["person_id"]))
        persons.append(p)
        conds.append(c)
        audit.append({"panel": panel, "years": f"{spec['year1']}-{spec['year1'] + 1}",
                      "persons_in_file": n0, "both_years_positive_weight": n_both,
                      "died_year2": died, "analysis_sample": len(p),
                      "with_any_condition": c["person_id"].nunique()})

    d = pd.concat(persons, ignore_index=True)
    c = pd.concat(conds, ignore_index=True)
    d["uid"] = d["panel"].astype(str) + "_" + d["person_id"]
    c["uid"] = c["panel"].astype(str) + "_" + c["person_id"]

    # Pooled weight: each panel's longitudinal weight represents the population
    # once, so dividing by the number of panels gives an average-year
    # population across the five pairs.
    d["weight"] = d["weight_panel"] / d["panel"].nunique()

    # MEPS assigns XXX000 (BLD000, MBD000, ...) where a condition could be
    # placed only at body-system level, typically because its ICD-10 code is
    # suppressed. These are not CCSR categories: they carry the body system
    # and nothing else. They are kept for body-system counts and excluded from
    # category flags, condition counts and group definitions.
    c["valid_category"] = ~c["ccsr"].str.endswith("000")
    valid = c[c["valid_category"]]
    any_mbd = set(valid.loc[valid["ccsr"].str.startswith("MBD"), "uid"])
    d["any_mbd"] = d["uid"].isin(any_mbd).astype(int)
    d["n_conditions"] = d["uid"].map(valid.groupby("uid")["ccsr"].nunique()).fillna(0)

    assert d["uid"].is_unique, "a person appears twice"
    assert c["uid"].isin(set(d["uid"])).all(), "conditions for unknown people"
    return d, c, audit


def main():
    d, c, audit = build()
    d.to_pickle(config.DERIVED / "prospective.pkl")
    c.to_pickle(config.DERIVED / "conditions_y1.pkl")
    a = pd.DataFrame(audit)
    a.to_csv(config.TABLES / "table0_sample_flow.csv", index=False)

    w = d["weight"]
    print("=== Sample flow ===")
    print(a.to_string(index=False))
    print(f"\n  analysis sample {len(d):,} persons; weighted "
          f"{w.sum() / 1e6:,.1f} million a year")
    print(f"  year-2 spending: weighted mean ${np.average(d['y'], weights=w):,.0f}, "
          f"median ${d['y'].median():,.0f}, zero share "
          f"{np.average(d['y'] == 0, weights=w):.1%}")
    top = d["y"].quantile(0.99)
    print(f"  top 1% threshold ${top:,.0f}; max ${d['y'].max():,.0f}; "
          f"above ${config.OUTCOME_CAP:,}: {int((d['y'] > config.OUTCOME_CAP).sum())}")
    print(f"  year-1 conditions: {c['ccsr'].nunique()} CCSR categories, "
          f"{np.average(d['n_conditions'] > 0, weights=w):.1%} with any")
    print(f"  correlation of year-1 and year-2 spending: "
          f"{np.corrcoef(d['spend_y1'].fillna(0), d['y'])[0, 1]:.3f}")
    print("\n  wrote data/derived/prospective.pkl, conditions_y1.pkl, table0")


if __name__ == "__main__":
    main()
