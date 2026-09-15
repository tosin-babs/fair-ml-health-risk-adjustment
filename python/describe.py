"""
Sample description: who is in the benchmark and what they spend.

Weighted estimates with Taylor-linearized standard errors for the stratified
cluster design (svy.py, shared across the programme's papers). Domain means
zero the weights outside the domain rather than dropping rows, so standard
errors reflect the random number of sampled PSUs in each domain.

Writes Table 1 and Table A6 (the CCSR categories that enter the features).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
import features
import svy

RACE = {1: "Hispanic", 2: "Non-Hispanic White", 3: "Non-Hispanic Black",
        4: "Non-Hispanic Asian", 5: "Non-Hispanic other or multiple"}
POV = {1: "Below 100% FPL", 2: "100-124% FPL", 3: "125-199% FPL",
       4: "200-399% FPL", 5: "400% FPL or more"}
INS = {1: "Any private coverage", 2: "Public coverage only", 3: "Uninsured all year"}


def main():
    d, c = features.load()
    des = svy.Design(d, "weight", "stratum", "cluster")
    rows = []

    def add(label, values, scale=1.0, subset=None):
        dd = des if subset is None else des.subset(subset)
        est, se = dd.mean(values)
        n = int(np.isfinite(np.asarray(values, float)).sum() if subset is None
                else (subset & np.isfinite(np.asarray(values, float))).sum())
        rows.append({"characteristic": label, "estimate": scale * est,
                     "se": scale * se, "n": n})

    rows.append({"characteristic": "Persons", "estimate": float(len(d)), "se": np.nan,
                 "n": len(d)})
    rows.append({"characteristic": "Weighted population, millions a year",
                 "estimate": d["weight"].sum() / 1e6, "se": np.nan, "n": len(d)})
    add("Age in year 1, mean", d["age"])
    add("Age 65 and over, %", (d["age"] >= 65).astype(float), 100)
    add("Female, %", d["female"].astype(float), 100)
    for k, lab in RACE.items():
        add(f"{lab}, %", (d["race"] == k).astype(float).where(d["race"].notna()), 100)
    for k, lab in POV.items():
        add(f"{lab}, %", (d["povcat"] == k).astype(float).where(d["povcat"].notna()), 100)
    for k, lab in INS.items():
        add(f"{lab}, %", (d["inscov"] == k).astype(float).where(d["inscov"].notna()), 100)
    add("Fair or poor perceived health, %",
        (d["health"] >= 4).astype(float).where(d["health"].notna()), 100)
    add("Needs help with ADL or IADL, %", d["any_function_help"].astype(float), 100)
    add("Any year-1 condition (CCSR category), %", (d["n_conditions"] > 0).astype(float), 100)
    add("Year-1 CCSR categories, mean", d["n_conditions"])
    add("Any mental health condition, %", d["any_mbd"].astype(float), 100)
    add("Year-1 spending, mean $", d["spend_y1"])
    add("Year-2 spending, mean $", d["y"])
    add("Year-2 spending of zero, %", (d["y"] == 0).astype(float), 100)
    for q, lab in ((0.5, "median"), (0.9, "90th percentile"), (0.99, "99th percentile")):
        rows.append({"characteristic": f"Year-2 spending, {lab} $",
                     "estimate": float(des.quantile(d["y"], q)[0]), "se": np.nan,
                     "n": len(d)})
    top5 = d["y"] >= des.quantile(d["y"], 0.95)[0]
    rows.append({"characteristic": "Share of year-2 spending by the top 5% of spenders, %",
                 "estimate": 100 * float(np.sum(d["weight"][top5] * d["y"][top5])
                                         / np.sum(d["weight"] * d["y"])),
                 "se": np.nan, "n": len(d)})
    t1 = pd.DataFrame(rows)
    t1.to_csv(config.TABLES / "table1_sample.csv", index=False)

    lab = pd.read_csv(config.DERIVED / "ccsr_labels.csv").set_index("ccsr")["description"]
    _, keep = features.ccsr_flags(d, c)
    valid = c[c["valid_category"]]
    prev = valid.groupby("ccsr")["uid"].nunique() / len(d)
    def describe(k):
        return " / ".join(lab.get(part, "") for part in k.split("+"))
    ta = pd.DataFrame({"ccsr": keep, "description": [describe(k) for k in keep],
                       "prevalence": [float(prev[k.split("+")[0]]) for k in keep]})
    ta.sort_values("prevalence", ascending=False).to_csv(
        config.TABLES / "tableA_ccsr_features.csv", index=False)

    print("=== Table 1: sample characteristics ===")
    with pd.option_context("display.width", 160):
        print(t1.round(2).to_string(index=False))
    print(f"\n  {len(keep)} CCSR categories enter the feature sets")
    print("wrote table 1 and table A6")


if __name__ == "__main__":
    main()
