"""
Feature matrices F1 to F4, and the attributes used only for evaluation.

  F1  age band by sex, region
  F2  F1 + year-1 CCSR condition flags, condition and body-system counts
  F3  F2 + year-1 use and spending
  F4  F3 + income, insurance, perceived health and functional help

Race and ethnicity never enter any feature set. They are returned in the
attribute frame, which models never see.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


def load():
    d = pd.read_pickle(config.DERIVED / "prospective.pkl").reset_index(drop=True)
    c = pd.read_pickle(config.DERIVED / "conditions_y1.pkl")
    return d, c


def _age_band(age):
    labels = [f"{lo}-{hi}" if hi < 200 else f"{lo}+" for lo, hi in config.AGE_BANDS]
    bins = [lo for lo, _ in config.AGE_BANDS] + [10_000]
    return pd.cut(age, bins=bins, right=False, labels=labels)


def _dummies(s, prefix):
    s = s.astype("object").where(s.notna(), "missing").astype(str)
    return pd.get_dummies(s, prefix=prefix, dtype=np.float32)


def ccsr_flags(d, c, min_prevalence=None):
    """Binary year-1 CCSR flags for categories above the prevalence floor."""
    floor = config.CCSR_MIN_PREVALENCE if min_prevalence is None else min_prevalence
    c = c[c["valid_category"]] if "valid_category" in c else c
    prev = c.groupby("ccsr")["uid"].nunique() / len(d)
    keep = sorted(prev[prev >= floor].index)
    sub = c[c["ccsr"].isin(keep)]
    wide = (pd.crosstab(sub["uid"], sub["ccsr"]).clip(upper=1)
              .reindex(index=d["uid"], columns=keep, fill_value=0))
    wide.index = d.index
    # MEPS releases three-digit ICD-10 codes, and CCSR assigns some of them
    # to two categories at once (E11 to END002 and END005), which makes two
    # flags identical for every person. One is kept, under a combined name,
    # so that no coefficient is split arbitrarily and a coding perturbation
    # cannot switch on one member of an inseparable pair.
    merged, drop = {}, set()
    cols = list(wide.columns)
    for i, a in enumerate(cols):
        if a in drop:
            continue
        for b in cols[i + 1:]:
            if b not in drop and (wide[a].to_numpy() == wide[b].to_numpy()).all():
                drop.add(b)
                merged.setdefault(a, []).append(b)
    wide = wide.drop(columns=sorted(drop))
    wide.columns = [f"ccsr_{k}" + ("+" + "+".join(merged[k]) if k in merged else "") for k in wide.columns]
    keep = [c[5:] for c in wide.columns]
    return wide.astype(np.float32), keep


def build(feature_set=None, min_prevalence=None, data=None):
    """Return X, y, w, clusters, strata, attrs for one feature set."""
    fs = feature_set or config.PRIMARY_FEATURE_SET
    d, c = data if data is not None else load()

    band = _age_band(d["age"])
    f1 = _dummies(band, "age")
    f1["female"] = d["female"].astype(np.float32)
    for col in list(f1.columns):
        if col.startswith("age_"):
            f1[f"{col}_x_female"] = f1[col] * f1["female"]
    f1 = f1.join(_dummies(d["region"], "region"))
    blocks = [f1]

    if fs in ("F2", "F3", "F4"):
        flags, _ = ccsr_flags(d, c, min_prevalence)
        body = c.assign(system=c["ccsr"].str[:3]).groupby("uid")["system"].nunique()
        counts = pd.DataFrame({
            "n_conditions": d["n_conditions"].astype(np.float32),
            "n_body_systems": d["uid"].map(body).fillna(0).astype(np.float32),
        })
        blocks += [flags, counts]

    if fs in ("F3", "F4"):
        use = pd.DataFrame({
            "log_spend_y1": np.log1p(d["spend_y1"].fillna(0)),
            "any_spend_y1": (d["spend_y1"].fillna(0) > 0).astype(float),
            "log_office_visits_y1": np.log1p(d["office_visits_y1"].fillna(0)),
            "log_er_visits_y1": np.log1p(d["er_visits_y1"].fillna(0)),
            "log_inpatient_y1": np.log1p(d["inpatient_y1"].fillna(0)),
            "log_rx_fills_y1": np.log1p(d["rx_fills_y1"].fillna(0)),
        }).astype(np.float32)
        blocks.append(use)

    if fs == "F4":
        social = pd.concat([
            _dummies(d["povcat"], "povcat"),
            _dummies(d["inscov"], "inscov"),
            _dummies(d["health"], "health"),
            _dummies(d["mental_health"], "mental_health"),
            d[["adl_help", "iadl_help"]].astype(np.float32),
        ], axis=1)
        blocks.append(social)

    X = pd.concat(blocks, axis=1).astype(np.float32)
    assert not any(k in X.columns for k in ("race", "race_1", "RACETHX")), \
        "race entered the feature matrix"

    attrs = d[["uid", "panel", "age", "female", "race", "povcat", "inscov",
               "any_mbd", "any_function_help", "n_conditions"]].copy()
    attrs["died"] = d["died_y2"].astype(int)
    return (X, d["y"].to_numpy(float), d["weight"].to_numpy(float),
            d["cluster"].to_numpy(), d["stratum"].to_numpy(), attrs)


def group_masks(attrs):
    """Boolean membership for each evaluated group, with a minimum size."""
    ops = {"le": np.less_equal, "ge": np.greater_equal, "eq": np.equal}
    out = {}
    for name, (col, op, val) in config.FAIRNESS_GROUPS.items():
        m = ops[op](attrs[col].to_numpy(float), val) & attrs[col].notna().to_numpy()
        if m.sum() >= config.MIN_GROUP_N:
            out[name] = m
    return out


if __name__ == "__main__":
    for fs in config.FEATURE_SETS:
        X, y, w, cl, st, attrs = build(fs)
        print(f"{fs}: {X.shape[1]:>4} features, {X.shape[0]:,} persons, "
              f"{len(np.unique(cl)):,} clusters")
    for name, m in group_masks(attrs).items():
        print(f"  {name:<26} n={m.sum():>6,}  weighted share "
              f"{w[m].sum() / w.sum():6.1%}")
