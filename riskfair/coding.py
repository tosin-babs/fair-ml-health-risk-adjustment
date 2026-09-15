"""
Coding-intensity stress test.

`perturb` switches on condition flags a person did not have and updates any
count features that summarise those flags. Everything else about the person,
including prior use and spending, is left as it was, so any rise in a model's
prediction is a response to coding alone.

`stress` runs the perturbation for a fitted model and returns the percentage
rise in total predicted spending and the dollar rise per added code.
"""

from __future__ import annotations

import numpy as np

# Chronic categories analogous to payment-model hierarchical condition groups,
# as AHRQ CCSR codes. Labels checked against the CCSR v2026.1 reference file.
CHRONIC_CCSR = {
    "END003": "Diabetes mellitus with complication",
    "END005": "Diabetes mellitus, Type 2",
    "CIR008": "Hypertension with complications and secondary hypertension",
    "CIR011": "Coronary atherosclerosis and other heart disease",
    "CIR017": "Cardiac dysrhythmias",
    "CIR019": "Heart failure",
    "CIR026": "Peripheral and visceral vascular disease",
    "RSP008": "Chronic obstructive pulmonary disease and bronchiectasis",
    "GEN003": "Chronic kidney disease",
    "MBD002": "Depressive disorders",
    "NVS011": "Neurocognitive disorders",
    "END009": "Obesity",
}


def perturb(X, columns, rows, pool, pool_weights, k, rng, count_col=None,
            system_col=None, prefix="ccsr_"):
    """Return a copy of X with k absent pool flags switched on for each row.

    X            2-D array, one row per person
    columns      column names of X
    rows         row indices to perturb
    pool         candidate flag columns
    pool_weights mapping column -> non-negative sampling weight
    count_col    optional column counting distinct flags (incremented)
    system_col   optional column counting distinct body systems, taken as the
                 three characters after `prefix` (incremented when new)
    """
    Xp = X.copy()
    idx = {c: columns.index(c) for c in pool}
    flag_cols = [i for i, c in enumerate(columns) if c.startswith(prefix)]
    j_cnt = columns.index(count_col) if count_col else None
    j_sys = columns.index(system_col) if system_col else None
    added = 0
    for r in rows:
        absent = [c for c in pool if Xp[r, idx[c]] == 0]
        if not absent:
            continue
        p = np.array([pool_weights[c] for c in absent], dtype=float)
        if p.sum() <= 0:
            continue
        take = rng.choice(len(absent), size=min(k, len(absent)), replace=False,
                          p=p / p.sum())
        systems = {columns[i][len(prefix):len(prefix) + 3]
                   for i in flag_cols if Xp[r, i] == 1}
        for t in take:
            c = absent[t]
            Xp[r, idx[c]] = 1
            if j_cnt is not None:
                Xp[r, j_cnt] += 1
            sys_ = c[len(prefix):len(prefix) + 3]
            if j_sys is not None and sys_ not in systems:
                Xp[r, j_sys] += 1
                systems.add(sys_)
            added += 1
    return Xp, added


def flag_gains(predict, X, w, columns, pool, count_col=None, system_col=None,
               prefix="ccsr_"):
    """Weighted mean rise in a model's prediction when each flag in `pool` is
    switched on for the rows that lack it, counts updated as in `perturb`.

    This is what a coder who can query the model learns: which codes pay most.
    Returns a dict column -> dollars per added code (NaN if every row has it).
    """
    p0 = predict(X)
    j_cnt = columns.index(count_col) if count_col else None
    j_sys = columns.index(system_col) if system_col else None
    flag_cols = [i for i, c in enumerate(columns) if c.startswith(prefix)]
    out = {}
    for c in pool:
        j = columns.index(c)
        rows = np.flatnonzero(X[:, j] == 0)
        if not len(rows):
            out[c] = np.nan
            continue
        Xp = X[rows].copy()
        Xp[:, j] = 1
        if j_cnt is not None:
            Xp[:, j_cnt] += 1
        if j_sys is not None:
            same = [i for i in flag_cols
                    if columns[i][len(prefix):len(prefix) + 3] == c[len(prefix):len(prefix) + 3]]
            new_system = X[np.ix_(rows, same)].sum(axis=1) == 0
            Xp[new_system, j_sys] += 1
        out[c] = float(np.average(predict(Xp) - p0[rows], weights=w[rows]))
    return out


def stress(predict, X, w, columns, pool, pool_weights, share, k, seed,
           count_col=None, system_col=None):
    """Percentage rise in total prediction and dollars per added code."""
    rng = np.random.default_rng(seed)
    chosen = rng.random(len(X)) < share
    p0 = predict(X)
    Xp, added = perturb(X, columns, np.flatnonzero(chosen), pool, pool_weights,
                        k, rng, count_col, system_col)
    p1 = predict(Xp)
    rise_all = float(np.sum(w * (p1 - p0)))
    per_code = (float(np.sum(w[chosen] * (p1 - p0)[chosen]))
                / (float(np.sum(w[chosen])) * added / max(int(chosen.sum()), 1))
                if added else np.nan)
    return {"pct_rise_total": 100 * rise_all / float(np.sum(w * p0)),
            "dollars_per_code": per_code, "codes_added": added,
            "base_total": float(np.sum(w * p0)), "rise_total": rise_all,
            "affected_weight": float(np.sum(w[chosen]))}
