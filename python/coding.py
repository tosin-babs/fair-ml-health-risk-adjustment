"""
RQ4: how much does each model's predicted spending rise when coding intensifies?

For a random share of held-out people, one or two condition flags they did not
have are switched on, drawn either in proportion to each category's prevalence
among people with any condition or only from chronic, payment-relevant
categories (riskfair.coding.CHRONIC_CCSR). The condition and body-system
counts are updated to match; use and spending are left as they were, so any
rise in a prediction is a response to coding alone.

Models are fitted on the training PSUs of each fold of the first repeat and
stressed on that fold's held-out PSUs. Results are pooled across folds as
weighted totals.

Writes Table 5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import common
import config
import features
from common import coding, fairness
from riskfair.models import GBM, WLS, NeuralNet, TweedieGLM, TwoPart

MODELS = {
    "WLS": lambda: WLS(),
    "Tweedie GLM": lambda: TweedieGLM(),
    "Two-part": lambda: TwoPart(),
    "LightGBM (Tweedie)": lambda: GBM("tweedie", seed=config.SEED),
    "LightGBM (squared error)": lambda: GBM("regression", seed=config.SEED),
    "CANN": lambda: NeuralNet(cann=True, seed=config.SEED),
    "Constrained WLS": None,
}


def main():
    data = features.load()
    X, y, w, cl, st, attrs = features.build(config.PRIMARY_FEATURE_SET, data=data)
    cols = list(X.columns)
    Xn = X.to_numpy(np.float64)
    masks = features.group_masks(attrs)
    target = [g for g in config.FAIR_TARGET_GROUPS if g in masks]
    folds = common.folds(cl, st, repeats=1)

    ccsr_cols = [c for c in cols if c.startswith("ccsr_")]
    has_any = Xn[:, cols.index("n_conditions")] > 0
    prev = {c: float(Xn[has_any, cols.index(c)].mean()) for c in ccsr_cols}
    chronic = [c for c in ccsr_cols if c[5:] in coding.CHRONIC_CCSR]
    pools = {"prevalence": (ccsr_cols, prev),
             "chronic": (chronic, {c: prev[c] for c in chronic})}
    print(f"chronic pool: {len(chronic)} of {len(coding.CHRONIC_CCSR)} listed "
          f"categories are features: {', '.join(c[5:] for c in chronic)}")

    rows = []
    for mname, make in MODELS.items():
        acc = {}
        for _, fold, tr, te in folds:
            if make is None:
                m = fairness.FairWLS(constrained=True).fit(
                    Xn[tr], y[tr], w[tr], [masks[g][tr] for g in target])
                predict = m.predict
            else:
                m = make().fit(Xn[tr].astype(np.float32), y[tr], w[tr], clusters=cl[tr])
                predict = lambda A, m=m: m.predict(A.astype(np.float32))
            for pool in config.CODING_POOLS:
                pc, pw = pools[pool]
                for share in config.CODING_SHARES:
                    for k in config.CODING_ADDED:
                        r = coding.stress(predict, Xn[te], w[te], cols, pc, pw, share, k,
                                          seed=config.SEED + 100 * fold + 10 * k
                                          + int(share * 1000),
                                          count_col="n_conditions",
                                          system_col="n_body_systems")
                        a = acc.setdefault((pool, share, k), np.zeros(4))
                        a += [r["base_total"], r["rise_total"],
                              r["dollars_per_code"] * r["affected_weight"]
                              * r["codes_added"] / max(1, r["codes_added"])
                              if np.isfinite(r["dollars_per_code"]) else 0.0,
                              r["affected_weight"]]
        for (pool, share, k), (base, rise, per_code_w, aff_w) in acc.items():
            rows.append({"model": mname, "pool": pool, "share_affected": share,
                         "codes_added": k,
                         "pct_rise_total_predicted": 100 * rise / base,
                         "dollars_per_added_code": per_code_w / aff_w if aff_w else np.nan})
        ref = [x for x in rows if x["model"] == mname and x["pool"] == "chronic"
               and x["share_affected"] == 0.10 and x["codes_added"] == 1][0]
        print(f"  {mname:<26} 10% gain one chronic code: total predicted "
              f"{ref['pct_rise_total_predicted']:+.2f}%, "
              f"${ref['dollars_per_added_code']:,.0f} per code")
    pd.DataFrame(rows).to_csv(config.TABLES / "table5_coding_sensitivity.csv", index=False)
    print("\nwrote table 5")


if __name__ == "__main__":
    main()
