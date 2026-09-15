"""
RQ3: what drives the predictions, and whether the drivers are stable.

TreeSHAP values for the squared-error LightGBM are computed on each held-out
fold of the first repeat, from the model fitted on that fold's training PSUs.
The squared-error model is explained rather than the Tweedie one because its
SHAP values are in dollars; the Tweedie model's are in log units. The two
models have the same out-of-sample accuracy (Table 2). Global
importance is the weighted mean absolute SHAP value. Stability is the Spearman
correlation of importance ranks between folds. For the neural models, which
TreeSHAP does not cover, permutation importance on the held-out fold is used
instead (the loss in weighted R-squared when a feature is shuffled). GLM
coefficients give the actuarial comparison.

Two sense checks follow the actuarial logic the spec asks for: predicted
spending should rise with the number of year-1 conditions and with year-1
spending, so the SHAP contribution of each should be positively rank-correlated
with the feature value.

Writes Tables 4, 4b and 4c.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats

import common
import config
import features
from common import metrics
from riskfair.models import GBM, NeuralNet

SAMPLE = int(os.environ.get("P6_SHAP_SAMPLE", 4000))


def labels():
    lab = pd.read_csv(config.DERIVED / "ccsr_labels.csv").set_index("ccsr")["description"]

    def name(col):
        if col.startswith("ccsr_"):
            code = col[5:]
            desc = " / ".join(lab.get(part, "") for part in code.split("+"))
            return f"{code} {desc}".strip()
        return col
    return name


def main():
    import shap
    data = features.load()
    X, y, w, cl, st, attrs = features.build(config.PRIMARY_FEATURE_SET, data=data)
    cols = list(X.columns)
    Xn = X.to_numpy(np.float32)
    folds = common.folds(cl, st, repeats=1)
    rng = np.random.default_rng(config.SEED)
    name = labels()

    imp, checks, perm = [], [], []
    for _, fold, tr, te in folds:
        m = GBM("regression").fit(Xn[tr], y[tr], w[tr], clusters=cl[tr])
        s = rng.choice(te, size=min(SAMPLE, len(te)), replace=False)
        sv = shap.TreeExplainer(m.m_).shap_values(Xn[s])
        ws = w[s] / w[s].sum()
        imp.append(pd.Series(np.abs(sv).T @ ws, index=cols, name=fold))
        for feat in ("n_conditions", "log_spend_y1", "n_body_systems"):
            j = cols.index(feat)
            rho = stats.spearmanr(Xn[s, j], sv[:, j]).statistic
            checks.append({"fold": fold, "feature": feat, "spearman_value_vs_shap": rho})

        nn = NeuralNet(cann=True).fit(Xn[tr], y[tr], w[tr], clusters=cl[tr])
        base = metrics.r2(y[te], nn.predict(Xn[te]), w[te])
        top = imp[-1].sort_values(ascending=False).index[:20]
        for feat in top:
            j = cols.index(feat)
            Xp = Xn[te].copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            perm.append({"fold": fold, "feature": feat,
                         "r2_loss_cann": base - metrics.r2(y[te], nn.predict(Xp), w[te])})
        print(f"  fold {fold}: SHAP on {len(s):,} held-out persons; "
              f"CANN held-out R2 {base:.3f}")

    I = pd.concat(imp, axis=1)
    t4 = pd.DataFrame({"feature": I.index, "label": [name(c) for c in I.index],
                       "mean_abs_shap": I.mean(axis=1).values,
                       "sd_across_folds": I.std(axis=1).values})
    t4 = t4.sort_values("mean_abs_shap", ascending=False)
    t4["rank"] = np.arange(1, len(t4) + 1)
    P = pd.DataFrame(perm).groupby("feature")["r2_loss_cann"].mean()
    t4["cann_permutation_r2_loss"] = t4["feature"].map(P)
    t4.to_csv(config.TABLES / "table4_importance.csv", index=False)

    ranks = I.rank(ascending=False)
    rho = []
    for a in ranks.columns:
        for b in ranks.columns:
            if a < b:
                top = I[[a, b]].mean(axis=1).sort_values(ascending=False).index[:30]
                rho.append({"fold_a": a, "fold_b": b,
                            "spearman_all": stats.spearmanr(ranks[a], ranks[b]).statistic,
                            "spearman_top30": stats.spearmanr(ranks.loc[top, a],
                                                              ranks.loc[top, b]).statistic})
    t4b = pd.DataFrame(rho)
    t4b.to_csv(config.TABLES / "table4b_stability.csv", index=False)
    t4c = pd.DataFrame(checks)
    t4c.to_csv(config.TABLES / "table4c_sense_checks.csv", index=False)

    print("\n=== Top 15 drivers, squared-error LightGBM ($) ===")
    for _, r in t4.head(15).iterrows():
        print(f"  {int(r['rank']):>2} {r['label'][:52]:<52} {r['mean_abs_shap']:>8,.0f} "
              f"(sd {r['sd_across_folds']:,.0f})")
    print(f"\n  rank stability across folds: Spearman {t4b['spearman_all'].mean():.3f} "
          f"(top 30: {t4b['spearman_top30'].mean():.3f})")
    print(t4c.groupby("feature")["spearman_value_vs_shap"].mean().round(3).to_string())
    print("\nwrote tables 4, 4b, 4c")


if __name__ == "__main__":
    main()
