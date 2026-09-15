"""
Tests for the guarantees the benchmark and the riskfair package depend on.

    ../.venv/bin/python -m pytest -q tests

Tests that need the MEPS microdata skip themselves when it is absent, so the
package tests run anywhere (including CI).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from riskfair import coding, fairness, metrics, surveycv  # noqa: E402
from riskfair.models import WLS  # noqa: E402

try:
    import config  # noqa: E402
    DATA = (config.DERIVED / "prospective.pkl").exists()
except Exception:  # pragma: no cover - pipeline absent
    config, DATA = None, False


# --------------------------------------------------------------- leakage ----
@pytest.mark.skipif(config is None, reason="pipeline config absent")
def test_year1_rounds_only():
    assert config.HEALTH_ROUND <= 2, "round 3 is interviewed in year 2"
    assert config.FUNCTION_ROUND <= 2, "round 3 is interviewed in year 2"


@pytest.mark.skipif(config is None, reason="pipeline config absent")
def test_conditions_come_from_year1_files():
    known_year = {"h207.dta": 2018, "h214.dta": 2019, "h222.dta": 2020,
                  "h231.dta": 2021, "h241.dta": 2022}
    for panel, spec in config.PANELS.items():
        assert known_year[spec["conditions"]] == spec["year1"], panel


@pytest.mark.skipif(not DATA, reason="derived MEPS data not built")
def test_no_outcome_or_race_in_features():
    import features
    for fs in config.FEATURE_SETS:
        X, y, *_ = features.build(fs)
        cols = " ".join(X.columns).lower()
        for banned in ("y2", "totexp", "race", "hisp", "black", "asian"):
            assert banned not in cols, (fs, banned)
        corr = np.array([abs(np.corrcoef(X[c], y)[0, 1]) if X[c].std() > 0 else 0
                         for c in X.columns])
        assert corr.max() < 0.95, "a feature is nearly the outcome"


@pytest.mark.skipif(not DATA, reason="derived MEPS data not built")
def test_placeholder_codes_are_not_categories():
    import features
    X, *_ = features.build("F2")
    assert not any(c.endswith("000") for c in X.columns if c.startswith("ccsr_"))


# ---------------------------------------------------------------- folds ----
def _toy_design(n_strata=12, psu_per=3, per_psu=20, seed=1):
    rng = np.random.default_rng(seed)
    st, cl = [], []
    for s in range(n_strata):
        for p in range(psu_per):
            k = per_psu + int(rng.integers(0, 5))
            st += [f"s{s}"] * k
            cl += [f"s{s}_p{p}"] * k
    return np.array(cl), np.array(st)


def test_folds_are_disjoint_by_psu():
    cl, st = _toy_design()
    folds = surveycv.survey_folds(cl, st, n_splits=3, repeats=2, seed=7)
    assert surveycv.check_disjoint(folds, cl)


def test_every_person_tested_once_per_repeat():
    cl, st = _toy_design()
    folds = surveycv.survey_folds(cl, st, n_splits=3, repeats=2, seed=7)
    for rep in (0, 1):
        tested = np.concatenate([te for r, _, _, te in folds if r == rep])
        assert len(tested) == len(cl) and len(set(tested)) == len(cl)


def test_inner_split_keeps_clusters_whole():
    cl, _ = _toy_design()
    tr, va = surveycv.inner_split(cl, frac=0.2, seed=3)
    assert not set(cl[tr]) & set(cl[va])


# -------------------------------------------------------------- metrics ----
def test_metric_identities():
    rng = np.random.default_rng(0)
    y = rng.gamma(0.5, 5000, 2000)
    w = rng.uniform(0.5, 2.0, 2000)
    assert metrics.r2(y, y, w) == pytest.approx(1.0)
    assert metrics.cpm(y, y, w) == pytest.approx(1.0)
    ybar = np.full_like(y, np.average(y, weights=w))
    assert metrics.r2(y, ybar, w) == pytest.approx(0.0, abs=1e-12)
    assert metrics.accuracy(y, ybar, w)["pr_overall"] == pytest.approx(1.0)


def test_net_compensation_sign():
    y = np.array([100.0, 100.0, 300.0, 300.0])
    p = np.array([150.0, 150.0, 200.0, 200.0])
    g = metrics.group_fairness(y, p, np.ones(4), {"high": np.array([0, 0, 1, 1], bool)})
    assert g["high"]["nc"] == pytest.approx(-100.0)
    assert g["high"]["pr"] == pytest.approx(200 / 300)


# ------------------------------------------------------------- fairness ----
def test_constrained_wls_zero_net_compensation_in_sample():
    rng = np.random.default_rng(3)
    n = 3000
    X = rng.normal(size=(n, 5))
    grp = rng.random(n) < 0.2
    y = 1000 + X @ np.array([200, -50, 80, 0, 30]) + 900 * grp + rng.normal(0, 300, n)
    w = rng.uniform(0.5, 2, n)
    m = fairness.FairWLS(constrained=True).fit(X, y, w, [grp])
    nc = metrics.group_fairness(y, m.predict(X), w, {"g": grp})["g"]["nc"]
    assert abs(nc) < 1e-6


def test_penalty_reduces_net_compensation_monotonically():
    rng = np.random.default_rng(5)
    n = 4000
    X = rng.normal(size=(n, 4))
    grp = rng.random(n) < 0.15
    y = 500 + X @ np.array([100, 50, -20, 10]) + 800 * grp + rng.normal(0, 200, n)
    w = np.ones(n)
    ncs = [abs(metrics.group_fairness(
        y, fairness.FairWLS(lam=lam).fit(X, y, w, [grp]).predict(X), w,
        {"g": grp})["g"]["nc"]) for lam in (0.0, 1.0, 10.0, 100.0)]
    assert all(a >= b - 1e-9 for a, b in zip(ncs, ncs[1:]))


def test_penalized_lambda_zero_equals_wls():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(500, 3))
    y = X @ np.array([1.0, 2.0, 3.0]) + rng.normal(size=500)
    w = np.ones(500)
    a = fairness.FairWLS(lam=0.0).fit(X, y, w, [X[:, 0] > 0]).predict(X)
    b = WLS().fit(X, y, w).predict(X)
    assert np.allclose(a, b, atol=1e-6)


def test_stacked_constrained_gbm_corrects_undercompensated_group():
    pytest.importorskip("lightgbm")
    rng = np.random.default_rng(8)
    n = 8000
    X = rng.normal(size=(n, 6))
    grp = (0.6 * X[:, 0] + rng.normal(0, 1, n)) > 1.0      # only partly observable
    y = np.exp(7 + 0.3 * X[:, 1] + 1.0 * grp + rng.normal(0, 0.6, n))
    w = rng.uniform(0.5, 2.0, n)
    cl = np.arange(n) // 8
    tr = cl % 4 != 0
    te = ~tr
    from riskfair.models import GBM
    plain = GBM("tweedie", seed=1).fit(X[tr], y[tr], w[tr], clusters=cl[tr])
    fair = fairness.StackedFairGBM(constrained=True, k=2, seed=1).fit(
        X[tr], y[tr], w[tr], [grp[tr]], clusters=cl[tr])
    nc = lambda p: metrics.group_fairness(y[te], p, w[te], {"g": grp[te]})["g"]["nc"]
    assert nc(plain.predict(X[te])) < 0, "setup: plain model should under-pay"
    assert abs(nc(fair.predict(X[te]))) < 0.5 * abs(nc(plain.predict(X[te])))


# --------------------------------------------------------------- coding ----
def test_perturb_adds_only_absent_flags_and_updates_counts():
    cols = ["ccsr_CIR019", "ccsr_END005", "ccsr_MBD002", "n_conditions", "n_body_systems"]
    X = np.array([[1, 0, 0, 1, 1], [0, 0, 0, 0, 0]], dtype=float)
    rng = np.random.default_rng(0)
    pool = cols[:3]
    Xp, added = coding.perturb(X, cols, [0, 1], pool, {c: 1.0 for c in pool}, 2, rng,
                               count_col="n_conditions", system_col="n_body_systems")
    assert added == 4
    assert (Xp[:, :3].sum(axis=1) == [3, 2]).all()
    assert (Xp[:, 3] == [3, 2]).all()
    assert (Xp[:, 3] >= Xp[:, 4]).all()


def test_linear_model_rise_equals_coefficient():
    rng = np.random.default_rng(1)
    cols = ["ccsr_CIR019", "n_conditions", "n_body_systems"]
    X = np.column_stack([np.zeros(1000), np.zeros(1000), np.zeros(1000)])
    y = rng.gamma(1, 1000, 1000)
    w = np.ones(1000)
    predict = lambda A: 100 + 250 * A[:, 0]
    r = coding.stress(predict, X, w, cols, ["ccsr_CIR019"], {"ccsr_CIR019": 1.0},
                      share=1.0, k=1, seed=0)
    assert r["dollars_per_code"] == pytest.approx(250.0)


def test_flag_gains_counts_new_body_systems_only_once():
    cols = ["ccsr_CIR019", "ccsr_CIR017", "n_conditions", "n_body_systems"]
    X = np.array([[0, 0, 0, 0], [0, 1, 1, 1]], dtype=float)
    w = np.ones(2)
    predict = lambda A: 100 + 300 * A[:, 0] + 50 * A[:, 1] + 20 * A[:, 2] + 10 * A[:, 3]
    g = coding.flag_gains(predict, X, w, cols, ["ccsr_CIR019"],
                          count_col="n_conditions", system_col="n_body_systems")
    # row 0 gains a new body system (+330); row 1 already has a circulatory code (+320)
    assert g["ccsr_CIR019"] == pytest.approx(325.0)


# ------------------------------------------------------------- bootstrap ----
def test_rao_wu_draws_one_fewer_psu_and_rescales_weights():
    # stratum 0 has 2 PSUs of 3 rows each; stratum 1 has 4 PSUs of 2 rows each
    clusters = np.array([0] * 3 + [1] * 3 + [2, 2, 3, 3, 4, 4, 5, 5])
    strata = np.array([0] * 6 + [1] * 8)
    rng = np.random.default_rng(0)
    for _ in range(20):
        idx, mult = metrics.bootstrap_index(clusters, strata, rng)
        s0, s1 = strata[idx] == 0, strata[idx] == 1
        assert s0.sum() == 3 and s1.sum() == 6          # n_h - 1 PSUs drawn in each stratum
        assert np.allclose(mult[s0], 2.0) and np.allclose(mult[s1], 4 / 3)
        # rescaled weight total per stratum equals the stratum's full weight
        assert mult[s0].sum() == pytest.approx(6.0) and mult[s1].sum() == pytest.approx(8.0)
    idx, mult = metrics.bootstrap_index(clusters, strata, rng, rao_wu=False)
    assert len(idx) == len(clusters) and np.allclose(mult, 1.0)
