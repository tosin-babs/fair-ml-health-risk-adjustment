"""
Tests for riskfair.robust and riskfair.adversary on synthetic data.

    ../.venv/bin/python -m pytest -q tests/test_adversary.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from riskfair import adversary, robust  # noqa: E402
from riskfair.models import PaymentWLS  # noqa: E402

COLS = ["age_0", "age_1", "female", "age_0_x_female", "age_1_x_female",
        "ccsr_END005", "ccsr_END003", "ccsr_CIR019", "n_conditions", "n_body_systems"]
POOL = ["ccsr_END005", "ccsr_END003", "ccsr_CIR019"]


def synthetic(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    old = rng.random(n) < 0.5
    fem = rng.random(n) < 0.5
    flags = rng.random((n, 3)) < [0.2, 0.05, 0.08]
    X = np.zeros((n, len(COLS)))
    X[:, 0], X[:, 1], X[:, 2] = ~old, old, fem
    X[:, 3], X[:, 4] = (~old) & fem, old & fem
    X[:, 5:8] = flags
    X[:, 8] = flags.sum(axis=1)
    X[:, 9] = (flags[:, :2].any(axis=1)).astype(float) + flags[:, 2]
    y = (2000 + 3000 * old + 4000 * flags[:, 0] + 9000 * flags[:, 1] + 12000 * flags[:, 2]
         + rng.gamma(2.0, 1500, n))
    w = rng.uniform(0.5, 2.0, n)
    return X, y, w


class _Const:
    def __init__(self, v):
        self.v = v

    def predict(self, X):
        return np.full(len(X), self.v)


def test_gain_matrix_matches_direct_perturbation():
    X, y, w = synthetic()
    f = PaymentWLS().set_columns(COLS).fit(X, y, w)
    G = adversary.gain_matrix(f.predict, X, COLS, POOL, count_col="n_conditions",
                              system_col="n_body_systems")
    i = int(np.flatnonzero(X[:, 5] == 0)[0])
    Xp = X[[i]].copy()
    Xp[0, 5] = 1
    Xp[0, 8] += 1
    if X[i, 6] == 0:          # END005 and END003 share a body system
        Xp[0, 9] += 1
    assert np.isclose(G[i, 0], f.predict(Xp)[0] - f.predict(X[[i]])[0])
    assert np.isnan(G[X[:, 5] == 1, 0]).all()


def test_no_codes_outside_plausibility_set():
    X, y, w = synthetic()
    f = PaymentWLS().set_columns(COLS).fit(X, y, w)
    P = adversary.plausible_by_system(X, COLS, POOL)
    plan = adversary.Plan(cost_per_code=0.0, max_codes=3, reach=1.0, tilt=0.0, pool=POOL,
                          plausible=P, count_col="n_conditions", system_col="n_body_systems")
    added, _ = plan.code(f.predict, X, w, COLS)
    assert not (added & ~P).any()
    assert not (added & (X[:, 5:8] > 0)).any()


def test_extraction_zero_when_codes_too_dear_and_no_tilt():
    X, y, w = synthetic()
    f = PaymentWLS().set_columns(COLS).fit(X, y, w)
    plan = adversary.Plan(cost_model=_Const(0.0), cost_per_code=1e9, max_codes=2, reach=1.0,
                          tilt=0.0, pool=POOL, count_col="n_conditions", system_col="n_body_systems")
    added, s, Xc = plan.respond(f.predict, X, w, COLS)
    e = adversary.extraction(f.predict, X, Xc, w, s, y, added, plan.cost_per_code)
    assert added.sum() == 0 and np.allclose(s, 1.0)
    assert e["coding"] == 0 and e["selection"] == 0 and e["extraction"] == 0


def test_selection_keeps_size_and_favors_margin():
    X, y, w = synthetic()
    f = PaymentWLS().set_columns(COLS).fit(X, y, w)
    pay = f.predict(X)
    for rule in ("threshold", "smooth"):
        plan = adversary.Plan(cost_model=_Const(pay.mean()), tilt=0.3, rule=rule)
        s = plan.select(pay, w, np.full(len(X), pay.mean()))
        assert np.isclose(np.sum(w * s), w.sum())
        assert np.average(pay, weights=w * s) > np.average(pay, weights=w)


def test_reach_limits_reviewed_weight():
    X, y, w = synthetic()
    f = PaymentWLS().set_columns(COLS).fit(X, y, w)
    plan = adversary.Plan(cost_per_code=0.0, max_codes=1, reach=0.1, tilt=0.0, pool=POOL,
                          count_col="n_conditions", system_col="n_body_systems")
    added, _ = plan.code(f.predict, X, w, COLS)
    reviewed = added.any(axis=1)
    assert 0 < w[reviewed].sum() / w.sum() <= 0.1 + w.max() / w.sum()


def test_capped_coefficients_respect_caps():
    X, y, w = synthetic()
    caps = {"ccsr_END005": 1000.0, "ccsr_CIR019": 500.0}
    f = robust.CappedPaymentWLS(caps).set_columns(COLS).fit(X, y, w)
    c = f.coefficients()
    assert c["ccsr_END005"] <= 1000.0 + 1e-6 and c["ccsr_CIR019"] <= 500.0 + 1e-6
    assert c["ccsr_END003"] > 1000.0


def test_penalty_shrinks_only_penalized_flags():
    X, y, w = synthetic()
    base = PaymentWLS().set_columns(COLS).fit(X, y, w).coefficients()
    # Any added code also raises the condition counts, so they are penalized
    # with the flag: left free, they absorb what the flag loses.
    pw = {"ccsr_CIR019": 1.0, "n_conditions": 1.0, "n_body_systems": 1.0}
    pen = robust.PenalizedPaymentWLS(lam=50.0, pool_weights=pw).set_columns(COLS).fit(X, y, w).coefficients()
    assert pen["ccsr_CIR019"] < 0.1 * base["ccsr_CIR019"]
    assert pen["n_conditions"] < 0.1 * base["n_conditions"]
    # the unpenalized flags pick up what the count column was carrying for
    # them and land near their planted values (4,000 and 9,000)
    assert 3000 < pen["ccsr_END005"] < 5000 and 8000 < pen["ccsr_END003"] < 10000


def test_penalty_on_flags_alone_leaks_into_counts():
    X, y, w = synthetic()
    base = PaymentWLS().set_columns(COLS).fit(X, y, w).coefficients()
    pen = robust.PenalizedPaymentWLS(lam=50.0, pool_weights={"ccsr_CIR019": 1.0}).set_columns(COLS).fit(X, y, w).coefficients()
    assert pen["n_conditions"] > base["n_conditions"]


def test_cvar_is_worst_subgroup_mean():
    rng = np.random.default_rng(1)
    r = rng.normal(size=1000)
    w = np.ones(1000)
    c, top = robust._cvar(r, w, 0.1)
    assert len(top) == 100 and np.isclose(c, np.sort(r)[-100:].mean())


def test_dro_reduces_worst_expected_mispricing():
    X, y, w = synthetic()
    rng = np.random.default_rng(3)
    # a reference that knows an interaction the payment form cannot express
    ref = y - rng.gamma(2.0, 1500, len(y)) + 6000 * (X[:, 1] * X[:, 5])
    base = PaymentWLS().set_columns(COLS).fit(X, y, w)
    dro = robust.DROPaymentWLS(lam=1.0, alpha=0.1, reference=ref).set_columns(COLS).fit(X, y, w)
    def worst(m):
        e = ref - m.predict(X)
        return max(robust._cvar(e, w, 0.1)[0], robust._cvar(-e, w, 0.1)[0])
    assert worst(dro) < worst(base)
    assert dro.converged_
    # the fit is still anchored on y: accuracy does not collapse
    r2 = lambda m: 1 - np.sum(w * (y - m.predict(X)) ** 2) / np.sum(w * (y - np.average(y, weights=w)) ** 2)  # noqa: E731
    assert r2(dro) > 0.6 * r2(base)


def test_incremental_costs_recover_planted_effects():
    X, y, w = synthetic(n=20000)
    ic = robust.incremental_costs(X, y, w, COLS, POOL, n_conditions_col="n_conditions")
    assert 3000 < ic["ccsr_END005"] < 5000
    assert 10000 < ic["ccsr_CIR019"] < 14000


def test_train_returns_base_formula_when_plan_has_no_reach():
    X, y, w = synthetic()
    make = lambda: PaymentWLS().set_columns(COLS)  # noqa: E731
    plan = adversary.Plan(cost_model=_Const(0.0), cost_per_code=0.0, max_codes=1, reach=0.0, tilt=0.0,
                          pool=POOL, count_col="n_conditions", system_col="n_body_systems")
    f, path = adversary.train(make, plan, X, y, w, COLS, iters=5)
    assert len(path) == 1 and path[0]["change"] < 1e-9
    assert np.allclose(f.predict(X), make().fit(X, y, w).predict(X))


def test_train_lowers_extraction():
    X, y, w = synthetic(n=6000)
    make = lambda: PaymentWLS().set_columns(COLS)  # noqa: E731
    plan = adversary.Plan(cost_model=_Const(float(np.average(y, weights=w))), cost_per_code=200.0,
                          max_codes=1, reach=1.0, tilt=0.2, pool=POOL,
                          count_col="n_conditions", system_col="n_body_systems")
    f, path = adversary.train(make, plan, X, y, w, COLS, iters=10, tol=1e-4)
    added, s, Xc = plan.respond(f.predict, X, w, COLS)
    after = adversary.extraction(f.predict, X, Xc, w, s, y, added, plan.cost_per_code)
    assert after["coding"] < path[0]["coding"]


def test_dro_coding_penalty_matches_penalized_fit():
    X, y, w = synthetic()
    pw = {"ccsr_CIR019": 1.0, "n_conditions": 1.0, "n_body_systems": 1.0}
    a = robust.PenalizedPaymentWLS(lam=5.0, pool_weights=pw).set_columns(COLS).fit(X, y, w).coefficients()
    b = robust.DROPaymentWLS(lam=0.0, lam_code=5.0, pool_weights=pw, normalize=False).set_columns(COLS).fit(X, y, w).coefficients()
    for k in ("ccsr_CIR019", "ccsr_END003", "n_conditions", "age_1, male"):
        assert abs(a[k] - b[k]) < 0.02 * max(abs(a[k]), 100.0), k


def test_dro_is_budget_neutral():
    X, y, w = synthetic()
    ref = y + 5000 * (X[:, 1] * X[:, 5])
    dro = robust.DROPaymentWLS(lam=20.0, alpha=0.1, reference=ref).set_columns(COLS).fit(X, y, w)
    assert abs(np.sum(w * dro.predict(X)) / np.sum(w * y) - 1) < 1e-9


def test_dro_gradient_matches_finite_differences():
    X, y, w = synthetic(n=600)
    ref = y + 5000 * (X[:, 1] * X[:, 5])
    pw = {"ccsr_CIR019": 1.0, "n_conditions": 1.0}
    m = robust.DROPaymentWLS(lam=3.0, alpha=0.2, lam_code=2.0, pool_weights=pw, reference=ref).set_columns(COLS)
    m.fit(X, y, w)
    # rebuild the objective as fit() does and check the gradient numerically
    A = m._design(X)
    wn = w / w.sum()
    scale = float(np.sqrt(np.average(y ** 2, weights=wn)))
    b0 = np.concatenate([m.coef_ / m.factor_ / scale + 0.05, m.t_ / scale])
    import types
    captured = {}
    def fake_minimize(fun, x0, **kw):
        captured["obj"] = fun
        return types.SimpleNamespace(x=x0, success=True, fun=fun(x0)[0], message="ok")
    import scipy.optimize
    orig = scipy.optimize.minimize
    scipy.optimize.minimize = fake_minimize
    try:
        m.fit(X, y, w)
    finally:
        scipy.optimize.minimize = orig
    obj = captured["obj"]
    v0, g = obj(b0)
    num = np.zeros_like(b0)
    for j in range(len(b0)):
        e = np.zeros_like(b0); e[j] = 1e-6
        num[j] = (obj(b0 + e)[0] - obj(b0 - e)[0]) / 2e-6
    assert np.allclose(g, num, rtol=1e-4, atol=1e-6)


def test_audit_spreads_codes():
    X, y, w = synthetic(n=4000)
    f = PaymentWLS().set_columns(COLS).fit(X, y, w)
    base = dict(cost_per_code=0.0, max_codes=1, reach=0.5, tilt=0.0, pool=POOL,
                count_col="n_conditions", system_col="n_body_systems")
    a0, _ = adversary.Plan(**base).code(f.predict, X, w, COLS)
    a1, _ = adversary.Plan(audit=5000.0, **base).code(f.predict, X, w, COLS)
    used0 = (a0.sum(axis=0) > 0).sum()
    used1 = (a1.sum(axis=0) > 0).sum()
    assert used1 > used0 or (a1.sum(axis=0).max() < a0.sum(axis=0).max())


def test_selection_split_adds_up():
    X, y, w = synthetic()
    f = PaymentWLS().set_columns(COLS).fit(X, y, w)
    plan = adversary.Plan(cost_model=_Const(float(np.average(y, weights=w))), cost_per_code=0.0,
                          max_codes=1, reach=0.3, tilt=0.2, pool=POOL,
                          count_col="n_conditions", system_col="n_body_systems")
    added, s, Xc = plan.respond(f.predict, X, w, COLS)
    e = adversary.extraction(f.predict, X, Xc, w, s, y, added, 0.0)
    assert abs(e["selection_ungamed"] + e["selection_interaction"] - e["selection"]) < 1e-6 * max(1, abs(e["selection"]))


def test_audit_cost_is_charged():
    X, y, w = synthetic(n=4000)
    f = PaymentWLS().set_columns(COLS).fit(X, y, w)
    base = dict(cost_per_code=100.0, max_codes=1, reach=0.5, tilt=0.0, pool=POOL,
                count_col="n_conditions", system_col="n_body_systems")
    plan = adversary.Plan(audit=3000.0, **base)
    added, _ = plan.code(f.predict, X, w, COLS)
    n_codes = added.sum()
    assert plan.row_cost_.sum() > 100.0 * n_codes          # audit adds to the flat cost
    assert np.all(plan.row_cost_[added.sum(axis=1) == 0] == 0)


class _Stored:
    def __init__(self, values):
        self.values = np.asarray(values, float)

    def predict(self, X):
        assert len(X) == len(self.values)
        return self.values


def test_cross_fitted_training_runs_and_keeps_size():
    X, y, w = synthetic(n=3000)
    groups = np.arange(len(y)) // 10
    make = lambda: PaymentWLS().set_columns(COLS)  # noqa: E731
    plan = adversary.Plan(cost_model=_Stored(np.full(len(y), float(np.average(y, weights=w)))),
                          cost_per_code=200.0, max_codes=1, reach=0.2, tilt=0.2, pool=POOL,
                          count_col="n_conditions", system_col="n_body_systems")
    f, path = adversary.train(make, plan, X, y, w, COLS, iters=4, tol=1e-6, groups=groups, seed=1)
    assert len(path) >= 1 and np.isfinite(path[-1]["extraction"])
    halves = adversary._halves(groups, 1)
    assert not set(groups[halves[0]]) & set(groups[halves[1]])
