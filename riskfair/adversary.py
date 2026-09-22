"""
A strategic plan, and formulas trained against it.

The plan sees a payment formula f and responds through two channels:

  coding     it adds a diagnosis code to a person's record when the payment
             rise exceeds the cost of adding it, only for codes that are
             plausible for that person, at most `max_codes` per person, and
             only for the share `reach` of enrollees with most to gain
  selection  it recruits harder where it expects to be paid more than the
             person will cost, using its own cost model, tilting enrollment
             by a factor 1 + tilt above and 1 - tilt below the median
             expected margin (or smoothly, in proportion to exp(eta * margin)),
             holding its size fixed

`Plan.respond` returns the codes added and the selection weights,
`extraction` prices what the plan gains, and `train` alternates the plan's
response with a refit of the formula on the data the response produces
(repeated risk minimization in the sense of Perdomo et al. 2020).

Everything here is a simulation of incentives. The plan is a model, and
every parameter of it is meant to be swept.
"""

from __future__ import annotations

import numpy as np


def gain_matrix(predict, X, columns, pool, plausible=None, count_col=None,
                system_col=None, prefix="ccsr_"):
    """Payment rise for each person and each pool flag switched on alone,
    counts updated as in `coding.perturb`. NaN where the person already has
    the flag or it is not plausible for them.

    Returns an (n, len(pool)) array. One prediction pass per flag.
    """
    X = np.asarray(X, float)
    n = len(X)
    p0 = predict(X)
    j_cnt = columns.index(count_col) if count_col else None
    j_sys = columns.index(system_col) if system_col else None
    flag_cols = [i for i, c in enumerate(columns) if c.startswith(prefix)]
    systems = {}
    for i in flag_cols:
        systems.setdefault(columns[i][len(prefix):len(prefix) + 3], []).append(i)
    G = np.full((n, len(pool)), np.nan)
    for q, c in enumerate(pool):
        j = columns.index(c)
        ok = X[:, j] == 0
        if plausible is not None:
            ok &= plausible[:, q]
        rows = np.flatnonzero(ok)
        if not len(rows):
            continue
        Xp = X[rows].copy()
        Xp[:, j] = 1
        if j_cnt is not None:
            Xp[:, j_cnt] += 1
        if j_sys is not None:
            same = systems.get(c[len(prefix):len(prefix) + 3], [])
            new_system = X[np.ix_(rows, same)].sum(axis=1) == 0
            Xp[new_system, j_sys] += 1
        G[rows, q] = predict(Xp) - p0[rows]
    return G


def plausible_by_system(X, columns, pool, prefix="ccsr_"):
    """A flag is plausible for a person who already carries a flag in the
    same body system (the three characters after the prefix)."""
    X = np.asarray(X, float)
    flag_cols = [i for i, c in enumerate(columns) if c.startswith(prefix)]
    sys_of = lambda c: c[len(prefix):len(prefix) + 3]  # noqa: E731
    have = {}
    for i in flag_cols:
        have.setdefault(sys_of(columns[i]), []).append(i)
    P = np.zeros((len(X), len(pool)), bool)
    for q, c in enumerate(pool):
        cols = have.get(sys_of(c), [])
        if cols:
            P[:, q] = X[:, cols].sum(axis=1) > 0
    return P


def apply_codes(X, columns, pool, added, count_col=None, system_col=None, prefix="ccsr_"):
    """Return X with the flags in `added` (n x len(pool) bool) switched on,
    counts and body-system counts updated."""
    X = np.asarray(X, float).copy()
    j_cnt = columns.index(count_col) if count_col else None
    j_sys = columns.index(system_col) if system_col else None
    flag_cols = [i for i, c in enumerate(columns) if c.startswith(prefix)]
    sys_of = lambda c: c[len(prefix):len(prefix) + 3]  # noqa: E731
    before = {}
    for i in flag_cols:
        before.setdefault(sys_of(columns[i]), []).append(i)
    had_system = {s: X[:, cols].sum(axis=1) > 0 for s, cols in before.items()}
    new_systems = np.zeros(len(X))
    seen = {s: had_system[s].copy() for s in had_system}
    for q, c in enumerate(pool):
        rows = np.flatnonzero(added[:, q] & (X[:, columns.index(c)] == 0))
        if not len(rows):
            continue
        X[rows, columns.index(c)] = 1
        if j_cnt is not None:
            X[rows, j_cnt] += 1
        s = sys_of(c)
        if j_sys is not None:
            fresh = rows[~seen[s][rows]]
            new_systems[fresh] += 1
            seen[s][fresh] = True
    if j_sys is not None:
        X[:, j_sys] += new_systems
    return X


class Plan:
    """A plan that best-responds to a payment formula.

    cost_model     object with predict(X) giving the plan's own expected cost,
                   cross-fitted so it never saw the rows it is applied to
    cost_per_code  dollars the plan spends to add one code
    max_codes      codes it can add per person
    reach          share of enrollee weight it can review for coding, taken
                   from those with most to gain
    tilt           selection tilt: weights 1 + tilt above the median expected
                   margin and 1 - tilt below (rule "threshold"), or
                   proportional to exp(eta * margin / mean cost) with
                   eta = tilt (rule "smooth"); 0 turns selection off
    plausible      (n, len(pool)) bool matrix of which codes each person could
                   be given; None allows every absent code
    """

    def __init__(self, cost_model=None, cost_per_code=500.0, max_codes=1, reach=0.25,
                 tilt=0.2, rule="threshold", pool=(), plausible=None,
                 count_col=None, system_col=None, prefix="ccsr_", audit=0.0):
        self.cost_model = cost_model
        self.cost_per_code, self.max_codes, self.reach = cost_per_code, int(max_codes), reach
        # Audit exposure: each 1% of enrollees already given code j raises the
        # cost of giving it once more by `audit` dollars. With audit = 0 a
        # linear formula's best response puts the single most lucrative code
        # on every reviewed chart; with audit > 0 the plan spreads its codes.
        self.audit = float(audit)
        self.tilt, self.rule = tilt, rule
        self.pool, self.plausible = list(pool), plausible
        self.count_col, self.system_col, self.prefix = count_col, system_col, prefix

    def subset(self, idx):
        """The same plan restricted to rows idx (plausibility and the stored
        cost predictions subset alongside)."""
        import copy
        p = copy.copy(self)
        if self.plausible is not None:
            p.plausible = self.plausible[idx]
        cm = self.cost_model
        if cm is not None and hasattr(cm, "values"):
            p.cost_model = copy.copy(cm)
            p.cost_model.values = np.asarray(cm.values)[idx]
        return p

    # ------------------------------------------------------------ coding --
    def code(self, predict, X, w, columns, plausible=None, G=None):
        """Which codes the plan adds: (n, len(pool)) bool, and the gain matrix.
        Pass a precomputed gain matrix G to reuse it across plan settings; it
        depends on the formula and the plausibility rule, not on the plan's
        cost, reach or tilt."""
        P = self.plausible if plausible is None else plausible
        if G is None:
            G = gain_matrix(predict, X, columns, self.pool, P, self.count_col,
                            self.system_col, self.prefix)
        added = np.zeros(G.shape, bool)
        self.row_cost_ = np.zeros(G.shape[0])
        if not self.pool or self.reach <= 0 or self.max_codes <= 0:
            return added, G
        if self.audit > 0:
            return self._code_with_audit(G, w), G
        net = np.where(np.isnan(G), -np.inf, G - self.cost_per_code)
        # best `max_codes` codes per person, by net gain, only if positive
        order = np.argsort(-net, axis=1)[:, :self.max_codes]
        take = np.take_along_axis(net, order, axis=1) > 0
        avail = np.where(take, np.take_along_axis(net, order, axis=1), 0.0).sum(axis=1)
        # review the enrollees with most to gain until `reach` of the weight
        rows = np.argsort(-avail)
        cw = np.cumsum(w[rows]) / w.sum()
        n_review = int(np.searchsorted(cw, self.reach, side="left")) + 1
        review = rows[:n_review]
        review = review[avail[review] > 0]
        for r in review:
            for q, ok in zip(order[r], take[r]):
                if ok:
                    added[r, q] = True
        self.row_cost_ = self.cost_per_code * added.sum(axis=1).astype(float)
        return added, G

    def _code_with_audit(self, G, w):
        """Greedy best response when a code's cost rises with how often the
        plan has already used it. Enrollees are reviewed in descending order of
        their best available gain until `reach` of the weight is covered; each
        takes up to `max_codes` codes, each chosen to maximize gain minus the
        code's current marginal cost."""
        n, J = G.shape
        wshare = w / w.sum()
        used = np.zeros(J)                       # weight share already given each code
        added = np.zeros(G.shape, bool)
        self.row_cost_ = np.zeros(n)             # what the plan pays for each person's codes
        best = np.where(np.isnan(G), -np.inf, G).max(axis=1)
        order = np.argsort(-best)
        covered = 0.0
        for r in order:
            if covered >= self.reach or best[r] <= self.cost_per_code:
                break
            covered += wshare[r]
            for _ in range(self.max_codes):
                cost = self.cost_per_code + self.audit * 100.0 * used
                net = np.where(np.isnan(G[r]) | added[r], -np.inf, G[r] - cost)
                j = int(np.argmax(net))
                if net[j] <= 0:
                    break
                added[r, j] = True
                self.row_cost_[r] += cost[j]
                used[j] += wshare[r]
        return added

    # --------------------------------------------------------- selection --
    def select(self, payment, w, expected_cost):
        """Selection weights s with sum(w s) = sum(w)."""
        n = len(w)
        if self.tilt <= 0 or self.cost_model is None:
            return np.ones(n)
        margin = payment - expected_cost
        if self.rule == "smooth":
            z = self.tilt * margin / max(float(np.average(expected_cost, weights=w)), 1.0)
            s = np.exp(z - z.max())
        else:
            o = np.argsort(-margin)
            cw = np.cumsum(w[o]) / w.sum()
            s = np.full(n, 1.0 - self.tilt)
            s[o[cw <= 0.5]] = 1.0 + self.tilt
        return s * w.sum() / np.sum(w * s)

    # ---------------------------------------------------------- response --
    def respond(self, predict, X, w, columns, plausible=None, G=None):
        """Coding then selection. Returns (added, s, X_coded)."""
        added, _ = self.code(predict, X, w, columns, plausible, G)
        Xc = apply_codes(X, columns, self.pool, added, self.count_col, self.system_col, self.prefix)
        pay = predict(Xc)
        cost = self.cost_model.predict(np.asarray(X, float)) if self.cost_model is not None else pay
        s = self.select(pay, w, cost)
        return added, s, Xc


def extraction(predict, X, Xc, w, s, y, added, cost_per_code, per=1000.0, row_cost=None):
    """What the plan gains, per `per` enrollees of weight.

    payment_base      sum w f(x)
    coding            sum w (f(xc) - f(x)) - cost of the codes: payment for
                      codes that change nothing about cost
    selection_shift   sum w (s - 1) f(xc): extra payment from the tilt
    selection         sum w (s - 1) (f(xc) - y): payment above cost from the
                      tilt, which is what favorable selection earns (needs y)
    extraction        coding + selection
    profit_base       sum w (f(x) - y)
    profit            sum w s (f(xc) - y) - coding cost
    """
    p0, p1 = predict(np.asarray(X, float)), predict(np.asarray(Xc, float))
    codes = float(np.sum(w * added.sum(axis=1)))
    k = per / w.sum()
    paid = float(np.sum(w * row_cost)) if row_cost is not None else cost_per_code * codes
    coding = float(np.sum(w * (p1 - p0))) - paid
    shift = float(np.sum(w * (s - 1.0) * p1))
    selection = float(np.sum(w * (s - 1.0) * (p1 - y))) if y is not None else shift
    out = {
        "payment_base": float(np.sum(w * p0)) * k,
        "coding": coding * k,
        "selection_shift": shift * k,
        "selection": selection * k,
        "codes_added": codes * k,
        "extraction": (coding + selection) * k,
    }
    if y is not None:
        # selection split: on what the formula pays for the ungamed record,
        # and the extra from tilting toward people who were coded
        out["selection_ungamed"] = float(np.sum(w * (s - 1.0) * (p0 - y))) * k
        out["selection_interaction"] = float(np.sum(w * (s - 1.0) * (p1 - p0))) * k
    if y is not None:
        out["profit_base"] = float(np.sum(w * (p0 - y))) * k
        out["profit"] = (float(np.sum(w * s * (p1 - y))) - cost_per_code * codes) * k
    return out


def _halves(groups, seed):
    g = np.asarray(groups)
    u = np.unique(g)
    rng = np.random.default_rng(seed)
    side = dict(zip(u, rng.permutation(len(u)) % 2))
    h = np.array([side[x] for x in g])
    return [np.flatnonzero(h == 0), np.flatnonzero(h == 1)]


def respond_cross_fitted(make_formula, plan, X, y, w, columns, halves, Xfit=None, wfit=None):
    """The plan's response on every row, each half answered against a formula
    fitted on the other half, so the plan never games a formula's fit to the
    rows it is responding on. Xfit, wfit: the data the formulas are fitted on
    (the previous round's coded records and tilted weights); the plan codes
    the true records X. Returns added, s (normalized within each half), Xc,
    row_cost and the out-of-fold payments before and after coding."""
    X = np.asarray(X, float)
    Xfit = X if Xfit is None else Xfit
    wfit = w if wfit is None else wfit
    n = len(X)
    added = np.zeros((n, len(plan.pool)), bool)
    s = np.ones(n)
    Xc = X.copy()
    row_cost = np.zeros(n)
    p0 = np.zeros(n)
    p1 = np.zeros(n)
    for h, other in ((halves[0], halves[1]), (halves[1], halves[0])):
        g = make_formula()
        if hasattr(g, "take_rows"):          # row-aligned inputs such as a DRO reference
            g.take_rows(other)
        g.fit(Xfit[other], y[other], wfit[other])
        sub = plan.subset(h)
        a_h, s_h, Xc_h = sub.respond(g.predict, X[h], w[h], columns)
        added[h], s[h], Xc[h] = a_h, s_h, Xc_h
        row_cost[h] = sub.row_cost_
        p0[h], p1[h] = g.predict(X[h]), g.predict(Xc_h)
    return added, s, Xc, row_cost, p0, p1


def train(make_formula, plan, X, y, w, columns, iters=20, tol=0.01, damping=0.0,
          plausible=None, callback=None, groups=None, seed=0):
    """Alternate the plan's best response with a refit of the formula on the
    data the response produces: the formula at step t is fitted to the coded
    features and the tilted weights the plan produced against step t - 1, with
    the true cost y (repeated risk minimization).

    With `groups` (cluster ids), the plan's response is cross-fitted: the rows
    are split into two cluster-disjoint halves and each half responds to a
    formula fitted on the other half's current data, so the plan never games
    in-sample noise. Without it the plan responds to the in-sample fit.

    make_formula  () -> object with fit(X, y, w) and predict(X)
    Returns (formula, path) where path lists the mean absolute change in
    payment and the plan's gains on the training rows at each step.
    """
    X = np.asarray(X, float)
    f = make_formula().fit(X, y, w)
    halves = _halves(groups, seed) if groups is not None else None
    path = []
    prev = f.predict(X)
    Xfit, wfit = X, w
    k = 1000.0 / w.sum()
    for t in range(1, iters + 1):
        if halves is not None:
            added, s, Xc, row_cost, p0, p1 = respond_cross_fitted(make_formula, plan, X, y, w, columns, halves,
                                                                   Xfit, wfit)
        else:
            added, s, Xc = plan.respond(f.predict, X, w, columns, plausible)
            row_cost = getattr(plan, "row_cost_", plan.cost_per_code * added.sum(axis=1))
            p0, p1 = f.predict(X), f.predict(Xc)
        g = make_formula().fit(Xc, y, w * s)
        cur = g.predict(X)
        change = float(np.average(np.abs(cur - prev), weights=w) / max(np.average(np.abs(prev), weights=w), 1.0))
        coding = (float(np.sum(w * (p1 - p0))) - float(np.sum(w * row_cost))) * k
        selection = float(np.sum(w * (s - 1.0) * (p1 - y))) * k
        path.append({"iter": t, "change": change, "coding": coding, "selection": selection,
                     "extraction": coding + selection, "payment_base": float(np.sum(w * p0)) * k})
        if callback:
            callback(t, g, path[-1])
        f, prev = g, cur
        Xfit, wfit = Xc, w * s
        if change < tol:
            break
    return f, path
