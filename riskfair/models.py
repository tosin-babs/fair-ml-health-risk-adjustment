"""
Health-cost prediction models behind one interface: fit(X, y, w) and predict(X).

  WLS            weighted least squares, the linear additive payment form
  TweedieGLM     Tweedie GLM with log link, the actuarial standard for claims
  TwoPart        logistic model for any spending times a gamma GLM for amount
  ElasticNet     elastic net on raw spending
  GBM            LightGBM with a Tweedie or squared-error objective
  RandomForest   random forest
  NeuralNet      feed-forward network with Tweedie deviance and log link;
                 with cann=True, a combined actuarial neural network: the
                 Tweedie GLM's linear predictor as a fixed skip connection plus
                 a network correction initialised at zero (Schelldorfer and
                 Wuthrich 2019)

Every model with a regularization or architecture choice tunes it on an inner
split of whole clusters taken from the training data passed to fit, so a test
observation never influences a hyperparameter and no model family is handicapped
by a fixed setting another family gets to tune. Pass the training rows' cluster
identifiers as `clusters`.

Multiplicative models (log link, Tweedie or gamma deviance) are rescaled so
their weighted predictions sum to weighted spending on the training data: the
balance property risk adjustment requires and those deviances do not guarantee.
"""

from __future__ import annotations

import numpy as np

from .surveycv import inner_folds, inner_split

TWEEDIE_POWER = 1.5
DEFAULT_SEED = 2026
# Inner folds for the inexpensive models' hyperparameter selection. Boosting
# and neural networks use a single early-stopping split instead.
INNER_K = 3


def _balance(y, p, w):
    """Factor that makes weighted predictions sum to weighted spending.

    Estimated on the model's own training data, never on test data.
    """
    s = float(np.sum(w * p))
    return float(np.sum(w * y)) / s if s > 0 else 1.0


def _r2(y, p, w):
    return 1 - np.sum(w * (y - p) ** 2) / np.sum(w * (y - np.average(y, weights=w)) ** 2)


def _clusters_or_rows(clusters, n):
    return np.arange(n) if clusters is None else np.asarray(clusters)


def _import_torch():
    """Import PyTorch safely alongside LightGBM and scikit-learn.

    Each library ships its own OpenMP runtime. On macOS, tested with LightGBM
    4.7 and PyTorch 2.14, loading PyTorch before LightGBM segfaults, and
    running a multi-threaded PyTorch after LightGBM deadlocks. Loading
    LightGBM first and restricting PyTorch to one thread avoids both, and
    costs little for networks this size. LightGBM keeps all its threads.
    """
    try:
        import lightgbm  # noqa: F401  load its OpenMP runtime first
    except ImportError:
        pass
    import torch
    torch.set_num_threads(1)
    return torch


class WLS:
    name = "WLS"

    def fit(self, X, y, w, clusters=None):
        A = np.column_stack([np.ones(len(X)), X])
        sw = np.sqrt(w / w.mean())
        G = (A * sw[:, None]).T @ (A * sw[:, None])
        G[np.diag_indices_from(G)] += 1e-6
        self.coef_ = np.linalg.solve(G, (A * sw[:, None]).T @ (y * sw))
        return self

    def predict(self, X):
        return self.coef_[0] + X @ self.coef_[1:]


class _Scaled:
    def _fit_scale(self, X):
        self.mu_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0)
        self.sd_[self.sd_ == 0] = 1.0

    def _scale(self, X):
        return (X - self.mu_) / self.sd_


class TweedieGLM(_Scaled):
    """Tweedie GLM, log link. The L2 penalty is tuned unless `alpha` is given.

    A log-link GLM extrapolates exponentially: a rare combination of condition
    flags can produce a prediction several times the largest observed
    spending. Too weak a penalty lets that dominate squared-error accuracy,
    so the penalty is chosen on inner validation over a wide grid.
    """

    name = "Tweedie GLM"
    ALPHAS = (1e-2, 1e-1, 1.0, 10.0)

    def __init__(self, power=TWEEDIE_POWER, alpha=None):
        self.power, self.alpha = power, alpha

    def _fit_alpha(self, Z, y, sw, alpha):
        from sklearn.linear_model import TweedieRegressor
        return TweedieRegressor(power=self.power, link="log", alpha=alpha,
                                max_iter=3000).fit(Z, y, sample_weight=sw)

    def fit(self, X, y, w, clusters=None):
        self._fit_scale(X)
        Z = self._scale(X)
        sw = w / w.mean()
        alpha = self.alpha
        if alpha is None:
            splits = inner_folds(_clusters_or_rows(clusters, len(y)), k=INNER_K)
            best = None
            for a in self.ALPHAS:
                scores = []
                for tr, va in splits:
                    m = self._fit_alpha(Z[tr], y[tr], sw[tr], a)
                    b = _balance(y[tr], m.predict(Z[tr]), sw[tr])
                    scores.append(_r2(y[va], b * m.predict(Z[va]), sw[va]))
                s = float(np.mean(scores))
                if best is None or s > best[0]:
                    best = (s, a)
            alpha = best[1]
        self.params_ = {"alpha": alpha}
        self.m_ = self._fit_alpha(Z, y, sw, alpha)
        self.balance_ = _balance(y, np.exp(self.linear_predictor(X)), w)
        return self

    def linear_predictor(self, X):
        """Unbalanced linear predictor, used as the CANN skip connection."""
        return self.m_.intercept_ + self._scale(X) @ self.m_.coef_

    def predict(self, X):
        return self.balance_ * np.exp(self.linear_predictor(X))


class TwoPart(_Scaled):
    """P(any spending) from logistic regression times E(spending | any) from a
    gamma GLM. The logistic penalty is tuned on validation log loss, the gamma
    penalty on validation R-squared of the combined prediction."""

    name = "Two-part"
    CS = (0.1, 1.0)
    ALPHAS = (1e-1, 1.0, 10.0)

    def fit(self, X, y, w, clusters=None):
        from sklearn.linear_model import GammaRegressor, LogisticRegression
        self._fit_scale(X)
        Z = self._scale(X)
        sw = w / w.mean()
        anyy = (y > 0).astype(int)
        splits = inner_folds(_clusters_or_rows(clusters, len(y)), k=INNER_K)

        best_c = None
        for C in self.CS:
            lls = []
            for tr, va in splits:
                lg = LogisticRegression(C=C, max_iter=3000).fit(Z[tr], anyy[tr], sample_weight=sw[tr])
                p = np.clip(lg.predict_proba(Z[va])[:, 1], 1e-9, 1 - 1e-9)
                lls.append(-np.average(anyy[va] * np.log(p) + (1 - anyy[va]) * np.log(1 - p),
                                       weights=sw[va]))
            if best_c is None or np.mean(lls) < best_c[0]:
                best_c = (float(np.mean(lls)), C)
        C = best_c[1]
        logits = [LogisticRegression(C=C, max_iter=3000).fit(Z[tr], anyy[tr], sample_weight=sw[tr])
                  for tr, _ in splits]
        best_a = None
        for a in self.ALPHAS:
            scores = []
            for (tr, va), lg in zip(splits, logits):
                pos_tr = tr[y[tr] > 0]
                g = GammaRegressor(alpha=a, max_iter=3000).fit(Z[pos_tr], y[pos_tr],
                                                              sample_weight=sw[pos_tr])
                b = _balance(y[tr], lg.predict_proba(Z[tr])[:, 1] * g.predict(Z[tr]), sw[tr])
                scores.append(_r2(y[va], b * lg.predict_proba(Z[va])[:, 1] * g.predict(Z[va]),
                                  sw[va]))
            if best_a is None or np.mean(scores) > best_a[0]:
                best_a = (float(np.mean(scores)), a)
        alpha = best_a[1]
        self.params_ = {"C": C, "alpha": alpha}
        self.p_ = LogisticRegression(C=C, max_iter=3000).fit(Z, anyy, sample_weight=sw)
        pos = y > 0
        self.g_ = GammaRegressor(alpha=alpha, max_iter=3000).fit(Z[pos], y[pos], sample_weight=sw[pos])
        self.balance_ = 1.0
        self.balance_ = _balance(y, self.predict(X), w)
        return self

    def predict(self, X):
        Z = self._scale(X)
        return self.balance_ * self.p_.predict_proba(Z)[:, 1] * self.g_.predict(Z)


class ElasticNet(_Scaled):
    name = "Elastic net"
    GRID = [(a, r) for a in (1.0, 10.0, 100.0) for r in (0.2, 0.8)]

    def fit(self, X, y, w, clusters=None):
        from sklearn.linear_model import ElasticNet as EN
        self._fit_scale(X)
        Z = self._scale(X)
        sw = w / w.mean()
        splits = inner_folds(_clusters_or_rows(clusters, len(y)), k=INNER_K)
        best = None
        for alpha, l1 in self.GRID:
            scores = []
            for tr, va in splits:
                m = EN(alpha=alpha, l1_ratio=l1, max_iter=5000)
                m.fit(Z[tr], y[tr], sample_weight=sw[tr])
                scores.append(_r2(y[va], m.predict(Z[va]), sw[va]))
            s = float(np.mean(scores))
            if best is None or s > best[0]:
                best = (s, alpha, l1)
        self.params_ = {"alpha": best[1], "l1_ratio": best[2]}
        self.m_ = EN(max_iter=5000, **self.params_).fit(Z, y, sample_weight=sw)
        return self

    def predict(self, X):
        return self.m_.predict(self._scale(X))


class GBM:
    GRID = [(31, 50), (15, 100), (63, 100)]

    def __init__(self, objective="tweedie", seed=DEFAULT_SEED):
        self.objective, self.seed = objective, seed
        self.name = ("LightGBM (Tweedie)" if objective == "tweedie"
                     else "LightGBM (squared error)")

    def _params(self, leaves, min_child):
        p = dict(n_estimators=4000, learning_rate=0.03, num_leaves=leaves,
                 min_child_samples=min_child, subsample=0.8, subsample_freq=1,
                 colsample_bytree=0.8, reg_lambda=1.0, random_state=self.seed,
                 n_jobs=-1, verbose=-1)
        if self.objective == "tweedie":
            p.update(objective="tweedie", tweedie_variance_power=TWEEDIE_POWER)
        else:
            p.update(objective="regression")
        return p

    def fit(self, X, y, w, clusters=None):
        import lightgbm as lgb
        sw = w / w.mean()
        tr, va = inner_split(_clusters_or_rows(clusters, len(y)))
        best = None
        for leaves, mc in self.GRID:
            m = lgb.LGBMRegressor(**self._params(leaves, mc))
            m.fit(X[tr], y[tr], sample_weight=sw[tr], eval_X=(X[va],), eval_y=(y[va],),
                  eval_sample_weight=[sw[va]],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
            s = _r2(y[va], m.predict(X[va]), sw[va])
            if best is None or s > best[0]:
                best = (s, leaves, mc, m.best_iteration_ or m.n_estimators)
        _, leaves, mc, n_iter = best
        self.params_ = {"num_leaves": leaves, "min_child_samples": mc,
                        "n_estimators": int(n_iter)}
        p = self._params(leaves, mc)
        p["n_estimators"] = max(int(n_iter * 1.1), 50)
        self.m_ = lgb.LGBMRegressor(**p).fit(X, y, sample_weight=sw)
        self.balance_ = (_balance(y, self.m_.predict(X), w)
                         if self.objective == "tweedie" else 1.0)
        return self

    def predict(self, X):
        return self.balance_ * self.m_.predict(X)


class RandomForest:
    name = "Random forest"

    def __init__(self, seed=DEFAULT_SEED):
        self.seed = seed

    def fit(self, X, y, w, clusters=None):
        from sklearn.ensemble import RandomForestRegressor
        self.m_ = RandomForestRegressor(n_estimators=300, min_samples_leaf=25,
                                        max_features=0.33, n_jobs=-1,
                                        random_state=self.seed)
        self.m_.fit(X, y, sample_weight=w / w.mean())
        return self

    def predict(self, X):
        return self.m_.predict(X)


def _tweedie_deviance_loss(mu, y, w, p=TWEEDIE_POWER):
    import torch
    mu = torch.clamp(mu, min=1e-6)
    dev = 2 * (torch.pow(torch.clamp(y, min=0), 2 - p) / ((1 - p) * (2 - p))
               - y * torch.pow(mu, 1 - p) / (1 - p)
               + torch.pow(mu, 2 - p) / (2 - p))
    return torch.sum(w * dev) / torch.sum(w)


class NeuralNet(_Scaled):
    """Feed-forward network with log link; CANN when cann=True.

    The learning rate and width are tuned on inner validation deviance. For the
    CANN, the correction's output layer starts at zero, so the network begins
    exactly at the GLM and moves away only where validation deviance improves;
    the GLM underneath is tuned first and its penalty is then held fixed.
    """

    GRID = ((1e-3, (64, 32)), (3e-3, (64, 32)), (3e-3, (128, 64)), (1e-2, (32,)))

    def __init__(self, cann=False, epochs=300, dropout=0.1, patience=25,
                 weight_decay=1e-5, seed=DEFAULT_SEED):
        self.cann, self.epochs, self.dropout = cann, epochs, dropout
        self.patience, self.weight_decay, self.seed = patience, weight_decay, seed
        self.name = "CANN" if cann else "Neural network"

    def _net(self, d_in, hidden):
        import torch.nn as nn
        layers, prev = [], d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = h
        last = nn.Linear(prev, 1)
        if self.cann:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        layers.append(last)
        return nn.Sequential(*layers)

    def _train(self, torch, Zt, yt, wt, ot, base, tr, va, lr, hidden):
        torch.manual_seed(self.seed)
        net = self._net(Zt.shape[1], hidden)
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=self.weight_decay)
        rng = np.random.default_rng(self.seed)
        vt = torch.as_tensor(va)
        with torch.no_grad():
            best = float(_tweedie_deviance_loss(torch.exp(net(Zt[vt]).squeeze(1) + ot[vt] + base),
                                                yt[vt], wt[vt]))
        best_state = {k: v.clone() for k, v in net.state_dict().items()}
        bad, epochs = 0, 0
        for epoch in range(self.epochs):
            net.train()
            perm = rng.permutation(tr)
            for b in np.array_split(perm, max(1, len(perm) // 1024)):
                bt = torch.as_tensor(b)
                eta = net(Zt[bt]).squeeze(1) + ot[bt] + base
                loss = _tweedie_deviance_loss(torch.exp(eta), yt[bt], wt[bt])
                opt.zero_grad()
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vloss = float(_tweedie_deviance_loss(
                    torch.exp(net(Zt[vt]).squeeze(1) + ot[vt] + base), yt[vt], wt[vt]))
            epochs = epoch + 1
            if vloss < best * (1 - 1e-5):
                best, bad = vloss, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(best_state)
        return net, best, epochs

    def fit(self, X, y, w, clusters=None):
        torch = _import_torch()
        cl = _clusters_or_rows(clusters, len(y))
        self._fit_scale(X)
        Z = self._scale(X).astype(np.float32)
        if self.cann:
            self.glm_ = TweedieGLM().fit(X, y, w, clusters=cl)
            offset = np.array(self.glm_.linear_predictor(X), dtype=np.float32)
            base = 0.0
        else:
            offset = np.zeros(len(X), dtype=np.float32)
            base = float(np.log(np.average(y, weights=w)))
        self.base_ = base
        tr, va = inner_split(cl)
        T = lambda a: torch.as_tensor(np.array(a, dtype=np.float32))
        Zt, yt, wt, ot = T(Z), T(y), T(w / w.mean()), T(offset)
        best = None
        for lr, hidden in self.GRID:
            net, vloss, epochs = self._train(torch, Zt, yt, wt, ot, base, tr, va, lr, hidden)
            if best is None or vloss < best[0]:
                best = (vloss, net, lr, hidden, epochs)
        _, self.net_, lr, hidden, epochs = best
        self.params_ = {"lr": lr, "hidden": str(hidden), "epochs": epochs}
        self.balance_ = 1.0
        self.balance_ = _balance(y, self.predict(X), w)
        return self

    def predict(self, X):
        torch = _import_torch()
        Z = torch.as_tensor(np.array(self._scale(X), dtype=np.float32))
        offset = self.glm_.linear_predictor(X) if self.cann else np.zeros(len(X))
        self.net_.eval()
        with torch.no_grad():
            eta = self.net_(Z).squeeze(1).numpy() + offset + self.base_
        return self.balance_ * np.exp(eta)


def registry(seed=DEFAULT_SEED):
    """Key -> constructor, in the order results are tabulated."""
    return {
        "wls": WLS,
        "tweedie": TweedieGLM,
        "twopart": TwoPart,
        "enet": ElasticNet,
        "gbm": lambda: GBM("tweedie", seed=seed),
        "gbm_mse": lambda: GBM("regression", seed=seed),
        "rf": lambda: RandomForest(seed=seed),
        "mlp": lambda: NeuralNet(cann=False, seed=seed),
        "cann": lambda: NeuralNet(cann=True, seed=seed),
    }
