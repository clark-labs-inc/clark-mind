"""
Cortical column dynamics circuit: growing local transition operators (no backprop)
-----------------------------------------------------------------------------------
Tests the design's core claim that the prototype-memory swap failed to show on
language: a cell should be a LOCAL PREDICTIVE OPERATOR, not a prototype + vote.

Here we isolate the "local transition operator" component (design section 9):

    next_state ~= A_i (x - c_i) + b_i           # local linear predictor around c_i
    y_hat      = sum_i a_i (A_i (x-c_i) + b_i)   # precision-gated mixture
    r          = y - y_hat                       # local residual
    A_i       += eta * a_i * outer(r, x-c_i)     # local delta, no backprop
    b_i       += eta * a_i * r
    c_i       += eta_c * a_i * (x - c_i)
grown by repeated-residual compression, forgotten when it stops paying off.

This is a growing mixture of locally-weighted linear regressors (RFWR/LWPR-style)
trained purely by local rules. We pit it against the things it must beat to
justify the design:
    - persistence (predict last value)
    - global linear AR (one ridge-regression model = single global operator)
    - k-NN memory (the prototype-memory analogue: average neighbours' next value)

Task: 1-step prediction of the Mackey-Glass chaotic series. Local linear models
are classically strong here and prototype/global-linear models classically weak,
so it is a clean, falsifiable test. Metric: NRMSE (lower is better).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

np.seterr(over="ignore", invalid="ignore", divide="ignore")


# -----------------------------------------------------------------------------
# Data: Mackey-Glass chaotic time series + delay embedding
# -----------------------------------------------------------------------------


def mackey_glass(n, tau=17, beta=0.2, gamma=0.1, p=10, dt=1.0, burn=1000, seed=0):
    rng = np.random.default_rng(seed)
    total = n + burn + tau + 1
    x = np.zeros(total, dtype=np.float64)
    x[: tau + 1] = 1.2 + 0.01 * rng.standard_normal(tau + 1)
    for t in range(tau, total - 1):
        x[t + 1] = x[t] + dt * (beta * x[t - tau] / (1.0 + x[t - tau] ** p) - gamma * x[t])
    return x[burn: burn + n].astype(np.float32)


def henon_map(n, a=1.4, b=0.3, burn=1000, seed=0):
    rng = np.random.default_rng(seed)
    total = n + burn
    x = np.zeros(total, np.float64)
    x[0], x_prev = 0.1 + 0.01 * rng.standard_normal(), 0.0
    for t in range(total - 1):
        x[t + 1] = 1.0 - a * x[t] ** 2 + b * x_prev
        x_prev = x[t]
    return x[burn:].astype(np.float32)


def logistic_map(n, r=3.9, burn=1000, seed=0):
    rng = np.random.default_rng(seed)
    total = n + burn
    x = np.zeros(total, np.float64)
    x[0] = 0.3 + 0.01 * rng.standard_normal()
    for t in range(total - 1):
        x[t + 1] = r * x[t] * (1.0 - x[t])
    return x[burn:].astype(np.float32)


def gen_series(system, n, seed):
    if system == "henon":
        return henon_map(n, seed=seed)
    if system == "logistic":
        return logistic_map(n, seed=seed)
    return mackey_glass(n, seed=seed)


def embed(series, window):
    X = np.stack([series[t - window:t] for t in range(window, len(series))])
    Y = series[window:][:, None]
    return X.astype(np.float32), Y.astype(np.float32)


def nrmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)) / (np.std(true) + 1e-12))


# -----------------------------------------------------------------------------
# Baselines (all no-backprop)
# -----------------------------------------------------------------------------


def ridge_ar(Xtr, Ytr, Xte, lam=1e-2):
    Xtr1 = np.hstack([Xtr, np.ones((len(Xtr), 1), np.float32)])
    Xte1 = np.hstack([Xte, np.ones((len(Xte), 1), np.float32)])
    A = Xtr1.T @ Xtr1 + lam * np.eye(Xtr1.shape[1], dtype=np.float32)
    W = np.linalg.solve(A, Xtr1.T @ Ytr)
    return Xte1 @ W


def knn_predict(Xtr, Ytr, Xte, k=8):
    out = np.zeros((len(Xte), Ytr.shape[1]), np.float32)
    for i in range(len(Xte)):
        d2 = np.sum((Xtr - Xte[i]) ** 2, axis=1)
        nn = np.argpartition(d2, k)[:k]
        out[i] = Ytr[nn].mean(0)
    return out


# -----------------------------------------------------------------------------
# Growing local linear predictor (the cortical column's transition circuit)
# -----------------------------------------------------------------------------


@dataclass
class LLPConfig:
    in_dim: int
    out_dim: int = 1
    k: int = 8
    max_cells: int = 2000
    lr: float = 0.3
    lr_center: float = 0.02
    width: float = 2.0
    grow_err: float = 0.4      # grow only if local error large ...
    grow_novel: float = 1.5    # ... and region novel (sq-dist to nearest center) ...
    grow_resid: float = 0.25   # ... and that region has persistently failed
    resid_decay: float = 0.05
    seed: int = 0


class GrowingLocalLinearPredictor:
    def __init__(self, cfg: LLPConfig):
        self.cfg = cfg
        self.C = np.empty((0, cfg.in_dim), np.float32)
        self.W = np.empty((0, cfg.out_dim, cfg.in_dim), np.float32)
        self.b = np.empty((0, cfg.out_dim), np.float32)
        self.rel = np.empty((0,), np.float32)
        self.resid_ema = np.empty((0,), np.float32)

    def __len__(self):
        return len(self.C)

    def _active(self, x):
        d2 = np.sum((self.C - x) ** 2, axis=1)
        k = min(self.cfg.k, len(d2))
        idx = np.argpartition(d2, k - 1)[:k]
        idx = idx[np.argsort(d2[idx])]
        a = self.rel[idx] * np.exp(-0.5 * d2[idx] / (self.cfg.width ** 2))
        s = a.sum()
        a = a / s if s > 1e-12 else np.full(len(idx), 1.0 / len(idx), np.float32)
        return idx, a.astype(np.float32), d2[idx]

    def _forward(self, x, idx, a):
        z = x[None, :] - self.C[idx]
        y = np.einsum("koi,ki->ko", self.W[idx], z) + self.b[idx]
        return (a[:, None] * y).sum(0), z

    def add_cell(self, x, y):
        c = self.cfg
        self.C = np.vstack([self.C, x[None]])
        self.W = np.concatenate([self.W, np.zeros((1, c.out_dim, c.in_dim), np.float32)])
        self.b = np.vstack([self.b, y[None].astype(np.float32)])
        self.rel = np.append(self.rel, np.float32(1.0))
        self.resid_ema = np.append(self.resid_ema, np.float32(0.0))

    def predict(self, x):
        if len(self.C) == 0:
            return np.zeros(self.cfg.out_dim, np.float32)
        idx, a, _ = self._active(x)
        return self._forward(x, idx, a)[0]

    def update(self, x, y):
        c = self.cfg
        if len(self.C) == 0:
            self.add_cell(x, y)
            return
        idx, a, d2 = self._active(x)
        pred, z = self._forward(x, idx, a)
        r = (y - pred).astype(np.float32)
        rn = float(np.linalg.norm(r))
        self.W[idx] += c.lr * np.einsum("k,o,ki->koi", a, r, z)
        self.b[idx] += c.lr * a[:, None] * r[None, :]
        self.C[idx] += c.lr_center * a[:, None] * z
        self.rel[idx] = 0.99 * self.rel[idx] + 0.01 / (1.0 + rn)
        self.resid_ema[idx] = (1 - c.resid_decay) * self.resid_ema[idx] + c.resid_decay * rn
        if (rn > c.grow_err and d2[0] > c.grow_novel
                and self.resid_ema[idx[0]] > c.grow_resid and len(self.C) < c.max_cells):
            self.add_cell(x, y)


def run_llp(Xtr, Ytr, Xte, epochs, seed, **kw):
    cfg = LLPConfig(in_dim=Xtr.shape[1], out_dim=Ytr.shape[1], seed=seed, **kw)
    m = GrowingLocalLinearPredictor(cfg)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for i in rng.permutation(len(Xtr)):
            m.update(Xtr[i], Ytr[i])
    pred = np.stack([m.predict(Xte[i]) for i in range(len(Xte))])
    return pred, len(m)


def main():
    import argparse, time
    p = argparse.ArgumentParser()
    p.add_argument("--system", choices=["henon", "logistic", "mackey"], default="henon")
    p.add_argument("--n", type=int, default=12000)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--knn_k", type=int, default=8)
    p.add_argument("--width", type=float, default=2.0)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--max_cells", type=int, default=2000)
    p.add_argument("--grow_err", type=float, default=0.15)
    p.add_argument("--grow_novel", type=float, default=0.5)
    p.add_argument("--grow_resid", type=float, default=0.1)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    series = gen_series(args.system, args.n, args.seed)
    X, Yclean = embed(series, args.window)         # clean dynamics (eval target)
    if args.noise > 0:                             # observation noise on inputs+train targets
        rng = np.random.default_rng(args.seed + 1)
        obs = series + args.noise * series.std() * rng.standard_normal(len(series))
        X, Ytrain = embed(obs, args.window)        # learn from noisy observations
    else:
        Ytrain = Yclean
    n_tr = int(0.7 * len(X))                        # temporal split: predict the future
    Xtr, Xte = X[:n_tr], X[n_tr:]
    Ytr = Ytrain[:n_tr]                             # train on (noisy) observed next value
    Yte = Yclean[n_tr:]                             # but score against the TRUE dynamics
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    ymu, ysd = Ytr.mean(0), Ytr.std(0) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Ytr, Yte = (Ytr - ymu) / ysd, (Yte - ymu) / ysd
    print(f"system={args.system} noise={args.noise}: train={len(Xtr)} "
          f"test={len(Xte)} window={args.window}")

    res = {}
    res["persistence"] = (nrmse(Xte[:, -1:], Yte), 0)
    res["ridge_AR(global linear)"] = (nrmse(ridge_ar(Xtr, Ytr, Xte), Yte), 0)
    res["kNN memory (prototype)"] = (nrmse(knn_predict(Xtr, Ytr, Xte, args.knn_k), Yte), len(Xtr))
    pred, cells = run_llp(Xtr, Ytr, Xte, args.epochs, args.seed,
                          k=args.k, width=args.width, max_cells=args.max_cells,
                          grow_err=args.grow_err, grow_novel=args.grow_novel,
                          grow_resid=args.grow_resid)
    res["local linear charts"] = (nrmse(pred, Yte), cells)

    print("\n=== TIME-SERIES SUMMARY (no backprop; lower NRMSE = better) ===")
    print(f"  {'model':28s} {'NRMSE':>8s} {'cells':>8s}")
    for name in ("persistence", "ridge_AR(global linear)", "kNN memory (prototype)",
                 "local linear charts"):
        e, c = res[name]
        print(f"  {name:28s} {e:8.4f} {c:8d}")


if __name__ == "__main__":
    main()
