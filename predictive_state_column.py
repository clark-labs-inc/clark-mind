"""
PSC-2: Predictive State Column for action-conditioned dynamics (no backprop)
----------------------------------------------------------------------------
Stage-2 world-model test. State = predictive equivalence class (a REGIME),
inferred from the recent predictive signature, not the current point in state
space. Each state owns z_{t+1} ~= A_s z_t + B_s a_t + b_s; a sticky Bayesian
belief over operators infers the active regime; operators learn by
responsibility-weighted local LMS (online EM). No backprop / optimizer / BPTT.

Benchmark: switching linear system whose regimes OVERLAP in state space (so the
same (x,a) implies different x' by hidden regime). This breaks proximity/averaging
methods and isolates the value of inferring state from predictive history.

This version adds MULTI-STEP ROLLOUT (the real world-model test): warm up on
observations, then roll forward under KNOWN actions feeding predictions back. For
PSC-2 the belief evolves open-loop by its sticky prior alone (no observations).
"""

from __future__ import annotations
import argparse
import numpy as np

np.seterr(over="ignore", invalid="ignore", divide="ignore")

ACTIONS = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], np.float32)


def gen_slds(n, regimes=2, theta=0.6, scale=0.97, push=0.25, switch_p=0.01,
             noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    # qualitatively DISTINCT regimes: +rot, -rot, contraction, fast +rot, fast -rot
    angles = [theta, -theta, 0.0, 2 * theta, -2 * theta]
    scales = [scale, scale, 0.55, 0.9, 0.9]
    As, Bs = [], []
    for r in range(regimes):
        ang, sc = angles[r % 5], scales[r % 5]
        c, s = np.cos(ang), np.sin(ang)
        As.append(sc * np.array([[c, -s], [s, c]], np.float32))
        B = np.zeros((2, 4), np.float32)
        for k in range(4):
            B[:, k] = push * ACTIONS[k] * (1.0 if r % 2 == 0 else -1.0)
        Bs.append(B)
    x = (rng.standard_normal(2) * 0.5).astype(np.float32)
    r = 0
    X = np.zeros((n + 1, 2), np.float32)
    acts = np.zeros((n, 4), np.float32)
    reg = np.zeros(n, np.int64)
    X[0] = x
    for t in range(n):
        if rng.random() < switch_p:
            r = int(rng.integers(0, regimes))
        a = int(rng.integers(0, 4))
        acts[t, a] = 1.0
        reg[t] = r
        X[t + 1] = As[r] @ X[t] + Bs[r] @ acts[t]
    obs = X + noise * rng.standard_normal(X.shape).astype(np.float32)
    return obs[:-1], acts, X[1:], reg


def nrmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)) / (np.std(true) + 1e-12))


# -----------------------------------------------------------------------------
# Models: each exposes fit + a (state, step) rollout interface
# -----------------------------------------------------------------------------

def feat_xa(X, A):
    return np.hstack([X, A, np.ones((len(X), 1), np.float32)])


def ridge_fit(Xtr, Atr, Ytr, lam=1e-2):
    F = feat_xa(Xtr, Atr)
    return np.linalg.solve(F.T @ F + lam * np.eye(F.shape[1], dtype=np.float32), F.T @ Ytr)


def knn_pred(Xtr, Atr, Ytr, Xte, Ate, k=12):
    Ftr, Fte = np.hstack([Xtr, Atr]), np.hstack([Xte, Ate])
    out = np.zeros((len(Fte), Ytr.shape[1]), np.float32)
    for i in range(len(Fte)):
        nn = np.argpartition(np.sum((Ftr - Fte[i]) ** 2, 1), k)[:k]
        out[i] = Ytr[nn].mean(0)
    return out


class ESN:
    def __init__(self, din, dout, n_res=400, sr=0.9, leak=0.3, seed=0):
        rng = np.random.default_rng(seed)
        self.leak, self.n_res = leak, n_res
        self.Win = (rng.standard_normal((n_res, din + 1)) * 0.5).astype(np.float32)
        W = rng.standard_normal((n_res, n_res)).astype(np.float32)
        self.W = (W * sr / (np.max(np.abs(np.linalg.eigvals(W))) + 1e-8)).astype(np.float32)
        self.Wout = None

    def _upd(self, h, x, a):
        u = np.concatenate([x, a, [1.0]]).astype(np.float32)
        return (1 - self.leak) * h + self.leak * np.tanh(self.Win @ u + self.W @ h)

    def collect(self, X, A, h0):
        h = h0.copy(); H = np.zeros((len(X), self.n_res), np.float32)
        for t in range(len(X)):
            h = self._upd(h, X[t], A[t]); H[t] = h
        return H, h

    def fit(self, Xtr, Atr, Ytr, lam=1e-2):
        H, self.hlast = self.collect(Xtr, Atr, np.zeros(self.n_res, np.float32))
        self.Wout = np.linalg.solve(H.T @ H + lam * np.eye(self.n_res, dtype=np.float32), H.T @ Ytr)
        return self


class PSC2:
    def __init__(self, dim, adim, n_states=6, sigma=0.25, leak=0.04, lr=0.2,
                 lam=1e-2, seed=0):
        self.dim, self.adim = dim, adim
        self.sigma, self.leak, self.lr, self.lam = sigma, leak, lr, lam
        rng = np.random.default_rng(seed + 7)
        self.A = [(0.95 * np.eye(dim) + 0.5 * rng.standard_normal((dim, dim))).astype(np.float32)
                  for _ in range(n_states)]
        self.B = [np.zeros((dim, adim), np.float32) for _ in range(n_states)]
        self.b = [np.zeros(dim, np.float32) for _ in range(n_states)]
        self.belief = np.full(n_states, 1.0 / n_states, np.float32)
        self.win = np.zeros(n_states, np.float32)

    def _preds(self, x, a):
        return [self.A[k] @ x + self.B[k] @ a + self.b[k] for k in range(len(self.A))]

    def predict_with(self, belief, x, a):
        preds = self._preds(x, a)
        return sum(belief[k] * preds[k] for k in range(len(preds)))

    def filter_belief(self, belief, x, a, y):           # belief update from an observation
        preds = self._preds(x, a)
        res = np.array([np.sum((y - p) ** 2) for p in preds], np.float32)
        K = len(preds)
        bel = (1 - self.leak) * belief + self.leak / K
        bel = bel * np.exp(-0.5 * res / (self.sigma ** 2))
        s = bel.sum()
        return (bel / s) if s > 1e-20 else np.full(K, 1.0 / K, np.float32), preds

    def roll_belief(self, belief, p_switch=0.005):       # open-loop: regime persists
        K = len(self.A)
        bel = (1 - p_switch) * belief + p_switch / K     # sticky self-transition, not washout
        return bel / bel.sum()

    def step(self, x, a, y, learn=True):                 # online training step
        belief, preds = self.filter_belief(self.belief, x, a, y)
        self.belief = belief
        pred = sum(belief[k] * preds[k] for k in range(len(preds)))
        if learn:
            xa, aa = x @ x + self.lam, a @ a + self.lam
            for k in range(len(self.A)):
                resp = float(belief[k])
                if resp < 1e-4:
                    continue
                rk = (y - preds[k]).astype(np.float32)
                self.A[k] += self.lr * resp * np.outer(rk, x) / xa
                self.B[k] += self.lr * resp * np.outer(rk, a) / aa
                self.b[k] += self.lr * resp * rk
            self.win += belief
        return pred

    def active_states(self, frac=0.01):
        return int(np.sum(self.win > frac * self.win.sum()))

    def _param(self, k):
        return np.concatenate([self.A[k].ravel(), self.B[k].ravel(), self.b[k]])

    def _keep(self, idx):
        self.A = [self.A[k] for k in idx]; self.B = [self.B[k] for k in idx]
        self.b = [self.b[k] for k in idx]; self.win = self.win[idx]
        self.belief = np.full(len(idx), 1.0 / len(idx), np.float32)

    def consolidate(self, merge_thresh=0.25, prune_frac=0.02):
        """Sleep-phase compression economy: prune dead states, then merge states
        with equivalent futures (near-identical operators), usage-weighted. Keeps
        state count bounded -> the 'no landfill' guarantee for lifelong training."""
        n0 = len(self.A)
        keep = [k for k in range(len(self.A)) if self.win[k] > prune_frac * (self.win.sum() + 1e-9)]
        self._keep(keep if keep else list(range(len(self.A))))
        merged = True
        while merged and len(self.A) > 1:
            merged = False
            P = [self._param(k) for k in range(len(self.A))]
            best, bd = None, np.inf
            for i in range(len(self.A)):
                for j in range(i + 1, len(self.A)):
                    d = np.linalg.norm(P[i] - P[j]) / (np.linalg.norm(P[i]) + np.linalg.norm(P[j]) + 1e-8)
                    if d < bd:
                        bd, best = d, (i, j)
            if best and bd < merge_thresh:                # merge predictively-equivalent states
                i, j = best
                wi, wj = float(self.win[i]), float(self.win[j]); w = wi + wj + 1e-8
                self.A[i] = (wi * self.A[i] + wj * self.A[j]) / w
                self.B[i] = (wi * self.B[i] + wj * self.B[j]) / w
                self.b[i] = (wi * self.b[i] + wj * self.b[j]) / w
                self.win[i] = wi + wj
                self._keep([k for k in range(len(self.A)) if k != j])
                merged = True
        return n0, len(self.A)


# -----------------------------------------------------------------------------
# Multi-step rollout evaluation (warm up on obs, roll under known actions)
# -----------------------------------------------------------------------------


def rollout_eval(obs, acts, tgt, ntr, horizons, n_start, warm, ridgeW, esn, psc, seed):
    rng = np.random.default_rng(seed + 3)
    H = max(horizons)
    lo, hi = ntr + warm, len(obs) - H - 1
    starts = rng.choice(np.arange(lo, hi), size=min(n_start, hi - lo), replace=False)
    names = ["persistence", "ridge_global", "echo-state net", "PSC-2"]
    se = {nm: {h: [] for h in horizons} for nm in names}

    for s in starts:
        aseq = acts[s:s + H]
        true = tgt[s:s + H]                              # clean future states
        # persistence
        for h in horizons:
            se["persistence"][h].append(np.sum((obs[s] - true[h - 1]) ** 2))
        # ridge: iterate the single global operator
        x = obs[s].copy()
        for j in range(H):
            x = feat_xa(x[None], aseq[j][None])[0] @ ridgeW
            if (j + 1) in horizons:
                se["ridge_global"][j + 1].append(np.sum((x - true[j]) ** 2))
        # ESN: warm reservoir on prefix (teacher-forced), then roll feeding preds back
        h = np.zeros(esn.n_res, np.float32)
        for t in range(s - warm, s):
            h = esn._upd(h, obs[t], acts[t])
        x = obs[s].copy()
        for j in range(H):
            h = esn._upd(h, x, aseq[j]); x = h @ esn.Wout
            if (j + 1) in horizons:
                se["echo-state net"][j + 1].append(np.sum((x - true[j]) ** 2))
        # PSC-2: filter belief on prefix, then roll with sticky-prior-only belief
        bel = np.full(len(psc.A), 1.0 / len(psc.A), np.float32)
        for t in range(s - warm, s):
            bel, _ = psc.filter_belief(bel, obs[t], acts[t], obs[t + 1])
        x = obs[s].copy()
        for j in range(H):
            x = psc.predict_with(bel, x, aseq[j]); bel = psc.roll_belief(bel)
            if (j + 1) in horizons:
                se["PSC-2"][j + 1].append(np.sum((x - true[j]) ** 2))

    var = {h: np.var(np.stack([tgt[s:s + H][h - 1] for s in starts])) for h in horizons}
    out = {}
    for nm in names:
        out[nm] = {h: float(np.sqrt(np.mean(se[nm][h]) / (var[h] + 1e-12))) for h in horizons}
    return out, list(horizons)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=80000)
    p.add_argument("--regimes", type=int, default=2)
    p.add_argument("--noise", type=float, default=0.1)
    p.add_argument("--switch_p", type=float, default=0.01)
    p.add_argument("--n_states", type=int, default=6)
    p.add_argument("--n_start", type=int, default=400)
    p.add_argument("--warm", type=int, default=40)
    p.add_argument("--consolidate", action="store_true")
    p.add_argument("--merge_thresh", type=float, default=0.25)
    p.add_argument("--prune_frac", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    obs, acts, tgt, reg = gen_slds(args.n, regimes=args.regimes, switch_p=args.switch_p,
                                   noise=args.noise, seed=args.seed)
    ntr = int(0.7 * len(obs))
    mu, sd = obs[:ntr].mean(0), obs[:ntr].std(0) + 1e-8
    obs = (obs - mu) / sd; tgt = (tgt - mu) / sd
    Xtr, Atr, Ytr = obs[:ntr], acts[:ntr], tgt[:ntr]
    print(f"SLDS regimes={args.regimes} noise={args.noise}: train={ntr} test={len(obs)-ntr}")

    ridgeW = ridge_fit(Xtr, Atr, Ytr)
    esn = ESN(din=Xtr.shape[1] + Atr.shape[1], dout=Ytr.shape[1], seed=args.seed).fit(Xtr, Atr, Ytr)
    psc = PSC2(dim=Xtr.shape[1], adim=Atr.shape[1], n_states=args.n_states, seed=args.seed)
    for t in range(ntr):
        psc.step(Xtr[t], Atr[t], Ytr[t], learn=True)

    horizons = [1, 5, 20, 100]

    def rollout_table(tag):
        res, hs = rollout_eval(obs, acts, tgt, ntr, horizons, args.n_start, args.warm,
                               ridgeW, esn, psc, args.seed)
        print(f"\n=== ROLLOUT NRMSE {tag} (PSC-2 states={len(psc.A)}, "
              f"active={psc.active_states()}) ===")
        print("  " + f"{'model':16s}" + "".join(f"k={h:<7d}" for h in hs))
        for nm in ("persistence", "ridge_global", "echo-state net", "PSC-2"):
            print("  " + f"{nm:16s}" + "".join(f"{res[nm][h]:<9.3f}" for h in hs))

    rollout_table(f"[BEFORE consolidation, seeded {args.n_states}]")
    if args.consolidate:
        n0, n1 = psc.consolidate(merge_thresh=args.merge_thresh, prune_frac=args.prune_frac)
        print(f"\n  >>> sleep/consolidation: {n0} states -> {n1} "
              f"(true regimes={args.regimes})")
        rollout_table("[AFTER consolidation]")


if __name__ == "__main__":
    main()
