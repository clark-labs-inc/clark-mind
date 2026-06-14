"""substrate.py: the shared predictive core of the one brain (no backprop).
UniversalPSC = a variable-order backoff count model over a single vocabulary;
_sample = temperature/top-p sampling. Everything the brain knows lives in one
UniversalPSC instance's count tables. Extracted as the lean substrate.
"""
from __future__ import annotations
import math
import numpy as np

RNG = np.random.default_rng(0)
MISSING = -1


# =============================================================================
# The ONE universal learner (zero modality-specific branches)
# =============================================================================
class UniversalPSC:
    def __init__(self, vocab, offsets, pos_dims, alpha=0.02, ab=4.0):
        self.K, self.offsets, self.pos_dims = vocab, offsets, pos_dims
        self.alpha, self.ab = alpha, ab
        self.t = [{} for _ in range(len(offsets) + 1)]      # one count table per backoff depth

    def _ctx(self, E, A):
        vals = [E.get(tuple(a + o for a, o in zip(A, off)), MISSING) for off in self.offsets]
        return tuple(A[d] for d in self.pos_dims), vals

    def fit(self, samples):
        for shape, E, G in samples:
            for A, value in E.items():
                posk, vals = self._ctx(E, A)
                for j in range(len(self.offsets) + 1):
                    d = self.t[j].setdefault((G, posk, tuple(vals[:j])), {})
                    d[value] = d.get(value, 0) + 1

    def size(self):
        return sum(len(tj) for tj in self.t)

    def consolidate(self, budget, decay=0.97):
        """SLEEP: bound the model by forgetting what it can RECONSTRUCT from
        backoff. Novel math for a backoff count model: a depth-j context is
        predictively REDUNDANT if its next-symbol distribution matches its
        depth-(j-1) backoff (Jensen-Shannon ~ 0) -- deleting it costs almost
        nothing because prediction falls back to the parent. Score each context
        by usage-weighted JS-from-backoff; decay counts (temporal forgetting);
        evict the lowest-scoring deepest contexts until size <= budget. The
        unbounded n-gram model self-prunes into a bounded variable-order /
        causal-state model, keeping exactly the information-bearing contexts."""
        for tj in self.t:                                    # temporal decay
            for ctx in list(tj):
                d = tj[ctx]
                for v in list(d):
                    d[v] *= decay
                    if d[v] < 0.5:
                        del d[v]
                if not d:
                    del tj[ctx]
        if self.size() <= budget:
            return
        def js(d, par):                                      # weighted JS divergence
            keys = set(d) | set(par)
            sd = sum(d.values()); sp = sum(par.values())
            div = 0.0
            for k in keys:
                p = d.get(k, 0) / sd; q = par.get(k, 0) / sp; m = 0.5*(p+q)
                if p > 0: div += 0.5*p*math.log(p/m)
                if q > 0: div += 0.5*q*math.log(q/m)
            return div
        cand = []
        for j in range(len(self.t) - 1, 0, -1):              # deepest first
            tj, tp = self.t[j], self.t[j-1]
            for ctx, d in tj.items():
                G, posk, vt = ctx
                par = tp.get((G, posk, vt[:-1]))
                score = 1e18 if par is None else sum(d.values()) * js(d, par)
                cand.append((score, j, ctx))                 # low score = redundant
        cand.sort(key=lambda x: x[0])
        for _, j, ctx in cand[:self.size() - budget]:
            self.t[j].pop(ctx, None)

    def _interp(self, p, d):
        a = np.full(self.K, self.alpha)
        for k, v in d.items(): a[k] += v
        c = a.sum() - self.alpha * self.K; lam = c / (c + self.ab)
        return lam * (a / a.sum()) + (1 - lam) * p

    def predict(self, E, G, A):
        posk, vals = self._ctx(E, A)
        p = np.full(self.K, 1.0 / self.K)
        for j in range(len(self.offsets) + 1):              # shallow->deep; deep dominates
            d = self.t[j].get((G, posk, tuple(vals[:j])))
            if d: p = self._interp(p, d)
        return p

    def infer_globals(self, E, candidates):                 # universal "classify what I see"
        best, bs = candidates[0], -1e18
        for G in candidates:
            s = sum(math.log(self.predict(E, G, A)[v] + 1e-12) for A, v in E.items())
            if s > bs: bs, best = s, G
        return best

    def complete(self, modality, shape, E, G, temp=0.9, top_p=0.95):
        E = dict(E)
        for A in modality.order(shape, E):
            if A in E: continue
            v = _sample(self.predict(E, G, A), temp, top_p)
            E[A] = v
            if modality.stop is not None and v == modality.stop: break
        return E


def _sample(p, temp, top_p):
    logp = np.log(p + 1e-12) / max(temp, 1e-3); q = np.exp(logp - logp.max()); q /= q.sum()
    o = np.argsort(-q); cut = o[:max(1, int(np.searchsorted(np.cumsum(q[o]), top_p)) + 1)]
    return int(RNG.choice(cut, p=q[cut] / q[cut].sum()))


# =============================================================================
# Modalities = codecs only. Each implements the same small interface.
