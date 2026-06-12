"""
Universal Predictive Chart Memory (no backprop)
-----------------------------------------------
Replaces the prototype-and-vote neuron with a growing set of LOCAL PREDICTIVE
CHARTS. Each cell is a locally-weighted linear predictor (an affine operator in
local coordinates around its center) -- i.e. a growing mixture of local linear
experts (cf. Receptive-Field Weighted Regression / LWPR), trained only by local
residual correction, grown by a compression (MDL-style) criterion, and forgotten
when it stops paying for itself. Modality-agnostic: input is any feature vector,
output is a byte/event slot, so the same object does vision, language, etc.

Cell i:
    c_i      center (context key, unit-norm)
    A_i      local predictive operator   (out_dim x local_dim)
    b_i      prediction at the center     (out_dim)        # b_i = predict at x=c_i
    prec_i   precision (sharpness of the receptive field)
    rel_i    reliability;  resid_ema_i  running unexplained-residual energy

Shared random chart basis R (local_dim x feat_dim) gives local coordinates
    z_i = R (x - c_i)
so A_i is small even when features are high-dimensional.

Prediction (sparse, top-k active):
    a_i    = rel_i * exp(-0.5 * prec_i * ||x - c_i||^2)         (precision gating)
    y_hat  = sum_i a_i (A_i z_i + b_i)
Local update (no global backward pass):
    r          = target - softmax(y_hat)            (residual on candidate slots)
    A_i       += lr * a_i * outer(r, z_i)
    b_i       += lr * a_i * r
    c_i       += lr_c * a_i * (x - c_i)
Growth (compression, not error): add a cell only where residual is large AND the
region is novel AND the nearest cell has *persistently* failed (repeated, hence
compressible, structure) -- not on every mistake.

No loss.backward(), no optimizer, no backprop through layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

np.seterr(over="ignore", invalid="ignore", divide="ignore")


def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


@dataclass
class ChartConfig:
    feat_dim: int
    out_dim: int = 256
    local_dim: int = 24
    k: int = 16
    max_cells: int = 6000
    lr: float = 0.5            # local linear operator learning rate
    lr_center: float = 0.02
    prec_init: float = 2.0
    grow_conf: float = 0.5     # grow if true-slot prob below this
    grow_novel: float = 0.55   # ... and nearest center farther than this (sq dist)
    grow_resid: float = 0.30   # ... and nearest cell's running residual above this
    resid_decay: float = 0.03
    seed: int = 0


class UniversalPredictiveChartMemory:
    def __init__(self, cfg: ChartConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed + 4242)
        self.R = (self.rng.standard_normal((cfg.local_dim, cfg.feat_dim))
                  / np.sqrt(cfg.feat_dim)).astype(np.float32)
        self.C = np.empty((0, cfg.feat_dim), dtype=np.float32)
        self.A = np.empty((0, cfg.out_dim, cfg.local_dim), dtype=np.float32)
        self.b = np.empty((0, cfg.out_dim), dtype=np.float32)
        self.prec = np.empty((0,), dtype=np.float32)
        self.rel = np.empty((0,), dtype=np.float32)
        self.usage = np.empty((0,), dtype=np.float32)
        self.resid_ema = np.empty((0,), dtype=np.float32)

    def __len__(self):
        return len(self.C)

    def _active(self, x):
        d2 = np.maximum(0.0, 2.0 - 2.0 * (self.C @ x))   # x, C unit-norm
        k = min(self.cfg.k, len(d2))
        idx = np.argpartition(d2, k - 1)[:k]
        idx = idx[np.argsort(d2[idx])]
        a = self.rel[idx] * np.exp(-0.5 * self.prec[idx] * d2[idx])
        s = a.sum()
        a = a / s if s > 1e-12 else np.full(len(idx), 1.0 / len(idx), dtype=np.float32)
        return idx, a.astype(np.float32), d2[idx]

    def _forward(self, x, idx, a):
        z = (x[None, :] - self.C[idx]) @ self.R.T               # (k, local_dim)
        y = np.einsum("kol,kl->ko", self.A[idx], z) + self.b[idx]
        logits = (a[:, None] * y).sum(0)
        return logits, z

    def add_cell(self, x, target_byte, residual=None):
        c = self.cfg
        b = np.zeros(c.out_dim, dtype=np.float32)
        if residual is not None:
            b += residual
        b[int(target_byte)] += 1.0                               # predict target at center
        self.C = np.vstack([self.C, x[None].astype(np.float32)])
        self.A = np.concatenate([self.A, np.zeros((1, c.out_dim, c.local_dim), np.float32)])
        self.b = np.vstack([self.b, b[None]])
        self.prec = np.append(self.prec, np.float32(c.prec_init))
        self.rel = np.append(self.rel, np.float32(1.0))
        self.usage = np.append(self.usage, np.float32(1.0))
        self.resid_ema = np.append(self.resid_ema, np.float32(0.0))

    def predict_logits(self, x, candidates):
        if len(self.C) == 0:
            return np.zeros(self.cfg.out_dim, dtype=np.float32)
        idx, a, _ = self._active(x)
        logits, _ = self._forward(x, idx, a)
        return logits

    def predict_byte(self, x, candidates):
        logits = self.predict_logits(x, candidates)
        return int(candidates[int(np.argmax(logits[candidates]))])

    def predict_probs(self, x, candidates):
        return _softmax(self.predict_logits(x, candidates)[candidates])

    def train_one(self, x, target_byte, candidates):
        c = self.cfg
        if len(self.C) == 0:
            self.add_cell(x, target_byte)
            return
        idx, a, d2 = self._active(x)
        logits, z = self._forward(x, idx, a)
        p = _softmax(logits[candidates])
        pos = int(np.where(candidates == int(target_byte))[0][0])
        r = np.zeros(c.out_dim, dtype=np.float32)
        r[candidates] = -p
        r[int(target_byte)] += 1.0
        rnorm = float(np.linalg.norm(r[candidates]))

        # local residual updates on active charts (no backward pass)
        self.A[idx] += c.lr * np.einsum("k,o,kl->kol", a, r, z)
        self.b[idx] += c.lr * a[:, None] * r[None, :]
        self.C[idx] += c.lr_center * a[:, None] * (x[None, :] - self.C[idx])
        self.C[idx] /= np.linalg.norm(self.C[idx], axis=1, keepdims=True) + 1e-8
        self.rel[idx] = 0.99 * self.rel[idx] + 0.01 * float(p[pos])
        self.usage[idx] = 0.999 * self.usage[idx] + 0.001
        self.resid_ema[idx] = (1 - c.resid_decay) * self.resid_ema[idx] + c.resid_decay * rnorm

        # compression-style growth: novel + persistently-unexplained region only
        if (float(p[pos]) < c.grow_conf and d2[0] > c.grow_novel
                and self.resid_ema[idx[0]] > c.grow_resid and len(self.C) < c.max_cells):
            self.add_cell(x, target_byte, residual=r)
        elif len(self.C) > c.max_cells:
            self.prune()

    def prune(self):
        score = self.rel + 0.1 * self.usage - 0.05 * self.resid_ema
        keep = np.argsort(score)[-self.cfg.max_cells:]
        for name in ("C", "A", "b", "prec", "rel", "usage", "resid_ema"):
            setattr(self, name, getattr(self, name)[keep])
