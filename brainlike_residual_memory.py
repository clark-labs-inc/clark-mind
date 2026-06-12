"""
Brain-like Residual Memory: local unsupervised feature cortex + residual memory
-------------------------------------------------------------------------------
This extends `byte_multimodal_residual_memory.py` WITHOUT introducing backprop.

The diagnosis from earlier runs: the byte-event sketch encoder is a *frozen random
hash*, so the model never learns representations and stalls on real MNIST (~69%)
while memorizing most of the training set as prototypes.

Fix (stays true to the model): insert one or more LOCAL, UNSUPERVISED feature
layers between the sketch encoder and the growing residual memory. Each layer is a
competitive k-winners-take-all Hebbian sparse-coding layer -- biologically this is
a sheet of cortical units with lateral inhibition and homeostatic duty-cycle
control. Learning is purely local:

    s   = W @ z                      # cosine similarity to each unit's prototype
    win = top-k(s + homeostatic_boost)   # lateral inhibition / k-WTA
    W[win] += lr * (z - W[win]); renormalize   # instar / competitive Hebbian
    duty-cycle update keeps every unit used (no dead/dominant units)

There is no loss.backward(), no optimizer, no gradient through layers. The output
is an overcomplete sparse code that the existing residual memory routes over.

Usage:
    # A/B both arms on real MNIST under identical seed/data:
    python brainlike_residual_memory.py --mode mnist_byte --arm both \
        --train_limit 6000 --test_limit 1000 --epochs 2

    # just the new brain-like arm:
    python brainlike_residual_memory.py --mode mnist_byte --arm hebbian
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# NumPy 2.0's float32 matmul SIMD path emits spurious over/invalid warnings even
# when all operands and results are finite (verified). Silence only those.
np.seterr(over="ignore", invalid="ignore", divide="ignore")

from byte_multimodal_residual_memory import (
    ByteSketchConfig,
    ByteEventSketcher,
    ByteGRMConfig,
    ByteGrowingResidualMemory,
    load_mnist_byte_features,
    make_toy_multimodal_features,
)


# -----------------------------------------------------------------------------
# Local unsupervised k-WTA Hebbian sparse-coding layer (a "cortical sheet")
# -----------------------------------------------------------------------------


@dataclass
class HebbianLayerConfig:
    in_dim: int
    n_features: int = 1024          # overcomplete dictionary size
    k_active: int = 32              # winners after lateral inhibition (k-WTA)
    lr: float = 0.05                # competitive Hebbian learning rate
    duty_decay: float = 0.02        # running duty-cycle adaptation speed
    boost_strength: float = 5.0     # homeostasis: pull rare units up, frequent down
    seed: int = 0


class HebbianSparseLayer:
    """One self-organizing competitive layer. No backprop -- local rules only."""

    def __init__(self, cfg: HebbianLayerConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        W = self.rng.standard_normal((cfg.n_features, cfg.in_dim)).astype(np.float32)
        W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-8
        self.W = W
        self.target = cfg.k_active / cfg.n_features
        # duty cycle = running fraction of inputs for which a unit wins
        self.freq = np.full(cfg.n_features, self.target, dtype=np.float32)

    def init_from_data(self, X: np.ndarray) -> None:
        """Seed the dictionary from real inputs (k-means-style) -- the single
        biggest cure for dead units: every unit starts inside the data manifold."""
        sel = self.rng.integers(0, len(X), size=self.cfg.n_features)
        W = X[sel].astype(np.float32) + 0.01 * self.rng.standard_normal(
            (self.cfg.n_features, self.cfg.in_dim)).astype(np.float32)
        self.W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-8)

    def _winners(self, z: np.ndarray, train: bool) -> Tuple[np.ndarray, np.ndarray]:
        s = self.W @ z  # cosine similarity (rows of W and z are unit-norm)
        if train:
            # lateral-inhibition + homeostasis: boost under-used units so the
            # whole sheet specializes instead of a few units dominating.
            score = s + self.cfg.boost_strength * (self.target - self.freq)
        else:
            score = s
        k = self.cfg.k_active
        win = np.argpartition(score, -k)[-k:]
        return win, s

    def learn_one(self, z: np.ndarray) -> None:
        win, _ = self._winners(z, train=True)
        # competitive Hebbian (instar): each winner moves toward the input.
        self.W[win] += self.cfg.lr * (z - self.W[win])
        self.W[win] /= np.linalg.norm(self.W[win], axis=1, keepdims=True) + 1e-8
        # homeostatic duty-cycle update.
        self.freq *= 1.0 - self.cfg.duty_decay
        self.freq[win] += self.cfg.duty_decay

    def dead_fraction(self) -> float:
        return float(np.mean(self.freq < 0.2 * self.target))


def transform_layer(layer: HebbianSparseLayer, X: np.ndarray) -> np.ndarray:
    """Batched inference: sparse, rectified, L2-normalized code (no learning)."""
    S = X @ layer.W.T
    k = layer.cfg.k_active
    idx = np.argpartition(S, -k, axis=1)[:, -k:]
    rows = np.arange(S.shape[0])[:, None]
    out = np.zeros_like(S)
    out[rows, idx] = np.maximum(S[rows, idx], 0.0)  # rectified winner activations
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-8
    return out.astype(np.float32)


def build_stack(in_dim: int, dims: List[int], k_active: int, lr: float,
                seed: int) -> List[HebbianSparseLayer]:
    layers: List[HebbianSparseLayer] = []
    cur = in_dim
    for li, d in enumerate(dims):
        layers.append(HebbianSparseLayer(HebbianLayerConfig(
            in_dim=cur, n_features=d,
            k_active=min(k_active, max(1, d // 8)),
            lr=lr, seed=seed + 17 * li,
        )))
        cur = d
    return layers


def pretrain_stack(layers: List[HebbianSparseLayer], X: np.ndarray,
                   epochs: int, seed: int) -> np.ndarray:
    """Greedy layer-wise unsupervised pretraining (classic no-backprop deep stack)."""
    rng = np.random.default_rng(seed)
    cur = X
    for li, layer in enumerate(layers):
        layer.init_from_data(cur)  # data-driven dictionary seed for this layer
        for ep in range(epochs):
            for i in rng.permutation(len(cur)):
                layer.learn_one(cur[i])
            print(f"  [feat L{li}] epoch {ep + 1}/{epochs} "
                  f"dead_units {layer.dead_fraction():.3f}")
        cur = transform_layer(layer, cur)  # feed sparse code to next layer
    return cur


def transform_stack(layers: List[HebbianSparseLayer], X: np.ndarray) -> np.ndarray:
    cur = X
    for layer in layers:
        cur = transform_layer(layer, cur)
    return cur


# -----------------------------------------------------------------------------
# V1-style visual cortex: patch dictionary (simple cells) + spatial pooling
# (complex cells). Whitening + competitive Hebbian learning. Still no backprop.
# -----------------------------------------------------------------------------


def shift_images(imgs, dys, dxs):
    out = np.empty_like(imgs)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            m = (dys == dy) & (dxs == dx)
            if m.any():
                out[m] = np.roll(np.roll(imgs[m], dy, axis=1), dx, axis=2)
    return out


def augment_translation(imgs, y, copies, seed):
    """Brain-plausible data augmentation: extra copies under small random shifts.
    Gives translation robustness the pooling layer only partially provides."""
    rng = np.random.default_rng(seed + 5)
    out_i, out_y = [imgs], [y]
    for _ in range(copies):
        dys = rng.integers(-2, 3, size=len(imgs))
        dxs = rng.integers(-2, 3, size=len(imgs))
        out_i.append(shift_images(imgs, dys, dxs))
        out_y.append(y)
    return np.concatenate(out_i), np.concatenate(out_y)


def load_mnist_raw(train_limit, test_limit):
    from torchvision.datasets import MNIST
    tr = MNIST(root="./data", train=True, download=True)
    te = MNIST(root="./data", train=False, download=True)
    lt = None if train_limit is None or train_limit < 0 else train_limit
    le = None if test_limit is None or test_limit < 0 else test_limit
    Xtr = tr.data.numpy().astype(np.float32)[:lt] / 255.0
    ytr = tr.targets.numpy().astype(np.int64)[:lt]
    Xte = te.data.numpy().astype(np.float32)[:le] / 255.0
    yte = te.targets.numpy().astype(np.int64)[:le]
    return Xtr, ytr, Xte, yte


def extract_patches(imgs, patch, stride):
    n, h, w = imgs.shape
    cols = []
    for y in range(0, h - patch + 1, stride):
        for x in range(0, w - patch + 1, stride):
            cols.append(imgs[:, y:y + patch, x:x + patch].reshape(n, -1))
    grid = len(range(0, h - patch + 1, stride))
    return np.stack(cols, axis=1), grid  # (n, grid*grid, patch*patch), grid


def contrast_norm(P, eps=0.03):
    """Per-patch brightness/contrast normalization (Coates&Ng). Blank patches
    collapse toward zero instead of dominating the competition."""
    Pc = P - P.mean(axis=1, keepdims=True)
    Pc /= np.sqrt(P.var(axis=1, keepdims=True) + eps)
    return Pc.astype(np.float32)


def fit_zca(P, eps=0.1):
    P = contrast_norm(P)
    mu = P.mean(0)
    Pc = P - mu
    cov = (Pc.T @ Pc) / max(1, len(Pc))
    U, S, _ = np.linalg.svd(cov)
    Wz = (U @ np.diag(1.0 / np.sqrt(S + eps)) @ U.T).astype(np.float32)
    return mu.astype(np.float32), Wz


def whiten_norm(P, mu, Wz):
    Pw = (contrast_norm(P) - mu) @ Wz
    Pw /= np.linalg.norm(Pw, axis=1, keepdims=True) + 1e-8
    return Pw.astype(np.float32)


def _pool_codes(A, pool):
    grid = A.shape[1]
    step = max(1, grid // pool)
    feats = []
    for i in range(pool):                                       # complex-cell pooling
        for j in range(pool):
            region = A[:, i * step:(i + 1) * step, j * step:(j + 1) * step, :]
            feats.append(region.sum(axis=(1, 2)))
    F = np.concatenate(feats, axis=1)
    F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-8
    return F.astype(np.float32)


def v1_encode(imgs, mu, Wz, layer, patch, stride, pool, encode="triangle",
              chunk=2000):
    """Memory-safe batched encoding. `triangle` = Coates&Ng soft activation
    (atoms closer than average fire); `kwta` = hard top-k. Both are inference-only
    -- the dictionary itself was learned with competitive k-WTA Hebbian updates."""
    n = len(imgs)
    atoms = layer.W.shape[0]
    out = None
    grid = 0
    for s in range(0, n, chunk):
        P, grid = extract_patches(imgs[s:s + chunk], patch, stride)
        b = P.shape[0]
        flat = whiten_norm(P.reshape(-1, P.shape[-1]), mu, Wz)
        S = flat @ layer.W.T                                    # (b*grid*grid, atoms)
        if encode == "triangle":
            A = np.maximum(0.0, S - S.mean(axis=1, keepdims=True))
        else:
            k = layer.cfg.k_active
            idx = np.argpartition(S, -k, axis=1)[:, -k:]
            rows = np.arange(S.shape[0])[:, None]
            A = np.zeros_like(S)
            A[rows, idx] = np.maximum(0.0, S[rows, idx])
        F = _pool_codes(A.reshape(b, grid, grid, atoms), pool)
        if out is None:
            out = np.zeros((n, F.shape[1]), dtype=np.float32)
        out[s:s + chunk] = F
    return out, grid


def build_v1_features(args):
    print("\n--- V1 cortex (patch dictionary + pooling, no backprop) ---")
    Xtr, ytr, Xte, yte = load_mnist_raw(args.train_limit, args.test_limit)
    patch, stride, pool, atoms = args.patch, args.stride, args.pool, args.atoms
    t0 = time.time()
    # learn the dictionary on whitened training patches (subsampled for speed).
    # Only extract patches from a subset of images to bound memory at dense stride.
    rng = np.random.default_rng(args.seed)
    n_dict_imgs = min(len(Xtr), 6000)
    P, grid = extract_patches(Xtr[:n_dict_imgs], patch, stride)
    flat = P.reshape(-1, P.shape[-1])
    # drop near-blank patches so the dictionary spends capacity on real strokes
    keep = flat.std(axis=1) > 0.05
    flat = flat[keep] if keep.any() else flat
    sub = flat[rng.permutation(len(flat))[:min(len(flat), 200000)]]
    mu, Wz = fit_zca(sub)
    subw = whiten_norm(sub, mu, Wz)
    layer = HebbianSparseLayer(HebbianLayerConfig(
        in_dim=patch * patch, n_features=atoms,
        k_active=args.patch_k, lr=args.feat_lr,
        boost_strength=args.patch_boost, seed=args.seed))
    layer.init_from_data(subw)
    for ep in range(args.feat_epochs):
        for i in rng.permutation(len(subw)):
            layer.learn_one(subw[i])
        print(f"  [V1 dict] epoch {ep + 1}/{args.feat_epochs} "
              f"dead_units {layer.dead_fraction():.3f}")
    if getattr(args, "augment", 0) > 0:
        before = len(Xtr)
        Xtr, ytr = augment_translation(Xtr, ytr, args.augment, args.seed)
        print(f"  augmented train {before} -> {len(Xtr)} (x{1 + args.augment} via ±2px jitter)")
    chunk = int(getattr(args, "chunk", 1000))
    Ctr, grid = v1_encode(Xtr, mu, Wz, layer, patch, stride, pool, args.encode, chunk)
    Cte, _ = v1_encode(Xte, mu, Wz, layer, patch, stride, pool, args.encode, chunk)
    dim = Ctr.shape[1]
    print(f"  patch={patch} stride={stride} grid={grid} pool={pool} atoms={atoms} "
          f"encode={args.encode} -> feature dim={dim} ({time.time() - t0:.1f}s)")
    return Ctr, ytr, Cte, yte, dim


# -----------------------------------------------------------------------------
# Experiment driver
# -----------------------------------------------------------------------------


def load_features(args, sketcher: ByteEventSketcher):
    if args.mode == "mnist_byte":
        return load_mnist_byte_features(args.train_limit, args.test_limit, sketcher)
    return make_toy_multimodal_features(args.train_limit, args.test_limit, sketcher, args.seed)


def run_memory(Xtr, ytr, Xte, yte, dim, args, tag: str):
    cfg = ByteGRMConfig(
        dim=dim, k=args.k, steps=args.steps,
        max_neurons=args.max_neurons, seed=args.seed,
    )
    mem = ByteGrowingResidualMemory(cfg)
    t0 = time.time()
    mem.fit_digits(Xtr, ytr, epochs=args.epochs)
    acc = mem.score_digits(Xte, yte)
    dt = time.time() - t0
    print(f"[{tag}|memory] test_acc={acc:.4f} neurons={len(mem.C)} dim={dim} "
          f"train_time={dt:.1f}s")
    return acc, len(mem.C)


def run_linear(Xtr, ytr, Xte, yte, dim, args, tag: str, epochs: int = 15,
               lr: float = 0.2):
    """Single-layer local readout trained by the delta/Widrow-Hoff rule:
        W += lr * (onehot(y) - softmax(Wx)) (x)
    This is local, error-modulated Hebbian plasticity -- NOT backprop (no error is
    propagated through any hidden layer). Biologically: one sheet of output cells
    whose synapses change with a three-factor (pre x post-error x learning) signal."""
    epochs = int(getattr(args, "lin_epochs", epochs))
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    C = 10
    W = np.zeros((C, dim), dtype=np.float32)
    b = np.zeros(C, dtype=np.float32)
    for _ in range(epochs):
        for i in rng.permutation(len(Xtr)):
            z = W @ Xtr[i] + b
            z -= z.max()
            p = np.exp(z)
            p /= p.sum() + 1e-12
            e = -p
            e[int(ytr[i])] += 1.0
            W += lr * np.outer(e, Xtr[i])
            b += lr * e
    pred = (Xte @ W.T + b).argmax(1)
    acc = float((pred == yte).mean())
    print(f"[{tag}|linear] test_acc={acc:.4f} params={C * dim} dim={dim} "
          f"train_time={time.time() - t0:.1f}s")
    return acc, C * dim


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["mnist_byte", "toy_multimodal"], default="mnist_byte")
    p.add_argument("--arm", type=str, default="baseline,v1",
                   help="comma list of: baseline, sheet, v1 (v1 needs mnist_byte)")
    p.add_argument("--train_limit", type=int, default=6000)
    p.add_argument("--test_limit", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--dim", type=int, default=256)            # sketch dim
    p.add_argument("--k", type=int, default=24)               # memory routing fan-in
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--max_neurons", type=int, default=6000)
    p.add_argument("--seed", type=int, default=0)
    # whole-image Hebbian "sheet" cortex hyperparameters
    p.add_argument("--feat_dims", type=str, default="1024")
    p.add_argument("--k_active", type=int, default=32)
    p.add_argument("--feat_lr", type=float, default=0.05)
    p.add_argument("--feat_epochs", type=int, default=4)
    # V1 patch-cortex hyperparameters
    p.add_argument("--patch", type=int, default=7)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--pool", type=int, default=2)
    p.add_argument("--atoms", type=int, default=256)
    p.add_argument("--patch_k", type=int, default=3)
    p.add_argument("--patch_boost", type=float, default=25.0)
    p.add_argument("--encode", choices=["triangle", "kwta"], default="triangle")
    p.add_argument("--chunk", type=int, default=1000)
    p.add_argument("--lin_epochs", type=int, default=15)
    p.add_argument("--readout", type=str, default="memory,linear",
                   help="comma list of readouts to evaluate: memory, linear")
    args = p.parse_args()

    arms = [a.strip() for a in args.arm.split(",") if a.strip()]
    readouts = [r.strip() for r in args.readout.split(",") if r.strip()]

    # 1) build the feature representation for each requested arm (the "front end")
    feats = {}  # arm -> (Xtr, ytr, Xte, yte, dim)
    if any(a in ("baseline", "sheet") for a in arms):
        sketcher = ByteEventSketcher(ByteSketchConfig(dim=args.dim, seed=args.seed))
        print(f"encoding {args.mode} features (sketch dim={args.dim}) ...")
        Xtr, ytr, Xte, yte = load_features(args, sketcher)
        print(f"train={len(Xtr)} test={len(Xte)}")
        if "baseline" in arms:
            feats["baseline"] = (Xtr, ytr, Xte, yte, args.dim)
        if "sheet" in arms:
            dims = [int(x) for x in args.feat_dims.split(",") if x.strip()]
            print(f"\n--- whole-image Hebbian sheet {dims} k_active={args.k_active} ---")
            layers = build_stack(args.dim, dims, args.k_active, args.feat_lr, args.seed)
            t0 = time.time()
            pretrain_stack(layers, Xtr, args.feat_epochs, args.seed)
            Ctr = transform_stack(layers, Xtr)
            Cte = transform_stack(layers, Xte)
            print(f"  unsupervised pretrain+transform: {time.time() - t0:.1f}s")
            feats["sheet"] = (Ctr, ytr, Cte, yte, dims[-1])
    if "v1" in arms:
        feats["v1"] = build_v1_features(args)

    # 2) evaluate every (front end x readout) cell -- no backprop in any path
    runner = {"memory": run_memory, "linear": run_linear}
    results = {}
    for arm in arms:
        if arm not in feats:
            continue
        Xtr, ytr, Xte, yte, dim = feats[arm]
        for ro in readouts:
            print()
            results[(arm, ro)] = runner[ro](Xtr, ytr, Xte, yte, dim, args, arm)

    print("\n=== SUMMARY (no backprop in any arm) ===")
    print(f"  {'front end':10s} {'readout':8s} {'test_acc':>9s} {'size':>8s}")
    for arm in arms:
        for ro in readouts:
            if (arm, ro) in results:
                acc, size = results[(arm, ro)]
                print(f"  {arm:10s} {ro:8s} {acc:9.4f} {size:8d}")


if __name__ == "__main__":
    main()
