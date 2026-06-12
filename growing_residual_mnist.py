"""
Growing Residual Memory for MNIST
---------------------------------
A no-backprop classifier prototype:
- fixed unsupervised encoder: PCA or random projection
- dynamic neurons: local prototypes with class-message vectors
- residual learning: update only neurons that participated
- operation choice: each neuron learns which message operation helps
- bounded recursion: neurons can fire in multiple routing steps with a repeat penalty
- forgetting/pruning: weak / unused / harmful neurons are removed when capacity is exceeded

Run:
    pip install numpy scikit-learn torch torchvision
    python growing_residual_mnist.py --train_limit 12000 --test_limit 2000 --epochs 2
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)


@dataclass
class GRMConfig:
    dim: int = 64
    classes: int = 10
    k: int = 16
    steps: int = 3
    lr_vote: float = 0.35
    lr_center: float = 0.04
    lr_q: float = 0.03
    add_conf: float = 0.62
    add_dist: float = 1.35
    max_neurons: int = 4000
    min_sigma: float = 0.35
    init_sigma: float = 1.0
    state_mix: float = 0.12
    explore_ops: float = 0.03
    seed: int = 0


class GrowingResidualMemory:
    """A small non-backprop, residual-growing memory network."""

    def __init__(self, cfg: GRMConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.C = np.empty((0, cfg.dim), dtype=np.float32)          # centers / prototypes
        self.V = np.empty((0, cfg.classes), dtype=np.float32)      # class-message vectors
        self.sigma = np.empty((0,), dtype=np.float32)              # receptive field radius
        self.rel = np.empty((0,), dtype=np.float32)                # reliability
        self.usage = np.empty((0,), dtype=np.float32)              # use trace
        self.age = np.empty((0,), dtype=np.int32)
        self.op_q = np.empty((0, 3), dtype=np.float32)             # operation scores per neuron

    def _target(self, y: int) -> np.ndarray:
        # Slightly negative non-target entries make the residual also learn inhibition.
        t = np.full(self.cfg.classes, -0.1 / (self.cfg.classes - 1), dtype=np.float32)
        t[y] = 1.0
        return t

    def _nearest(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.C) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        d2 = np.einsum("ij,ij->i", self.C - z, self.C - z)
        k = min(self.cfg.k, len(d2))
        idx = np.argpartition(d2, k - 1)[:k]
        idx = idx[np.argsort(d2[idx])]
        return idx.astype(np.int64), d2[idx].astype(np.float32)

    def _op_multiplier(self, op: int, d2: float, sig: float, rel: float) -> float:
        # Operation 0: normal message.
        # Operation 1: sharper locality gate.
        # Operation 2: reliability-amplified message.
        if op == 0:
            return 1.0
        if op == 1:
            return float(1.0 / (1.0 + d2 / (sig * sig + 1e-6)))
        return float(np.clip(0.5 + rel, 0.2, 1.5))

    def _route(self, z0: np.ndarray, train: bool = False):
        """Bounded recursive routing. Returns logits and path events."""
        if len(self.C) == 0:
            return np.zeros(self.cfg.classes, dtype=np.float32), []

        state = z0.copy()
        logits = np.zeros(self.cfg.classes, dtype=np.float32)
        path = []
        repeat_count = {}

        for step in range(self.cfg.steps):
            idx, d2 = self._nearest(state)
            if len(idx) == 0:
                break

            raw_a = []
            chosen_ops = []
            for j, dist2 in zip(idx, d2):
                if train and self.rng.random() < self.cfg.explore_ops:
                    op = int(self.rng.integers(0, 3))
                else:
                    op = int(np.argmax(self.op_q[j]))

                sig = max(float(self.sigma[j]), self.cfg.min_sigma)
                rel = max(float(self.rel[j]), 0.05)
                mult = self._op_multiplier(op, float(dist2), sig, rel)
                repeat_penalty = 1.0 / (1.0 + repeat_count.get(int(j), 0))
                a = np.exp(-float(dist2) / (2.0 * sig * sig)) * rel * mult * repeat_penalty
                raw_a.append(a)
                chosen_ops.append(op)

            a = np.asarray(raw_a, dtype=np.float32)
            if float(a.sum()) <= 1e-12:
                a = np.ones_like(a) / len(a)
            else:
                a = a / (a.sum() + 1e-12)

            # Add messages. Later recursive steps get slightly less weight.
            step_gain = 1.0 / (1.0 + 0.25 * step)
            logits += step_gain * np.sum(a[:, None] * self.V[idx], axis=0)

            # State update: move a little toward the currently activated local manifold.
            local_center = np.sum(a[:, None] * self.C[idx], axis=0)
            state = (1.0 - self.cfg.state_mix) * state + self.cfg.state_mix * local_center
            state = state / (np.linalg.norm(state) + 1e-8)

            for j, aj, op in zip(idx, a, chosen_ops):
                ji = int(j)
                repeat_count[ji] = repeat_count.get(ji, 0) + 1
                path.append((ji, float(aj * step_gain), int(op)))

        return logits, path

    def add_neuron(self, z: np.ndarray, y: int, residual: Optional[np.ndarray] = None) -> None:
        if residual is None:
            residual = self._target(y)
        v = residual.astype(np.float32).copy()
        v[y] += 1.0  # give the true class an initial attractor
        self.C = np.vstack([self.C, z[None].astype(np.float32)])
        self.V = np.vstack([self.V, v[None].astype(np.float32)])
        self.sigma = np.append(self.sigma, np.float32(self.cfg.init_sigma))
        self.rel = np.append(self.rel, np.float32(1.0))
        self.usage = np.append(self.usage, np.float32(1.0))
        self.age = np.append(self.age, np.int32(0))
        self.op_q = np.vstack([self.op_q, np.zeros((1, 3), dtype=np.float32)])

    def train_one(self, z: np.ndarray, y: int) -> None:
        target = self._target(y)
        if len(self.C) == 0:
            self.add_neuron(z, y, target)
            return

        logits, path = self._route(z, train=True)
        p = softmax(logits)
        residual = target - p.astype(np.float32)

        if not path:
            self.add_neuron(z, y, residual)
            return

        # Combine duplicate recursive events for each neuron, but update operation Q per event.
        total_a = {}
        for j, aj, op in path:
            total_a[j] = total_a.get(j, 0.0) + aj
            # Local credit: does this neuron's selected operation point in the residual direction?
            help_score = float(np.dot(self.V[j], residual) * aj)
            self.op_q[j, op] = (1.0 - self.cfg.lr_q) * self.op_q[j, op] + self.cfg.lr_q * help_score

        for j, aj in total_a.items():
            aj = float(min(1.0, aj))
            help_score = float(np.dot(self.V[j], residual) * aj)

            # Residual message learning: no chain rule, no backward pass.
            self.V[j] += self.cfg.lr_vote * aj * residual

            # Move the receptive field toward examples it seems to help.
            # Harmful neurons are not pulled toward the sample; their reliability falls instead.
            if help_score > -0.02 or int(np.argmax(self.V[j])) == y:
                self.C[j] += self.cfg.lr_center * aj * (z - self.C[j])
                self.C[j] /= np.linalg.norm(self.C[j]) + 1e-8

            self.rel[j] = 0.995 * self.rel[j] + 0.005 * (1.0 / (1.0 + np.exp(-help_score)))
            self.usage[j] = 0.999 * self.usage[j] + 0.001
            self.age[j] += 1

        idx, d2 = self._nearest(z)
        nearest_dist = float(np.sqrt(d2[0])) if len(d2) else np.inf

        # Growth rule: create a new local expert when residual remains high or routing is far.
        if p[y] < self.cfg.add_conf or nearest_dist > self.cfg.add_dist:
            self.add_neuron(z, y, residual)

        if len(self.C) > self.cfg.max_neurons:
            self.prune()

    def prune(self) -> None:
        # Forget weak, unused, or harmful neurons. This is deletion, not gradient decay.
        vote_strength = np.linalg.norm(self.V, axis=1)
        score = self.rel + 0.15 * self.usage + 0.03 * vote_strength
        keep = np.argsort(score)[-self.cfg.max_neurons:]
        self.C = self.C[keep]
        self.V = self.V[keep]
        self.sigma = self.sigma[keep]
        self.rel = self.rel[keep]
        self.usage = self.usage[keep]
        self.age = self.age[keep]
        self.op_q = self.op_q[keep]

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1) -> None:
        for ep in range(epochs):
            order = self.rng.permutation(len(X))
            correct = 0
            for n, i in enumerate(order, start=1):
                pred = self.predict_one(X[i]) if len(self.C) else -1
                correct += int(pred == int(y[i]))
                self.train_one(X[i], int(y[i]))
                if n % 2000 == 0:
                    print(f"epoch {ep+1} seen {n:5d} neurons {len(self.C):5d} online_acc {correct/n:.3f}")

    def predict_one(self, z: np.ndarray) -> int:
        logits, _ = self._route(z, train=False)
        return int(np.argmax(logits))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = np.array([self.predict_one(x) for x in X], dtype=np.int64)
        return float(np.mean(preds == y))


def load_mnist(train_limit: Optional[int], test_limit: Optional[int]):
    from torchvision.datasets import MNIST

    root = "./data"
    train = MNIST(root=root, train=True, download=True)
    test = MNIST(root=root, train=False, download=True)

    X_train = train.data.numpy().astype(np.float32).reshape(-1, 784) / 255.0
    y_train = train.targets.numpy().astype(np.int64)
    X_test = test.data.numpy().astype(np.float32).reshape(-1, 784) / 255.0
    y_test = test.targets.numpy().astype(np.int64)

    if train_limit is not None:
        X_train, y_train = X_train[:train_limit], y_train[:train_limit]
    if test_limit is not None:
        X_test, y_test = X_test[:test_limit], y_test[:test_limit]
    return X_train, y_train, X_test, y_test


def make_features(X_train: np.ndarray, X_test: np.ndarray, dim: int, seed: int):
    # PCA is an unsupervised fixed encoder, not backprop. If sklearn is unavailable,
    # fall back to a fixed random projection.
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=dim, whiten=True, random_state=seed)
        Z_train = pca.fit_transform(X_train).astype(np.float32)
        Z_test = pca.transform(X_test).astype(np.float32)
    except Exception as exc:  # pragma: no cover
        print(f"PCA unavailable ({exc}); using fixed random projection.")
        rng = np.random.default_rng(seed)
        W = rng.normal(0.0, 1.0 / np.sqrt(X_train.shape[1]), size=(X_train.shape[1], dim)).astype(np.float32)
        Z_train = np.tanh(X_train @ W)
        Z_test = np.tanh(X_test @ W)

    return normalize_rows(Z_train.astype(np.float32)), normalize_rows(Z_test.astype(np.float32))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_limit", type=int, default=12000)
    parser.add_argument("--test_limit", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--max_neurons", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_mnist(args.train_limit, args.test_limit)
    Z_train, Z_test = make_features(X_train, X_test, dim=args.dim, seed=args.seed)

    cfg = GRMConfig(
        dim=args.dim,
        k=args.k,
        steps=args.steps,
        max_neurons=args.max_neurons,
        seed=args.seed,
    )
    model = GrowingResidualMemory(cfg)
    model.fit(Z_train, y_train, epochs=args.epochs)
    acc = model.score(Z_test, y_test)
    print(f"test_acc={acc:.4f} neurons={len(model.C)} no_backprop=True")


if __name__ == "__main__":
    main()
