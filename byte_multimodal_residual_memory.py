"""
Byte/Event-Level Multimodal Growing Residual Memory
---------------------------------------------------
A no-backprop prototype that treats text, images, audio, labels, and metadata as
one common stream of byte-level events.

Core ideas:
- Every modality becomes events: (modality, role, channel, position/time, byte value).
- A fixed hash/sketch encoder maps event streams into vectors. This encoder is not
  trained and uses no backpropagation.
- A growing residual memory learns local byte-output messages from residuals.
- The same memory can route over image bytes, text bytes, audio bytes, or mixtures.
- Output is byte-level: for MNIST-style classification, the answer is the byte
  ord('0')..ord('9'), not a special neural-net class head.

Examples:
    python byte_multimodal_residual_memory.py --mode toy_multimodal --epochs 2
    python byte_multimodal_residual_memory.py --mode mnist_byte --train_limit 6000 --test_limit 1000 --epochs 2

No loss.backward(), no optimizer.step(), no learned layered weights.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Stable hashing utilities
# -----------------------------------------------------------------------------

MASK64 = (1 << 64) - 1


def splitmix64(x: int) -> int:
    """Deterministic 64-bit integer mixer."""
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return (x ^ (x >> 31)) & MASK64


def stable_hash(*fields: int) -> int:
    """Hash a short tuple of small integers into a deterministic uint64."""
    h = 0xD1B54A32D192ED03
    for f in fields:
        h ^= splitmix64(int(f) & MASK64)
        h = splitmix64(h)
    return h


def normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(x))
    if n < eps:
        return x.astype(np.float32)
    return (x / n).astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return (e / (np.sum(e) + 1e-12)).astype(np.float32)


# -----------------------------------------------------------------------------
# Universal byte/event sketch encoder
# -----------------------------------------------------------------------------

# Modality tags. These are not separate encoders; they are fields in the same
# byte-event language so the memory can learn cross-modal bridges.
MOD_TEXT = 1
MOD_IMAGE = 2
MOD_AUDIO = 3
MOD_META = 4

# Role tags.
ROLE_OBSERVED = 1
ROLE_QUERY = 2
ROLE_TARGET = 3
ROLE_CONTEXT = 4


@dataclass
class ByteSketchConfig:
    dim: int = 256
    seed: int = 0
    image_threshold: int = 8
    include_zero_pixels: bool = False
    image_bin_size: int = 16
    audio_bin_size: int = 16
    use_bigrams: bool = True


class ByteEventSketcher:
    """
    Fixed, non-learned byte/event sketcher.

    It converts heterogeneous byte streams into one vector by hashing event fields.
    This is similar in spirit to feature hashing: cheap, sparse, deterministic, and
    not trained by backpropagation.
    """

    def __init__(self, cfg: ByteSketchConfig):
        self.cfg = cfg
        self.dim = int(cfg.dim)
        self.seed = int(cfg.seed) & 0xFFFF_FFFF

    def _add(self, z: np.ndarray, weight: float, *fields: int) -> None:
        h = stable_hash(self.seed, *fields)
        j = h % self.dim
        sign = 1.0 if ((h >> 63) & 1) == 0 else -1.0
        z[j] += np.float32(sign * weight)

    def add_text_bytes(self, z: np.ndarray, data: bytes, role: int = ROLE_QUERY) -> None:
        prev: Optional[int] = None
        for pos, b in enumerate(data):
            coarse = pos // 8
            fine = pos % 8
            # Exact positional byte event.
            self._add(z, 1.00, MOD_TEXT, role, 0, coarse, fine, int(b))
            # Position-free byte event, allowing text bytes to generalize across prompts.
            self._add(z, 0.35, MOD_TEXT, role, 1, 0, 0, int(b))
            # Adjacent byte relation, useful for words and markup-like prompts.
            if self.cfg.use_bigrams and prev is not None:
                self._add(z, 0.70, MOD_TEXT, role, 2, coarse, int(prev), int(b))
            prev = int(b)

    def add_image_bytes(self, z: np.ndarray, image: np.ndarray, role: int = ROLE_OBSERVED) -> None:
        if image.ndim != 2:
            raise ValueError(f"Expected a 2D grayscale image, got shape {image.shape}")
        h, w = image.shape
        bin_size = max(1, int(self.cfg.image_bin_size))
        threshold = int(self.cfg.image_threshold)
        for y in range(h):
            prev: Optional[int] = None
            for x in range(w):
                b = int(image[y, x])
                if (not self.cfg.include_zero_pixels) and b <= threshold:
                    prev = b
                    continue
                bbin = min(255, b) // bin_size
                amp = 0.25 + 0.75 * (min(255, b) / 255.0)
                # Exact-ish byte at exact position.
                self._add(z, amp, MOD_IMAGE, role, 0, y, x, b)
                # Binned byte at exact position.
                self._add(z, 0.75 * amp, MOD_IMAGE, role, 1, y, x, bbin)
                # Coarse patch event, encouraging local shape generalization.
                self._add(z, 0.60 * amp, MOD_IMAGE, role, 2, y // 4, x // 4, bbin)
                # Row and column events, weak global structure.
                self._add(z, 0.30 * amp, MOD_IMAGE, role, 3, y, 0, bbin)
                self._add(z, 0.30 * amp, MOD_IMAGE, role, 4, 0, x, bbin)
                if self.cfg.use_bigrams and prev is not None:
                    self._add(z, 0.30 * amp, MOD_IMAGE, role, 5, y, int(prev) // bin_size, bbin)
                prev = b

    def add_audio_bytes(self, z: np.ndarray, audio: np.ndarray | bytes, role: int = ROLE_OBSERVED) -> None:
        if isinstance(audio, bytes):
            values = np.frombuffer(audio, dtype=np.uint8)
        else:
            values = np.asarray(audio, dtype=np.uint8).reshape(-1)
        bin_size = max(1, int(self.cfg.audio_bin_size))
        prev: Optional[int] = None
        for t, b0 in enumerate(values):
            b = int(b0)
            bbin = b // bin_size
            coarse_t = t // 8
            fine_t = t % 8
            amp = 0.5 + 0.5 * (abs(b - 127) / 128.0)
            self._add(z, amp, MOD_AUDIO, role, 0, coarse_t, fine_t, b)
            self._add(z, 0.65 * amp, MOD_AUDIO, role, 1, coarse_t, fine_t, bbin)
            self._add(z, 0.35 * amp, MOD_AUDIO, role, 2, coarse_t, 0, bbin)
            if self.cfg.use_bigrams and prev is not None:
                self._add(z, 0.55 * amp, MOD_AUDIO, role, 3, coarse_t, int(prev) // bin_size, bbin)
            prev = b

    def add_meta_byte(self, z: np.ndarray, key: int, value: int, role: int = ROLE_CONTEXT) -> None:
        self._add(z, 1.0, MOD_META, role, 0, int(key), 0, int(value) & 255)

    def encode(
        self,
        *,
        image: Optional[np.ndarray] = None,
        text: Optional[bytes | str] = None,
        audio: Optional[np.ndarray | bytes] = None,
        meta: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> np.ndarray:
        z = np.zeros(self.dim, dtype=np.float32)
        if image is not None:
            self.add_image_bytes(z, np.asarray(image, dtype=np.uint8), ROLE_OBSERVED)
        if text is not None:
            if isinstance(text, str):
                text = text.encode("utf-8", errors="replace")
            self.add_text_bytes(z, text, ROLE_QUERY)
        if audio is not None:
            self.add_audio_bytes(z, audio, ROLE_OBSERVED)
        if meta is not None:
            for k, v in meta:
                self.add_meta_byte(z, k, v, ROLE_CONTEXT)
        return normalize(z)


# -----------------------------------------------------------------------------
# Growing residual memory with byte-level outputs
# -----------------------------------------------------------------------------

DIGIT_BYTES = np.array([ord(str(i)) for i in range(10)], dtype=np.int64)


@dataclass
class ByteGRMConfig:
    dim: int = 256
    output_bytes: int = 256
    k: int = 24
    steps: int = 4
    lr_msg: float = 0.35
    lr_center: float = 0.035
    lr_q: float = 0.035
    add_conf: float = 0.58
    add_dist: float = 1.25
    max_neurons: int = 6000
    min_sigma: float = 0.35
    init_sigma: float = 1.0
    state_mix: float = 0.10
    explore_ops: float = 0.04
    seed: int = 0


class ByteGrowingResidualMemory:
    """Local residual memory with byte-output messages."""

    def __init__(self, cfg: ByteGRMConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.C = np.empty((0, cfg.dim), dtype=np.float32)            # centers
        self.M = np.empty((0, cfg.output_bytes), dtype=np.float32)   # byte-message logits
        self.sigma = np.empty((0,), dtype=np.float32)
        self.rel = np.empty((0,), dtype=np.float32)
        self.usage = np.empty((0,), dtype=np.float32)
        self.age = np.empty((0,), dtype=np.int32)
        self.op_q = np.empty((0, 4), dtype=np.float32)

    def _nearest(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.C) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        d2 = np.einsum("ij,ij->i", self.C - z, self.C - z)
        k = min(int(self.cfg.k), len(d2))
        idx = np.argpartition(d2, k - 1)[:k]
        idx = idx[np.argsort(d2[idx])]
        return idx.astype(np.int64), d2[idx].astype(np.float32)

    def _op_multiplier(self, op: int, d2: float, sig: float, rel: float, step: int) -> float:
        # op 0: normal byte message
        # op 1: sharper locality gate
        # op 2: reliability-amplified message
        # op 3: recursive/late-step amplifier, useful when multiple passes clarify state
        if op == 0:
            return 1.0
        if op == 1:
            return float(1.0 / (1.0 + d2 / (sig * sig + 1e-6)))
        if op == 2:
            return float(np.clip(0.5 + rel, 0.2, 1.6))
        return float(0.75 + 0.25 * step)

    def _route(self, z0: np.ndarray, train: bool = False):
        if len(self.C) == 0:
            return np.zeros(self.cfg.output_bytes, dtype=np.float32), []

        state = z0.copy()
        logits = np.zeros(self.cfg.output_bytes, dtype=np.float32)
        path = []
        repeat_count: dict[int, int] = {}

        for step in range(int(self.cfg.steps)):
            idx, d2 = self._nearest(state)
            if len(idx) == 0:
                break

            raw = []
            ops = []
            for j, dist2 in zip(idx, d2):
                if train and self.rng.random() < self.cfg.explore_ops:
                    op = int(self.rng.integers(0, self.op_q.shape[1]))
                else:
                    op = int(np.argmax(self.op_q[j]))
                sig = max(float(self.sigma[j]), self.cfg.min_sigma)
                rel = max(float(self.rel[j]), 0.03)
                repeat_penalty = 1.0 / (1.0 + repeat_count.get(int(j), 0))
                a = np.exp(-float(dist2) / (2.0 * sig * sig)) * rel
                a *= self._op_multiplier(op, float(dist2), sig, rel, step)
                a *= repeat_penalty
                raw.append(a)
                ops.append(op)

            a = np.asarray(raw, dtype=np.float32)
            if float(a.sum()) <= 1e-12:
                a[:] = 1.0 / len(a)
            else:
                a /= float(a.sum()) + 1e-12

            step_gain = 1.0 / (1.0 + 0.25 * step)
            logits += step_gain * np.sum(a[:, None] * self.M[idx], axis=0)

            # Recursive state update: pull state toward the local activated manifold.
            local_center = np.sum(a[:, None] * self.C[idx], axis=0)
            state = (1.0 - self.cfg.state_mix) * state + self.cfg.state_mix * local_center
            state = normalize(state)

            for j, aj, op in zip(idx, a, ops):
                ji = int(j)
                repeat_count[ji] = repeat_count.get(ji, 0) + 1
                path.append((ji, float(aj * step_gain), int(op)))

        return logits, path

    def _masked_probs(self, logits: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        return softmax(logits[candidates])

    def _residual(self, logits: np.ndarray, target_byte: int, candidates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p = self._masked_probs(logits, candidates)
        target_local = np.zeros(len(candidates), dtype=np.float32)
        match = np.where(candidates == int(target_byte))[0]
        if len(match) != 1:
            raise ValueError("target_byte must appear exactly once in candidates")
        target_local[int(match[0])] = 1.0
        residual = np.zeros(self.cfg.output_bytes, dtype=np.float32)
        residual[candidates] = target_local - p
        return residual, p

    def add_neuron(self, z: np.ndarray, target_byte: int, residual: Optional[np.ndarray] = None) -> None:
        if residual is None:
            residual = np.zeros(self.cfg.output_bytes, dtype=np.float32)
            residual[int(target_byte)] = 1.0
        msg = residual.astype(np.float32).copy()
        msg[int(target_byte)] += 1.0
        self.C = np.vstack([self.C, z[None].astype(np.float32)])
        self.M = np.vstack([self.M, msg[None].astype(np.float32)])
        self.sigma = np.append(self.sigma, np.float32(self.cfg.init_sigma))
        self.rel = np.append(self.rel, np.float32(1.0))
        self.usage = np.append(self.usage, np.float32(1.0))
        self.age = np.append(self.age, np.int32(0))
        self.op_q = np.vstack([self.op_q, np.zeros((1, 4), dtype=np.float32)])

    def train_one(self, z: np.ndarray, target_byte: int, candidates: np.ndarray = DIGIT_BYTES) -> None:
        if len(self.C) == 0:
            self.add_neuron(z, target_byte)
            return

        logits, path = self._route(z, train=True)
        residual, p = self._residual(logits, int(target_byte), candidates)

        if not path:
            self.add_neuron(z, target_byte, residual)
            return

        # Local operation credit and combined activation per neuron.
        total_a: dict[int, float] = {}
        for j, aj, op in path:
            total_a[j] = total_a.get(j, 0.0) + aj
            reward = float(np.dot(self.M[j], residual) * aj)
            self.op_q[j, op] = (1.0 - self.cfg.lr_q) * self.op_q[j, op] + self.cfg.lr_q * reward

        true_pos = int(np.where(candidates == int(target_byte))[0][0])
        true_prob = float(p[true_pos])

        for j, aj in total_a.items():
            aj = float(min(1.0, aj))
            help_score = float(np.dot(self.M[j], residual) * aj)
            # Local residual byte-message update. This is the main learning rule.
            self.M[j] += self.cfg.lr_msg * aj * residual
            # Useful or class-consistent neurons move toward this event sketch.
            if help_score > -0.02 or int(np.argmax(self.M[j][candidates])) == true_pos:
                self.C[j] += self.cfg.lr_center * aj * (z - self.C[j])
                self.C[j] = normalize(self.C[j])
            self.rel[j] = 0.995 * self.rel[j] + 0.005 * (1.0 / (1.0 + np.exp(-help_score)))
            self.usage[j] = 0.999 * self.usage[j] + 0.001
            self.age[j] += 1

        idx, d2 = self._nearest(z)
        nearest_dist = float(np.sqrt(d2[0])) if len(d2) else np.inf
        if true_prob < self.cfg.add_conf or nearest_dist > self.cfg.add_dist:
            self.add_neuron(z, target_byte, residual)

        if len(self.C) > self.cfg.max_neurons:
            self.prune()

    def predict_byte(self, z: np.ndarray, candidates: np.ndarray = DIGIT_BYTES) -> int:
        logits, _ = self._route(z, train=False)
        local = logits[candidates]
        return int(candidates[int(np.argmax(local))])

    def predict_digit(self, z: np.ndarray) -> int:
        return self.predict_byte(z, DIGIT_BYTES) - ord("0")

    def prune(self) -> None:
        vote_strength = np.linalg.norm(self.M, axis=1)
        # Entropy-like confusion penalty over digit bytes.
        digit_logits = self.M[:, DIGIT_BYTES]
        digit_probs = np.exp(digit_logits - digit_logits.max(axis=1, keepdims=True))
        digit_probs /= digit_probs.sum(axis=1, keepdims=True) + 1e-12
        entropy = -np.sum(digit_probs * np.log(digit_probs + 1e-12), axis=1) / np.log(10)
        score = self.rel + 0.15 * self.usage + 0.03 * vote_strength - 0.08 * entropy
        keep = np.argsort(score)[-int(self.cfg.max_neurons):]
        self.C = self.C[keep]
        self.M = self.M[keep]
        self.sigma = self.sigma[keep]
        self.rel = self.rel[keep]
        self.usage = self.usage[keep]
        self.age = self.age[keep]
        self.op_q = self.op_q[keep]

    def fit_digits(self, X: np.ndarray, y: np.ndarray, epochs: int = 1) -> None:
        for ep in range(epochs):
            order = self.rng.permutation(len(X))
            correct = 0
            for n, i in enumerate(order, start=1):
                pred = self.predict_digit(X[i]) if len(self.C) else -1
                correct += int(pred == int(y[i]))
                self.train_one(X[i], ord(str(int(y[i]))), DIGIT_BYTES)
                if n % 1000 == 0:
                    print(
                        f"epoch {ep + 1} seen {n:5d} neurons {len(self.C):5d} "
                        f"online_acc {correct / n:.3f}"
                    )

    def score_digits(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = np.array([self.predict_digit(x) for x in X], dtype=np.int64)
        return float(np.mean(preds == y))


# -----------------------------------------------------------------------------
# Datasets: MNIST-as-bytes and a built-in multimodal toy dataset
# -----------------------------------------------------------------------------

PROMPTS = [
    b"answer digit byte:",
    b"what digit is in the image?",
    b"classify visual bytes ->",
    b"read pixels; output 0-9:",
]


def load_mnist_byte_features(
    train_limit: Optional[int],
    test_limit: Optional[int],
    sketcher: ByteEventSketcher,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from torchvision.datasets import MNIST

    train = MNIST(root="./data", train=True, download=True)
    test = MNIST(root="./data", train=False, download=True)

    X_train_img = train.data.numpy().astype(np.uint8)
    y_train = train.targets.numpy().astype(np.int64)
    X_test_img = test.data.numpy().astype(np.uint8)
    y_test = test.targets.numpy().astype(np.int64)

    if train_limit is not None:
        X_train_img, y_train = X_train_img[:train_limit], y_train[:train_limit]
    if test_limit is not None:
        X_test_img, y_test = X_test_img[:test_limit], y_test[:test_limit]

    def encode_many(images: np.ndarray, split_tag: int) -> np.ndarray:
        Z = np.zeros((len(images), sketcher.dim), dtype=np.float32)
        for i, img in enumerate(images):
            prompt = PROMPTS[i % len(PROMPTS)]
            # Meta split tag is deliberately weak; it demonstrates metadata bytes.
            Z[i] = sketcher.encode(image=img, text=prompt, meta=[(1, split_tag)])
            if (i + 1) % 2000 == 0:
                print(f"encoded {i + 1} byte/event samples")
        return Z

    return encode_many(X_train_img, 1), y_train, encode_many(X_test_img, 2), y_test


SEGMENTS = {
    0: "abcedf".replace("e", "e"),  # explicit but harmless; see map below
    1: "bc",
    2: "abged",
    3: "abgcd",
    4: "fgbc",
    5: "afgcd",
    6: "afgecd",
    7: "abc",
    8: "abcdefg",
    9: "abfgcd",
}

# Segment coordinate map for an 8x8 seven-segment-like byte image.
SEG_COORDS = {
    "a": [(0, x) for x in range(2, 6)],
    "b": [(y, 6) for y in range(1, 4)],
    "c": [(y, 6) for y in range(4, 7)],
    "d": [(7, x) for x in range(2, 6)],
    "e": [(y, 1) for y in range(4, 7)],
    "f": [(y, 1) for y in range(1, 4)],
    "g": [(3, x) for x in range(2, 6)],
}


def seven_segment_image(label: int, rng: np.random.Generator) -> np.ndarray:
    img = rng.integers(0, 18, size=(8, 8), dtype=np.uint8)
    for seg in SEGMENTS[int(label)]:
        for y, x in SEG_COORDS[seg]:
            val = int(rng.integers(200, 256))
            img[y, x] = np.uint8(val)
    # Add light salt noise.
    for _ in range(int(rng.integers(0, 4))):
        y = int(rng.integers(0, 8))
        x = int(rng.integers(0, 8))
        img[y, x] = np.uint8(rng.integers(40, 180))
    return img


def digit_audio(label: int, rng: np.random.Generator, length: int = 48) -> np.ndarray:
    t = np.arange(length, dtype=np.float32)
    freq = float(label + 1)
    wave = 127.0 + 62.0 * np.sin(2.0 * np.pi * freq * t / length)
    wave += rng.normal(0.0, 5.0, size=length)
    return np.clip(wave, 0, 255).astype(np.uint8)


def make_toy_multimodal_features(
    n_train: int,
    n_test: int,
    sketcher: ByteEventSketcher,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Built-in multimodal toy task:
    - image bytes: noisy seven-segment drawing
    - audio bytes: noisy waveform whose frequency encodes the same digit
    - text bytes: a query prompt, not the answer
    - target: ASCII byte '0'..'9'
    """
    rng = np.random.default_rng(seed)

    def make_split(n: int, split_tag: int) -> Tuple[np.ndarray, np.ndarray]:
        Z = np.zeros((n, sketcher.dim), dtype=np.float32)
        y = np.zeros(n, dtype=np.int64)
        for i in range(n):
            label = int(rng.integers(0, 10))
            img = seven_segment_image(label, rng)
            aud = digit_audio(label, rng)
            prompt = PROMPTS[i % len(PROMPTS)]
            Z[i] = sketcher.encode(image=img, audio=aud, text=prompt, meta=[(1, split_tag)])
            y[i] = label
        return Z, y

    return (*make_split(n_train, 1), *make_split(n_test, 2))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["toy_multimodal", "mnist_byte"], default="toy_multimodal")
    parser.add_argument("--train_limit", type=int, default=6000)
    parser.add_argument("--test_limit", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--max_neurons", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image_threshold", type=int, default=8)
    parser.add_argument("--include_zero_pixels", action="store_true")
    args = parser.parse_args()

    sketcher = ByteEventSketcher(
        ByteSketchConfig(
            dim=args.dim,
            seed=args.seed,
            image_threshold=args.image_threshold,
            include_zero_pixels=bool(args.include_zero_pixels),
        )
    )

    if args.mode == "mnist_byte":
        X_train, y_train, X_test, y_test = load_mnist_byte_features(args.train_limit, args.test_limit, sketcher)
    else:
        X_train, y_train, X_test, y_test = make_toy_multimodal_features(
            int(args.train_limit), int(args.test_limit), sketcher, int(args.seed)
        )

    cfg = ByteGRMConfig(
        dim=args.dim,
        k=args.k,
        steps=args.steps,
        max_neurons=args.max_neurons,
        seed=args.seed,
    )
    model = ByteGrowingResidualMemory(cfg)
    model.fit_digits(X_train, y_train, epochs=args.epochs)
    acc = model.score_digits(X_test, y_test)
    print(
        f"mode={args.mode} test_acc={acc:.4f} neurons={len(model.C)} "
        f"byte_level=True multimodal=True no_backprop=True"
    )


if __name__ == "__main__":
    main()
