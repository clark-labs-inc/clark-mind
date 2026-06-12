"""
Byte-level Language Modeling with the Growing Residual Memory (no backprop)
---------------------------------------------------------------------------
Tests whether the SAME architecture used for vision generalizes to sequence
prediction. Language modeling is actually the most natural fit: the memory already
predicts a *byte* from a byte-event sketch, so next-byte (next-character)
prediction needs no new output machinery.

Pipeline (no loss.backward, no optimizer, local rules only):

    context (last W chars)
      -> CharLMSketcher: hash suffix n-grams + recent positions  (byte-event encoder)
      -> readout predicts the next byte:
           (a) growing residual memory  (local residual + dynamic neurons), or
           (b) local linear delta-rule readout (error-modulated Hebbian)
    residual = target - prediction -> local update

Baseline: an interpolated char n-gram (orders 0..K) with add-delta smoothing.

Metrics: next-char accuracy and bits-per-character (cross-entropy, the standard
language-model metric -- lower is better).

    python byte_language_model.py --train_chars 30000 --test_chars 5000
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from typing import List, Tuple

import numpy as np

np.seterr(over="ignore", invalid="ignore", divide="ignore")  # spurious f32 matmul warns

from byte_multimodal_residual_memory import (
    stable_hash, normalize, softmax,
    ByteGRMConfig, ByteGrowingResidualMemory,
)
from universal_chart_memory import ChartConfig, UniversalPredictiveChartMemory


# -----------------------------------------------------------------------------
# Byte-event context encoder for sequences (suffix n-grams + recent positions)
# -----------------------------------------------------------------------------


class CharLMSketcher:
    """Fixed (non-learned) hash sketch of a character context. Two byte-event
    families: suffix n-grams (shared long suffix -> very similar sketch, like an
    n-gram backoff) and recency-weighted positional bytes."""

    def __init__(self, dim=1024, max_order=8, max_pos=12, seed=0):
        self.dim = int(dim)
        self.max_order = int(max_order)
        self.max_pos = int(max_pos)
        self.seed = int(seed) & 0xFFFFFFFF

    def _add(self, z, weight, *fields):
        h = stable_hash(self.seed, *fields)
        z[h % self.dim] += np.float32((1.0 if ((h >> 63) & 1) == 0 else -1.0) * weight)

    def encode(self, ctx: bytes) -> np.ndarray:
        z = np.zeros(self.dim, dtype=np.float32)
        L = len(ctx)
        # suffix n-grams: longer shared context -> stronger match
        for n in range(1, self.max_order + 1):
            if L >= n:
                self._add(z, float(n), 7, n, *ctx[L - n:])
        # recency-weighted positional bytes (most recent char matters most)
        for off in range(min(L, self.max_pos)):
            self._add(z, 1.0 / (1.0 + off), 11, off, int(ctx[L - 1 - off]))
        return normalize(z)


class VSASketcher:
    """Vector-symbolic (MAP-style) context encoder. Each byte has a fixed random
    bipolar atom; a suffix n-gram is BOUND by elementwise product of position-
    permuted atoms (compositional, order-sensitive), and suffix n-grams are BUNDLED
    (summed). This represents structure algebraically rather than by hashing slots."""

    def __init__(self, dim=4096, max_order=8, seed=0):
        self.dim = int(dim)
        self.max_order = int(max_order)
        rng = np.random.default_rng(seed)
        self.V = rng.choice([-1.0, 1.0], size=(256, self.dim)).astype(np.float32)
        # precompute position-permuted atoms (roll by shift) for shifts 0..max_order-1
        self.rolled = np.stack([np.roll(self.V, s, axis=1)
                                for s in range(self.max_order)])  # (shift,256,dim)

    def encode(self, ctx: bytes) -> np.ndarray:
        z = np.zeros(self.dim, dtype=np.float32)
        L = len(ctx)
        for n in range(1, self.max_order + 1):
            if L < n:
                break
            g = np.ones(self.dim, dtype=np.float32)
            for j in range(n):                      # char ctx[L-n+j] at shift n-1-j
                g *= self.rolled[n - 1 - j, ctx[L - n + j]]
            z += float(n) * g                       # bundle, longer n-grams weighted up
        return normalize(z)


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------


def build_dataset(path, train_chars, test_chars, window, sketcher):
    data = open(path, "rb").read()
    data = data[: train_chars + test_chars + window + 1]
    vocab = sorted(set(data))
    byte2idx = {b: i for i, b in enumerate(vocab)}
    cand = np.array(vocab, dtype=np.int64)

    def make(lo, hi):
        Z = np.zeros((hi - lo, sketcher.dim), dtype=np.float32)
        yb = np.zeros(hi - lo, dtype=np.int64)   # target byte value
        for n, i in enumerate(range(lo, hi)):
            ctx = data[max(0, i - window):i]
            Z[n] = sketcher.encode(ctx)
            yb[n] = data[i]
        yi = np.array([byte2idx[b] for b in yb], dtype=np.int64)  # vocab index
        return Z, yb, yi

    a = window
    Ztr, ybtr, yitr = make(a, a + train_chars)
    Zte, ybte, yite = make(a + train_chars, a + train_chars + test_chars)
    return (Ztr, ybtr, yitr), (Zte, ybte, yite), cand, vocab, data


# -----------------------------------------------------------------------------
# Baseline: interpolated char n-gram (orders 0..K), add-delta smoothing
# -----------------------------------------------------------------------------


def ngram_baseline(data, a, train_chars, test_chars, vocab, K=6, delta=0.05):
    V = len(vocab)
    b2i = {b: i for i, b in enumerate(vocab)}
    tables = [defaultdict(lambda: np.zeros(V, dtype=np.float64)) for _ in range(K + 1)]
    for i in range(a, a + train_chars):
        nxt = b2i[data[i]]
        for n in range(K + 1):
            ctx = data[i - n:i]
            tables[n][bytes(ctx)][nxt] += 1.0

    def dist(ctx_full: bytes) -> np.ndarray:
        p = np.zeros(V, dtype=np.float64)
        wsum = 0.0
        for n in range(K + 1):
            ctx = ctx_full[len(ctx_full) - n:] if n else b""
            c = tables[n].get(bytes(ctx))
            if c is None:
                continue
            lam = float(n + 1)                       # favor longer matched context
            pn = (c + delta) / (c.sum() + delta * V)
            p += lam * pn
            wsum += lam
        if wsum == 0.0:
            return np.full(V, 1.0 / V)
        return p / wsum

    correct = 0.0
    bits = 0.0
    lo, hi = a + train_chars, a + train_chars + test_chars
    for i in range(lo, hi):
        p = dist(data[i - K:i])
        t = b2i[data[i]]
        correct += int(p.argmax() == t)
        bits += -np.log2(max(p[t], 1e-12))
    return correct / test_chars, bits / test_chars


# -----------------------------------------------------------------------------
# Predictive causal states (computational-mechanics / variable-order Markov):
# group histories by the FUTURE they predict. A deeper context becomes its own
# state only if its next-byte distribution diverges from its parent (split by
# future-disagreement); otherwise it merges into the parent (predictive
# equivalence). As eps->0 every context splits => exact n-gram in the limit.
# Pure counting + local divergence tests. No backprop, no gradients.
# -----------------------------------------------------------------------------


def _js(p, q):
    m = 0.5 * (p + q)
    return 0.5 * np.sum(p * np.log2(p / m + 1e-12)) + 0.5 * np.sum(q * np.log2(q / m + 1e-12))


def causal_state_lm(data, a, train_chars, test_chars, vocab,
                    D=10, eps=0.04, min_count=2, alpha=0.05, alpha_back=2.0):
    V = len(vocab)
    b2i = {b: i for i, b in enumerate(vocab)}
    counts = {}                                          # context bytes -> counts[V]
    for i in range(a, a + train_chars):
        nxt = b2i[data[i]]
        for m in range(D + 1):
            if i - m < 0:
                break
            key = data[i - m:i]
            arr = counts.get(key)
            if arr is None:
                arr = np.zeros(V, np.float64)
                counts[key] = arr
            arr[nxt] += 1.0

    def dist(arr):
        return (arr + alpha) / (arr.sum() + alpha * V)

    # split decision: keep a context as its own state iff its future distribution
    # diverges from its parent (the same context with the OLDEST char dropped).
    kept = set()
    for key, arr in counts.items():
        if len(key) == 0 or arr.sum() < min_count:
            continue
        parr = counts.get(key[1:])                       # parent = shorter recent suffix
        if parr is None:
            continue
        if _js(dist(arr), dist(parr)) > eps:
            kept.add(key)

    root = dist(counts[b""])

    def predict(h):
        p = root.copy()
        for m in range(1, D + 1):                        # backoff over KEPT states only
            if len(h) < m:
                break
            ctx = h[m and -m:]
            if ctx in kept:
                arr = counts[ctx]
                c = arr.sum()
                lam = c / (c + alpha_back)
                p = lam * dist(arr) + (1.0 - lam) * p
        return p

    correct, bits = 0, 0.0
    lo, hi = a + train_chars, a + train_chars + test_chars
    for i in range(lo, hi):
        p = predict(data[i - D:i])
        t = b2i[data[i]]
        correct += int(p.argmax() == t)
        bits += -np.log2(max(p[t], 1e-12))
    return correct / test_chars, bits / test_chars, len(kept) + 1, len(counts)


def hierarchical_causal_lm(data, a, train_chars, test_chars, vocab,
                           D=8, eps=0.08, min_count=2, alpha=0.05, alpha_back=2.0,
                           q_order=16, lam=0.5, chunk=False):
    """Two-level predictive-state hierarchy (tests 'hierarchy replaces depth').
    L0 = char causal states (as above). Each position gets the id of its deepest
    kept state -> a STATE-TRAJECTORY stream. L1 = a causal model over the last
    q state-ids predicting the next byte (longer, ABSTRACTED context than L0's
    char suffix). Final P = lam*L0 + (1-lam)*L1. Pure counting, no backprop."""
    V = len(vocab)
    b2i = {b: i for i, b in enumerate(vocab)}
    counts = {}
    for i in range(a, a + train_chars):
        nxt = b2i[data[i]]
        for m in range(D + 1):
            if i - m < 0:
                break
            key = data[i - m:i]
            arr = counts.get(key)
            if arr is None:
                arr = np.zeros(V, np.float64); counts[key] = arr
            arr[nxt] += 1.0

    def dist(arr):
        return (arr + alpha) / (arr.sum() + alpha * V)

    kept = set()
    for key, arr in counts.items():
        if len(key) == 0 or arr.sum() < min_count:
            continue
        parr = counts.get(key[1:])
        if parr is not None and _js(dist(arr), dist(parr)) > eps:
            kept.add(key)
    root = dist(counts[b""])
    ids = {b"": 0}
    for k in kept:
        ids[k] = len(ids)

    def l0(h):                                     # -> (state_id, P0)
        p = root.copy(); sid = 0
        for m in range(1, D + 1):
            if len(h) < m:
                break
            ctx = h[m and -m:]
            if ctx in kept:
                arr = counts[ctx]; c = arr.sum(); l = c / (c + alpha_back)
                p = l * dist(arr) + (1.0 - l) * p
                sid = ids[ctx]
        return sid, p

    # L1: state-trajectory -> next byte. If chunk=True, run-length-compress the
    # state stream so L1 sees genuinely longer-range (chunked) context.
    l1 = [defaultdict(lambda: np.zeros(V, np.float64)) for _ in range(q_order + 1)]

    def push(hist, sid):
        if chunk:
            if not hist or hist[-1] != sid:
                hist.append(sid)
        else:
            hist.append(sid)

    hist = []
    for pos in range(train_chars):
        sid = l0(data[a + pos - D:a + pos])[0]
        push(hist, sid)                                # include current state, then count
        nxt = b2i[data[a + pos]]
        for q in range(1, q_order + 1):
            if len(hist) < q:
                break
            l1[q][tuple(hist[-q:])][nxt] += 1.0

    def l1_dist(h):
        p = root.copy()
        for q in range(1, q_order + 1):
            if len(h) < q:
                break
            arr = l1[q].get(tuple(h[-q:]))
            if arr is not None and arr.sum() >= min_count:
                c = arr.sum(); l = c / (c + alpha_back)
                p = l * dist(arr) + (1.0 - l) * p
        return p

    correct0 = correctH = 0
    bits0 = bitsH = 0.0
    lo, hi = a + train_chars, a + train_chars + test_chars
    for i in range(lo, hi):
        sid, p0 = l0(data[i - D:i])
        push(hist, sid)                                # consistent with training
        p1 = l1_dist(hist)
        ph = lam * p0 + (1.0 - lam) * p1
        t = b2i[data[i]]
        correct0 += int(p0.argmax() == t); bits0 += -np.log2(max(p0[t], 1e-12))
        correctH += int(ph.argmax() == t); bitsH += -np.log2(max(ph[t], 1e-12))
    n = test_chars
    return ((correct0 / n, bits0 / n), (correctH / n, bitsH / n),
            len(kept) + 1, sum(len(t) for t in l1))


# -----------------------------------------------------------------------------
# Readouts (both no-backprop)
# -----------------------------------------------------------------------------


def run_memory_lm(train, test, cand, args):
    Ztr, ybtr, _ = train
    Zte, ybte, _ = test
    mem = ByteGrowingResidualMemory(ByteGRMConfig(
        dim=Ztr.shape[1], k=args.k, steps=args.steps,
        max_neurons=args.max_neurons, seed=args.seed))
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    for ep in range(args.epochs):
        correct = 0
        for n, i in enumerate(rng.permutation(len(Ztr)), 1):
            if len(mem.C):
                correct += int(mem.predict_byte(Ztr[i], cand) == int(ybtr[i]))
            mem.train_one(Ztr[i], int(ybtr[i]), cand)
            if n % 5000 == 0:
                print(f"  [mem] epoch {ep+1} seen {n} neurons {len(mem.C)} "
                      f"online_acc {correct/n:.3f}")
    # evaluate
    correct, bits = 0, 0.0
    for i in range(len(Zte)):
        logits, _ = mem._route(Zte[i], train=False)
        p = softmax(logits[cand])
        t = int(np.where(cand == int(ybte[i]))[0][0])
        correct += int(int(cand[int(p.argmax())]) == int(ybte[i]))
        bits += -np.log2(max(float(p[t]), 1e-12))
    acc, bpc = correct / len(Zte), bits / len(Zte)
    print(f"[memory] acc={acc:.4f} bits/char={bpc:.3f} neurons={len(mem.C)} "
          f"time={time.time()-t0:.1f}s")
    return acc, bpc


def run_chart_lm(train, test, cand, args):
    Ztr, ybtr, _ = train
    Zte, ybte, _ = test
    cm = UniversalPredictiveChartMemory(ChartConfig(
        feat_dim=Ztr.shape[1], local_dim=args.local_dim, k=args.k,
        max_cells=args.max_neurons, lr=args.chart_lr, grow_conf=args.grow_conf,
        grow_novel=args.grow_novel, grow_resid=args.grow_resid, seed=args.seed))
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    for ep in range(args.epochs):
        correct = 0
        for n, i in enumerate(rng.permutation(len(Ztr)), 1):
            if len(cm):
                correct += int(cm.predict_byte(Ztr[i], cand) == int(ybtr[i]))
            cm.train_one(Ztr[i], int(ybtr[i]), cand)
            if n % 5000 == 0:
                print(f"  [chart] epoch {ep+1} seen {n} cells {len(cm)} "
                      f"online_acc {correct/n:.3f}")
    correct, bits = 0, 0.0
    for i in range(len(Zte)):
        p = cm.predict_probs(Zte[i], cand)
        t = int(np.where(cand == int(ybte[i]))[0][0])
        correct += int(int(cand[int(p.argmax())]) == int(ybte[i]))
        bits += -np.log2(max(float(p[t]), 1e-12))
    acc, bpc = correct / len(Zte), bits / len(Zte)
    print(f"[chart] acc={acc:.4f} bits/char={bpc:.3f} cells={len(cm)} "
          f"time={time.time()-t0:.1f}s")
    return acc, bpc


def run_linear_lm(train, test, V, args, lr=0.2):
    Ztr, _, yitr = train
    Zte, _, yite = test
    dim = Ztr.shape[1]
    W = np.zeros((V, dim), dtype=np.float32)
    b = np.zeros(V, dtype=np.float32)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    for _ in range(args.lin_epochs):
        for i in rng.permutation(len(Ztr)):
            z = W @ Ztr[i] + b
            z -= z.max()
            e = np.exp(z)
            e /= e.sum() + 1e-12
            g = -e
            g[yitr[i]] += 1.0
            W += lr * np.outer(g, Ztr[i])
            b += lr * g
    logits = Zte @ W.T + b
    logits -= logits.max(axis=1, keepdims=True)
    P = np.exp(logits)
    P /= P.sum(axis=1, keepdims=True) + 1e-12
    acc = float((P.argmax(1) == yite).mean())
    bpc = float(np.mean(-np.log2(np.clip(P[np.arange(len(yite)), yite], 1e-12, 1.0))))
    print(f"[linear] acc={acc:.4f} bits/char={bpc:.3f} params={V*dim} "
          f"time={time.time()-t0:.1f}s")
    return acc, bpc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", default="/tmp/tiny.txt")
    p.add_argument("--train_chars", type=int, default=30000)
    p.add_argument("--test_chars", type=int, default=5000)
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--max_order", type=int, default=8)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--k", type=int, default=24)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--max_neurons", type=int, default=12000)
    p.add_argument("--local_dim", type=int, default=24)
    p.add_argument("--chart_lr", type=float, default=0.5)
    p.add_argument("--grow_conf", type=float, default=0.5)
    p.add_argument("--grow_novel", type=float, default=0.55)
    p.add_argument("--grow_resid", type=float, default=0.30)
    p.add_argument("--lin_epochs", type=int, default=8)
    p.add_argument("--ngram_k", type=int, default=6)
    p.add_argument("--causal_D", type=int, default=10)
    p.add_argument("--causal_eps", type=float, default=0.04)
    p.add_argument("--min_count", type=int, default=2)
    p.add_argument("--causal_alpha", type=float, default=0.05)
    p.add_argument("--alpha_back", type=float, default=2.0)
    p.add_argument("--q_order", type=int, default=16)
    p.add_argument("--hier_lam", type=float, default=0.5)
    p.add_argument("--hier_chunk", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--encoder", choices=["hash", "vsa"], default="hash")
    p.add_argument("--arms", default="ngram,memory,linear")
    args = p.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    res = {}
    need_feats = any(x in ("memory", "chart", "linear") for x in arms)
    if need_feats:
        sk = (VSASketcher if args.encoder == "vsa" else CharLMSketcher)(
            dim=args.dim, max_order=args.max_order, seed=args.seed)
        print(f"encoder={args.encoder}; building features "
              f"(train={args.train_chars} test={args.test_chars}) ...")
        train, test, cand, vocab, data = build_dataset(
            args.path, args.train_chars, args.test_chars, args.window, sk)
    else:                                              # count-based arms need only raw bytes
        data = open(args.path, "rb").read()[: args.train_chars + args.test_chars + args.window + 1]
        vocab = sorted(set(data))
        cand = np.array(vocab, dtype=np.int64)
        train = test = None
    print(f"vocab={len(vocab)} chars, window={args.window}, train={args.train_chars}")

    if "ngram" in arms:
        t0 = time.time()
        acc, bpc = ngram_baseline(data, args.window, args.train_chars,
                                  args.test_chars, vocab, K=args.ngram_k)
        print(f"[ngram K={args.ngram_k}] acc={acc:.4f} bits/char={bpc:.3f} "
              f"time={time.time()-t0:.1f}s")
        res["ngram"] = (acc, bpc)
    if "causal" in arms:
        t0 = time.time()
        acc, bpc, nstates, ncontexts = causal_state_lm(
            data, args.window, args.train_chars, args.test_chars, vocab,
            D=args.causal_D, eps=args.causal_eps, min_count=args.min_count,
            alpha=args.causal_alpha, alpha_back=args.alpha_back)
        print(f"[causal eps={args.causal_eps}] acc={acc:.4f} bits/char={bpc:.3f} "
              f"states={nstates} (of {ncontexts} contexts) time={time.time()-t0:.1f}s")
        res["causal"] = (acc, bpc, nstates)
    if "hier" in arms:
        t0 = time.time()
        (l0r, hr, nstates, l1size) = hierarchical_causal_lm(
            data, args.window, args.train_chars, args.test_chars, vocab,
            D=args.causal_D, eps=args.causal_eps, min_count=args.min_count,
            alpha=args.causal_alpha, alpha_back=args.alpha_back,
            q_order=args.q_order, lam=args.hier_lam, chunk=args.hier_chunk)
        print(f"[L0 only ] acc={l0r[0]:.4f} bits/char={l0r[1]:.3f} states={nstates}")
        print(f"[L0+L1   ] acc={hr[0]:.4f} bits/char={hr[1]:.3f} "
              f"L1_entries={l1size} time={time.time()-t0:.1f}s")
        res["L0"] = l0r
        res["L0+L1"] = hr
    if "memory" in arms:
        res["memory"] = run_memory_lm(train, test, cand, args)
    if "chart" in arms:
        res["chart"] = run_chart_lm(train, test, cand, args)
    if "linear" in arms:
        res["linear"] = run_linear_lm(train, test, len(vocab), args)

    print("\n=== CHAR-LM SUMMARY (no backprop; lower bits/char = better) ===")
    print(f"  {'model':8s} {'next-char acc':>14s} {'bits/char':>11s} {'states':>9s}")
    for name in ("ngram", "causal", "L0", "L0+L1", "memory", "chart", "linear"):
        if name in res:
            acc, bpc = res[name][0], res[name][1]
            st = res[name][2] if len(res[name]) > 2 else 0
            print(f"  {name:8s} {acc:14.4f} {bpc:11.3f} {st:9d}")


if __name__ == "__main__":
    main()
