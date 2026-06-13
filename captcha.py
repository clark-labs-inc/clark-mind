"""CAPTCHA-style COGNITIVE benchmarks: develop & measure thinking skills.
-------------------------------------------------------------------------------
SCOPE / ETHICS: these are SYNTHETIC, self-generated puzzles with known answers
(verifiable reward). They exercise the perception and reasoning skills CAPTCHAs
probe -- as a cognitive curriculum, the way MNIST or ARC are benchmarks. This
is NOT a tool to defeat deployed anti-bot systems (no real CAPTCHA endpoints,
no evasion); modern CAPTCHAs are behavioural/risk-scored, so nothing here would
transfer to bypassing them. The point is the brain's THINKING, measured.

Two grid-native tasks (the brain's modality):
  COUNT       "how many objects?"  -- perception: connected-component counting.
  ODD-ONE-OUT 4 tiles, 3 are symmetry-transforms of one shape + 1 different;
              name the odd one -- RELATIONAL reasoning, solved by the symmetry
              canonicalizer (symmetry.py): tiles in the same orbit share a
              signature, the singleton orbit is the odd one. This is a real
              "thinking skill", and it REUSES a primitive we already built.

Run:  python3 captcha.py
"""
from __future__ import annotations
import random, numpy as np
from symmetry import canonical

RNG = random.Random(0)
NRNG = np.random.default_rng(0)


# ---- perception: count connected components (the "select all X" skill) ----
def _components(g):
    g = np.asarray(g); H, W = g.shape
    seen = np.zeros((H, W), bool); n = 0
    for y in range(H):
        for x in range(W):
            if g[y, x] and not seen[y, x]:
                n += 1; stack = [(y, x)]; seen[y, x] = True
                while stack:
                    cy, cx = stack.pop()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < H and 0 <= nx < W and g[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True; stack.append((ny, nx))
    return n


def count_puzzle():
    g = np.zeros((16, 16), int); k = RNG.randint(1, 6); placed = 0
    while placed < k:
        y, x = RNG.randint(1, 13), RNG.randint(1, 13)
        if g[y-1:y+2, x-1:x+2].any():
            continue                                   # keep blobs separated
        g[y, x] = RNG.randint(1, 9)
        if RNG.random() < 0.5:
            g[y, x + 1] = g[y, x]                       # 2-cell blob sometimes
        placed += 1
    return g, k


def count_skill():
    ok = sum(_components(g) == k for g, k in (count_puzzle() for _ in range(300)))
    return 100 * ok / 300


# ---- reasoning: odd-one-out via symmetry orbits ----
def _rand_tile():
    return NRNG.integers(0, 3, size=(4, 4)) + NRNG.integers(0, 2)  # colors 0..3


def _transform(g):
    g = np.rot90(g, int(NRNG.integers(0, 4)))
    if NRNG.integers(0, 2):
        g = np.fliplr(g)
    ncol = int(g.max()) + 1
    return NRNG.permutation(ncol)[g] if ncol > 1 else g


def odd_one_out_puzzle():
    base = _rand_tile()
    odd = _rand_tile()
    # ensure the odd tile is a DIFFERENT orbit than base
    while canonical(odd) == canonical(base):
        odd = _rand_tile()
    tiles = [_transform(base) for _ in range(3)] + [odd]
    order = list(range(4)); random.Random(NRNG.integers(1 << 30)).shuffle(order)
    tiles = [tiles[i] for i in order]
    answer = order.index(3)                              # where the odd tile went
    return tiles, answer


def solve_odd_one_out(tiles):
    """The odd tile is the one whose symmetry orbit is unique among the four."""
    sigs = [canonical(t) for t in tiles]
    for i, s in enumerate(sigs):
        if sum(o == s for o in sigs) == 1:
            return i
    return 0


def odd_skill():
    ok = 0
    for _ in range(300):
        tiles, ans = odd_one_out_puzzle()
        ok += (solve_odd_one_out(tiles) == ans)
    return 100 * ok / 300


def main():
    print("CAPTCHA-STYLE COGNITIVE BENCHMARKS (synthetic, verifiable, no backprop)\n")
    print(f"   COUNT objects        : {count_skill():4.0f}%   "
          f"(perception, connected components; chance ~17%)")
    print(f"   ODD-ONE-OUT          : {odd_skill():4.0f}%   "
          f"(relational reasoning via symmetry orbits; chance 25%)")
    print("\nThe odd-one-out 'thinking skill' is solved by the symmetry primitive:")
    print("three tiles share an orbit, the fourth doesn't -- the singleton is odd.")
    print("Perception is a strength; relational reasoning works HERE because the")
    print("relation (symmetry) is one we have a primitive for. No backprop, all")
    print("verifiable. (Synthetic cognitive benchmark -- not a real-CAPTCHA tool.)")


if __name__ == "__main__":
    main()
