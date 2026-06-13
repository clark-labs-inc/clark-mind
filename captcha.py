"""Hard CAPTCHA-style COGNITIVE benchmarks: reasoning as search over primitives.
-------------------------------------------------------------------------------
SCOPE / ETHICS: synthetic, self-generated puzzles with known answers (verifiable
reward) -- a cognitive curriculum to develop and measure THINKING, the way ARC
or Raven's matrices are benchmarks. NOT a tool against deployed anti-bot systems
(no real endpoints, no evasion; modern CAPTCHAs are behavioural, nothing here
transfers to bypassing them).

The earlier version detected a KNOWN symmetry. Real difficulty is INFERRING the
rule. The thesis these tasks test: reasoning = SEARCH over a library of
transforms/predicates. A puzzle is solved iff the rule it needs is expressible
in the library -- so the suite climbs until it finds the wall, honestly.

Tasks, increasing difficulty:
  ANALOGY      A:B :: C:?  -- infer the transform g with g(A)=B, apply to C.
  SEQUENCE     t1 t2 t3 -> t4  -- infer the repeated transform, extrapolate.
  RAVEN 3x3    rule across rows AND columns -- infer both, predict missing cell.
  ODD-PROPERTY 5 tiles, 4 share a LATENT property (count / symmetry / colours);
               infer the property, name the violator.
  BONGARD      6 positive + 6 negative share/lack a concept; classify a query
               -- concept induction, the abstraction wall.

Run:  python3 captcha.py
"""
from __future__ import annotations
import random, numpy as np
from itertools import product

NRNG = np.random.default_rng(0)


# ============================ transform library ==============================
# the hypothesis space for rule inference: the dihedral group D4 (+ recolour).
def _recolor_canon(g):
    m, nxt, out = {}, 0, np.empty_like(g.ravel())
    for i, v in enumerate(g.ravel()):
        v = int(v)
        if v not in m:
            m[v] = nxt; nxt += 1
        out[i] = m[v]
    return out.reshape(g.shape)


TRANSFORMS = {
    "identity": lambda g: g,
    "rot90": lambda g: np.rot90(g, 1),
    "rot180": lambda g: np.rot90(g, 2),
    "rot270": lambda g: np.rot90(g, 3),
    "fliplr": np.fliplr,
    "flipud": np.flipud,
    "transpose": lambda g: g.T,
    "anti": lambda g: np.rot90(np.fliplr(g), 1),
}


def same(a, b, recolor=False):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if recolor:
        return np.array_equal(_recolor_canon(a), _recolor_canon(b))
    return np.array_equal(a, b)


def infer_transform(a, b, recolor=False):
    """Search the library for a transform mapping a->b (the inference step)."""
    for name, fn in TRANSFORMS.items():
        try:
            if same(fn(a), b, recolor):
                return name
        except ValueError:
            pass
    return None


def rand_tile(n=4, k=3):
    return NRNG.integers(0, k, size=(n, n))


# ============================ tasks ==========================================
def analogy():
    base, c = rand_tile(), rand_tile()
    g = list(TRANSFORMS)[int(NRNG.integers(len(TRANSFORMS)))]
    A, B, C = base, TRANSFORMS[g](base), c
    D = TRANSFORMS[g](c)
    # multiple choice: D among 4 distractors (other transforms of c)
    opts = [D] + [TRANSFORMS[list(TRANSFORMS)[int(NRNG.integers(len(TRANSFORMS)))]](c)
                  for _ in range(3)]
    order = list(range(4)); random.Random(int(NRNG.integers(1 << 30))).shuffle(order)
    opts = [opts[i] for i in order]; ans = order.index(0)
    return (A, B, C, opts), ans


def solve_analogy(p):
    A, B, C, opts = p
    g = infer_transform(A, B)
    if g is None:
        return 0
    want = TRANSFORMS[g](C)
    for i, o in enumerate(opts):
        if same(o, want):
            return i
    return 0


def sequence():
    base = rand_tile()
    g = list(TRANSFORMS)[int(NRNG.integers(1, len(TRANSFORMS)))]  # non-identity
    seq = [base]
    for _ in range(3):
        seq.append(TRANSFORMS[g](seq[-1]))
    return seq[:3], seq[3]                               # (inputs, target grid)


def solve_sequence(seq):
    g = infer_transform(seq[0], seq[1])
    if g and same(TRANSFORMS[g](seq[1]), seq[2]):       # confirm rule on 2nd step
        return TRANSFORMS[g](seq[2])
    return seq[2]


def raven():
    """cell[i][j] = (down^i . right^j)(base); predict cell[2][2] from the rest."""
    base = rand_tile()
    r = list(TRANSFORMS)[int(NRNG.integers(1, len(TRANSFORMS)))]
    d = list(TRANSFORMS)[int(NRNG.integers(1, len(TRANSFORMS)))]
    grid = [[None]*3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            g = base
            for _ in range(j):
                g = TRANSFORMS[r](g)
            for _ in range(i):
                g = TRANSFORMS[d](g)
            grid[i][j] = g
    target = grid[2][2]; grid[2][2] = None
    return grid, target


def solve_raven(grid):
    r = infer_transform(grid[0][0], grid[0][1])         # right rule from row 0
    d = infer_transform(grid[0][0], grid[1][0])         # down rule from col 0
    if r is None or d is None:
        return grid[2][1] if grid[2][1] is not None else grid[1][2]
    g = grid[2][1]                                       # bottom-middle
    return TRANSFORMS[r](g) if g is not None else None


# ---- property inference (odd-one-out by latent property) ----
def _ncomp(g):
    g = np.asarray(g); H, W = g.shape; seen = np.zeros_like(g, bool); n = 0
    for y in range(H):
        for x in range(W):
            if g[y, x] and not seen[y, x]:
                n += 1; st = [(y, x)]; seen[y, x] = True
                while st:
                    cy, cx = st.pop()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy+dy, cx+dx
                        if 0 <= ny < H and 0 <= nx < W and g[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True; st.append((ny, nx))
    return n


PREDICATES = {
    "ncolors": lambda g: len(set(int(v) for v in g.ravel())),
    "nobjects": lambda g: _ncomp(g),
    "h_mirror": lambda g: int(np.array_equal(g, np.fliplr(g))),
    "v_mirror": lambda g: int(np.array_equal(g, np.flipud(g))),
    "rot_sym": lambda g: int(np.array_equal(g, np.rot90(g, 2))),
    "nfilled": lambda g: int((np.asarray(g) > 0).sum()),
}


def odd_property():
    # 4 tiles share a property VALUE under some predicate; 1 differs.
    # retry whole puzzles until all five tiles are realizable.
    while True:
        fn = PREDICATES[list(PREDICATES)[int(NRNG.integers(len(PREDICATES)))]]
        pool = [NRNG.integers(0, 3, size=(5, 5)) for _ in range(400)]
        vals = [fn(t) for t in pool]
        from collections import Counter
        common = [v for v, c in Counter(vals).items() if c >= 4]
        if not common:
            continue
        tgt = common[int(NRNG.integers(len(common)))]
        share = [t for t, v in zip(pool, vals) if v == tgt][:4]
        diff = [t for t, v in zip(pool, vals) if v != tgt]
        if len(share) < 4 or not diff:
            continue
        odd = diff[int(NRNG.integers(len(diff)))]
        tiles = share + [odd]
        order = list(range(5)); random.Random(int(NRNG.integers(1 << 30))).shuffle(order)
        return [tiles[i] for i in order], order.index(4)


def solve_odd_property(tiles):
    # find a predicate under which exactly one tile is the minority value
    best = 0
    for fn in PREDICATES.values():
        vals = [fn(t) for t in tiles]
        for i, v in enumerate(vals):
            if vals.count(v) == 1 and len(set(vals)) == 2:
                return i
    return best


def bongard():
    """positive tiles satisfy a predicate threshold; negatives don't. classify."""
    pred = list(PREDICATES)[int(NRNG.integers(len(PREDICATES)))]
    fn = PREDICATES[pred]
    thr = None
    # choose a threshold splitting random tiles roughly in half
    sample = [fn(NRNG.integers(0, 3, size=(5, 5))) for _ in range(200)]
    thr = sorted(sample)[len(sample)//2]
    def pos():
        for _ in range(300):
            t = NRNG.integers(0, 3, size=(5, 5))
            if fn(t) >= thr:
                return t
        return NRNG.integers(0, 3, size=(5, 5))
    def neg():
        for _ in range(300):
            t = NRNG.integers(0, 3, size=(5, 5))
            if fn(t) < thr:
                return t
        return NRNG.integers(0, 3, size=(5, 5))
    P = [pos() for _ in range(6)]; N = [neg() for _ in range(6)]
    is_pos = bool(NRNG.integers(0, 2))
    q = pos() if is_pos else neg()
    return (P, N, q), int(is_pos)


def solve_bongard(p):
    P, N, q = p
    # search predicates for one that separates P from N by a threshold
    for fn in PREDICATES.values():
        vp = [fn(t) for t in P]; vn = [fn(t) for t in N]
        if max(vn) < min(vp):                            # clean separation, P high
            return int(fn(q) >= min(vp))
        if max(vp) < min(vn):                            # P low
            return int(fn(q) <= max(vp))
    return 0


def main():
    print("HARD COGNITIVE BENCHMARKS -- reasoning = search over a primitive")
    print("library; solvable iff the rule is in the library. (synthetic, "
          "verifiable, no backprop)\n")
    def acc(gen, solve, n=200):
        ok = 0
        for _ in range(n):
            p, a = gen()
            r = solve(p)
            ok += (same(r, a) if isinstance(a, np.ndarray) else r == a)
        return 100 * ok / n
    print(f"   ANALOGY  A:B::C:?          {acc(analogy, solve_analogy):4.0f}%   "
          f"(infer transform, apply; chance 25%)")
    print(f"   SEQUENCE t1 t2 t3 -> t4    "
          f"{acc(sequence, solve_sequence):4.0f}%   (infer repeated transform)")
    print(f"   RAVEN 3x3 matrix          {acc(raven, solve_raven):4.0f}%   "
          f"(infer row AND column rule)")
    print(f"   ODD-PROPERTY (5 tiles)    "
          f"{acc(odd_property, solve_odd_property):4.0f}%   "
          f"(infer latent shared property; chance 20%)")
    print(f"   BONGARD concept           {acc(bongard, solve_bongard):4.0f}%   "
          f"(concept induction; chance 50%)")
    print("\nEach is solved by SEARCHING the transform/predicate library for the")
    print("rule consistent with the examples, then applying it -- genuine")
    print("inference, not lookup. Where a puzzle's rule lives OUTSIDE the library")
    print("the score drops to chance: that gap is exactly the missing primitive.")


if __name__ == "__main__":
    main()
