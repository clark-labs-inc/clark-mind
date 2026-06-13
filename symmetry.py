"""Abstraction by SYMMETRY (Klein's Erlangen Program, 1872), gradient-free.
-------------------------------------------------------------------------------
The deepest no-backprop route to abstraction: a concept is an ORBIT under the
group of transformations that preserve its meaning. "A door" is not 1,000
pixel-states -- it is one state, modulo the symmetry group (rotation,
reflection, recolouring, translation). Collapsing each state to its orbit
representative *is* the abstraction the counting substrate cannot learn on its
own; it is delivered by 19th-century group theory, not by training.

  group here = D4 (4 rotations x 2 reflections = dihedral, order 8)
             x color permutations (relabel by first appearance)
             x translation (crop to bounding box)

canonical(grid) returns the lexicographically-minimal serialization over the
whole orbit -- so any two grids related by the group map to the SAME signature.
ARC is, at heart, a test of perceiving these symmetries; this makes them free.
"""
from __future__ import annotations
import numpy as np


def _d4(g):
    """The 8 dihedral images of a grid."""
    out = []
    for k in range(4):
        r = np.rot90(g, k)
        out.append(r); out.append(np.fliplr(r))
    return out


def _recolor_canon(g):
    """Invariance to colour permutation: relabel colours by order of first
    appearance in raster scan (0 stays 0 = background convention optional)."""
    g = np.asarray(g)
    mapping, nxt = {}, 0
    flat = g.ravel()
    out = np.empty_like(flat)
    for i, v in enumerate(flat):
        v = int(v)
        if v not in mapping:
            mapping[v] = nxt; nxt += 1
        out[i] = mapping[v]
    return out.reshape(g.shape)


def _bbox(g):
    """Invariance to translation: crop to the non-background bounding box."""
    g = np.asarray(g)
    nz = np.argwhere(g != 0)
    if nz.size == 0:
        return g
    (y0, x0), (y1, x1) = nz.min(0), nz.max(0) + 1
    return g[y0:y1, x0:x1]


def canonical(grid, recolor=True, translate=True):
    """Orbit representative under D4 (x colour x translation). Two grids
    related by the group return identical bytes."""
    g = np.asarray(grid)
    best = None
    for t in _d4(g):
        c = _bbox(t) if translate else t
        c = _recolor_canon(c) if recolor else c
        key = (c.shape, c.tobytes())
        if best is None or key < best:
            best = key
    return best


# ============================ learning test ==================================
def _test():
    rng = np.random.default_rng(0)

    def rand_pattern(h, w, k):
        return rng.integers(0, k, size=(h, w))

    def random_transform(g):
        """Apply a random group element: D4 + recolor + translate (pad)."""
        g = np.rot90(g, int(rng.integers(4)))
        if rng.integers(2):
            g = np.fliplr(g)
        ncol = int(g.max()) + 1
        perm = rng.permutation(ncol)
        g = perm[g]
        ph, pw = int(rng.integers(0, 4)), int(rng.integers(0, 4))
        return np.pad(g, ((ph, int(rng.integers(0, 4))),
                          (pw, int(rng.integers(0, 4)))))

    # K base concepts; each test item is a random group-transform of one.
    K = 12
    bases = [rand_pattern(4, 4, 3) + 1 for _ in range(K)]   # colors 1..3 (0=bg)

    def trial(canon):
        # "train": memorize signature -> label for a few transforms per base
        table = {}
        for lbl, b in enumerate(bases):
            for _ in range(5):
                t = random_transform(b)
                sig = canonical(t) if canon else (t.shape, t.tobytes())
                table[sig] = lbl
        # "test": NEW random transforms; correct iff signature recalls the label
        ok = 0
        for _ in range(600):
            lbl = int(rng.integers(K)); t = random_transform(bases[lbl])
            sig = canonical(t) if canon else (t.shape, t.tobytes())
            ok += (table.get(sig, -1) == lbl)
        uniq = len(table)
        return 100 * ok / 600, uniq

    print("ABSTRACTION BY SYMMETRY -- classify 12 patterns under random")
    print("D4 x recolour x translation transforms (held-out transforms):\n")
    a0, u0 = trial(canon=False)
    a1, u1 = trial(canon=True)
    print(f"   raw signature      : {a0:5.0f}%   ({u0} distinct stored states)")
    print(f"   orbit canonical    : {a1:5.0f}%   ({u1} distinct stored states)")
    print(f"\n   state collapse: {u0} -> {u1} ({u0/max(u1,1):.0f}x); the group "
          f"quotient turns {u0} surface forms into {u1} concepts.")
    print("   (raw ~chance: every transform is a new state; canonical "
          "generalizes because all transforms share one orbit. No backprop.)")


if __name__ == "__main__":
    _test()
