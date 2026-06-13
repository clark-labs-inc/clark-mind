# Ancient math, gradient-free: fundamental primitives backprop can't use

Mainstream ML rests on one slice of math — differentiable optimization — and
abandoned huge classical branches *because* they aren't differentiable. This
architecture has no gradients to protect, so it can use exactly that math.
Each primitive below is proven with a crisp learning test.

## Symmetry / group theory (Klein's Erlangen Program, 1872) — `symmetry.py`

**Abstraction = quotient by a symmetry group.** A concept is an orbit:
"a door" is one state modulo rotation, reflection, recolouring, translation.
Collapsing each state to its orbit representative *is* the abstraction the
counting substrate can't learn — delivered by group theory, not training.

Learning test (classify 12 patterns under random D4 × recolour × translation,
held-out transforms):

```
raw signature   :   0%   (60 distinct stored states)
orbit canonical : 100%   (14 distinct stored states)
```

60 surface forms → 14 concepts; perfect generalization to unseen transforms,
no backprop.

## Spectral graph theory (the spectral theorem, 1800s) — `spectral.py`

The agent's world model is a graph; the **Fiedler vector** (2nd Laplacian
eigenvector) is the smoothest coordinate on it, so its sign cuts the graph at
its weakest link and its steepest edge is the **bottleneck = subgoal**. Found
in closed form — the temporal abstraction the BFS agent never names.

Test (two rooms joined by one doorway):

```
Fiedler sign separates the two rooms : YES
bottleneck edge detected             : == the true doorway, no search
```

## Hyperbolic geometry (Poincaré/Lobachevsky, 1800s) — `geometry.py`

Concept hierarchies are trees; trees don't fit Euclidean space (volume grows
polynomially) but fit hyperbolic space (volume grows exponentially, matching
branching). Closed-form placement, no training.

Test (balanced binary tree, 127 nodes, worst-case distance distortion):

```
Euclidean  : 72.5x
Hyperbolic :  8.3x   (9x better, same 2 dimensions)
```

## The honest integration caveat (symmetry → ARC)

The symmetry primitive is powerful but cannot be bolted onto the ARC retina
naively: you may only quotient by the symmetries a *given game actually has*.
A gravity/orientation game is **not** rotation-invariant — blind D4
canonicalization would collapse genuinely different states and hurt. Correct
integration needs **per-game symmetry detection**: discover which transforms
preserve transitions/reward (the same action-independence logic as clock
habituation, applied to geometric transforms). That detection step is the real
next build; the primitive is ready for it.
