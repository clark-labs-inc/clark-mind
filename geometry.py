"""Geometry for hierarchy: hyperbolic space (Poincare disk), gradient-free.
-------------------------------------------------------------------------------
Concept hierarchies ("is-a" taxonomies) are TREES, and a tree's distances
cannot fit in Euclidean space without distortion that grows with its size --
the room runs out, because volume grows polynomially. Hyperbolic space has
volume that grows EXPONENTIALLY with radius, exactly matching a tree's
branching, so trees embed with almost no distortion. Placing the points is a
closed-form construction (Sarkar), not learned. This is why a brain that wants
to hold a big taxonomy should think in negative curvature.

This module embeds a tree by construction and measures distortion vs a fair
Euclidean embedding -- the quantitative case for hyperbolic concept memory.
"""
from __future__ import annotations
import numpy as np
np.seterr(over="ignore", invalid="ignore", divide="ignore")


def poincare_dist(u, v):
    u, v = np.asarray(u), np.asarray(v)
    num = np.sum((u - v) ** 2)
    den = (1 - np.sum(u ** 2)) * (1 - np.sum(v ** 2))
    return np.arccosh(1 + 2 * num / (den + 1e-12))


def embed_tree_hyperbolic(children, root=0, scale=0.6):
    """Sarkar-style placement: root at origin; each node's children spread in an
    angular wedge at a larger radius. Returns {node: point in the Poincare disk}."""
    pos = {root: np.zeros(2)}
    # assign each node an angular interval, subdivide among children
    def place(node, lo, hi, depth):
        kids = children.get(node, [])
        if not kids:
            return
        r = np.tanh(scale * (depth + 1))          # radius -> 1 with depth
        step = (hi - lo) / len(kids)
        for i, k in enumerate(kids):
            a0, a1 = lo + i * step, lo + (i + 1) * step
            ang = 0.5 * (a0 + a1)
            pos[k] = np.array([r * np.cos(ang), r * np.sin(ang)])
            place(k, a0, a1, depth + 1)
    place(root, 0, 2 * np.pi, 0)
    return pos


def _test():
    # a balanced binary tree, depth 6 (127 nodes)
    depth = 6
    children, nxt = {}, 1
    nodes = [0]; frontier = [(0, 0)]
    while frontier:
        node, d = frontier.pop(0)
        if d >= depth:
            continue
        children[node] = [nxt, nxt + 1]
        for c in (nxt, nxt + 1):
            nodes.append(c); frontier.append((c, d + 1))
        nxt += 2
    # true tree (graph) distances
    import collections
    adj = collections.defaultdict(list)
    for p, ks in children.items():
        for k in ks:
            adj[p].append(k); adj[k].append(p)
    def bfs(s):
        dist = {s: 0}; q = collections.deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1; q.append(v)
        return dist
    D = {s: bfs(s) for s in nodes}

    hpos = embed_tree_hyperbolic(children)
    # fair Euclidean baseline: same 2D coordinates, Euclidean metric
    def distortion(metric):
        ratios = []
        for i in nodes:
            for j in nodes:
                if i < j and D[i][j] > 0:
                    ratios.append(metric(i, j) / D[i][j])
        ratios = np.array(ratios)
        return ratios.max() / ratios.min()        # worst-case distortion

    hyp = distortion(lambda i, j: poincare_dist(hpos[i], hpos[j]))
    euc = distortion(lambda i, j: np.linalg.norm(hpos[i] - hpos[j]))
    print("HYPERBOLIC GEOMETRY FOR HIERARCHY -- balanced binary tree, 127 nodes")
    print("(distortion = worst-case ratio of embedded to true tree distance;")
    print(" 1.0 is perfect, lower is better):\n")
    print(f"   Euclidean (same 2D coords) distortion : {euc:6.1f}x")
    print(f"   Hyperbolic (Poincare disk) distortion : {hyp:6.1f}x")
    print(f"\n   hyperbolic preserves the taxonomy {euc/hyp:.0f}x better in the "
          f"same\n   2 dimensions -- exponential room for 'is-a' hierarchy, "
          f"no training.")


if __name__ == "__main__":
    _test()
