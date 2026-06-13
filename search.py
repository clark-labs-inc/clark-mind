"""Global search for a no-backprop brain: verifier-guided best-first search.
-------------------------------------------------------------------------------
Constraint satisfaction and hard combinatorial problems need SEARCH, which a
forward predictive pass cannot do. The brain's gradient-free version: propose
candidates, SCORE them with a verifier (the verifiable reward), expand the
promising ones, backtrack on dead ends. The checker is the value signal -- no
gradients. This is "System 2" laid over System 1.

Generic over any problem exposing:
  start()                 -> initial partial state
  expand(state)           -> iterable of (next_state, is_terminal)
  score(state)            -> float, higher is better (a verifier / energy)
The model's own predictions (if any) can bias `expand` ordering, but the
search itself is exact and backprop-free.
"""
from __future__ import annotations
import heapq, itertools


def best_first(problem, beam=2000, max_expand=200000):
    """Best-first search with a bounded frontier (beam). Returns the highest-
    scoring terminal state found. Pure search, verifier-scored."""
    counter = itertools.count()
    start = problem.start()
    frontier = [(-problem.score(start), next(counter), start)]
    best_term, best_val = None, -1e18
    expansions = 0
    while frontier and expansions < max_expand:
        negscore, _, state = heapq.heappop(frontier)
        expansions += 1
        for nxt, terminal in problem.expand(state):
            v = problem.score(nxt)
            if terminal:
                if v > best_val:
                    best_val, best_term = v, nxt
            else:
                heapq.heappush(frontier, (-v, next(counter), nxt))
        if len(frontier) > beam:                    # keep the best `beam` open nodes
            frontier = heapq.nsmallest(beam, frontier)
            heapq.heapify(frontier)
    return best_term, best_val, expansions


def relax(variables, domains, consistent, sweeps=100):
    """Constraint satisfaction by iterative local consistency (AC-3 flavour,
    gradient-free): repeatedly prune domain values that satisfy no assignment
    of a neighbour, until a fixed point. Returns reduced domains (a solution
    when every domain is a singleton)."""
    dom = {v: set(domains[v]) for v in variables}
    for _ in range(sweeps):
        changed = False
        for v in variables:
            keep = {a for a in dom[v]
                    if all(any(consistent(v, a, u, b) for b in dom[u])
                           for u in variables if u != v)}
            if keep != dom[v]:
                dom[v] = keep; changed = True
        if not changed:
            break
    return dom
