"""Subgoals in closed form: spectral graph theory (the spectral theorem, 1800s).
-------------------------------------------------------------------------------
The agent explores its world-graph by brute search (BFS frontier), which is why
long-horizon games stall. Spectral graph theory gives the GLOBAL structure of a
graph without search: the Fiedler vector (eigenvector of the graph Laplacian
for the 2nd-smallest eigenvalue) is the smoothest non-trivial coordinate on the
graph, so its sign bipartitions the graph at its weakest link and its steepest
edge is the BOTTLENECK -- the natural subgoal / "option" boundary. This is
Mahadevan's proto-value-function idea, almost unused in deep RL, and entirely
gradient-free (one eigendecomposition of what the agent already counted).

  fiedler(A) -> the 2nd Laplacian eigenvector
  bottleneck = the edge of largest Fiedler gradient (the doorway)
  partition  = sign(Fiedler) (the two rooms / subtasks)
"""
from __future__ import annotations
import numpy as np


def fiedler(A):
    """2nd-smallest Laplacian eigenvector of a symmetric adjacency matrix A."""
    A = np.asarray(A, dtype=np.float64)
    d = A.sum(1)
    L = np.diag(d) - A
    w, V = np.linalg.eigh(L)
    order = np.argsort(w)
    return V[:, order[1]]                     # skip the constant (lambda_0=0)


def bottleneck_edge(A, f=None):
    """The graph edge whose endpoints differ most in the Fiedler coordinate --
    the weakest cut, i.e. the subgoal doorway."""
    A = np.asarray(A)
    f = fiedler(A) if f is None else f
    best, be = -1.0, None
    n = A.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j]:
                g = abs(f[i] - f[j])
                if g > best:
                    best, be = g, (i, j)
    return be


def partition(A, f=None):
    f = fiedler(A) if f is None else f
    return (f >= 0).astype(int)


# ============================ test ===========================================
def _two_room_grid(room=4, seed=0):
    """Two room x room rooms joined by a single doorway cell -- the canonical
    bottleneck world. Returns adjacency, cell list, true doorway cells, and the
    true room labels."""
    cells = []
    # room A: x in [0,room); room B: x in [room+1, 2*room+1); doorway at x=room,
    # y=room//2 only.
    for y in range(room):
        for x in range(room):
            cells.append((y, x))
    door = (room // 2, room)
    cells.append(door)
    for y in range(room):
        for x in range(room + 1, 2 * room + 1):
            cells.append((y, x))
    idx = {c: i for i, c in enumerate(cells)}
    n = len(cells)
    A = np.zeros((n, n))
    for (y, x) in cells:
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = (y + dy, x + dx)
            if q in idx:
                A[idx[(y, x)], idx[q]] = A[idx[q], idx[(y, x)]] = 1
    rooms = {c: (0 if c[1] < room else 1) for c in cells}
    rooms[door] = -1                          # doorway belongs to neither
    return A, cells, door, rooms, idx


def _test():
    A, cells, door, rooms, idx = _two_room_grid(room=4)
    f = fiedler(A)
    part = partition(A, f)
    be = bottleneck_edge(A, f)
    # did the Fiedler sign separate the two rooms?
    sideA = {part[idx[c]] for c in cells if rooms[c] == 0}
    sideB = {part[idx[c]] for c in cells if rooms[c] == 1}
    separated = len(sideA) == 1 and len(sideB) == 1 and sideA != sideB
    # does the detected bottleneck edge touch the true doorway?
    bcells = (cells[be[0]], cells[be[1]])
    found = door in bcells

    print("SPECTRAL SUBGOAL DISCOVERY -- two rooms joined by one doorway:")
    print(f"   graph: {len(cells)} cells, true doorway at {door}")
    print(f"   Fiedler sign separates the two rooms : "
          f"{'YES' if separated else 'no'}")
    print(f"   bottleneck edge detected             : {bcells[0]}-{bcells[1]}"
          f"   {'== doorway' if found else '(missed)'}")
    print("\n   one eigendecomposition of the learned graph reveals the subgoal")
    print("   with NO search -- the option boundary the BFS agent never names.")


if __name__ == "__main__":
    _test()
