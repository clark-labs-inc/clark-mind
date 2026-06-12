"""
clark-mind plays ARC-AGI-3 via the GENERIC predictive agent (no backprop).
--------------------------------------------------------------------------
This file is ONLY an adapter: it encodes ARC frames into state signatures and
maps the agent's abstract actions to the ARC API. The agent itself
(predictive_agent.GenericPredictiveAgent) is task/modality-agnostic -- the exact
same class solves GridWorld. The adapter is the RETINA: receptive-field wiring
(here: connected-component object segmentation) is allowed to be
modality-specific; the learning rules are not. Adapter provides:
  - RESET as ordinary action 0 (the agent LEARNS what reset does); forced only
    as a long-streak backstop
  - coordinate actions 6/7 become composite (action, object-slot) ids: the
    frame is segmented into same-color connected components and each click
    targets a specific OBJECT's center cell -- deterministic, so the episodic
    model sees repeatable transitions instead of random-cell noise
  - rewards: +10*level on level-up, +10*(levels+1) on win, -0.5 on game-over

Run in the py3.12 venv:
    .venv-arc/bin/python clark_arc_agent.py --steps 600 --game vc33
"""
from __future__ import annotations
import argparse, hashlib, os, pickle, time, numpy as np
from collections import Counter
import arc_agi
from arcengine.enums import GameAction, GameState
from predictive_agent import GenericPredictiveAgent, SigStack, Habituator
from micro_cortex import MicroCortex

ACT = {i: getattr(GameAction, f"ACTION{i}") for i in range(1, 8)}
COORD = {6, 7}                 # ARC pointer/click actions need (x,y)
SIMPLE = [1, 2, 3, 4, 5]
NSLOT = 64                     # max click targets (object points) per frame


# -------- the ONLY ARC-specific code: encode a frame, map actions --------
def _h(obj):
    """Stable 64-bit hash (process-independent, unlike hash())."""
    b = obj if isinstance(obj, bytes) else repr(obj).encode()
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "little")


def segment(grid):
    """Same-color connected components (4-connectivity): the frame as OBJECTS.
    Returns [(size, color, cy, cx)] with a deterministic centroid cell each."""
    g = np.asarray(grid)
    H, W = g.shape
    lab = np.full((H, W), -1, np.int32)
    objs = []
    for y0 in range(H):
        for x0 in range(W):
            if lab[y0, x0] >= 0:
                continue
            c = int(g[y0, x0]); idx = len(objs)
            lab[y0, x0] = idx; stack = [(y0, x0)]; cells = []
            while stack:
                y, x = stack.pop(); cells.append((y, x))
                if y and lab[y-1, x] < 0 and g[y-1, x] == c:
                    lab[y-1, x] = idx; stack.append((y-1, x))
                if y < H-1 and lab[y+1, x] < 0 and g[y+1, x] == c:
                    lab[y+1, x] = idx; stack.append((y+1, x))
                if x and lab[y, x-1] < 0 and g[y, x-1] == c:
                    lab[y, x-1] = idx; stack.append((y, x-1))
                if x < W-1 and lab[y, x+1] < 0 and g[y, x+1] == c:
                    lab[y, x+1] = idx; stack.append((y, x+1))
            my = sum(p[0] for p in cells) / len(cells)
            mx = sum(p[1] for p in cells) / len(cells)
            cy, cx = min(cells, key=lambda p: (p[0]-my)**2 + (p[1]-mx)**2)
            size = len(cells)
            objs.append((size, c, cy, cx))
            if size >= 64:
                # a LARGE component is likely a board/canvas where position
                # within it matters: add its bbox quarter-points as extra
                # click targets (still deterministic), not just the centroid
                ys = [p[0] for p in cells]; xs = [p[1] for p in cells]
                yl, yh, xl, xh = min(ys), max(ys), min(xs), max(xs)
                for fy, fx in ((1, 1), (1, 3), (3, 1), (3, 3)):
                    qy = yl + (yh - yl) * fy // 4
                    qx = xl + (xh - xl) * fx // 4
                    if g[qy, qx] == c and (qy, qx) != (cy, cx):
                        objs.append((size, c, qy, qx))
    return objs


def diversify(objs, cap=NSLOT):
    """Click-slot ordering by TYPE DIVERSITY: one representative of every
    distinct (size, color) object type first (small types first -- buttons are
    small, boards/backgrounds are big), then second instances, and so on."""
    groups = {}
    for o in sorted(objs):
        groups.setdefault((o[0], o[1]), []).append(o)
    keys = sorted(groups)
    out, rank = [], 0
    while len(out) < cap and keys:
        keys = [k for k in keys if len(groups[k]) > rank]
        for k in keys:
            out.append(groups[k][rank])
            if len(out) >= cap:
                break
        rank += 1
    return out


class Retina(Habituator):
    """Adapter WIRING only (the habituation rule lives in the generic
    Habituator): elements = grid cells, a click's cause = its location bucket
    (slot ids shuffle across frames; same place = same cause), masked cells
    render as a sentinel color and are not objects."""
    SENT = 99

    def learn(self, g0, a, g1):
        if g0.shape != g1.shape:
            return
        self.observe({((int(y), int(x)), (int(g0[y, x]), int(g1[y, x])))
                      for y, x in zip(*np.nonzero(g0 != g1))}, a)

    def apply(self, g):
        if not self.mask:
            return g
        g = g.copy()
        H, W = g.shape
        for y, x in self.mask:
            if y < H and x < W:
                g[y, x] = self.SENT
        return g


RET = Retina()


def ctx(stack, best):
    """Bind the session-best level into the FINE signature (Markov fix): the
    level-up reward is paid only at a new session-best, so the same frame is
    worth 10 at best=0 and 0 at best>=1 -- without `best` in the state the
    reward is non-stationary and every unpaid re-completion DILUTES the win
    pair's mean (v4: re-completion went 4 -> 733 steps, consolidation erased).
    Both EXACT levels (raw + habituated) carry `best`; layout/inventory stay
    best-free so what-a-click-does transfers across contexts."""
    return SigStack(((stack[0], best), (stack[1], best)) + tuple(stack[2:]))


_CACHE = {}                                  # fine sig -> (SigStack, slots)
_CACHE_V = 0                                 # retina mask version of the cache
def perceive(frame):
    """Frame -> (multi-resolution SigStack, click slots, grid).
    Levels, fine to coarse (the agent backs off through them like an n-gram
    model): exact pixels -> object layout (types at quantized positions) ->
    object inventory (type counts only). A novel exact frame whose LAYOUT or
    INVENTORY was seen before is no longer a blank slate. Clock/HUD cells are
    masked out first (Retina), so frames differing only by a counter tick are
    the SAME state at every level."""
    global _CACHE_V
    arr = np.asarray(frame)
    if arr.size == 0 or arr.ndim < 2:
        return None, None, None
    g = arr[-1] if arr.ndim == 3 else arr
    if _CACHE_V != RET.version:
        _CACHE.clear(); _CACHE_V = RET.version
    fine_raw = _h(g.tobytes())
    if fine_raw not in _CACHE:
        if len(_CACHE) > 60000:
            _CACHE.clear()
        # "The same scene MODULO CLOCKS" is an abstraction RUNG, not a
        # replacement identity: raw exact -> habituated exact -> layout ->
        # inventory. Games whose canvas IS the state keep a Markov raw
        # identity (full masking made ft09 non-Markov, 1000 -> 86 states);
        # games that never repeat exactly (counters, reshuffles) back off ONE
        # rung to the sharp habituated level where skills accumulate (vc33's
        # button). Slots come from the masked scene: clocks are not targets.
        gm = RET.apply(g)
        iobjs = [o for o in segment(gm) if o[1] != Retina.SENT]
        mid = _h(tuple(sorted((s, c, y // 4, x // 4) for s, c, y, x in iobjs)))
        coarse = _h(tuple(sorted(
            Counter((c, int(s).bit_length()) for s, c, _, _ in iobjs).items())))
        _CACHE[fine_raw] = (SigStack((fine_raw, _h(gm.tobytes()), mid, coarse)),
                            diversify(iobjs))
    stack, slots = _CACHE[fine_raw]
    return stack, slots, g


def action_space():
    return [0] + SIMPLE + [(i, s) for i in sorted(COORD) for s in range(NSLOT)]


# -------- micro-features: local receptive fields for the cortex --------
def patch_bytes(g, cy, cx, k=2):
    """(2k+1)^2 window around a cell, -1 padded -- 'what am I clicking on'."""
    p = np.full((2 * k + 1, 2 * k + 1), -1, np.int16)
    y0, y1 = max(0, cy - k), min(g.shape[0], cy + k + 1)
    x0, x1 = max(0, cx - k), min(g.shape[1], cx + k + 1)
    p[y0 - cy + k:y1 - cy + k, x0 - cx + k:x1 - cx + k] = g[y0:y1, x0:x1]
    return p.tobytes()


def make_feats(grid, avail, objs):
    """Per-action micro-feature sets. A click's features describe the OBJECT it
    targets (local patch, color, log-size) so the cortex's value/novelty
    generalizes across every instance and recurrence of that object type."""
    g = np.asarray(grid)
    hist = (np.bincount(g.ravel().astype(np.int64) % 16, minlength=16) // 32).tobytes()
    feats, cells = {}, {}
    for a in avail:
        if isinstance(a, tuple):
            i, s = a
            size, c, cy, cx = objs[s]
            feats[a] = (("patch", i, patch_bytes(g, cy, cx)),
                        ("obj", i, c, int(size).bit_length()),
                        ("act", i))
            cells[a] = (cy, cx)
        else:
            feats[a] = (("act", a), ("ctx", a, hist))
    return feats, cells


def action_names():
    nm = {0: "RESET"}
    nm.update({(i, s): f"CLICK{i}#o{s}" for i in sorted(COORD) for s in range(NSLOT)})
    return nm


def available_now(fr, objs):
    """Map the env's offered actions into the agent's composite action space."""
    if fr.state in (GameState.WIN, GameState.GAME_OVER):
        return [0]                                   # only reset is meaningful
    ids = {int(a) for a in fr.available_actions}
    out = [0] + [i for i in SIMPLE if i in ids]
    out += [(i, s) for i in sorted(COORD & ids) for s in range(len(objs))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--game", type=str, default="0",
                    help="index OR game-id prefix; the API shuffles env order "
                         "per call, so prefixes are the only stable selector")
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--verbose_every", type=int, default=100)
    ap.add_argument("--stuck_reset", type=int, default=40)
    ap.add_argument("--brain", type=str, default="",
                    help="path to persist the world model across sessions")
    args = ap.parse_args()

    arc = arc_agi.Arcade()
    envs = arc.get_environments()
    if args.game.isdigit():
        gid = envs[int(args.game)].game_id
    else:
        gid = next(e.game_id for e in envs if e.game_id.startswith(args.game))
    env = arc.make(gid); fr = env.reset()
    print(f"clark-mind (generic agent) playing ARC-AGI-3: {gid}\n")

    agent = GenericPredictiveAgent(actions=action_space(), depth=args.depth,
                                   beta=0.8, names=action_names(),
                                   cortex=MicroCortex(beta=0.5),
                                   max_pairs=60000, sleep_every=2000)
    if args.brain and os.path.exists(args.brain):
        if os.path.exists(args.brain + ".retina"):
            with open(args.brain + ".retina", "rb") as f:
                RET.restore(pickle.load(f))
        agent.load(args.brain)
        print(f"  loaded brain: {len(agent.states)} states, "
              f"{sum(1 for v in agent.rew.values() if v > 0)} rewarding pairs remembered, "
              f"{len(RET.mask)} clock cells masked\n")
    sig, slots, grid = perceive(fr.frame)
    best_level = fr.levels_completed; t0 = time.time()
    sig = ctx(sig, best_level)
    forced_resets = chosen_resets = 0

    for step in range(args.steps):
        avail = available_now(fr, slots)
        feats, cells = make_feats(grid, avail, slots)
        a, qs, monologue = agent.act(sig, avail, feats=feats)
        if (a != 0 and agent.no_change_streak >= args.stuck_reset
                and agent.no_change_streak % args.stuck_reset == 0 and 0 in avail):
            a = 0; forced_resets += 1                # backstop: long no-effect streak
            print(f"  !! step {step}: stuck {agent.no_change_streak} steps -> forced RESET")
        prev_grid, prev_level, prev_state = grid, fr.levels_completed, fr.state
        data, click = None, ""
        if a == 0:
            chosen_resets += 1
            try:
                nf = env.step(GameAction.RESET)
            except Exception:
                nf = None
        else:
            i = a[0] if isinstance(a, tuple) else a
            if isinstance(a, tuple):
                cy, cx = cells[a]                    # the object cell this slot means
                data = {"x": cx, "y": cy}; click = f" @({cx},{cy})o{a[1]}"
            try:
                nf = env.step(ACT[i], data=data)
            except Exception:
                nf = None
        if nf is None or nf.frame is None:
            agent.learn(sig, a, -0.05, sig, avail, feats=feats); continue   # invalid: no move
        fr = nf
        # retina habituation: the causing action keyed by click LOCATION
        # bucket (slots shuffle across frames; same place = same cause)
        if a == 0:
            ra = 0
        elif isinstance(a, tuple):
            cy, cx = cells[a]; ra = ("c", cy // 16, cx // 16)
        else:
            ra = a
        arr = np.asarray(fr.frame)
        rg = arr[-1] if arr.ndim == 3 else arr
        if rg.ndim == 2:
            RET.learn(np.asarray(prev_grid), ra, rg)
        nsig, nslots, ngrid = perceive(fr.frame)
        if nsig is None:
            agent.learn(sig, a, 0.0, sig, avail, feats=feats); continue
        leveled = fr.levels_completed > prev_level
        won = fr.state == GameState.WIN and prev_state != GameState.WIN
        died = fr.state == GameState.GAME_OVER and prev_state != GameState.GAME_OVER
        # level/win reward must beat the exploration-optimism ceiling (~3.6) so
        # that, after replay propagates it, redoing a known win outranks novelty.
        # Higher levels pay MORE (xlevel), and ONLY a session-best counts: ARC's
        # real score is monotonic levels-completed, so re-completing a level
        # after reset earns nothing. Paying re-completions made FARMING level 1
        # every ~6 steps optimal (v3: 1300+ levelups/8k steps, zero level 2s);
        # unpaid re-completions dilute the win-pair's mean reward, values
        # flatten, and the value-gated frontier sweep resumes on the new level.
        reward = (10.0 * fr.levels_completed
                  if leveled and fr.levels_completed > best_level else 0.0) \
            + (10.0 * (fr.levels_completed + 1) if won else 0.0) \
            - (0.5 if died else 0.0)
        nsig = ctx(nsig, max(best_level, fr.levels_completed))
        agent.learn(sig, a, reward, nsig, avail, feats=feats)
        changed = int(np.sum(np.asarray(prev_grid) != np.asarray(ngrid)))
        nm = agent._aname(a)
        if leveled:
            print(f"  >>> step {step}: LEVEL UP -> {fr.levels_completed} via {nm}{click}")
        if died:
            print(f"  xx step {step}: GAME OVER via {nm}{click} (penalty -0.5)")
        if won:
            print(f"  *** step {step}: GAME WON via {nm}{click} ***")
        if step % args.verbose_every == 0:
            print(f"[step {step}] states={len(agent.states)} level={fr.levels_completed} "
                  f"-> {nm}{click}  (changed {changed}px)")
            print("  inner monologue:")
            for ln in monologue: print(ln)
        sig, slots, grid = nsig, nslots, ngrid
        best_level = max(best_level, fr.levels_completed)
        if fr.state == GameState.WIN:
            print(f"\n  GAME FULLY WON at step {step}!")
            break

    print(f"\n=== SUMMARY ({time.time()-t0:.0f}s) ===")
    print(f"  game={gid} steps={args.steps} levels_completed={best_level}")
    tried = sum(1 for v in agent.vis.values() if v > 0)
    print(f"  states discovered={len(agent.states)}  state-action pairs tried={tried}  "
          f"rewarding pairs={sum(1 for k, v in agent.rew.items() if v > 0)}")
    print(f"  resets: {chosen_resets} total ({forced_resets} forced by stuck-backstop)")
    print(f"  clock cells masked: {len(RET.mask)}")
    if args.brain:
        agent.save(args.brain)
        with open(args.brain + ".retina", "wb") as f:
            pickle.dump(RET.state(), f)
        print(f"  brain saved -> {args.brain}")


if __name__ == "__main__":
    main()
