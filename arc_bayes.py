"""ARC-AGI-3 adapter for the HEURISTIC-FREE BayesAgent (PSRL).
Reuses clark_arc_agent's retina (segmentation, multi-resolution SigStacks,
clock masking) by injection; differences from the heuristic agent's adapter:
  - reward = the environment's ACTUAL score change (new-best levels), nothing
    else: no death penalty, no invalid-step penalty, no shaping
  - clock masking decided by Bayes factor (Jeffreys), not count thresholds

Run: .venv-arc/bin/python arc_bayes.py --steps 3000 --game ft09
"""
from __future__ import annotations
import argparse, os, pickle, time, numpy as np
import arc_agi
from arcengine.enums import GameAction, GameState

import clark_arc_agent as A
from bayes_agent import BayesAgent, BayesHabituator


class BayesRetina(BayesHabituator):
    SENT = A.Retina.SENT

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--game", type=str, default="ft09")
    ap.add_argument("--brain", type=str, default="")
    args = ap.parse_args()

    A.RET = BayesRetina()                    # inject the Bayes-factor retina
    arc = arc_agi.Arcade()
    envs = arc.get_environments()
    gid = next(e.game_id for e in envs if e.game_id.startswith(args.game))
    env = arc.make(gid); fr = env.reset()
    print(f"BayesAgent (PSRL, heuristic-free) on ARC-AGI-3: {gid}\n")

    agent = BayesAgent(actions=A.action_space(), names=A.action_names())
    if args.brain and os.path.exists(args.brain):
        if os.path.exists(args.brain + ".retina"):
            with open(args.brain + ".retina", "rb") as f:
                A.RET.restore(pickle.load(f))
        agent.load(args.brain)
        print(f"  loaded brain: {len(agent.states)} states, "
              f"{len(A.RET.mask)} clock cells masked\n")
    sig, slots, grid = A.perceive(fr.frame)
    best = fr.levels_completed
    sig = A.ctx(sig, best)
    t0 = time.time()

    for step in range(args.steps):
        avail = A.available_now(fr, slots)
        feats, cells = A.make_feats(grid, avail, slots)
        a, _, _ = agent.act(sig, avail)
        data = None
        if a == 0:
            try:
                nf = env.step(GameAction.RESET)
            except Exception:
                nf = None
        else:
            i = a[0] if isinstance(a, tuple) else a
            if isinstance(a, tuple):
                cy, cx = cells[a]; data = {"x": cx, "y": cy}
            try:
                nf = env.step(A.ACT[i], data=data)
            except Exception:
                nf = None
        if nf is None or nf.frame is None:
            agent.learn(sig, a, 0.0, sig, avail); continue
        prev_grid, prev_level = grid, fr.levels_completed
        fr = nf
        if a == 0:
            ra = 0
        elif isinstance(a, tuple):
            cy, cx = cells[a]; ra = ("c", cy // 16, cx // 16)
        else:
            ra = a
        arr = np.asarray(fr.frame)
        rg = arr[-1] if arr.ndim == 3 else arr
        if rg.ndim == 2:
            A.RET.learn(np.asarray(prev_grid), ra, rg)
        nsig, nslots, ngrid = A.perceive(fr.frame)
        if nsig is None:
            agent.learn(sig, a, 0.0, sig, avail); continue
        # reward = the environment's actual score change, nothing else
        reward = float(max(0, fr.levels_completed - max(best, prev_level)))
        if fr.levels_completed > prev_level:
            print(f"  >>> step {step}: LEVEL UP -> {fr.levels_completed} "
                  f"via {agent.names.get(a, a)}")
        best = max(best, fr.levels_completed)
        nsig = A.ctx(nsig, best)
        agent.learn(sig, a, reward, nsig, avail)
        sig, slots, grid = nsig, nslots, ngrid
        if fr.state == GameState.WIN:
            print(f"\n  GAME FULLY WON at step {step}!"); break

    print(f"\n=== SUMMARY ({time.time()-t0:.0f}s) ===")
    print(f"  game={gid} steps={args.steps} levels_completed={best}")
    print(f"  states discovered={len(agent.states)}  "
          f"pairs={len(agent.levels[0].n)}  clock cells masked={len(A.RET.mask)}")
    if args.brain:
        agent.save(args.brain)
        with open(args.brain + ".retina", "wb") as f:
            pickle.dump(A.RET.state(), f)
        print(f"  brain saved -> {args.brain}")


if __name__ == "__main__":
    main()
