"""Record the brain playing ARC-AGI-3: every frame -> PNG, the run -> GIF.
Crosshairs mark where it clicks. Run with a trained brain to film a speedrun:
    .venv-arc/bin/python arc_record.py --game ft09 --brain /tmp/demo.pkl --steps 40
"""
from __future__ import annotations
import argparse, os, pickle, numpy as np
from PIL import Image, ImageDraw
import arc_agi
from arcengine.enums import GameAction, GameState

import clark_arc_agent as A
from predictive_agent import GenericPredictiveAgent
from micro_cortex import MicroCortex

# the ARC 16-color palette
PAL = [(0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64), (255, 220, 0),
       (170, 170, 170), (240, 18, 190), (255, 133, 27), (127, 219, 255),
       (135, 12, 37), (87, 36, 194), (46, 26, 71), (255, 255, 255),
       (90, 196, 177), (140, 80, 30), (250, 160, 200)]
SC = 8                                       # pixels per cell


def render(g, click=None, label=""):
    g = np.asarray(g)
    H, W = g.shape
    im = Image.new("RGB", (W * SC, H * SC + 18), (24, 24, 24))
    px = im.load()
    for y in range(H):
        for x in range(W):
            c = PAL[int(g[y, x]) % 16]
            for dy in range(SC):
                for dx in range(SC):
                    px[x * SC + dx, y * SC + dy] = c
    d = ImageDraw.Draw(im)
    if click is not None:
        cy, cx = click
        x0, y0 = cx * SC + SC // 2, cy * SC + SC // 2
        d.line([(x0 - 10, y0), (x0 + 10, y0)], fill=(255, 255, 255), width=2)
        d.line([(x0, y0 - 10), (x0, y0 + 10)], fill=(255, 255, 255), width=2)
        d.ellipse([x0 - 6, y0 - 6, x0 + 6, y0 + 6], outline=(255, 255, 255), width=2)
    d.text((4, H * SC + 3), label, fill=(230, 230, 230))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="ft09")
    ap.add_argument("--brain", default="")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--out", default="media")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    arc = arc_agi.Arcade()
    gid = next(e.game_id for e in arc.get_environments()
               if e.game_id.startswith(args.game))
    env = arc.make(gid); fr = env.reset()
    agent = GenericPredictiveAgent(actions=A.action_space(), depth=5, beta=0.8,
                                   names=A.action_names(),
                                   cortex=MicroCortex(beta=0.5),
                                   max_pairs=60000)
    if args.brain and os.path.exists(args.brain):
        if os.path.exists(args.brain + ".retina"):
            with open(args.brain + ".retina", "rb") as f:
                A.RET.restore(pickle.load(f))
        agent.load(args.brain)
    sig, slots, grid = A.perceive(fr.frame)
    best = fr.levels_completed
    sig = A.ctx(sig, best)
    frames = [render(grid, None, f"step 0   level {fr.levels_completed}")]

    for step in range(1, args.steps + 1):
        avail = A.available_now(fr, slots)
        feats, cells = A.make_feats(grid, avail, slots)
        a, _, _ = agent.act(sig, avail, feats=feats)
        if (a != 0 and agent.no_change_streak >= 40
                and agent.no_change_streak % 40 == 0 and 0 in avail):
            a = 0                            # the adapter's stuck backstop
        click, data = None, None
        if a == 0:
            nf = env.step(GameAction.RESET); what = "RESET"
        else:
            i = a[0] if isinstance(a, tuple) else a
            if isinstance(a, tuple):
                click = cells[a]; data = {"x": click[1], "y": click[0]}
                what = f"click ({click[1]},{click[0]})"
            else:
                what = f"ACTION{a}"
            nf = env.step(A.ACT[i], data=data)
        if nf is None or nf.frame is None:
            continue
        prev_level = fr.levels_completed
        fr = nf
        nsig, nslots, ngrid = A.perceive(fr.frame)
        if nsig is None:
            continue
        leveled = fr.levels_completed > prev_level
        reward = (10.0 * fr.levels_completed
                  if leveled and fr.levels_completed > best else 0.0)
        best = max(best, fr.levels_completed)
        nsig = A.ctx(nsig, best)
        agent.learn(sig, a, reward, nsig, avail, feats=feats)
        tag = f"step {step}   level {fr.levels_completed}   {what}"
        if leveled:
            tag += "   *** LEVEL UP ***"
        frames.append(render(grid, click, tag))     # frame BEFORE the effect
        frames.append(render(ngrid, None, tag))     # frame AFTER
        sig, slots, grid = nsig, nslots, ngrid
        if fr.state == GameState.WIN:
            break

    gif = os.path.join(args.out, f"{args.game}_play.gif")
    frames[0].save(gif, save_all=True, append_images=frames[1:],
                   duration=420, loop=0)
    frames[-1].save(os.path.join(args.out, f"{args.game}_final.png"))
    print(f"saved {gif} ({len(frames)} frames) and {args.game}_final.png")


if __name__ == "__main__":
    main()
