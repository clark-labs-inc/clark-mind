"""
Parse a multi-game clark-mind ARC-AGI-3 run log and compute official-style
metrics, including an RHAE (Relative Human Action Efficiency) estimate --
the ARC-AGI-3 paper's scoring: per level S = min(1, h/a)^2 with h = human
baseline action count; env score = sum(l * S_l) / (n(n+1)/2); total = mean
over all environments (levels never completed score 0).

Actions-per-level is estimated as steps between FIRST completions of
consecutive levels within a pass (the log does not track per-level action
counters; resets and farming inflate this, so the estimate is conservative).

Usage: .venv-arc/bin/python arc_report.py outputs/arc_full_run.out
"""
from __future__ import annotations
import re, sys
import arc_agi


def parse(path):
    runs = []   # list of dicts per session
    cur = None
    for ln in open(path, errors="replace"):
        m = re.match(r"##### PASS (\d+) GAME (\S+) #####", ln)
        if m:
            cur = {"pass": int(m[1]), "game": m[2], "ups": [], "first": {},
                   "levels": 0, "states": 0, "pairs": 0, "rew": 0, "won": False}
            runs.append(cur)
            continue
        if cur is None:
            continue
        m = re.search(r">>> step (\d+): LEVEL UP -> (\d+)", ln)
        if m:
            s, l = int(m[1]), int(m[2])
            cur["ups"].append((s, l))
            cur["first"].setdefault(l, s)
            continue
        if "GAME FULLY WON" in ln:
            cur["won"] = True
        m = re.search(r"game=(\S+) steps=\d+ levels_completed=(\d+)", ln)
        if m:
            cur["gid"], cur["levels"] = m[1], int(m[2])
        m = re.search(r"states discovered=(\d+)\s+state-action pairs tried=(\d+)\s+rewarding pairs=(\d+)", ln)
        if m:
            cur["states"], cur["pairs"], cur["rew"] = int(m[1]), int(m[2]), int(m[3])
    return runs


def rhae_env(first, baseline, won_all):
    """Env score per the ARC-AGI-3 paper: weighted by level index."""
    n = len(baseline)
    num = 0.0
    prev = 0
    for l in range(1, n + 1):
        if l in first:
            a = max(first[l] - prev, 1)     # steps spent inside this level
            prev = first[l]
            s = min(1.0, baseline[l - 1] / a) ** 2
        else:
            s = 0.0
        num += l * s
    return num / (n * (n + 1) / 2)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "outputs/arc_full_run.out"
    runs = parse(path)
    envs = arc_agi.Arcade().get_environments()
    # the API returns environments in a DIFFERENT order on every call, so all
    # joins must be by game_id, never by index
    meta = {e.game_id: (list(e.baseline_actions or []), ",".join(e.tags or []))
            for e in envs}

    by_pass = {}
    for r in runs:
        if "gid" in r:
            by_pass.setdefault(r["pass"], {})[r["gid"]] = r

    for p in sorted(by_pass):
        rows = by_pass[p]
        print(f"\n=== PASS {p} ===")
        print(f"{'game':18s} {'tags':16s} {'lvls':>9s} {'level-ups':>9s} "
              f"{'1st lvl@':>8s} {'states':>7s} {'RHAE%':>7s}")
        tot_levels = tot_possible = 0
        tot_rhae = 0.0
        solved_games = 0
        for gid in sorted(rows):
            r = rows[gid]
            base, tags = meta[gid]
            n = len(base)
            e = rhae_env(r["first"], base, r["won"])
            tot_rhae += e
            tot_levels += r["levels"]; tot_possible += n
            if r["won"]:
                solved_games += 1
            f1 = str(r["first"].get(1, "-"))
            print(f"{gid:18s} {tags:16s} {r['levels']:>4d}/{n:<4d} {len(r['ups']):>9d} "
                  f"{f1:>8s} {r['states']:>7d} {100*e:>6.2f}%")
        print(f"{'TOTAL':18s} {'':16s} {tot_levels:>4d}/{tot_possible:<4d} "
              f"games fully won: {solved_games}/25   "
              f"RHAE score: {100*tot_rhae/len(rows):.3f}%")


if __name__ == "__main__":
    main()
