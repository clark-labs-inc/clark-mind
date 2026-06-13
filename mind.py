"""clark-mind: ONE FRONT DOOR for everything the mind can do (no backprop).
-----------------------------------------------------------------------------
Natural prompts route to the right substrate (each stays fully generic; this
file is only wiring -- a mouth and ears, not a brain):

  ACT        "play vc33 for 2000 steps" | "play arc"        -> generic
             predictive agent on ARC-AGI-3 (borrows the lifelong run's
             consolidated brain, never fights it for the file)
  GENERATE   "make an image of a 7" | "generate piano music"
             "continue song.mid" | "complete digit.png"     -> UniversalPSC
             studio (one predictive-state learner, modality = codec)
  REFLECT    "status" | "report"                            -> lifelong-run
             progress: cycles, levels climbed, skills held, brain sizes
  LIFELONG   "stop learning" / "resume learning"            -> control the
             endless consolidation loop (outputs/run_forever.sh)

Run:  python3 mind.py "<prompt>"      or interactively:  python3 mind.py
"""
from __future__ import annotations
import glob, os, re, shutil, subprocess, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
ARC_PY = os.path.join(ROOT, ".venv-arc", "bin", "python")
GEN_PY = "python3"                       # has PIL/pretty_midi for the studio
GAMES = ("su15 m0r0 s5i5 sk48 wa30 vc33 lp85 ls20 sc25 tn36 ka59 re86 ft09 "
         "dc22 sb26 cn04 ar25 sp80 g50t bp35 tr87 cd82 r11l tu93 lf52").split()


def sh(cmd, **kw):
    return subprocess.run(cmd, cwd=ROOT, **kw)


# ---------------------------------------------------------------- ACT (ARC)
def act(prompt):
    p = prompt.lower()
    g = next((t for t in re.findall(r"[a-z][a-z0-9]{3}", p) if t in GAMES), None)
    steps = next((int(n) for n in re.findall(r"\b(\d{3,6})\b", p)), 2000)
    if g is None:
        print("which game? one of:", " ".join(GAMES))
        return
    # borrow the lifelong brain (copy-on-use: never write the loop's file)
    mine, life = f"outputs/brain_mind_{g}.pkl", f"outputs/brain_life_{g}.pkl"
    lp, mp = os.path.join(ROOT, life), os.path.join(ROOT, mine)
    if os.path.exists(lp) and (not os.path.exists(mp)
                               or os.path.getmtime(lp) > os.path.getmtime(mp)):
        shutil.copy(lp, mp)
        if os.path.exists(lp + ".retina"):
            shutil.copy(lp + ".retina", mp + ".retina")
        print(f"  (borrowed the lifelong brain for {g})")
    sh([ARC_PY, "clark_arc_agent.py", "--steps", str(steps), "--game", g,
        "--brain", mine, "--verbose_every", "500"])


# ------------------------------------------------------------- REFLECT
def status():
    logs = sorted(glob.glob(os.path.join(ROOT, "outputs/life_cycle_*.out")),
                  key=os.path.getmtime)
    if not logs:
        print("no lifelong run yet (start one: 'resume learning')")
        return
    last = logs[-1]
    txt = open(last, errors="replace").read()
    done = len(re.findall(r"^##### PASS", txt, re.M))
    fails = txt.count("FAILED")
    prog = re.findall(r"game=(\S+) steps=\d+ levels_completed=([1-9])", txt)
    l2 = sum(1 for ln in (open(f, errors="replace") for f in logs)
             for line in ln if re.search(r"LEVEL UP -> [2-9]", line))
    print(f"lifelong run: cycle {len(logs)} ({done}/25 sessions, {fails} failed)")
    for gid, lv in prog:
        print(f"  {gid}: level {lv}")
    print(f"  level-2+ completions across all cycles: {l2}")
    brains = sorted(glob.glob(os.path.join(ROOT, "outputs/brain_life_*.pkl")),
                    key=os.path.getsize)
    if brains:
        b = brains[-1]
        print(f"  {len(brains)} brains, largest {os.path.basename(b)} "
              f"{os.path.getsize(b) >> 20}MB (sleep-bounded)")


# ------------------------------------------------------------- LIFELONG
def lifelong(start):
    stop = os.path.join(ROOT, "outputs/STOP")
    if start:
        if os.path.exists(stop):
            os.remove(stop)
        subprocess.Popen(["zsh", "outputs/run_forever.sh"], cwd=ROOT,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("lifelong learning resumed (endless consolidation cycles)")
    else:
        open(stop, "w").close()
        print("STOP set -- the loop halts after the current game finishes")


# ------------------------------------------------------------- ROUTER
def dispatch(prompt):
    p = prompt.lower().strip()
    if not p:
        return
    if any(k in p for k in ("status", "report", "progress", "how are you")):
        return status()
    if "stop" in p and "learn" in p:
        return lifelong(False)
    if any(k in p for k in ("resume", "start", "run")) and ("learn" in p or "forever" in p):
        return lifelong(True)
    if any(k in p for k in ("science", "biology", "dna", "rna", "protein",
                            "chemistry", "molecule", "fold")):
        return sh([GEN_PY, "science.py"])          # verifiable science curriculum
    if any(k in p for k in ("math", "arithmetic", "add", "multiply", "think",
                            "calculus", "deriv")):
        return sh([GEN_PY, "lessons_think.py"])    # thinking / verifiable math
    if any(k in p for k in ("symmetry", "group", "abstract", "invariant")):
        return sh([GEN_PY, "symmetry.py"])
    if any(k in p for k in ("spectral", "subgoal", "bottleneck", "fiedler")):
        return sh([GEN_PY, "spectral.py"])
    if any(k in p for k in ("hyperbolic", "hierarchy", "taxonomy", "geometry")):
        return sh([GEN_PY, "geometry.py"])
    if any(k in p for k in ("captcha", "puzzle", "odd one", "count")):
        return sh([GEN_PY, "captcha.py"])
    if any(k in p for k in ("blurry", "rotated", "ocr", "text captcha",
                            "read text", "distorted")):
        return sh([GEN_PY, "captcha_vision.py"])
    if any(k in p for k in ("primitive", "compose", "compound", "library")):
        return sh([GEN_PY, "primitives.py"])
    if any(k in p for k in ("brain", "transfer", "single brain")):
        return sh([GEN_PY, "brain.py"])
    if any(k in p for k in ("language", "wikitext", "induction", "coreference",
                            "bpc", "llm")):
        return sh([GEN_PY, "language.py"])         # bpc scaling + fetch induction
    if any(k in p for k in ("play", "arc", " game", "level")) or \
            any(t in GAMES for t in re.findall(r"[a-z][a-z0-9]{3}", p)):
        return act(prompt)
    # everything else: the generation studio routes by its own modality
    # keywords (image/music/...) and supplied .mid/.png files
    r = sh([GEN_PY, "psc_studio.py", prompt])
    if r.returncode != 0:
        print('try: "play vc33 for 2000 steps" | "make an image of a 7" | '
              '"generate music" | "status" | "stop learning"')


def main():
    prompt = " ".join(sys.argv[1:]).strip()
    if prompt:
        return dispatch(prompt)
    print("clark-mind. talk to me (empty line to exit).")
    while True:
        try:
            line = input("mind> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            break
        dispatch(line)


if __name__ == "__main__":
    main()
