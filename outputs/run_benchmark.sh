#!/bin/zsh
# Full ARC-AGI-3 public-set benchmark: 25 games x 2 passes (pass 2 reloads each
# game's own brain = cross-session consolidation). Brains are keyed by GAME-ID
# prefix because the API shuffles env order on every call.
#   usage: zsh outputs/run_benchmark.sh <tag> [steps]
#   e.g.:  zsh outputs/run_benchmark.sh v10 8000 > outputs/arc_full_run_v10.out
TAG=${1:?usage: run_benchmark.sh <tag> [steps]}
STEPS=${2:-8000}
cd /Users/stan/Documents/git/clark-mind
GAMES=(su15 m0r0 s5i5 sk48 wa30 vc33 lp85 ls20 sc25 tn36 ka59 re86 ft09 dc22 sb26 cn04 ar25 sp80 g50t bp35 tr87 cd82 r11l tu93 lf52)
for pass in 1 2; do
  for g in $GAMES; do
    echo "##### PASS $pass GAME $g #####"
    .venv-arc/bin/python clark_arc_agent.py --steps $STEPS --game $g \
      --brain outputs/brain_${TAG}_$g.pkl --verbose_every 100000 \
      || echo "##### GAME $g FAILED #####"
  done
done
echo "##### ALL DONE #####"
