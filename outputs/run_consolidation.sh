#!/bin/zsh
# True consolidation test: same-game brain reload (gid-keyed, immune to the
# API's per-call env shuffling). Only the 5 games where fresh sessions found
# reward -- consolidation has zero signal to amplify in the other 20.
cd /Users/stan/Documents/git/clark-mind
for pass in 1 2 3; do
  for g in vc33 cd82 lp85 r11l sp80; do
    echo "##### PASS $pass GAME $g #####"
    .venv-arc/bin/python clark_arc_agent.py --steps 8000 --game $g \
      --brain outputs/brain_cons_$g.pkl --verbose_every 100000 \
      || echo "##### GAME $g FAILED #####"
  done
done
echo "##### ALL DONE #####"
