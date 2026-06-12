#!/bin/zsh
cd /Users/stan/Documents/git/clark-mind
GAMES=(cd82 ft09 lp85 ls20 r11l sp80 su15 tn36 tu93 vc33)
for pass in 3 4; do
  for g in $GAMES; do
    echo "##### PASS $pass GAME $g #####"
    .venv-arc/bin/python clark_arc_agent.py --steps 8000 --game $g \
      --brain outputs/brain_v5_$g.pkl --verbose_every 100000 \
      || echo "##### GAME $g FAILED #####"
  done
done
echo "##### ALL DONE #####"
