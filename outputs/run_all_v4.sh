#!/bin/zsh
# Full ARC-AGI-3 public-set benchmark v3 (Phase A multi-res backoff + Phase S
# systematic graph exploration): 25 games x 2 passes, brains keyed by GAME-ID
# prefix (the API shuffles env order per call, so indices are unstable).
# Pass 2 reloads each game's own brain -> true cross-session consolidation.
cd /Users/stan/Documents/git/clark-mind
GAMES=(su15 m0r0 s5i5 sk48 wa30 vc33 lp85 ls20 sc25 tn36 ka59 re86 ft09 dc22 sb26 cn04 ar25 sp80 g50t bp35 tr87 cd82 r11l tu93 lf52)
for pass in 1 2; do
  for g in $GAMES; do
    echo "##### PASS $pass GAME $g #####"
    .venv-arc/bin/python clark_arc_agent.py --steps 8000 --game $g \
      --brain outputs/brain_v4_$g.pkl --verbose_every 100000 \
      || echo "##### GAME $g FAILED #####"
  done
done
echo "##### ALL DONE #####"
