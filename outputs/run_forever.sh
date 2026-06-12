#!/bin/zsh
# Lifelong run: endless consolidation passes over all 25 public ARC-AGI-3
# games on PERSISTENT brains (outputs/brain_life_*.pkl). Safe to leave
# running indefinitely: episodic memory is bounded (max_pairs + sleep),
# skills are protected for life, long-term rungs keep compounding.
# One log per cycle: outputs/life_cycle_N.out. Stop with: touch outputs/STOP
cd /Users/stan/Documents/git/clark-mind
GAMES=(su15 m0r0 s5i5 sk48 wa30 vc33 lp85 ls20 sc25 tn36 ka59 re86 ft09 dc22 sb26 cn04 ar25 sp80 g50t bp35 tr87 cd82 r11l tu93 lf52)
# resume cycle numbering after restarts
n=1
for f in outputs/life_cycle_*.out(N); do
  m=${${f##*_}%%.out}
  (( m >= n )) && n=$((m + 1))
done
while [[ ! -e outputs/STOP ]]; do
  log=outputs/life_cycle_$n.out
  for g in $GAMES; do
    [[ -e outputs/STOP ]] && break
    echo "##### PASS $n GAME $g #####" >> $log
    .venv-arc/bin/python clark_arc_agent.py --steps 8000 --game $g \
      --brain outputs/brain_life_$g.pkl --verbose_every 100000 >> $log 2>&1 \
      || echo "##### GAME $g FAILED #####" >> $log
  done
  echo "##### CYCLE $n DONE #####" >> $log
  n=$((n+1))
done
echo "STOP file found -- lifelong run halted after $((n-1)) cycles."
