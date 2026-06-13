#!/bin/zsh
# Lifelong run: endless consolidation passes over all 25 public ARC-AGI-3
# games on PERSISTENT brains. HEURISTIC-FREE agent (hierarchical PSRL,
# arc_bayes.py) with ~1GB episodic capacity per brain; skills protected for
# life. Safe to leave running indefinitely.
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
    .venv-arc/bin/python arc_bayes.py --steps 8000 --game $g \
      --brain outputs/brain_bayes_$g.pkl >> $log 2>&1 \
      || echo "##### GAME $g FAILED #####" >> $log
  done
  # PRACTICE: train and exercise the other faculties every cycle --
  # the one-mind battery (draw/name/perceive/act/dream), generation, the
  # verifiable curricula (arithmetic with the locality/INFILL trick), and the
  # full SCIENCE set (biology via content-addressable fetch, chemistry,
  # protein folding via search). Metrics logged per cycle.
  if [[ ! -e outputs/STOP ]]; then
    echo "##### PASS $n PRACTICE #####" >> $log
    python3 one_mind.py >> $log 2>&1 || echo "PRACTICE one_mind FAILED" >> $log
    python3 science.py >> $log 2>&1 || echo "PRACTICE science FAILED" >> $log
    python3 lessons_think.py >> $log 2>&1 || echo "PRACTICE think FAILED" >> $log
    python3 lessons_skills.py >> $log 2>&1 || echo "PRACTICE skills FAILED" >> $log
    python3 psc_studio.py "generate piano music" >> $log 2>&1 \
      || echo "PRACTICE music FAILED" >> $log
    python3 psc_studio.py "make an image of a $((n % 10))" >> $log 2>&1 \
      || echo "PRACTICE image FAILED" >> $log
  fi
  echo "##### CYCLE $n DONE #####" >> $log
  n=$((n+1))
done
echo "STOP file found -- lifelong run halted after $((n-1)) cycles."
