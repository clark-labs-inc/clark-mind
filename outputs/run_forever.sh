#!/bin/zsh
# Lifelong learning for THE ONE BRAIN (the_brain.py): a single persistent model
# that keeps reading (byte-level language) each cycle -- genuine accumulation
# (heldout bpc drops) while its cognitive faculties are kept as a regression
# check. Counts add, nothing is forgotten. Stop with: touch outputs/STOP
cd /Users/stan/Documents/git/clark-mind
while [[ ! -e outputs/STOP ]]; do
  python3 the_brain.py --lifelong >> outputs/brain_life.out 2>&1
  sleep 2
done
echo "STOP -- single-brain lifelong learning halted."
