# Thinking & long-form generation, backprop-free

What reasoning can a no-backprop *counting* model actually do? These two
scripts answer it with verifiable rewards (a checker says correct/incorrect)
on held-out problems the model never saw.

## Chain-of-thought only helps if the steps are *local*

`lessons_think.py` teaches the SAME stream model 3-digit addition three ways
and scores 300 unseen sums:

| representation | held-out accuracy |
|---|---|
| DIRECT — `347+285=` → answer | **0%** |
| THINK — answer preceded by a carry trace | **0%** |
| INFILL — each result computed *immediately after* its digit pair | **100%** |

Same model, same data, same learning rule — only the **order tokens are laid
out in** changed.

Why: an LLM adds by using **attention** to fetch the right operand digit into
each reasoning step across arbitrary distance — a learned, backprop-trained
pointer. This substrate has *no attention*; it only sees the last few tokens.
So "thinking out loud" helps only when the reasoning is arranged so each
step's inputs are physically adjacent. When they are, it computes carries
perfectly. When they aren't (`a+b=`), it knows the *grammar* of a carry trace
but fills in the wrong digits — it can compute, but it cannot **fetch**.
Discovering the right arrangement is exactly what attention buys, and is the
one thing this architecture provably lacks.

## Long answers: yes, kilobytes — and *correct*, if the task is local

Because the model rolls one token at a time with no length limit, output is
unbounded. The honest split:

- **Free-form prose**: stays only *locally* coherent; a kilobyte of it drifts
  (no long-range plan — the project's recurring wall).
- **Structured / procedural** output where each step is local: stays
  **correct** for as long as you let it run, and **generalizes to any length**.

Trained on nothing larger than 3-digit addition, the INFILL adder correctly
adds far longer numbers, emitting kilobytes of correct worked steps:

```
50-digit  + 50-digit : CORRECT  (~0.5 KB of steps)
200-digit + 200-digit: CORRECT  (~2.0 KB)
800-digit + 800-digit: CORRECT  (~7.8 KB)
```

Length-generalization is perfect because each step is local and
length-invariant — the model learned the *algorithm*, not a table.

## The takeaway

This is not an LLM and won't become one without attention/backprop. But within
its actual strength — local, verifiable, procedural reasoning — it can be
*taught* to think step by step and produce unbounded, correct, long-form
output, with no gradients anywhere. The skill is in the **representation**
(arrange dependencies to be local); the learning rule stays pure counting.

Run: `python3 lessons_think.py` and `python3 lessons.py`
