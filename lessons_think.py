"""Thinking, backprop-free: what reasoning can a counting model actually do?
-------------------------------------------------------------------------------
A clean, reproducible demonstration of the difference between an LLM's
reasoning and a no-backprop counting model's, on held-out 3-digit addition.
Same model (UniversalPSC), same data, same learning rule -- only the ORDER the
tokens are arranged in changes:

  DIRECT  "347+285=" -> answer .......................... 0%   non-local
  THINK   answer preceded by a carry trace .............. 0%   trace needs a
          ("347+285= : 7+5+0=2,c1 ; ... #...")               non-local COPY
  INFILL  each step computed RIGHT AFTER its digits ..... 100% every prediction
          ("7+5+0=2c1 4+8+1=3c1 3+2+1=6c0")                  is local

The lesson: an LLM adds by using ATTENTION to fetch the right operand digit
into each reasoning step across arbitrary distance -- a learned, backprop-
trained pointer. This substrate has no attention; it only sees the last few
tokens. So "thinking out loud" helps ONLY when the reasoning is laid out so
each step's inputs are physically adjacent. When they are, it computes carries
flawlessly and generalizes to unseen numbers. When they're not, it's helpless
-- not because it can't compute, but because it can't FETCH. Discovering the
right arrangement is exactly what attention buys, and is what this
architecture provably lacks.

Run:  python3 lessons_think.py
"""
from __future__ import annotations
import random, numpy as np
from psc_studio import UniversalPSC, _sample

BOS, SEP, EOS = 256, 257, 258
VOCAB = 259
RNG = random.Random(1)
BACKOFF = [(-1,), (-2,), (-3,), (-4,), (-5,), (-6,), (-7,), (-8,)]


def lsb(n):
    return str(n)[::-1]


def digit_pairs(a, b):
    da, db = str(a)[::-1], str(b)[::-1]
    w = max(len(da), len(db))
    return list(zip(da.ljust(w, "0"), db.ljust(w, "0")))


def carry_trace(a, b):
    """LSB-first worked addition; steps in order, each local to its pair."""
    steps, carry = [], 0
    for x, y in digit_pairs(a, b):
        s = int(x) + int(y) + carry
        steps.append(f"{x}+{y}+{carry}={s % 10}c{s // 10}"); carry = s // 10
    if carry:
        steps.append(f"+{carry}={carry}c0")
    return " ".join(steps)


def enc(s):
    return [BOS] + [ord(c) for c in s] + [EOS]


def make(n):
    seen, items = set(), []
    while len(items) < n:
        ab = (RNG.randint(0, 999), RNG.randint(0, 999))
        if ab in seen:
            continue
        seen.add(ab); items.append(ab)
    return items


def fit_model(bodies):
    psc = UniversalPSC(VOCAB, BACKOFF, ())
    psc.fit([((len(s),), {(i,): v for i, v in enumerate(s)}, ())
             for body in bodies for s in [enc(body)]])
    return psc


def run_direct_or_think(mode, train, test):
    bodies = []
    for a, b in train:
        ans = lsb(a + b)
        bodies.append(f"{a}+{b}=#{ans}" if mode == "direct"
                      else f"{a}+{b}= : {carry_trace(a, b)} #{ans}")
    psc = fit_model(bodies)

    def answer(a, b):
        seq = [BOS] + [ord(c) for c in f"{a}+{b}="]
        E = {(i,): v for i, v in enumerate(seq)}
        out, started = [], False
        for i in range(len(seq), len(seq) + (90 if mode == "think" else 12)):
            v = _sample(psc.predict(E, (), (i,)), 0.01, 1.0)
            if v == EOS:
                break
            E[(i,)] = v
            ch = chr(v) if v < 256 else ""
            if ch == "#":
                started = True; continue
            if started and ch.isdigit():
                out.append(ch)
            elif started:
                break
        return "".join(out)

    return 100 * sum(answer(a, b) == lsb(a + b) for a, b in test) / len(test)


def run_infill(train, test):
    """Each result digit + carry is predicted immediately after its pair, so
    every prediction is a local function of the last few tokens."""
    psc = fit_model([carry_trace(a, b) for a, b in train])

    def solve(a, b):
        seq, E = [BOS], {}
        def feed(s):
            for c in s:
                t = c if isinstance(c, int) else ord(c)
                E[(len(seq),)] = t; seq.append(t)
        carry, digits = 0, []
        for x, y in digit_pairs(a, b):
            feed(f"{x}+{y}+{carry}=")
            v = _sample(psc.predict(E, (), (len(seq),)), 0.01, 1.0); feed([v])
            d = chr(v) if v < 256 else "?"
            feed("c")
            v2 = _sample(psc.predict(E, (), (len(seq),)), 0.01, 1.0); feed([v2])
            digits.append(d)
            carry = int(chr(v2)) if v2 < 256 and chr(v2).isdigit() else 0
            feed(" ")
        if carry:
            digits.append(str(carry))
        return "".join(digits)

    return 100 * sum(solve(a, b) == lsb(a + b) for a, b in test) / len(test)


def main():
    train, test = make(4000), make(300)
    print("addition, 3-digit, held-out 300 unseen sums (answer LSB-first):\n")
    rows = [
        ("DIRECT  (prompt -> answer)", run_direct_or_think("direct", train, test)),
        ("THINK   (prompt -> carry trace -> answer)", run_direct_or_think("think", train, test)),
        ("INFILL  (each step computed right after its digits)", run_infill(train, test)),
    ]
    for label, acc in rows:
        print(f"  {label:52s} {acc:5.0f}%")
    print("\nSame model, same data, only token ORDER differs. DIRECT/THINK need a\n"
          "non-local COPY (attention); INFILL makes every step's inputs adjacent,\n"
          "so the counting substrate adds flawlessly and generalizes. No backprop.")


if __name__ == "__main__":
    main()
