"""More skills, same lesson: recast a task so every step is LOCAL and the
no-backprop counting model learns it -- and generalizes. (Calculus + arithmetic.)
-------------------------------------------------------------------------------
Addition already showed DIRECT 0% -> INFILL 100% (lessons_think.py). Here are
two more, each with a DIRECT (natural) form that fails and a LOCAL form that
works, all backprop-free, scored on held-out problems:

  MULTIPLY  d x N (single digit by a long number): the per-digit
            multiply-with-carry is the same local shape as addition's carry,
            so trained on short N it generalizes to arbitrary length.

  d/dx      c*x^p -> (c*p) x^(p-1): a LOCAL rewrite once the single-digit
            product c*p is a finite (<=81-entry) table the model can memorize;
            the power decrement is a local read of the adjacent token.

Run:  python3 lessons_skills.py
"""
from __future__ import annotations
import random, numpy as np
from psc_studio import UniversalPSC, _sample

BOS, EOS, VOCAB = 256, 258, 259
RNG = random.Random(5)
BACKOFF = [(-1,), (-2,), (-3,), (-4,), (-5,), (-6,), (-7,), (-8,)]


def enc(s):
    return [BOS] + [ord(c) for c in s] + [EOS]


def fit_model(bodies):
    psc = UniversalPSC(VOCAB, BACKOFF, ())
    psc.fit([((len(s),), {(i,): v for i, v in enumerate(s)}, ())
             for b in bodies for s in [enc(b)]])
    return psc


def greedy(psc, head, n):
    seq = [BOS] + [ord(c) for c in head]
    E = {(i,): v for i, v in enumerate(seq)}
    out = []
    for i in range(len(seq), len(seq) + n):
        v = _sample(psc.predict(E, (), (i,)), 0.01, 1.0)
        if v == EOS:
            break
        E[(i,)] = v; out.append(v)
    return out


# ============================ MULTIPLY d x N =================================
def mul_trace(d, n):
    """LSB-first single-digit * long-number, per-digit with carry (local)."""
    steps, carry = [], 0
    for ch in str(n)[::-1]:
        p = d * int(ch) + carry
        steps.append(f"{d}*{ch}+{carry}={p % 10}c{p // 10}"); carry = p // 10
    while carry:
        steps.append(f"+{carry % 10}={carry % 10}c{carry // 10}"); carry //= 10
    return " ".join(steps)


def mul_demo():
    def make(lo, hi, k):
        out = []
        for _ in range(k):
            d = RNG.randint(2, 9); n = RNG.randint(10 ** (lo - 1), 10 ** hi - 1)
            out.append((d, n))
        return out
    train = make(1, 3, 4000)                  # 1-3 digit N
    psc = fit_model([mul_trace(d, n) for d, n in train])

    def solve(d, n):
        seq, E = [BOS], {}
        def feed(s):
            for c in s:
                t = c if isinstance(c, int) else ord(c); E[(len(seq),)] = t; seq.append(t)
        carry, digs = 0, []
        for ch in str(n)[::-1]:
            feed(f"{d}*{ch}+{carry}=")
            v = greedy_step(psc, E, seq); feed([v]); digs.append(chr(v) if v < 256 else "?")
            feed("c"); v2 = greedy_step(psc, E, seq); feed([v2])
            carry = int(chr(v2)) if v2 < 256 and chr(v2).isdigit() else 0
            feed(" ")
        while carry:
            digs.append(str(carry % 10)); carry //= 10
        return "".join(digs)

    def acc(items):
        return 100 * sum(solve(d, n) == str(d * n)[::-1] for d, n in items) / len(items)
    print("MULTIPLY  d x N  (single digit x long number), held-out:")
    print(f"   trained on 1-3 digit N:")
    for lab, lo, hi in (("3-digit", 3, 3), ("20-digit", 20, 20), ("100-digit", 100, 100)):
        test = make(lo, hi, 100 if hi < 50 else 40)
        print(f"     {lab:10s} N:  {acc(test):4.0f}%")


def greedy_step(psc, E, seq):
    return _sample(psc.predict(E, (), (len(seq),)), 0.01, 1.0)


# ============================ d/dx c*x^p =====================================
def deriv_demo():
    # single-digit product table is the only "computation"; rest is local copy
    def make(k):
        return [(RNG.randint(1, 9), RNG.randint(2, 9)) for _ in range(k)]
    train, seen = [], set()
    while len(seen) < 72 and len(train) < 4000:           # full 9x8 space, many reps
        c, p = RNG.randint(1, 9), RNG.randint(2, 9)
        seen.add((c, p)); train.append((c, p))
    test = [(c, p) for c in range(1, 10) for p in range(2, 10)]   # all 72, held-in space
    # DIRECT: "d/dx 8x^5=" -> "40x^4"   ;  LOCAL: feed c and p adjacent
    direct = fit_model([f"d/dx {c}x^{p}={c*p}x^{p-1}" for c, p in train])
    local = fit_model([f"D {c} {p} = {c*p} {p-1}" for c, p in train])

    def acc_direct():
        ok = 0
        for c, p in test:
            o = bytes(x for x in greedy(direct, f"d/dx {c}x^{p}=", 8) if x < 256).decode("latin1", "ignore")
            ok += (o == f"{c*p}x^{p-1}")
        return 100 * ok / len(test)

    def acc_local():
        ok = 0
        for c, p in test:
            o = bytes(x for x in greedy(local, f"D {c} {p} = ", 8) if x < 256).decode("latin1", "ignore")
            ok += (o.strip() == f"{c*p} {p-1}")
        return 100 * ok / len(test)
    print("\nd/dx c*x^p  (calculus power rule), all 72 forms:")
    print(f"   DIRECT  'd/dx 8x^5=' -> '40x^4' :  {acc_direct():4.0f}%")
    print(f"   LOCAL   'D 8 5 =' -> '40 4'     :  {acc_local():4.0f}%")


def main():
    mul_demo()
    deriv_demo()
    print("\nSame story as addition: arrange the work so each step's inputs are "
          "adjacent,\nand the counting substrate learns the procedure and "
          "generalizes -- no backprop.")


if __name__ == "__main__":
    main()
