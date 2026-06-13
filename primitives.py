"""A primitive library: atomic shared ops every skill composes from.
-------------------------------------------------------------------------------
brain.py showed transfer is POSITIVE only for genuinely IDENTICAL shared
sub-procedures (and negative for merely-similar ones). So a compounding brain is
built deliberately: a registry of finite ATOMIC primitives, each learned once
into ONE shared model, and every complex skill is a COMPOSITION of primitive
queries against that model. Learning a primitive benefits every skill that
calls it; adding a new skill that reuses existing primitives is free.

Each primitive has a distinct leading tag (so different primitives don't pollute
each other at shallow contexts), but a primitive shared across skills is the
SAME query every time (so it transfers perfectly). That is the whole design.

  PRIMITIVES (finite, exact tables)            SKILLS (compositions)
    a  x y c -> digit-add-with-carry             add(a,b)        = chain of `a`
    m  d x c -> digit multiply-accumulate        mul(d,n)        = chain of `m`
    p  base  -> DNA complement                   revcomp(seq)    = chain of `p`
    r  base  -> DNA->RNA                          transcribe(seq) = chain of `r`
    g  codon -> amino acid                        translate(seq)  = chain of `g`

Run:  python3 primitives.py
"""
from __future__ import annotations
import random, os, pickle, numpy as np
from psc_studio import UniversalPSC, _sample

VOCAB = 259
BOS, EOS = 256, 258
RNG = random.Random(0)
COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}
T2U = {"A": "A", "C": "C", "G": "G", "T": "U"}
_BASES = "ACGT"
CODON = {x + y + z: "ACDEFGHIKLMNPQRSTVWY*"[i % 21]
         for i, (x, y, z) in enumerate(
             (a, b, c) for a in _BASES for b in _BASES for c in _BASES)}


# ---- the registry: each primitive yields its exact facts as tagged strings ----
def prim_facts():
    facts = []
    for x in range(10):
        for y in range(10):
            for c in range(3):                       # incoming carry 0..2
                t = x + y + c
                facts += [f"a {x} {y} {c}={t % 10}{t // 10}"] * 6
    for d in range(10):
        for x in range(10):
            for c in range(9):                       # mac incoming carry 0..8
                t = d * x + c
                facts += [f"m {d} {x} {c}={t % 10}{t // 10}"] * 3
    for b, v in COMP.items():
        facts += [f"p {b}={v}"] * 40
    for b, v in T2U.items():
        facts += [f"r {b}={v}"] * 40
    for k, v in CODON.items():
        facts += [f"g {k}={v}"] * 40
    return facts


class PrimitiveBrain:
    """One shared model trained on the primitive library; skills query it."""
    def __init__(self):
        self.psc = UniversalPSC(VOCAB, [(-i,) for i in range(1, 9)], ())
        self.learned = set()

    def learn(self, facts):
        seqs = [[BOS] + [b for b in s.encode()] + [EOS] for s in facts]
        self.psc.fit([((len(s),), {(i,): v for i, v in enumerate(s)}, ())
                      for s in seqs])

    def query(self, prefix, k):
        """Greedy-complete k chars after `prefix=` (a primitive lookup)."""
        seq = [BOS] + [b for b in prefix.encode()]
        E = {(i,): v for i, v in enumerate(seq)}
        out = []
        for i in range(len(seq), len(seq) + k):
            v = _sample(self.psc.predict(E, (), (i,)), 0.01, 1.0)
            if v == EOS or v >= 256:
                break
            E[(i,)] = v; out.append(chr(v))
        return "".join(out)

    # ---- primitive calls ----
    def p_add(self, x, y, c):
        o = self.query(f"a {x} {y} {c}=", 2)
        return (int(o[0]), int(o[1])) if len(o) == 2 and o.isdigit() else (0, 0)

    def p_mac(self, d, x, c):
        o = self.query(f"m {d} {x} {c}=", 2)
        return (int(o[0]), int(o[1])) if len(o) == 2 and o.isdigit() else (0, 0)

    def p_comp(self, b):
        return self.query(f"p {b}=", 1)

    def p_rna(self, b):
        return self.query(f"r {b}=", 1)

    def p_codon(self, k):
        return self.query(f"g {k}=", 1)


# ---- skills as COMPOSITIONS of primitive queries (no skill-specific data) ----
def add(brain, a, b):
    da, db = str(a)[::-1], str(b)[::-1]
    c, out = 0, []
    for i in range(max(len(da), len(db))):
        x = int(da[i]) if i < len(da) else 0
        y = int(db[i]) if i < len(db) else 0
        s, c = brain.p_add(x, y, c); out.append(str(s))
    if c:
        out.append(str(c))
    return "".join(out)                                  # LSB-first


def mul(brain, d, n):
    c, out = 0, []
    for ch in str(n)[::-1]:
        s, c = brain.p_mac(d, int(ch), c); out.append(str(s))
    while c:
        out.append(str(c % 10)); c //= 10
    return "".join(out)


def transcribe(brain, seq):
    return "".join(brain.p_rna(b) for b in seq)


def revcomp(brain, seq):
    return "".join(brain.p_comp(b) for b in reversed(seq))


def translate(brain, seq):
    return "".join(brain.p_codon(seq[i:i + 3]) for i in range(0, len(seq) - 2, 3))


def main():
    brain = PrimitiveBrain()
    facts = prim_facts()
    brain.learn(facts)
    print(f"ONE shared brain, taught {len(set(facts))} atomic primitive facts.\n")
    print("Every skill below is a COMPOSITION of those primitives -- the brain")
    print("saw ZERO full-skill examples; it chains primitive queries:\n")

    def acc(fn, n=200):
        return 100 * sum(fn() for _ in range(n)) / n

    a_ok = acc(lambda: add(brain, x := RNG.randint(0, 9999), y := RNG.randint(0, 9999))
               == str(x + y)[::-1])
    m_ok = acc(lambda: mul(brain, d := RNG.randint(2, 9), n := RNG.randint(0, 9999))
               == str(d * n)[::-1])
    def dna(n): return "".join(RNG.choice(_BASES) for _ in range(n))
    t_ok = acc(lambda: transcribe(brain, s := dna(40)) == s.replace("T", "U"))
    r_ok = acc(lambda: revcomp(brain, s := dna(40))
               == "".join(COMP[c] for c in s[::-1]))
    g_ok = acc(lambda: translate(brain, s := dna(3 * RNG.randint(2, 8)))
               == "".join(CODON[s[i:i+3]] for i in range(0, len(s) - 2, 3)))
    print(f"   add (4-digit)        {a_ok:4.0f}%      [composes primitive 'a']")
    print(f"   multiply (x 4-digit) {m_ok:4.0f}%      [composes primitive 'm']")
    print(f"   transcribe (40 base) {t_ok:4.0f}%      [composes primitive 'r']")
    print(f"   rev-complement       {r_ok:4.0f}%      [composes primitive 'p']")
    print(f"   translate protein    {g_ok:4.0f}%      [composes primitive 'g']")

    # COMPOUNDING: a NEW skill reusing an EXISTING primitive, zero new training
    def sum_list(nums):
        acc_s = "0"
        for v in nums:
            r = add(brain, int(acc_s[::-1]), v); acc_s = r
        return acc_s
    nl_ok = acc(lambda: sum_list(L := [RNG.randint(0, 999) for _ in range(5)])
                == str(sum(L))[::-1], n=100)
    print(f"\n   NEW skill 'sum a list' (reuses primitive 'a', no new training):"
          f" {nl_ok:4.0f}%")
    print("\nlearning a primitive benefits every skill that composes it; new "
          "skills\nthat reuse primitives are free. One brain, genuinely "
          "compounding -- no backprop.")


if __name__ == "__main__":
    main()
