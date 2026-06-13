"""Verifiable-reward curriculum: teach the one mind math/physics by CHECKING.
-------------------------------------------------------------------------------
No backprop means no loss functions. The equivalent here is a VERIFIABLE
REWARD: a checker that says correct/incorrect. A Task is anything that can
  - sample()  -> (prompt, answer)   a fresh problem and its truth
  - check(a)  -> bool               verify a proposed answer (the reward)
That's the whole interface; everything below is one of these. The SAME
UniversalPSC stream model (the thing that draws 7s and plays GridWorld) is
taught problem->answer sequences and then scored on HELD-OUT problems it has
never seen -- generalization, not memorization.

Honest expectation, consistent with this project's standing finding: the
substrate is a counting/retrieval machine. Tasks whose answer is a lookup or a
short LOCAL rewrite (digit arithmetic with carries, d/dx x^n -> n x^(n-1),
sequence-next, unit conversion) should learn and generalize. Multi-step
derivation will plateau. The table this prints is the verdict, not a promise.

Run:  python3 lessons.py            (train + held-out report)
      python3 lessons.py 20000      (more practice)
"""
from __future__ import annotations
import random, sys, numpy as np
from psc_studio import UniversalPSC, _sample

BOS, SEP, EOS = 256, 257, 258
VOCAB = 259
RNG = random.Random(0)


# ------------------------------- tasks ---------------------------------------
# Numbers are emitted LEAST-SIGNIFICANT-DIGIT FIRST so that a carry is a
# left-to-right LOCAL dependency the stream model can actually learn -- the
# single representational choice that makes arithmetic learnable by counting.
def lsb(n: int) -> str:
    return str(abs(n))[::-1]


class Add:
    name = "addition"
    def sample(self):
        a, b = RNG.randint(0, 999), RNG.randint(0, 999)
        return f"{a}+{b}=", lsb(a + b)
    def check(self, got, truth):
        return got == truth


class Sub:
    name = "subtraction"
    def sample(self):
        a, b = RNG.randint(0, 999), RNG.randint(0, 999)
        a, b = max(a, b), min(a, b)
        return f"{a}-{b}=", lsb(a - b)
    def check(self, got, truth):
        return got == truth


class Mod:
    name = "modulo"
    def sample(self):
        a, b = RNG.randint(0, 999), RNG.randint(2, 12)
        return f"{a}%{b}=", lsb(a % b)
    def check(self, got, truth):
        return got == truth


class SeqNext:
    name = "sequence-next"
    def sample(self):
        s = RNG.randint(0, 20); d = RNG.randint(-9, 9)
        terms = [s + d * i for i in range(4)]
        return " ".join(map(str, terms)) + " ?=", lsb(s + d * 4)
    def check(self, got, truth):
        return got == truth


class Deriv:
    name = "d/dx poly"
    def sample(self):
        # d/dx of c*x^p (rule: -> (c*p) x^(p-1)); a verifiable LOCAL rewrite
        c, p = RNG.randint(1, 9), RNG.randint(2, 9)
        ans = f"{c*p}x^{p-1}"
        return f"d/dx {c}x^{p}=", ans
    def check(self, got, truth):
        return got == truth


class Units:
    name = "unit convert"
    FAC = {"km": 1000, "m": 1, "cm": 0.01, "h": 3600, "min": 60, "s": 1}
    def sample(self):
        pairs = [("km", "m"), ("m", "cm"), ("h", "s"), ("min", "s")]
        u, v = RNG.choice(pairs); n = RNG.randint(1, 99)
        val = int(n * self.FAC[u] / self.FAC[v])
        return f"{n}{u} in {v}=", lsb(val)
    def check(self, got, truth):
        return got == truth


TASKS = [Add(), Sub(), Mod(), SeqNext(), Deriv(), Units()]


# --------------------------- teach + verify ----------------------------------
def encode(prompt, answer):
    return [BOS] + [ord(c) for c in prompt] + [SEP] + [ord(c) for c in answer] + [EOS]


def main():
    n_each = int(sys.argv[1]) // len(TASKS) if len(sys.argv) > 1 else 2500
    psc = UniversalPSC(VOCAB, [(-1,), (-2,), (-3,), (-4,), (-5,), (-6,)], ())

    # hold out the LAST problems of each task as an unseen test set
    # Collect UNIQUE problems per task, but bound the attempts: some tasks
    # have a small finite problem space (e.g. d/dx of c*x^p), so we take
    # whatever uniques exist instead of spinning forever. Hold out ~20% (<=200)
    # of each task's unique set as the unseen test; never let test eat the
    # whole space (then "held-out" would be meaningless).
    train, tests = [], {t.name: [] for t in TASKS}
    for t in TASKS:
        pool, seen, miss = [], set(), 0
        while len(pool) < n_each + 200 and miss < 4000:
            p, a = t.sample()
            if p in seen:
                miss += 1; continue
            seen.add(p); pool.append((p, a)); miss = 0
        RNG_local = __import__("random").Random(7)
        RNG_local.shuffle(pool)
        n_test = min(200, max(1, len(pool) // 5))
        tests[t.name] = pool[:n_test]
        train += [(t, p, a) for p, a in pool[n_test:]]

    psc.fit([((len(s),), {(i,): v for i, v in enumerate(s)}, ())
             for _, p, a in train for s in [encode(p, a)]])
    print(f"taught {len(train)} problems across {len(TASKS)} subjects\n")

    def answer(prompt):
        seq = [BOS] + [ord(c) for c in prompt] + [SEP]
        E = {(i,): v for i, v in enumerate(seq)}
        out = []
        for i in range(len(seq), len(seq) + 16):
            v = _sample(psc.predict(E, (), (i,)), temp=0.01, top_p=1.0)
            if v == EOS or v >= 256:
                break
            E[(i,)] = v; out.append(chr(v))
        return "".join(out)

    print(f"{'subject':16s} {'held-out':>10s} {'(n)':>6s}   example")
    print("-" * 62)
    for t in TASKS:
        ok = 0; ex = ""
        for p, truth in tests[t.name]:
            got = answer(p)
            good = t.check(got, truth)
            ok += good
            if not ex:
                # show the answer in human (MSB) order
                sh = got[::-1] if t.name != "d/dx poly" else got
                tr = truth[::-1] if t.name != "d/dx poly" else truth
                ex = f"{p}{sh}  ({'ok' if good else 'want '+tr})"
        acc = 100 * ok / len(tests[t.name])
        print(f"{t.name:16s} {acc:9.0f}% {len(tests[t.name]):>6d}   {ex}")
    print("\n(answers generated least-significant-digit-first; verifiable "
          "reward = the checker, no backprop)")


if __name__ == "__main__":
    main()
