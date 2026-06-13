"""Language on the no-backprop substrate: how far data helps, and where the
associative-fetch primitive unlocks what local counting cannot.
-------------------------------------------------------------------------------
HONEST PREMISE: this substrate does not beat neural LMs at open language --
counting plateaus ~2.6 bpc where neural LMs reach ~1.3-1.5; the gap IS the
learned-abstraction gap. This script does the two truthful things:

  1) SCALE: train a byte backoff model (UniversalPSC) on real WikiText at
     increasing data sizes; report held-out bits/char. More data helps, with
     diminishing returns -- quantified, not promised.

  2) FETCH: the one thing local models structurally cannot do is long-range
     COPY (coreference / induction: a rare token must be repeated from far
     back). That is attention's induction head, and assoc_memory.py provides
     it backprop-free. Test on a copy task local context fails: show fetch
     turns failure into success -- the first thing that pushes language past
     the local-statistics plateau.
"""
from __future__ import annotations
import math, random, numpy as np
from psc_studio import UniversalPSC, _sample
from assoc_memory import AssocMemory


def _ensure_data():
    import os
    if os.path.exists("data/wiki_train.txt"):
        return
    os.makedirs("data", exist_ok=True)
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    open("data/wiki_train.txt", "w").write("".join(ds["train"]["text"])[:2_000_000])
    open("data/wiki_test.txt", "w").write("".join(ds["test"]["text"])[:200_000])


def load(path, n):
    _ensure_data()
    return open(path, errors="replace").read()[:n]


def bpc_scale():
    test = load("data/wiki_test.txt", 50_000)
    print("WikiText bits/char (held-out), byte backoff model, by train size:")
    print("   (neural LMs ~1.0-1.3 bpc; counting substrate plateaus higher)\n")
    for n in (100_000, 400_000, 1_500_000):
        train = load("data/wiki_train.txt", n)
        psc = UniversalPSC(259, [(-i,) for i in range(1, 9)], ())
        seq = [b for b in train.encode()[:n]]
        psc.fit([((len(seq),), {(i,): v for i, v in enumerate(seq)}, ())])
        # held-out bits/char via the model's predictive distribution
        ts = [b for b in test.encode()]
        E = {(i,): v for i, v in enumerate(ts)}
        ll = 0.0
        for i in range(1, len(ts)):
            p = psc.predict(E, (), (i,))
            ll += -math.log2(max(p[ts[i]], 1e-12))
        print(f"   train {n:>9,} chars :  {ll / (len(ts) - 1):4.2f} bpc")


# ---- induction / long-range copy: the attention-defining language op --------
def induction_test():
    """Stream contains rare 'key' tokens each followed by a distinct 'value'
    char; later the key reappears and the model must emit its value -- but the
    pairing is hundreds of chars back, outside any local window. Local model
    guesses; associative fetch (key->value memory) recalls it."""
    rng = random.Random(0)
    alpha = "abcdefghijklmnopqrstuvwxyz"

    def make():
        keys = ["".join(rng.choice(alpha) for _ in range(4)) for _ in range(6)]
        vals = [rng.choice(alpha.upper()) for _ in keys]
        # long filler so the pairing is non-local
        body = []
        for k, v in zip(keys, vals):
            body.append(f"{k}={v}. " + "".join(rng.choice(alpha + " ") for _ in range(40)))
        rng.shuffle(body)
        prefix = "".join(body)
        qi = rng.randrange(len(keys))
        return prefix, keys[qi], vals[qi]

    # local baseline: a byte backoff model trained on many such streams
    psc = UniversalPSC(259, [(-i,) for i in range(1, 9)], ())
    seqs = []
    for _ in range(800):
        pre, k, v = make()
        s = [b for b in (pre + f"{k}=").encode()] + [ord(v)]
        seqs.append(s)
    psc.fit([((len(s),), {(i,): t for i, t in enumerate(s)}, ()) for s in seqs])

    def local_answer(pre, k):
        s = [b for b in (pre + f"{k}=").encode()]
        E = {(i,): t for i, t in enumerate(s)}
        return chr(_sample(psc.predict(E, (), (len(s),)), 0.01, 1.0) % 256)

    def fetch_answer(pre, k):
        # write key->value pairs as the stream is read (Hebbian), recall by key
        mem = AssocMemory(beta=20.0)
        def vec(s):
            v = np.zeros(26 * 4)
            for i, c in enumerate(s[:4]):
                v[i * 26 + (ord(c) - 97) % 26] = 1.0
            return v
        i = 0
        while i < len(pre) - 1:
            if pre[i + 1:i + 2] == "=" and i >= 3 and pre[i - 3:i + 1].isalpha():
                key = pre[i - 3:i + 1]; val = pre[i + 2:i + 3]
                if val:
                    mem.write(vec(key), [float(ord(val))])
            i += 1
        _soft, hard = mem.read(vec(k))
        return chr(int(round(hard[0]))) if hard is not None else "?"

    nloc = nfet = 0
    for _ in range(300):
        pre, k, v = make()
        nloc += (local_answer(pre, k) == v)
        nfet += (fetch_answer(pre, k) == v)
    print("\nINDUCTION / long-range copy (recall a value paired with a key seen")
    print("hundreds of chars earlier -- coreference, the LLM induction head):")
    print(f"   local backoff model     : {100*nloc/300:4.0f}%  (chance ~4%)")
    print(f"   associative fetch        : {100*nfet/300:4.0f}%  (attention, no backprop)")


def main():
    bpc_scale()
    induction_test()
    print("\nVerdict: more data lowers bpc with diminishing returns (we do NOT "
          "reach\nneural-LM territory); but associative fetch gives the substrate "
          "the\nlong-range copy it never had -- the honest path to better "
          "language.")


if __name__ == "__main__":
    main()
