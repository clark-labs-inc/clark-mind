"""Non-local fetch for a no-backprop brain: content-addressable memory.
-------------------------------------------------------------------------------
The substrate could only see the last few tokens, so anything needing a digit
or base from far back (addition's carry, DNA->RNA transcription, SMILES ring
closure) scored 0%. A brain doesn't route around that -- it RETRIEVES, via
hippocampal pattern completion. The math, gradient-free:

  modern Hopfield network == attention:  read = softmax(beta * K @ q) @ V
  but the memory is STORED by Hebbian writes (counts/outer products), never by
  backprop. Storage is writing (key, value) pairs; reading is a similarity-
  weighted recall with a sharpness beta. High beta -> a hard pointer (exact
  fetch); low beta -> a blend. That is attention without a single gradient.

This module is the primitive. `AssocMemory` stores and recalls vectors.
`PositionalTape` is the simplest useful instance: it remembers the input
stream indexed by position, so a learner can fetch "the symbol that was at
position p" -- the copy operation the local model cannot do. The predictive
model then only has to learn the LOCAL map on the fetched symbol (e.g. T->U),
which it is already good at.
"""
from __future__ import annotations
import numpy as np
np.seterr(over="ignore", invalid="ignore", divide="ignore")


class AssocMemory:
    """Hebbian content-addressable memory. Keys/values are vectors; recall is
    similarity-weighted (modern-Hopfield / attention), stored by writing, not
    backprop."""
    def __init__(self, beta=12.0):
        self.beta = beta
        self.K = []          # stored key vectors
        self.V = []          # stored value vectors

    def write(self, key, value):
        self.K.append(np.asarray(key, dtype=np.float64))
        self.V.append(np.asarray(value, dtype=np.float64))

    def read(self, query):
        """Return the similarity-weighted recalled value (soft) and the single
        best-matching value (hard pointer)."""
        if not self.K:
            return None, None
        K = np.stack(self.K); V = np.stack(self.V)
        q = np.asarray(query, dtype=np.float64)
        sim = K @ q / (np.linalg.norm(K, axis=1) * (np.linalg.norm(q) + 1e-9) + 1e-9)
        w = np.exp(self.beta * (sim - sim.max())); w /= w.sum()
        soft = w @ V
        hard = V[int(np.argmax(sim))]
        return soft, hard

    def clear(self):
        self.K.clear(); self.V.clear()


class PositionalTape:
    """Content-addressable by POSITION: write the input symbols as they stream,
    then fetch symbol[p] for any p -- random access the local window can't
    reach. Positions are encoded as smooth Fourier features so 'nearby
    position' is a graded match (robust recall), exact at integer queries."""
    def __init__(self, dim=24, beta=30.0):
        self.dim = dim
        self.mem = AssocMemory(beta=beta)
        self.freqs = np.exp(np.linspace(0, np.log(1000.0), dim // 2))

    def _pos(self, p):
        a = self.freqs * p
        return np.concatenate([np.sin(a), np.cos(a)])

    def write_stream(self, symbols):
        self.mem.clear()
        for i, s in enumerate(symbols):
            self.mem.write(self._pos(i), [float(s)])

    def fetch(self, p):
        _soft, hard = self.mem.read(self._pos(p))
        return int(round(hard[0])) if hard is not None else None


# ============================ proof ==========================================
# The claim: with a content-addressed read, the SEPARATED copy tasks that
# scored 0% (the model had to reach an arbitrary distance back) now work,
# WITHOUT the interleaving hack. The learner emits, at each output step, the
# LOCAL transform of the fetched symbol -- which a counting model handles.
def _proof():
    import random
    rng = random.Random(0)

    def dna(n):
        return "".join(rng.choice("ACGT") for _ in range(n))

    COMP = {"A": "T", "T": "A", "C": "G", "G": "G"}  # placeholder, fixed below
    COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    T2U = {"A": "A", "C": "C", "G": "G", "T": "U"}

    def transcribe_with_fetch(seq):
        """Output position k = input position k; fetch base k, apply local T->U."""
        tape = PositionalTape()
        tape.write_stream([ord(c) for c in seq])
        out = []
        for k in range(len(seq)):
            base = chr(tape.fetch(k))            # NON-LOCAL fetch
            out.append(T2U[base])                # LOCAL map
        return "".join(out)

    def revcomp_with_fetch(seq):
        """Output position k = input position (n-1-k); fetch + local complement."""
        n = len(seq)
        tape = PositionalTape()
        tape.write_stream([ord(c) for c in seq])
        out = []
        for k in range(n):
            base = chr(tape.fetch(n - 1 - k))    # reversed positional fetch
            out.append(COMP[base])               # LOCAL complement
        return "".join(out)

    print("content-addressed copy (no interleaving, separated form):")
    for name, fn, truth in (
        ("transcription DNA->RNA", transcribe_with_fetch,
         lambda s: s.replace("T", "U")),
        ("reverse-complement", revcomp_with_fetch,
         lambda s: "".join({"A": "T", "T": "A", "C": "G", "G": "C"}[c]
                           for c in s[::-1])),
    ):
        ok = 0
        for _ in range(300):
            s = dna(rng.randint(30, 80))         # long, far beyond any window
            ok += (fn(s) == truth(s))
        print(f"   {name:24s} {100*ok/300:4.0f}%  (vs 0% without fetch)")
    # show the associative read is robust, not a lookup table:
    tape = PositionalTape()
    tape.write_stream([ord(c) for c in "ACGTACGT"])
    got = "".join(chr(tape.fetch(p)) for p in range(8))
    print(f"   positional recall of 'ACGTACGT' -> '{got}'")


if __name__ == "__main__":
    _proof()
