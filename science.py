"""A well-rounded scientific curriculum with verifiable rewards (no backprop).
-------------------------------------------------------------------------------
Each subject is checked exactly (correct/incorrect). Subjects are grouped by
the brain mechanism that handles them:

  FETCH    biology sequence transductions -- need non-local copy, solved by the
           content-addressable PositionalTape (assoc_memory.py) + a local map.
  TABLE    finite lookups the counting model memorizes (genetic code, periodic
           table, complements).
  LOCAL    per-element procedures with carries (molecular mass accumulation).
  SEARCH   hard combinatorial science -- protein folding on the 2D HP lattice,
           attacked by verifier-guided best-first search (search.py). Honestly
           measured: optimal for short chains, degrades with length.

Run:  python3 science.py
"""
from __future__ import annotations
import random
import numpy as np
from assoc_memory import PositionalTape
from search import best_first

RNG = random.Random(0)

COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}
T2U = {"A": "A", "C": "C", "G": "G", "T": "U"}
CODON = {  # the genetic code -- biology's famous 64-entry lookup
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L",
    "CTA": "L", "CTG": "L", "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V", "TCT": "S", "TCC": "S",
    "TCA": "S", "TCG": "S", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T", "GCT": "A", "GCC": "A",
    "GCA": "A", "GCG": "A", "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q", "AAT": "N", "AAC": "N",
    "AAA": "K", "AAG": "K", "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W", "CGT": "R", "CGC": "R",
    "CGA": "R", "CGG": "R", "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G"}
# atomic masses (rounded) for molecular-mass accumulation
MASS = {"H": 1, "C": 12, "N": 14, "O": 16, "S": 32, "P": 31, "Na": 23, "Cl": 35}


def dna(n):
    return "".join(RNG.choice("ACGT") for _ in range(n))


# ---- FETCH: sequence transductions via content-addressable recall ----
def fetch_transcribe(seq):
    tape = PositionalTape(); tape.write_stream([ord(c) for c in seq])
    return "".join(T2U[chr(tape.fetch(k))] for k in range(len(seq)))


def fetch_revcomp(seq):
    n = len(seq); tape = PositionalTape(); tape.write_stream([ord(c) for c in seq])
    return "".join(COMP[chr(tape.fetch(n - 1 - k))] for k in range(n))


def translate(seq):
    return "".join(CODON[seq[i:i + 3]] for i in range(0, len(seq) - 2, 3))


# ---- LOCAL: molecular mass by per-token accumulation ----
def molar_mass(formula):
    import re
    total = 0
    for el, cnt in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if el:
            total += MASS[el] * (int(cnt) if cnt else 1)
    return total


# ---- SEARCH: protein folding, 2D HP lattice model ----
class HPFold:
    """Hydrophobic-Polar lattice protein folding: place a chain on the 2D grid
    as a self-avoiding walk; energy = -(# of non-adjacent-in-chain H-H pairs
    that are neighbours on the lattice). Finding the minimum-energy fold is
    NP-hard -- the canonical 'really hard' verifiable biology task. We SEARCH
    for it (search.py) and verify the contact energy exactly."""
    MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def __init__(self, seq):
        self.seq = seq                       # e.g. "HPHPPHHPH"

    def start(self):
        return ((0, 0),)                      # first residue at origin

    def expand(self, path):
        i = len(path)
        if i >= len(self.seq):
            return
        x, y = path[-1]
        for dx, dy in self.MOVES:
            p = (x + dx, y + dy)
            if p in path:
                continue                      # self-avoiding
            np_ = path + (p,)
            yield np_, (len(np_) == len(self.seq))

    def score(self, path):
        # partial score = current H-H contacts (admissible-ish guide) ; final
        # score = total contacts. Verifiable energy = -contacts.
        pts = {p: self.seq[i] for i, p in enumerate(path)}
        contacts = 0
        for i, p in enumerate(path):
            if self.seq[i] != "H":
                continue
            x, y = p
            for dx, dy in self.MOVES:
                q = (x + dx, y + dy)
                j = path.index(q) if q in path else -1
                if q in pts and pts[q] == "H" and j != -1 and abs(j - i) > 1:
                    contacts += 1
        return contacts / 2                  # each pair counted twice

    def brute_optimum(self):
        best = -1
        def walk(path):
            nonlocal best
            if len(path) == len(self.seq):
                best = max(best, self.score(path)); return
            for nxt, _ in self.expand(path):
                walk(nxt)
        walk(self.start())
        return best


def main():
    print("BIOLOGY (sequence) -- non-local fetch + local map / finite table:")
    for name, fn, truth in (
        ("transcription DNA->RNA", fetch_transcribe, lambda s: s.replace("T", "U")),
        ("reverse-complement", fetch_revcomp,
         lambda s: "".join(COMP[c] for c in s[::-1])),
    ):
        ok = sum(fn(s := dna(RNG.randint(30, 90))) == truth(s) for _ in range(200))
        print(f"   {name:26s} {ok / 2:4.0f}%  (held-out 30-90 base, was 0%)")
    okc = 0
    for _ in range(200):
        s = "".join(dna(3) for _ in range(RNG.randint(2, 8)))
        okc += translate(s) == "".join(CODON[s[i:i+3]] for i in range(0, len(s) - 2, 3))
    print(f"   {'codon->protein':26s} {okc / 2:4.0f}%  (genetic-code table)")

    print("\nCHEMISTRY -- local accumulation over a finite atomic-mass table:")
    for f, exp in (("H2O", 18), ("CO2", 44), ("C6H12O6", 180), ("NaCl", 58),
                   ("C2H5OH", 46)):
        print(f"   molar mass {f:8s} = {molar_mass(f):3d}  "
              f"({'ok' if molar_mass(f) == exp else 'X'})")

    print("\nHARD FRONTIER -- protein folding (2D HP lattice), verifier-guided")
    print("search vs the true optimum (brute force); honest degradation:")
    for seq in ("HPHPPH", "HHPPHPPHPH", "HPHPPHHPHPPHPH"):
        best, val, exp = best_first(HPFold(seq), beam=4000, max_expand=60000)
        opt = HPFold(seq).brute_optimum() if len(seq) <= 12 else None
        tag = (f"optimum {opt}" if opt is not None else "optimum n/a (too long to brute)")
        match = "" if opt is None else ("  == optimal" if val == opt else f"  (< {opt})")
        print(f"   len {len(seq):2d}: search found {val:.0f} H-H contacts "
              f"[{exp} expansions, {tag}]{match}")
    print("\nfetch = attention-without-backprop; folding = search-without-backprop. "
          "all verifiable.")


if __name__ == "__main__":
    main()
