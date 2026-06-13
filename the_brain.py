"""THE BRAIN: one big persistent model holding every cognitive skill.
-------------------------------------------------------------------------------
Not a router over sub-brains -- ONE shared predictive model (UniversalPSC over a
single unified vocabulary) + ONE associative memory, that all skills read and
write, persisted to a single file and accumulating across runs. Skills are
COMPOSED from a shared primitive library (the only design that transfers, per
brain.py): identical primitives are the same query everywhere, tagged so they
don't pollute. Knowledge lives in the one model; group theory / search / the
codecs are innate operators it composes (like a brain's hardwired cortex ops).

Faculties, all from the SAME model instance:
  arithmetic   add / multiply        (compose digit primitives a, m)
  biology      transcribe / revcomp / translate  (primitives r, p, g)
  language     name a number         (primitive n)
  memory       remember K=V / recall (the associative-fetch primitive)
  vision       draw a digit          (unified vocab vision codes, like one_mind)

Run:  python3 the_brain.py            -> grows, persists, runs a mixed battery
      python3 the_brain.py "add 347 285" | "translate ATGCCG" | "draw 7" | ...
"""
from __future__ import annotations
import os, re, sys, math, pickle, random, numpy as np
from substrate import UniversalPSC, _sample
from psc_omni import VOCAB, VIS0, KV, BOS, EOS, SEP, txt, WORD
from assoc_memory import AssocMemory

np.seterr(over="ignore", invalid="ignore", divide="ignore")
RNG = random.Random(0)
PATH = "outputs/the_brain.pkl"
COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}
T2U = {"A": "A", "C": "C", "G": "G", "T": "U"}
_B = "ACGT"
CODON = {x+y+z: "ACDEFGHIKLMNPQRSTVWY*"[i % 21] for i, (x, y, z) in
         enumerate((a, b, c) for a in _B for b in _B for c in _B)}


class TheBrain:
    def __init__(self):
        self.psc = UniversalPSC(VOCAB, [(-i,) for i in range(1, 9)], ())
        self.mem = AssocMemory(beta=20.0)
        self.cod = None                                  # vision codec (lazy)
        self.taught = set()
        self.read_pos = 0                                # chars of text ingested

    # ---------- learning into the ONE model ----------
    def _teach(self, seqs):
        self.psc.fit([((len(s),), {(i,): v for i, v in enumerate(s)}, ())
                      for s in seqs])

    def grow(self, vision=True):
        """Populate the one model with the primitive library + language."""
        if "cog" not in self.taught:
            facts = []
            for x in range(10):
                for y in range(10):
                    for c in range(3):
                        t = x+y+c; facts += [f"a {x} {y} {c}={t%10}{t//10}"]*6
            for d in range(10):
                for x in range(10):
                    for c in range(9):
                        t = d*x+c; facts += [f"m {d} {x} {c}={t%10}{t//10}"]*3
            for b, v in COMP.items(): facts += [f"p {b}={v}"]*40
            for b, v in T2U.items(): facts += [f"r {b}={v}"]*40
            for k, v in CODON.items(): facts += [f"g {k}={v}"]*40
            for d in range(10): facts += [f"n {d}={WORD[d]}"]*40
            self._teach([[BOS]+[c for c in s.encode()]+[EOS] for s in facts])
            self.taught.add("cog")
        if vision and "vision" not in self.taught:
            self._grow_vision(); self.taught.add("vision")

    def _grow_vision(self, n=2500):
        from torchvision.datasets import MNIST
        from PIL import Image
        from psc_omni import Codecs
        mn = MNIST(root="./data", train=True, download=True)
        X = mn.data.numpy().astype(np.float32)[:n]/255.0; y = mn.targets.numpy()[:n]
        X32 = np.stack([np.asarray(Image.fromarray(np.uint8(x*255)).resize((32, 32)),
                                   np.float32)/255.0 for x in X])
        self.cod = Codecs(); self.cod.fit(X32, y)
        seqs = [[BOS]+txt(f"draw {int(y[i])}")+[SEP]+list(self.cod.vis_enc(X32[i]))+[EOS]
                for i in range(n)]
        self._teach(seqs)

    # ---------- one query interface against the ONE model ----------
    def _q(self, prefix, k):
        seq = [BOS]+[b for b in prefix.encode()]; E = {(i,): v for i, v in enumerate(seq)}
        out = []
        for i in range(len(seq), len(seq)+k):
            v = _sample(self.psc.predict(E, (), (i,)), 0.01, 1.0)
            if v == EOS or v >= 256: break
            E[(i,)] = v; out.append(chr(v))
        return "".join(out)

    def _digit(self, p, k=2):
        o = self._q(p, k); return o if o.isdigit() else "00"

    def add(self, a, b):
        da, db = str(a)[::-1], str(b)[::-1]; c, out = 0, []
        for i in range(max(len(da), len(db))):
            x = int(da[i]) if i < len(da) else 0; y = int(db[i]) if i < len(db) else 0
            o = self._digit(f"a {x} {y} {c}="); out.append(o[0]); c = int(o[1])
        if c: out.append(str(c))
        return int("".join(out)[::-1])

    def mul(self, d, n):
        c, out = 0, []
        for ch in str(n)[::-1]:
            o = self._digit(f"m {d} {ch} {c}="); out.append(o[0]); c = int(o[1])
        while c: out.append(str(c % 10)); c //= 10
        return int("".join(out)[::-1])

    def transcribe(self, s): return "".join(self._q(f"r {b}=", 1) for b in s)
    def revcomp(self, s): return "".join(self._q(f"p {b}=", 1) for b in reversed(s))
    def translate(self, s):
        return "".join(self._q(f"g {s[i:i+3]}=", 1) for i in range(0, len(s)-2, 3))
    def name(self, d): return self._q(f"n {d}=", 8)

    def remember(self, k, v):
        self.mem.write(self._kv(k), [float(ord(v[0]))])
    def recall(self, k):
        _s, hard = self.mem.read(self._kv(k))
        return chr(int(round(hard[0]))) if hard is not None else "?"
    def _kv(self, s):
        v = np.zeros(26*4)
        for i, ch in enumerate(s[:4].upper()):
            if "A" <= ch <= "Z": v[i*26 + ord(ch)-65] = 1.0
        return v

    def draw(self, d, path=None):
        toks = [t for t in self._roll(txt(f"draw {d}")+[SEP], 40)
                if VIS0 <= t < VIS0+KV][:16]
        toks = (toks + [VIS0]*16)[:16]
        img = self.cod.vis_dec(toks)
        from PIL import Image
        path = path or f"outputs/the_brain_draw_{d}.png"
        Image.fromarray(np.uint8(np.clip(img, 0, 1)*255)).resize((96, 96),
                        Image.NEAREST).save(path)
        return path

    def _roll(self, prefix, maxlen):
        seq = [BOS]+list(prefix); E = {(i,): v for i, v in enumerate(seq)}
        for i in range(len(seq), maxlen):
            v = _sample(self.psc.predict(E, (), (i,)), 0.5, 0.92)
            if v == EOS: break
            E[(i,)] = v; seq.append(v)
        return seq[len(prefix)+1:]

    # ---------- persistence: ONE file ----------
    def save(self):
        os.makedirs("outputs", exist_ok=True)
        with open(PATH, "wb") as f:
            pickle.dump({"t": self.psc.t, "taught": self.taught,
                         "K": self.mem.K, "V": self.mem.V, "cod": self.cod,
                         "read_pos": self.read_pos}, f)

    def load(self):
        if not os.path.exists(PATH):
            return False
        d = pickle.load(open(PATH, "rb"))
        self.psc.t = d["t"]; self.taught = d["taught"]
        self.mem.K = d["K"]; self.mem.V = d["V"]; self.cod = d["cod"]
        self.read_pos = d.get("read_pos", 0)
        return True

    # ---------- lifelong: the one brain keeps READING (genuine growth) ----------
    def ingest_text(self, chunk):
        self._teach([[BOS] + [b for b in chunk.encode()] + [EOS]])

    def heldout_bpc(self, text):
        ts = [b for b in text.encode()]
        E = {(i,): v for i, v in enumerate(ts)}
        ll = 0.0
        for i in range(1, len(ts)):
            p = self.psc.predict(E, (), (i,))
            ll += -math.log2(max(p[ts[i]], 1e-12))
        return ll / max(1, len(ts) - 1)

    def faculty_check(self):
        a = self.add(347, 285) == 632
        m = self.mul(7, 412) == 2884
        t = self.transcribe("ACGT") == "ACGU"
        n = self.name(5) == "five"
        self.remember("CAT", "M"); r = self.recall("CAT") == "M"
        return dict(add=a, mul=m, transcribe=t, name=n, recall=r)

    def lifelong(self, chunk=40000, test_n=15000):
        if not self.load():
            print("growing the one brain ..."); self.grow(vision=True); self.save()
        corpus = open("data/wiki_train.txt", errors="replace").read()
        test = corpus[-test_n:-test_n + 4000]
        train_end = max(1, len(corpus) - test_n)
        log = open("outputs/brain_life.out", "a")
        cyc = 0
        print("lifelong single-brain learning (reads more each cycle); "
              "stop: touch outputs/STOP")
        while not os.path.exists("outputs/STOP"):
            cyc += 1
            start = self.read_pos % max(1, train_end - chunk)
            self.ingest_text(corpus[start:start + chunk])
            self.read_pos += chunk
            bpc = self.heldout_bpc(test)
            fac = self.faculty_check()
            kept = sum(fac.values())
            line = (f"cycle {cyc}: read {self.read_pos:,} chars  "
                    f"heldout {bpc:.3f} bpc  faculties {kept}/5 "
                    f"{''.join('.' if v else 'X' for v in fac.values())}")
            print(line); log.write(line + "\n"); log.flush()
            self.save()

    # ---------- one mouth: parse intent, compose, answer ----------
    def ask(self, text):
        t = text.strip(); low = t.lower()
        m = re.search(r"(\d+)\s*[+]\s*(\d+)", t) or (
            re.search(r"add (\d+) (\d+)", low))
        if m: return str(self.add(int(m[1]), int(m[2])))
        m = re.search(r"(\d+)\s*[x*]\s*(\d+)", t) or re.search(r"multiply (\d+) (?:by )?(\d+)", low)
        if m: return str(self.mul(int(m[1]), int(m[2])))
        m = re.search(r"remember (\w+)\s*=\s*(\w+)", low)
        if m: self.remember(m[1], m[2]); return f"ok, {m[1]}={m[2]}"
        m = re.search(r"(?:recall|what is) (\w+)", low)
        if m: return f"{m[1]} = {self.recall(m[1])}"
        m = re.search(r"name (\d)", low) or re.search(r"spell (\d)", low)
        if m: return self.name(int(m[1]))
        m = re.search(r"draw (\d)", low) or re.search(r"image of (?:a )?(\d)", low)
        if m: return "drew -> " + self.draw(int(m[1]))
        m = re.search(r"translate ([acgt]+)", low)
        if m: return self.translate(m[1].upper())
        m = re.search(r"(?:reverse.?complement|revcomp|complement) ([acgt]+)", low)
        if m: return self.revcomp(m[1].upper())
        m = re.search(r"transcribe ([acgt]+)", low)
        if m: return self.transcribe(m[1].upper())
        return "I can: add/multiply, transcribe/complement/translate DNA, name a digit, draw a digit, remember/recall."


def main():
    if "--lifelong" in sys.argv:
        TheBrain().lifelong(); return
    b = TheBrain()
    fresh = not b.load()
    if fresh:
        print("growing one brain (cognitive + vision) ...")
        b.grow(vision=True); b.save()
    sizes = sum(len(d) for d in b.psc.t)
    print(f"ONE brain: {sizes:,} context entries in a SINGLE model, "
          f"{len(b.mem.K)} memories, vision={'yes' if b.cod else 'no'}\n")
    if len(sys.argv) > 1:
        print(b.ask(" ".join(sys.argv[1:]))); return
    print("a mixed battery, every answer from the SAME model:")
    b.remember("CAT", "M"); b.remember("DOG", "K")
    for q in ["add 347 285", "multiply 7 by 412", "name 5",
              "translate ATGCCGTGG", "revcomp ACGTACGT", "transcribe ACGT",
              "recall CAT", "recall DOG"]:
        print(f"   '{q}'  ->  {b.ask(q)}")
    if b.cod:
        print(f"   'draw 3'  ->  {b.ask('draw 3')}")
    print(f"\none model, one memory, one file ({PATH}). no router, no sub-brains.")


if __name__ == "__main__":
    main()
