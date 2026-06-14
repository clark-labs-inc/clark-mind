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
from psc_omni import VOCAB, VIS0, KV, AUD0, KA, BOS, EOS, SEP, FRAME, txt, WORD, kmeans
from assoc_memory import AssocMemory
from predictive_coding import PredictiveCoding      # the free-energy deep faculty

np.seterr(over="ignore", invalid="ignore", divide="ignore")
RNG = random.Random(0)
PATH = "outputs/the_brain.pkl"
MAX_CTX = 3_000_000          # context budget; sleep consolidates back to it
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
        self.images_seen = 0                             # cifar images ingested
        self.audio_seen = 0; self.video_seen = 0
        self.aud_C = None; self.aud_mu = None; self.aud_sd = None   # audio codec
        self.sem = None                                  # semantic embeddings (SVD of PPMI)
        self.sem_vocab = []; self.sem_idx = {}
        self.cortex = None                               # predictive-coding deep learner

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
        with open(PATH + ".tmp", "wb") as f:                 # atomic: temp + rename
            pickle.dump({"t": self.psc.t, "taught": self.taught,
                         "K": self.mem.K, "V": self.mem.V, "cod": self.cod,
                         "read_pos": self.read_pos, "sem": self.sem,
                         "sem_vocab": self.sem_vocab, "sem_idx": self.sem_idx,
                         "cortex": self.cortex, "images_seen": self.images_seen,
                         "audio_seen": self.audio_seen, "video_seen": self.video_seen,
                         "aud_C": self.aud_C, "aud_mu": self.aud_mu, "aud_sd": self.aud_sd}, f)
        os.replace(PATH + ".tmp", PATH)                      # never a truncated read

    def load(self):
        if not os.path.exists(PATH):
            return False
        d = pickle.load(open(PATH, "rb"))
        self.psc.t = d["t"]; self.taught = d["taught"]
        self.mem.K = d["K"]; self.mem.V = d["V"]; self.cod = d["cod"]
        self.read_pos = d.get("read_pos", 0)
        self.sem = d.get("sem"); self.sem_vocab = d.get("sem_vocab", [])
        self.sem_idx = d.get("sem_idx", {})
        self.cortex = d.get("cortex"); self.images_seen = d.get("images_seen", 0)
        self.audio_seen = d.get("audio_seen", 0); self.video_seen = d.get("video_seen", 0)
        self.aud_C = d.get("aud_C"); self.aud_mu = d.get("aud_mu"); self.aud_sd = d.get("aud_sd")
        return True

    # ---------- cortex: deep nonlinear learning by free-energy descent ----------
    def _mnist8(self, n=4000):
        from torchvision.datasets import MNIST
        from PIL import Image
        mn = MNIST(root="./data", train=True, download=True)
        X = mn.data.numpy().astype(np.float32) / 255.0; Y = mn.targets.numpy()
        idx = np.random.default_rng(0).choice(len(X), n, replace=False)
        XS = np.stack([np.asarray(Image.fromarray(np.uint8(X[i]*255)).resize((8, 8)),
                                  np.float32).ravel()/255.0 for i in idx])
        return XS, Y[idx]

    def grow_cortex(self, epochs=4):
        """Predictive coding learns digit RECOGNITION (pixels->label) -- a
        nonlinear task the counting model can't do; complements generative draw.
        This is the deep-abstraction faculty, same free-energy engine as unify."""
        XS, Y = self._mnist8()
        self.cortex = PredictiveCoding([64, 128, 10], lr=0.01, T=25)
        for _ in range(epochs):
            for i in np.random.default_rng().permutation(len(XS)):
                oh = np.full(10, -1.0); oh[Y[i]] = 1.0
                self.cortex.train_step(XS[i], oh)
        return self.cortex_eval()

    def cortex_eval(self, n=400):
        XS, Y = self._mnist8(n)
        return 100 * np.mean([self.cortex.predict(XS[j]).argmax() == Y[j]
                              for j in range(n)])

    # ---------- semantics: meaning = spectral factorization of the counts ----------
    def build_semantics(self, V=4000):
        """Turn local co-occurrence into a SEMANTIC space (SVD of PPMI). This is
        the operation that crosses 'statistics -> meaning' -- gradient-free."""
        from semantics import load_words, cooccur, ppmi, rsvd
        words = load_words()
        C, vocab, idx = cooccur(words, V=V)
        U, S = rsvd(ppmi(C), k=200)
        self.sem = (U * np.sqrt(S)).astype(np.float32)
        self.sem_vocab, self.sem_idx = vocab, idx
        return len(vocab)

    def similar(self, word, n=6):
        if self.sem is None or word not in self.sem_idx:
            return []
        v = self.sem[self.sem_idx[word]]
        s = self.sem @ v / (np.linalg.norm(self.sem, axis=1)*np.linalg.norm(v)+1e-9)
        return [self.sem_vocab[i] for i in np.argsort(-s)
                if self.sem_vocab[i] != word][:n]

    def analogy(self, a, b, c, n=3):
        if self.sem is None or not all(w in self.sem_idx for w in (a, b, c)):
            return []
        v = self.sem[self.sem_idx[b]] - self.sem[self.sem_idx[a]] + self.sem[self.sem_idx[c]]
        s = self.sem @ v / (np.linalg.norm(self.sem, axis=1)*np.linalg.norm(v)+1e-9)
        seen = {a, b, c}
        return [self.sem_vocab[i] for i in np.argsort(-s)
                if self.sem_vocab[i] not in seen][:n]

    # ---------- audio: mel-spectrogram -> codebook tokens (no torchcodec) ----------
    def grow_audio_codec(self, n=60):
        import io, soundfile as sf, librosa
        from datasets import load_dataset, Audio
        ds = load_dataset("openslr/librispeech_asr", split="train.clean.100",
                          streaming=True).cast_column("audio", Audio(decode=False))
        frames = []
        for ex in iter(ds):
            try:
                w, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]))
                if w.ndim > 1: w = w.mean(1)
                m = librosa.feature.melspectrogram(y=w.astype(float), sr=sr,
                                                   n_mels=32, hop_length=sr//10)
                frames.append(np.log(m + 1e-6).T)
            except Exception:
                pass
            if len(frames) >= n: break
        F = np.concatenate(frames)
        self.aud_mu = F.mean(0); self.aud_sd = F.std(0) + 1e-6
        self.aud_C = kmeans((F - self.aud_mu) / self.aud_sd, KA)

    def aud_tokens(self, w, sr, cap=40):
        import librosa
        if w.ndim > 1: w = w.mean(1)
        m = librosa.feature.melspectrogram(y=w.astype(float), sr=sr, n_mels=32,
                                           hop_length=sr//10)
        lm = (np.log(m + 1e-6).T - self.aud_mu) / self.aud_sd
        codes = ((lm[:, None] - self.aud_C[None])**2).sum(2).argmin(1)[:cap]
        return [AUD0 + int(c) for c in codes]

    def video_frames(self, url, k=3):
        """Download an mp4 and ffmpeg-extract k grayscale 32x32 frames -> vision
        codes (torchcodec is broken, so we shell out to ffmpeg)."""
        import urllib.request, subprocess, tempfile, os
        from PIL import Image
        d = tempfile.mkdtemp()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            open(f"{d}/v.mp4", "wb").write(urllib.request.urlopen(req, timeout=15).read())
            subprocess.run(["ffmpeg", "-y", "-i", f"{d}/v.mp4", "-vf",
                            "scale=32:32", "-vframes", str(k), f"{d}/f%02d.png"],
                           capture_output=True, timeout=30)
            out = []
            for fn in sorted(os.listdir(d)):
                if fn.endswith(".png"):
                    g = np.asarray(Image.open(f"{d}/{fn}").convert("L"), np.float32)/255.0
                    out.append(list(self.cod.vis_enc(g)))
            return out[:k]
        except Exception:
            return []
        finally:
            import shutil; shutil.rmtree(d, ignore_errors=True)

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
        """FULL-SCALE multimodal lifelong learning into the ONE brain:
        - TEXT: stream allenai/c4 (web text, effectively unlimited) -> genuine,
          unbounded language growth (no 2MB plateau).
        - IMAGE + IMAGE/TEXT: stream all of cifar10, encode each image to vision
          tokens paired with its label word ('img <label> SEP <codes>').
        - deep cortex keeps training (recognition). All in one persistent model."""
        import time
        from datasets import load_dataset
        if not self.load():
            print("growing the one brain ..."); self.grow(vision=True); self.save()
        if self.sem is None:
            self.build_semantics(); self.save()
        if self.cortex is None:
            self.grow_cortex(); self.save()
        if self.aud_C is None:
            print("fitting audio codec (mel -> codebook) ..."); self.grow_audio_codec(); self.save()
        import io, soundfile as sf
        from datasets import Audio
        cortex_X, cortex_Y = self._mnist8(4000)
        test = open("data/wiki_train.txt", errors="replace").read()[-test_n:-test_n+4000]
        LBL = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
        S0 = lambda **k: dict(streaming=True, **k)
        txt_it = iter(load_dataset("allenai/c4","en",split="train",streaming=True).shuffle(buffer_size=1000, seed=int(time.time())))
        img_it = iter(load_dataset("cifar10",split="train",streaming=True).shuffle(buffer_size=2000, seed=int(time.time())))
        spe_it = iter(load_dataset("openslr/librispeech_asr",split="train.clean.100",streaming=True).cast_column("audio",Audio(decode=False)))
        snd_it = iter(load_dataset("ashraq/esc50",split="train",streaming=True).cast_column("audio",Audio(decode=False)))
        vid_it = iter(load_dataset("TempoFunk/webvid-10M",split="train",streaming=True))
        log = open("outputs/brain_life.out", "a")
        cyc = 0
        print("FULL multimodal lifelong: text(c4) + image(cifar) + audio(speech/"
              "sound) + video(webvid). stop: touch outputs/STOP")
        def nxt(it):
            try: return next(it)
            except Exception: return None
        while not os.path.exists("outputs/STOP"):
            cyc += 1
            # ---- TEXT: unlimited web stream ----
            buf = ""
            for _ in range(2000):
                e = nxt(txt_it)
                if e is None: break
                buf += e["text"] + "\n"
                if len(buf) >= chunk: break
            self.ingest_text(buf[:chunk]); self.read_pos += len(buf[:chunk])
            # ---- IMAGE + label (image/text) ----
            ni = 0
            for _ in range(120):
                ex = nxt(img_it)
                if ex is None: break
                g = np.asarray(ex["img"].convert("L").resize((32, 32)), np.float32)/255.0
                self._teach([[BOS]+txt(f"img {LBL[ex['label']]} ")+[SEP]+list(self.cod.vis_enc(g))+[EOS]])
                ni += 1
            self.images_seen += ni
            # ---- AUDIO + text: speech (transcript) + sound (category) ----
            na = 0
            for it, kw, lab in ((spe_it, "say", "text"), (snd_it, "sound", "category")):
                for _ in range(12):
                    ex = nxt(it)
                    if ex is None: break
                    try:
                        w, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]))
                        tks = self.aud_tokens(np.asarray(w), sr)
                        cap = str(ex.get(lab, ""))[:40].lower()
                        self._teach([[BOS]+txt(f"{kw} {cap} ")+[SEP]+tks+[EOS]])
                        na += 1
                    except Exception:
                        pass
            self.audio_seen += na
            # ---- VIDEO + text: webvid frames + caption ----
            nv = 0
            for _ in range(2):
                ex = nxt(vid_it)
                if ex is None: break
                frames = self.video_frames(ex.get("contentUrl", ""), k=3)
                if frames:
                    seq = [BOS]+txt(f"video {str(ex.get('name',''))[:40].lower()} ")+[SEP]
                    for fr in frames: seq += [FRAME]+fr
                    self._teach([seq+[EOS]]); nv += 1
            self.video_seen += nv
            # ---- deep cortex keeps learning ----
            for i in np.random.default_rng(cyc).permutation(len(cortex_X))[:800]:
                oh = np.full(10, -1.0); oh[cortex_Y[i]] = 1.0
                self.cortex.train_step(cortex_X[i], oh)
            rec = self.cortex_eval(300)
            bpc = self.heldout_bpc(test); fac = self.faculty_check()
            slept = ""
            if cyc % 20 == 0 and self.psc.size() > MAX_CTX:    # SLEEP: bound memory
                before = self.psc.size()
                self.psc.consolidate(MAX_CTX, decay=0.97)
                slept = f"  [slept {before:,}->{self.psc.size():,}]"
            line = (f"cycle {cyc}: txt {self.read_pos//1000}k  img {self.images_seen}  "
                    f"aud {self.audio_seen}  vid {self.video_seen}  "
                    f"bpc {bpc:.2f}  cortex {rec:.0f}%  fac {sum(fac.values())}/5"
                    f"  ctx {self.psc.size()//1000}k{slept}")
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
        m = re.search(r"(\w+) is to (\w+) as (\w+) is to", low)
        if m:
            if self.sem is None: self.build_semantics()
            return ", ".join(self.analogy(m[1], m[2], m[3])) or "(unknown)"
        m = re.search(r"(?:similar to|like|meaning of|related to) (\w+)", low)
        if m:
            if self.sem is None: self.build_semantics()
            return ", ".join(self.similar(m[1])) or "(unknown word)"
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
