"""
PSC Studio: ONE universal predictive-state learner, prompted multimodally.
--------------------------------------------------------------------------
No per-modality logic in the learner. Everything is an event:

    event = (address tuple) -> value (int symbol)

The universal learner does exactly one thing: predict a value from the values at
a fixed set of relative address offsets (+ optional absolute position + optional
globals), with backoff. From that, three behaviours fall out as the SAME call:

    generate   = complete an empty canvas
    continue   = complete, keeping a prefix of the causal order
    complete   = complete, keeping a prefix (e.g. top half of an image)

A "modality" is only a codec: raw <-> events, plus a tiny spec (offsets, order,
vocab). Add a modality => the learner is unchanged. No backprop anywhere.

    python psc_studio.py "generate music"
    python psc_studio.py "make an image of a 7"
    python psc_studio.py "continue song.mid"        # supply music, it finishes it
    python psc_studio.py "complete drawing.png"      # supply an image, it finishes it
"""
from __future__ import annotations
import os, re, sys, pickle, time, math
from pathlib import Path
import numpy as np

np.seterr(over="ignore", invalid="ignore", divide="ignore")
os.makedirs("outputs/studio", exist_ok=True); os.makedirs("models", exist_ok=True)
RNG = np.random.default_rng()
MISSING = -1


# =============================================================================
# The ONE universal learner (zero modality-specific branches)
# =============================================================================
class UniversalPSC:
    def __init__(self, vocab, offsets, pos_dims, alpha=0.02, ab=4.0):
        self.K, self.offsets, self.pos_dims = vocab, offsets, pos_dims
        self.alpha, self.ab = alpha, ab
        self.t = [{} for _ in range(len(offsets) + 1)]      # one count table per backoff depth

    def _ctx(self, E, A):
        vals = [E.get(tuple(a + o for a, o in zip(A, off)), MISSING) for off in self.offsets]
        return tuple(A[d] for d in self.pos_dims), vals

    def fit(self, samples):
        for shape, E, G in samples:
            for A, value in E.items():
                posk, vals = self._ctx(E, A)
                for j in range(len(self.offsets) + 1):
                    d = self.t[j].setdefault((G, posk, tuple(vals[:j])), {})
                    d[value] = d.get(value, 0) + 1

    def _interp(self, p, d):
        a = np.full(self.K, self.alpha)
        for k, v in d.items(): a[k] += v
        c = a.sum() - self.alpha * self.K; lam = c / (c + self.ab)
        return lam * (a / a.sum()) + (1 - lam) * p

    def predict(self, E, G, A):
        posk, vals = self._ctx(E, A)
        p = np.full(self.K, 1.0 / self.K)
        for j in range(len(self.offsets) + 1):              # shallow->deep; deep dominates
            d = self.t[j].get((G, posk, tuple(vals[:j])))
            if d: p = self._interp(p, d)
        return p

    def infer_globals(self, E, candidates):                 # universal "classify what I see"
        best, bs = candidates[0], -1e18
        for G in candidates:
            s = sum(math.log(self.predict(E, G, A)[v] + 1e-12) for A, v in E.items())
            if s > bs: bs, best = s, G
        return best

    def complete(self, modality, shape, E, G, temp=0.9, top_p=0.95):
        E = dict(E)
        for A in modality.order(shape, E):
            if A in E: continue
            v = _sample(self.predict(E, G, A), temp, top_p)
            E[A] = v
            if modality.stop is not None and v == modality.stop: break
        return E


def _sample(p, temp, top_p):
    logp = np.log(p + 1e-12) / max(temp, 1e-3); q = np.exp(logp - logp.max()); q /= q.sum()
    o = np.argsort(-q); cut = o[:max(1, int(np.searchsorted(np.cumsum(q[o]), top_p)) + 1)]
    return int(RNG.choice(cut, p=q[cut] / q[cut].sum()))


# =============================================================================
# Modalities = codecs only. Each implements the same small interface.
# =============================================================================
class Modality:
    name = "base"; vocab = 0; offsets = []; pos_dims = (); candidates = []; stop = None
    keywords = (); batch = 1
    def default_shape(self): ...
    def order(self, shape, E): ...
    def train_samples(self): ...           # -> list of (shape, events, globals)
    def encode_file(self, path): ...       # -> (shape, events, native_globals_or_None)
    def render(self, shape, filled_list, outbase): ...   # -> human-inspectable files


class ImageModality(Modality):
    name = "image"; vocab = 256
    offsets = [(0, -1), (-1, 0), (-1, -1), (-1, 1)]   # left, up, up-left, up-right (causal)
    pos_dims = (0, 1); candidates = [(d,) for d in range(10)]
    keywords = ("image", "picture", "draw", "digit", "number", "sketch", "paint")
    batch = 16

    def __init__(self, patch=4, n_train=20000):
        import psc_image_gen as IG
        self.IG, self.patch, self.g = IG, patch, 28 // patch
        cache = Path(f"models/image_codec_{patch}.pkl")
        if cache.exists():
            d = pickle.load(open(cache, "rb"))
            self.codec = IG.PatchCodec(patch=patch, codes=256); self.codec.C = d["C"]; self.codec.seen = d["seen"]
            self._samples = d["samples"]
        else:
            print("[warming up: learning image patch codec…]")
            Xtr, ytr, _, _ = IG.load_mnist(n_train, 10)
            self.codec = IG.PatchCodec(patch=patch, codes=256); self.codec.fit(Xtr)
            self._samples = [((self.g, self.g), self._grid_to_events(self.codec.encode(x)), (int(y),))
                             for x, y in zip(Xtr, ytr)]
            pickle.dump({"C": self.codec.C, "seen": self.codec.seen, "samples": self._samples},
                        open(cache, "wb"))

    def _grid_to_events(self, grid):
        return {(y, x): int(grid[y, x]) for y in range(grid.shape[0]) for x in range(grid.shape[1])}

    def default_shape(self): return (self.g, self.g)
    def order(self, shape, E):
        return [(y, x) for y in range(shape[0]) for x in range(shape[1])]   # raster
    def train_samples(self): return self._samples

    def encode_file(self, path):
        from PIL import Image
        x = np.asarray(Image.open(path).convert("L").resize((28, 28)), np.float32) / 255.0
        if x.mean() > 0.5: x = 1.0 - x
        return (self.g, self.g), self._grid_to_events(self.codec.encode(x)), None

    def render(self, shape, filled_list, outbase):
        imgs = []
        for E in filled_list:
            grid = np.array([[E[(y, x)] for x in range(shape[1])] for y in range(shape[0])], np.int32)
            imgs.append(self.codec.decode(grid))
        out = outbase + ".png"
        self.IG.grid_png(imgs, out, cols=max(1, int(len(imgs) ** 0.5)), scale=6)
        return [out]


class MusicModality(Modality):
    name = "music"; vocab = 512
    offsets = [(-1,), (-2,), (-3,), (-4,), (-5,), (-6,)]   # previous events
    pos_dims = (); candidates = []; stop = 1                # EOS id
    keywords = ("music", "song", "midi", "piano", "melody", "tune", "compose", "play")
    batch = 1; MAXLEN = 1800

    def __init__(self, n_train=200):
        import psc_music_gen as MG
        self.MG, self.codec = MG, MG.MidiEventCodec()
        cache = Path("models/music_seqs.pkl")
        if cache.exists():
            seqs = pickle.load(open(cache, "rb"))
        else:
            print(f"[warming up: parsing {n_train} MAESTRO MIDI files…]")
            files = sorted(Path("data/maestro_midi").rglob("*.mid*")); RNG.shuffle(files)
            seqs = []
            for f in files[:n_train]:
                try:
                    e = self.codec.midi_to_events(f)[:5000]
                    if len(e) > 50: seqs.append(e)
                except Exception: pass
            pickle.dump(seqs, open(cache, "wb"))
        self._samples = [((len(s),), {(i,): int(v) for i, v in enumerate(s)}, ()) for s in seqs]

    def default_shape(self): return (self.MAXLEN,)
    def order(self, shape, E):
        return ((t,) for t in range(shape[0]))             # sequential, stops on EOS
    def train_samples(self): return self._samples

    def encode_file(self, path):
        ids = self.codec.midi_to_events(path)
        return (self.MAXLEN,), {(i,): int(v) for i, v in enumerate(ids)}, ()

    def render(self, shape, filled_list, outbase):
        E = filled_list[0]
        ids = [E[(t,)] for t in range(len(E))]
        pm = self.codec.events_to_midi(ids, outbase + ".mid")
        self.MG.save_pianoroll(pm, outbase + ".pianoroll.png")
        try:
            import soundfile as sf; sf.write(outbase + ".wav", pm.synthesize(fs=16000), 16000)
        except Exception: pass
        return [outbase + ".mid", outbase + ".wav", outbase + ".pianoroll.png"]


MODALITIES = [ImageModality, MusicModality]


# =============================================================================
# Driver + router: completely modality-agnostic
# =============================================================================
def train_or_load(modality):
    psc = UniversalPSC(modality.vocab, modality.offsets, modality.pos_dims)
    psc.fit(modality.train_samples())
    return psc


def run(modality, supplied=None, want_global=None, keep_frac=0.5):
    psc = train_or_load(modality)
    if supplied:                                           # continue / complete a given artifact
        shape, full, native = modality.encode_file(supplied)
        order = list(modality.order(shape, {}))
        keepN = int(keep_frac * len(order))
        E = {A: full[A] for A in order[:keepN] if A in full}
        G = native if native is not None else (
            psc.infer_globals(E, modality.candidates) if modality.candidates else ())
    else:                                                  # generate from scratch
        shape, E = modality.default_shape(), {}
        G = (want_global,) if want_global is not None else (
            (int(RNG.integers(0, len(modality.candidates))),) if modality.candidates else ())
    filled = [psc.complete(modality, shape, E, G) for _ in range(modality.batch)]
    outbase = f"outputs/studio/{modality.name}_{int(time.time())}"
    paths = modality.render(shape, filled, outbase)
    tag = f"(class {G[0]})" if G else ""
    print(f"  -> {modality.name} {tag} -> " + " , ".join(paths))
    return paths


def route(prompt):
    p = prompt.lower()
    files = re.findall(r"[\w./~-]+\.(?:midi|mid|png|jpe?g|bmp)", prompt, re.I)
    # pick modality: a supplied file's extension wins; else keyword match
    for path in files:
        for M in MODALITIES:
            inst = None
            if path.lower().endswith((".mid", ".midi")) and M is MusicModality: inst = M()
            if path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")) and M is ImageModality: inst = M()
            if inst: return inst, dict(supplied=path)
    for M in MODALITIES:
        if any(k in p for k in M.keywords):
            inst = M(); kw = {}
            if inst.candidates:                            # optional class from the prompt
                d = next((int(t) for t in re.findall(r"\b([0-9])\b", p)), None)
                if d is None:
                    words = {"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,
                             "six":6,"seven":7,"eight":8,"nine":9}
                    d = next((v for w, v in words.items() if w in p), None)
                if d is not None: kw["want_global"] = d
            return inst, kw
    return None, {}


def main():
    prompt = " ".join(sys.argv[1:]).strip()
    if not prompt:
        print('usage: python psc_studio.py "<prompt>"')
        print('  e.g. "generate music" | "make an image of a 3" | "continue x.mid" | "complete y.png"')
        return
    modality, kw = route(prompt)
    if modality is None:
        print(f'Could not route "{prompt}". Use a modality keyword or supply a .mid/.png file.'); return
    print(f'> "{prompt}"  [modality: {modality.name}]')
    t0 = time.time(); run(modality, **kw)
    print(f"  (done in {time.time()-t0:.1f}s, no backprop)")


if __name__ == "__main__":
    main()
