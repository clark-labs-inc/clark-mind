"""
Cross-modal binding with a HYBRID codebook (no backprop).
---------------------------------------------------------
Brain-like refinement: instead of one fully-shared codebook (which compromises
each sense's discriminability), use a HYBRID alphabet =
    shared core codes (common currency, fit on all senses)   -> cross-modal overlap
  + per-sense private codes (fit on that sense alone)         -> sharp within-sense detail

Everything else is unchanged and universal: paired (digit image, digit tone) ->
hybrid codes -> JOINT audio-anchored concept (no labels) -> UniversalPSC predicts
one sense's codes from that concept -> give one sense, complete the other.

    python psc_crossmodal.py --n 5000        # quick
    python psc_crossmodal.py --n 20000       # large run
"""
from __future__ import annotations
import argparse, os, numpy as np
from PIL import Image
np.seterr(over="ignore", invalid="ignore", divide="ignore")
os.makedirs("outputs/crossmodal", exist_ok=True)
RNG = np.random.default_rng(0)
D = 64
from psc_studio import UniversalPSC, _sample

GP, GS, GG = 8, 4, 7                                   # vision patches: 49 overlapping 8x8


def img_frames(x):
    return np.stack([x[y*GS:y*GS+GP, x2*GS:x2*GS+GP].ravel()
                     for y in range(GG) for x2 in range(GG)]).astype(np.float32)

def img_deframe(F):
    out = np.zeros((32, 32), np.float32); cnt = np.zeros((32, 32), np.float32)
    for i, f in enumerate(F):
        y, x = divmod(i, GG)
        out[y*GS:y*GS+GP, x*GS:x*GS+GP] += f.reshape(GP, GP); cnt[y*GS:y*GS+GP, x*GS:x*GS+GP] += 1
    return np.clip(out / np.maximum(cnt, 1), 0, 1)

def audio_frames(d, T=10):
    base = np.exp(-0.5 * ((np.arange(64) - (4 + d*5)) / 2.0) ** 2)
    return np.clip(base[None].repeat(T, 0) + 0.05*RNG.standard_normal((T, 64)), 0, None).astype(np.float32)

def audio_to_digit(F): return int(np.clip(round((np.argmax(F.mean(0)) - 4) / 5), 0, 9))

def audio_to_wav(F, path, dur=0.6, sr=16000):
    import soundfile as sf
    freq = 196.0 * 2 ** ((np.argmax(F.mean(0)) - 4) / 10.0)
    t = np.arange(int(dur*sr)) / sr
    sf.write(path, (0.3*np.sin(2*np.pi*freq*t)).astype(np.float32), sr)


def kmeans(X, G, iters=12, cap=150000):
    if len(X) > cap: X = X[RNG.choice(len(X), cap, replace=False)]
    C = X[RNG.choice(len(X), G, replace=False)].copy(); xn = (X**2).sum(1)[:, None]
    for _ in range(iters):
        a = (xn - 2*X@C.T + (C**2).sum(1)[None]).argmin(1)
        for g in range(G):
            if (a == g).any(): C[g] = X[a == g].mean(0)
    return C

def assign(x, C): return int(np.argmin(((C - x)**2).sum(1)))


class HybridCodec:
    def __init__(self, k_shared=256, k_priv=None):
        self.ks = k_shared; self.kp = k_priv or {"vision": 768, "audio": 128}
    def fit(self, by_mod):
        self.stat = {m: (F.mean(0), F.std(0)+1e-6) for m, F in by_mod.items()}
        Z = {m: (F-self.stat[m][0])/self.stat[m][1] for m, F in by_mod.items()}
        self.Cs = kmeans(np.concatenate(list(Z.values())), self.ks)     # shared core
        self.Cp = {m: kmeans(Z[m], self.kp[m]) for m in by_mod}         # private banks
        self.order = list(by_mod.keys()); self.off = {}; o = self.ks
        for m in self.order: self.off[m] = o; o += self.kp[m]
        self.K = o
        self.Call = np.concatenate([self.Cs] + [self.Cp[m] for m in self.order])
    def encode(self, m, F):
        z = (F-self.stat[m][0])/self.stat[m][1]; cand = np.concatenate([self.Cs, self.Cp[m]])
        idx = np.argmin(((z[:, None]-cand[None])**2).sum(2), 1)
        return np.where(idx < self.ks, idx, self.off[m] + (idx - self.ks)).astype(np.int32)
    def decode(self, m, gids): return self.Call[gids]*self.stat[m][1]+self.stat[m][0]


def hist(codes, K):
    h = np.bincount(codes, minlength=K).astype(np.float32)
    return h / (np.linalg.norm(h) + 1e-8)

def img_grid(imgs, path, cols, scale=6, gap=1):
    rows = (len(imgs)+cols-1)//cols; h = w = imgs[0].shape[0]
    cv = np.ones((rows*(h+gap)-gap, cols*(w+gap)-gap), np.float32)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols); cv[r*(h+gap):r*(h+gap)+h, c*(w+gap):c*(w+gap)+w] = im
    Image.fromarray(np.uint8(np.clip(cv,0,1)*255)).resize((cv.shape[1]*scale, cv.shape[0]*scale), Image.NEAREST).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--G", type=int, default=32); ap.add_argument("--Wa", type=float, default=6.0)
    ap.add_argument("--k_shared", type=int, default=256); ap.add_argument("--k_vis", type=int, default=768)
    args = ap.parse_args()
    import time; t0 = time.time()

    from torchvision.datasets import MNIST
    mn = MNIST(root="./data", train=True, download=True)
    Xi = mn.data.numpy().astype(np.float32)[:args.n]/255.0; yi = mn.targets.numpy()[:args.n]
    X32 = np.stack([np.asarray(Image.fromarray(np.uint8(x*255)).resize((32, 32)), np.float32)/255.0 for x in Xi])
    VF = [img_frames(x) for x in X32]; AF = [audio_frames(int(d)) for d in yi]
    print(f"n={args.n}: vision frames={len(VF)*49} audio frames={len(AF)*10}  ({time.time()-t0:.0f}s)")

    codec = HybridCodec(k_shared=args.k_shared, k_priv={"vision": args.k_vis, "audio": 128})
    codec.fit({"vision": np.concatenate(VF), "audio": np.concatenate(AF)})
    K = codec.K
    print(f"hybrid codebook: shared={codec.ks} vision_priv={codec.kp['vision']} "
          f"audio_priv={codec.kp['audio']}  total K={K}  ({time.time()-t0:.0f}s)")
    Vc = [codec.encode("vision", f) for f in VF]; Ac = [codec.encode("audio", f) for f in AF]

    VH = np.stack([hist(c, K) for c in Vc]); AH = np.stack([hist(c, K) for c in Ac])
    Cg = kmeans(np.concatenate([VH, args.Wa*AH], 1), args.G)
    Cv, Ca = Cg[:, :K], Cg[:, K:]
    gist = [assign(np.concatenate([VH[i], args.Wa*AH[i]]), Cg) for i in range(args.n)]
    print(f"joint concepts G={args.G} fit  ({time.time()-t0:.0f}s)")

    Paud = UniversalPSC(K, [(-1,), (-2,), (-3,)], ())
    Paud.fit([((len(Ac[i]),), {(t,): int(c) for t, c in enumerate(Ac[i])}, (gist[i],)) for i in range(args.n)])
    Pvis = UniversalPSC(K, [(0,-1), (-1,0), (-1,-1), (-1,1)], (0, 1))
    Pvis.fit([((GG,GG), {(y, x): int(Vc[i][y*GG+x]) for y in range(GG) for x in range(GG)}, (gist[i],)) for i in range(args.n)])
    print(f"learners fit  ({time.time()-t0:.0f}s)")

    # vision -> audio
    te = MNIST(root="./data", train=False, download=True)
    Xt = te.data.numpy().astype(np.float32)[:args.n_test]/255.0; yt = te.targets.numpy()[:args.n_test]
    correct = 0; saved = {}
    for j in range(len(Xt)):
        x32 = np.asarray(Image.fromarray(np.uint8(Xt[j]*255)).resize((32, 32)), np.float32)/255.0
        gv = assign(hist(codec.encode("vision", img_frames(x32)), K), Cv)
        E = {}
        for t in range(10): E[(t,)] = _sample(Paud.predict(E, (gv,), (t,)), 0.2, 0.9)
        F = codec.decode("audio", np.array([E[(t,)] for t in range(10)]))
        correct += int(audio_to_digit(F) == int(yt[j]))
        if int(yt[j]) not in saved:
            audio_to_wav(F, f"outputs/crossmodal/heard_digit{int(yt[j])}.wav"); saved[int(yt[j])] = 1
    va = 100*correct/len(Xt)

    # audio -> vision
    gens = []
    for d in range(10):
        ga = assign(args.Wa*hist(codec.encode("audio", audio_frames(d)), K), Ca)
        E = {}
        for y in range(GG):
            for x in range(GG): E[(y, x)] = _sample(Pvis.predict(E, (ga,), (y, x)), 0.7, 0.95)
        gens.append(img_deframe(codec.decode("vision", np.array([E[(y, x)] for y in range(GG) for x in range(GG)]))))
    img_grid(gens, "outputs/crossmodal/seen_from_sound_0to9.png", cols=10)
    print(f"\n=== CROSS-MODAL (no backprop, n={args.n}, K={K}) total {time.time()-t0:.0f}s ===")
    print(f"  VISION->AUDIO pitch-match accuracy: {va:.1f}%  (chance 10%)")
    print(f"  AUDIO->VISION: outputs/crossmodal/seen_from_sound_0to9.png")


if __name__ == "__main__":
    main()
