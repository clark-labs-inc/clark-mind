"""
Universal shared codec ("common neural currency") + consolidate + retrain.
--------------------------------------------------------------------------
Brain-like: every sense has a thin transducer to a COMMON feature space, and a
SINGLE shared codebook quantizes all of them into one alphabet of codes. Vision
and music then live in the same code vocabulary; the universal predictive-state
learner trains over those shared codes. No backprop anywhere.

    raw (image / midi)
      -> per-modality transducer -> 64-dim frames        (retina / cochlea)
      -> per-modality whitening (comparable scales)
      -> ONE shared k-means codebook (K codes)            (common cortical code)
      -> code-id event streams -> UniversalPSC            (association cortex)
      -> decode codes back per modality                    (motor output)

Outputs (outputs/brain/):
    shared_codebook.png     atoms decoded as image patches (the shared alphabet)
    image_recon.png         MNIST through the SHARED codebook
    music_recon.*           a MAESTRO piece through the SHARED codebook
    image_gen.png / music_gen.*   regenerated after consolidation
"""
from __future__ import annotations
import os, pickle, time
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.seterr(over="ignore", invalid="ignore", divide="ignore")
os.makedirs("outputs/brain", exist_ok=True)
RNG = np.random.default_rng(0)
D, K = 64, 512                  # common feature dim, shared codebook size


# ----------------------------------------------------------- transducers (senses)
class Vision:
    name = "vision"; patch = 8; size = 32; grid = 4          # 32/8 = 4x4 = 16 frames/image
    def frames(self, img32):                                  # (16, 64)
        p, g = self.patch, self.grid
        return np.stack([img32[y*p:(y+1)*p, x*p:(x+1)*p].ravel()
                         for y in range(g) for x in range(g)]).astype(np.float32)
    def deframe(self, F):                                     # (16,64) -> 32x32
        p, g = self.patch, self.grid
        out = np.zeros((self.size, self.size), np.float32)
        for i, f in enumerate(F):
            y, x = divmod(i, g); out[y*p:(y+1)*p, x*p:(x+1)*p] = f.reshape(p, p)
        return np.clip(out, 0, 1)


class Music:
    name = "music"; fs = 16                                   # 16 frames/sec piano roll
    def frames_from_midi(self, path):
        import pretty_midi
        roll = pretty_midi.PrettyMIDI(str(path)).get_piano_roll(fs=self.fs).T  # (T,128)
        roll = (roll > 0).astype(np.float32)
        return roll.reshape(roll.shape[0], 64, 2).max(2)      # 128 -> 64 (max-pool pairs)
    def deframe_to_midi(self, F, out):
        import pretty_midi
        roll = np.repeat((F > 0.5).astype(np.float32), 2, axis=1)  # 64 -> 128
        pm = pretty_midi.PrettyMIDI(); inst = pretty_midi.Instrument(0)
        for pitch in range(128):
            active = roll[:, pitch]; t = 0
            while t < len(active):
                if active[t]:
                    s = t
                    while t < len(active) and active[t]: t += 1
                    inst.notes.append(pretty_midi.Note(80, pitch, s/self.fs, t/self.fs))
                else: t += 1
        pm.instruments.append(inst); pm.write(str(out)); return pm


# ----------------------------------------------------------- the shared codec
class SharedCodec:
    def __init__(self, k=K, d=D): self.K, self.D = k, d
    def whiten_fit(self, F): return F.mean(0), F.std(0) + 1e-6
    def fit(self, frames_by_mod):
        self.stats = {m: self.whiten_fit(F) for m, F in frames_by_mod.items()}
        pooled = np.concatenate([(F - self.stats[m][0]) / self.stats[m][1]
                                 for m, F in frames_by_mod.items()])
        idx = RNG.choice(len(pooled), self.K, replace=False)
        self.C = pooled[idx].copy(); seen = np.full(self.K, 1e-3, np.float32)
        for n, f in enumerate(pooled[RNG.permutation(len(pooled))]):
            k = int(np.argmin(np.sum((self.C - f) ** 2, 1) / (1/np.sqrt(seen))))
            self.C[k] += (0.2/(1+3e-6*n)) * (f - self.C[k]); seen[k] += 1
        self.seen = seen
    def encode(self, m, F):
        z = (F - self.stats[m][0]) / self.stats[m][1]
        return np.argmin(((z[:, None] - self.C[None]) ** 2).sum(2), 1).astype(np.int32)
    def decode(self, m, codes):
        z = self.C[codes]; mu, sd = self.stats[m]
        return z * sd + mu


def img_grid(imgs, path, cols, scale=4, gap=1, channels=False):
    rows = (len(imgs)+cols-1)//cols; h = w = imgs[0].shape[0]
    cv = np.ones((rows*(h+gap)-gap, cols*(w+gap)-gap), np.float32)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols); cv[r*(h+gap):r*(h+gap)+h, c*(w+gap):c*(w+gap)+w] = im
    Image.fromarray(np.uint8(np.clip(cv,0,1)*255)).resize((cv.shape[1]*scale, cv.shape[0]*scale), Image.NEAREST).save(path)


def main():
    from psc_studio import UniversalPSC
    vis, mus = Vision(), Music()

    # --- gather frames from both senses ---
    from torchvision.datasets import MNIST
    mn = MNIST(root="./data", train=True, download=True)
    Xi = mn.data.numpy().astype(np.float32)[:6000] / 255.0
    yi = mn.targets.numpy()[:6000]
    X32 = np.stack([np.asarray(Image.fromarray(np.uint8(x*255)).resize((32,32)), np.float32)/255.0 for x in Xi])
    img_frames = [vis.frames(x) for x in X32]

    midis = sorted(Path("data/maestro_midi").rglob("*.midi")); RNG.shuffle(midis)
    music_frames = []
    for f in midis[:60]:
        try:
            fr = mus.frames_from_midi(f)
            if len(fr) > 40: music_frames.append(fr[:1500])
        except Exception: pass
    print(f"frames: vision={len(img_frames)*16}  music={sum(len(f) for f in music_frames)}")

    # --- ONE shared codebook over both modalities ---
    codec = SharedCodec()
    codec.fit({"vision": np.concatenate(img_frames),
               "music": np.concatenate(music_frames)})

    # --- consolidation stats: how shared is the alphabet? ---
    used_v = set(np.unique(np.concatenate([codec.encode("vision", f) for f in img_frames[:2000]])))
    used_m = set(np.unique(np.concatenate([codec.encode("music", f) for f in music_frames])))
    shared = used_v & used_m
    print(f"shared codebook K={K}: vision uses {len(used_v)}, music uses {len(used_m)}, "
          f"BOTH use {len(shared)} ({100*len(shared)/max(1,len(used_v|used_m)):.0f}% of active codes shared)")

    # --- reconstructions THROUGH the shared codebook ---
    recon_imgs = []
    for i in range(8):
        recon_imgs.append(X32[i]); recon_imgs.append(vis.deframe(codec.decode("vision", codec.encode("vision", img_frames[i]))))
    img_grid(recon_imgs, "outputs/brain/image_recon.png", cols=8, scale=5)
    mf = music_frames[0]
    mus.deframe_to_midi(codec.decode("music", codec.encode("music", mf)).reshape(-1,64), "outputs/brain/music_recon.mid")
    # shared codebook visualized as image patches
    atoms = []
    for k in range(min(256, K)):
        a = (codec.C[k]*codec.stats["vision"][1] + codec.stats["vision"][0]).reshape(8,8)
        atoms.append((a-a.min())/(a.max()-a.min()+1e-8))
    img_grid(atoms, "outputs/brain/shared_codebook.png", cols=16, scale=3)

    # --- consolidate -> events in the SHARED alphabet -> retrain UniversalPSC ---
    img_samples = [((4,4), {(i//4, i%4): int(c) for i, c in enumerate(codec.encode("vision", img_frames[n]))},
                    (int(yi[n]),)) for n in range(len(img_frames))]
    mus_samples = [((len(f),), {(t,): int(c) for t, c in enumerate(codec.encode("music", f))}, ())
                   for f in music_frames]

    pv = UniversalPSC(K, [(0,-1),(-1,0),(-1,-1),(-1,1)], (0,1)); pv.fit(img_samples)
    pm = UniversalPSC(K, [(-1,),(-2,),(-3,),(-4,)], ()); pm.fit(mus_samples)

    # regenerate a digit and a music clip from shared codes
    import psc_studio as S
    class _Vorder:
        order = lambda self, shape, E: [(y,x) for y in range(4) for x in range(4)]; stop=None
    gen = {}
    for y in range(4):
        for x in range(4):
            E = {k: gen[k] for k in gen}
            gen[(y,x)] = S._sample(pv.predict(gen, (3,), (y,x)), 0.9, 0.95)
    digit = vis.deframe(codec.decode("vision", np.array([gen[(y,x)] for y in range(4) for x in range(4)])))
    img_grid([digit], "outputs/brain/image_gen.png", cols=1, scale=8)

    seqE, codes = {}, []
    for t in range(400):
        c = S._sample(pm.predict(seqE, (), (t,)), 0.9, 0.95); seqE[(t,)] = c; codes.append(c)
    mus.deframe_to_midi(codec.decode("music", np.array(codes)).reshape(-1,64), "outputs/brain/music_gen.mid")
    try:
        import pretty_midi, soundfile as sf
        for n in ("music_recon","music_gen"):
            pmid = pretty_midi.PrettyMIDI(f"outputs/brain/{n}.mid")
            sf.write(f"outputs/brain/{n}.wav", pmid.synthesize(fs=16000), 16000)
    except Exception as e: print("wav skip", repr(e)[:60])
    print("wrote outputs/brain/{shared_codebook,image_recon,image_gen}.png, music_{recon,gen}.mid/.wav")


if __name__ == "__main__":
    main()
