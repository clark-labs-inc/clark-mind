"""
PSC-Omni: multimodal-in / multimodal-out, LLM-style prompting (no backprop).
----------------------------------------------------------------------------
ONE unified autoregressive predictive-state model over an interleaved token
stream. Everything is a token in a single vocabulary:

    [text bytes] [vision codes] [audio codes] [concept tokens] [BOS/EOS/SEP/FRAME]

The modality is just the token's id-range. A training example is a sequence:

    BOS  <input tokens>  [CONCEPT gist]  <instruction text>  SEP  <output tokens>  EOS

The learner is the universal backoff predictive-state model (1D, order-N), the
SAME object used for dynamics/music/images. Prompting = give a prefix, roll the
model forward to EOS; decode the output by its id-range. Tasks demonstrated in
ONE model: text->image, text->audio, text->text, image->text, audio->text,
audio->image, text->video. No loss.backward(), no optimizer, no transformer.
"""
from __future__ import annotations
import os, time, numpy as np
from PIL import Image
np.seterr(over="ignore", invalid="ignore", divide="ignore")
os.makedirs("outputs/omni", exist_ok=True)
RNG = np.random.default_rng(0)
from substrate import UniversalPSC, _sample

# ---- unified vocabulary layout ----
KV, KA, GC = 256, 64, 32
VIS0, AUD0, CON0 = 256, 256 + KV, 256 + KV + KA
BOS, EOS, SEP, FRAME = CON0 + GC, CON0 + GC + 1, CON0 + GC + 2, CON0 + GC + 3
VOCAB = FRAME + 1
WORD = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


# ---- codecs (raw <-> tokens) ----
def img_patches(x32, p=8, s=8, g=4):
    return np.stack([x32[y*s:y*s+p, c*s:c*s+p].ravel() for y in range(g) for c in range(g)]).astype(np.float32)

def audio_frames(d, T=8):
    base = np.exp(-0.5 * ((np.arange(64) - (4 + d*5)) / 2.0) ** 2)
    return np.clip(base[None].repeat(T, 0) + 0.05*RNG.standard_normal((T, 64)), 0, None).astype(np.float32)

def kmeans(X, G, iters=12, cap=120000):
    if len(X) > cap: X = X[RNG.choice(len(X), cap, replace=False)]
    C = X[RNG.choice(len(X), G, replace=False)].copy(); xn = (X**2).sum(1)[:, None]
    for _ in range(iters):
        a = (xn - 2*X@C.T + (C**2).sum(1)[None]).argmin(1)
        for g in range(G):
            if (a == g).any(): C[g] = X[a == g].mean(0)
    return C

def assign(x, C): return int(np.argmin(((C - x)**2).sum(1)))


class Codecs:
    def fit(self, imgs32, digits):
        P = np.concatenate([img_patches(x) for x in imgs32])
        self.mu_v, self.sd_v = P.mean(0), P.std(0)+1e-6
        self.Cv = kmeans((P-self.mu_v)/self.sd_v, KV)
        A = np.concatenate([audio_frames(int(d)) for d in digits[:2000]])
        self.mu_a, self.sd_a = A.mean(0), A.std(0)+1e-6
        self.Ca = kmeans((A-self.mu_a)/self.sd_a, KA)

    def vis_enc(self, x32):
        z = (img_patches(x32)-self.mu_v)/self.sd_v
        return VIS0 + np.argmin(((z[:, None]-self.Cv[None])**2).sum(2), 1)
    def vis_dec(self, toks):
        F = self.Cv[np.array(toks)-VIS0]*self.sd_v + self.mu_v
        out = np.zeros((32, 32), np.float32)
        for i, f in enumerate(F):
            y, x = divmod(i, 4); out[y*8:y*8+8, x*8:x*8+8] = f.reshape(8, 8)
        return np.clip(out, 0, 1)
    def aud_enc(self, d):
        z = (audio_frames(d)-self.mu_a)/self.sd_a
        return AUD0 + np.argmin(((z[:, None]-self.Ca[None])**2).sum(2), 1)
    def aud_dec(self, toks):
        return self.Ca[np.array(toks)-AUD0]*self.sd_a + self.mu_a


def txt(s): return [b for b in s.encode()]
def shift(x, dx): return np.clip(np.roll(x, dx, axis=1), 0, 1)   # video: translate the digit


def main():
    t0 = time.time()
    from torchvision.datasets import MNIST
    mn = MNIST(root="./data", train=True, download=True)
    n = 6000
    Xi = mn.data.numpy().astype(np.float32)[:n]/255.0; yi = mn.targets.numpy()[:n]
    X32 = np.stack([np.asarray(Image.fromarray(np.uint8(x*255)).resize((32, 32)), np.float32)/255.0 for x in Xi])
    cod = Codecs(); cod.fit(X32, yi)
    print(f"codecs fit, vocab={VOCAB}  ({time.time()-t0:.0f}s)")

    Vt = [list(cod.vis_enc(X32[i])) for i in range(n)]
    At = [list(cod.aud_enc(int(yi[i]))) for i in range(n)]
    # joint audio-anchored concept (the cross-modal bridge), no labels
    def hist(toks, lo, K):
        h = np.bincount(np.array(toks)-lo, minlength=K).astype(np.float32); return h/(np.linalg.norm(h)+1e-8)
    VH = np.stack([hist(Vt[i], VIS0, KV) for i in range(n)])
    AH = np.stack([hist(At[i], AUD0, KA) for i in range(n)])
    Cg = kmeans(np.concatenate([VH, 6.0*AH], 1), GC); Cv, Ca = Cg[:, :KV], Cg[:, KV:]
    gist = [assign(np.concatenate([VH[i], 6.0*AH[i]]), Cg) for i in range(n)]
    con = lambda g: [CON0 + g]
    print(f"concepts fit  ({time.time()-t0:.0f}s)")

    # ---- build interleaved multimodal instruction corpus ----
    seqs = []
    for i in range(n):
        d = int(yi[i]); v, a, c = Vt[i], At[i], con(gist[i])
        seqs += [
            [BOS] + txt(f"draw {d}") + [SEP] + v + [EOS],                  # text -> image
            [BOS] + txt(f"sound {d}") + [SEP] + a + [EOS],                 # text -> audio
            [BOS] + txt(f"name {d}") + [SEP] + txt(WORD[d]) + [EOS],       # text -> text
            [BOS] + v + txt("what digit") + c + [SEP] + txt(str(d)) + [EOS],   # image -> text
            [BOS] + a + txt("what digit") + c + [SEP] + txt(str(d)) + [EOS],   # audio -> text
            [BOS] + a + txt("show it") + c + [SEP] + v + [EOS],            # audio -> image
        ]
        if i < 3000:                                                       # text -> video (clip)
            vid = []
            for dx in (-4, 0, 4):
                vid += [FRAME] + list(cod.vis_enc(shift(X32[i], dx)))
            seqs.append([BOS] + txt(f"video {d}") + [SEP] + vid + [EOS])
    psc = UniversalPSC(VOCAB, [(-1,), (-2,), (-3,), (-4,), (-5,), (-6,), (-7,), (-8,)], ())
    psc.fit([((len(s),), {(t,): int(tok) for t, tok in enumerate(s)}, ()) for s in seqs])
    print(f"omni model fit on {len(seqs)} sequences  ({time.time()-t0:.0f}s)")

    def roll(prefix, maxlen=80, temp=0.5, top_p=0.92):
        E = {(t,): int(v) for t, v in enumerate(prefix)}; seq = list(prefix)
        for t in range(len(prefix), maxlen):
            v = _sample(psc.predict(E, (), (t,)), temp, top_p); E[(t,)] = v; seq.append(v)
            if v == EOS: break
        return seq[len(prefix):]

    def out_after(seq):
        return seq[:-1] if seq and seq[-1] == EOS else seq

    # ===================== PROMPT BATTERY =====================
    print("\n=== PROMPTS (one model, multimodal in/out, no backprop) ===")
    # text -> image (grid 0..9)
    gi = []
    for d in range(10):
        toks = [t for t in out_after(roll([BOS]+txt(f"draw {d}")+[SEP], 40)) if VIS0 <= t < VIS0+KV][:16]
        toks = (toks + [VIS0]*16)[:16]; gi.append(cod.vis_dec(toks))
    grid(gi, "outputs/omni/text2image_0to9.png")
    print('  "draw N"  -> outputs/omni/text2image_0to9.png')

    # text -> audio
    import soundfile as sf
    for d in (3, 8):
        toks = [t for t in out_after(roll([BOS]+txt(f"sound {d}")+[SEP], 30)) if AUD0 <= t < AUD0+KA]
        if toks:
            F = cod.aud_dec(toks); freq = 196*2**((np.argmax(F.mean(0))-4)/10.0)
            tt = np.arange(9600)/16000.; sf.write(f"outputs/omni/say{d}.wav", (0.3*np.sin(2*np.pi*freq*tt)).astype(np.float32), 16000)
    print('  "sound 3" / "sound 8"  -> outputs/omni/say3.wav, say8.wav')

    # text -> text
    for d in (2, 5, 9):
        o = bytes([t for t in out_after(roll([BOS]+txt(f"name {d}")+[SEP], 30)) if t < 256]).decode("latin1", "ignore")
        print(f'  "name {d}"  -> "{o}"')

    # image -> text  (accuracy) and audio -> text (accuracy)
    te = MNIST(root="./data", train=False, download=True)
    Xt = te.data.numpy().astype(np.float32)[:400]/255.0; yt = te.targets.numpy()[:400]
    vis_ok = aud_ok = 0
    for j in range(len(Xt)):
        x32 = np.asarray(Image.fromarray(np.uint8(Xt[j]*255)).resize((32, 32)), np.float32)/255.0
        v = list(cod.vis_enc(x32)); gv = assign(hist(v, VIS0, KV), Cv)
        o = bytes([t for t in out_after(roll([BOS]+v+txt("what digit")+con(gv)+[SEP], len(v)+24)) if 48 <= t <= 57])
        vis_ok += int(o[:1] == str(int(yt[j])).encode())
        a = list(cod.aud_enc(int(yt[j]))); ga = assign(6.0*hist(a, AUD0, KA), Ca)
        o2 = bytes([t for t in out_after(roll([BOS]+a+txt("what digit")+con(ga)+[SEP], len(a)+24)) if 48 <= t <= 57])
        aud_ok += int(o2[:1] == str(int(yt[j])).encode())
    print(f'  [image] "what digit"  -> {100*vis_ok/len(Xt):.0f}% correct   (the no-backprop vision ceiling)')
    print(f'  [audio] "what digit"  -> {100*aud_ok/len(Xt):.0f}% correct')

    # audio -> image
    gi = []
    for d in range(10):
        a = list(cod.aud_enc(d)); ga = assign(6.0*hist(a, AUD0, KA), Ca)
        toks = [t for t in out_after(roll([BOS]+a+txt("show it")+con(ga)+[SEP], len(a)+24)) if VIS0 <= t < VIS0+KV][:16]
        toks = (toks + [VIS0]*16)[:16]; gi.append(cod.vis_dec(toks))
    grid(gi, "outputs/omni/audio2image_0to9.png")
    print('  [audio of N] "show it"  -> outputs/omni/audio2image_0to9.png')

    # text -> video : structured decode (seed each FRAME, constrain frame tokens to vision range)
    def sample_vis(E, t):
        p = psc.predict(E, (), (t,)).copy(); p[:VIS0] = 0; p[VIS0+KV:] = 0
        if p.sum() < 1e-9: p[VIS0:VIS0+KV] = 1.0
        return _sample(p / p.sum(), 0.45, 0.95)
    for d in (3, 7):
        seq = [BOS]+txt(f"video {d}")+[SEP]; E = {(i,): v for i, v in enumerate(seq)}; ims = []
        for _ in range(3):
            seq.append(FRAME); E[(len(seq)-1,)] = FRAME; fr = []
            for _ in range(16):
                t = len(seq); v = sample_vis(E, t); E[(t,)] = v; seq.append(v); fr.append(v)
            ims.append(Image.fromarray(np.uint8(cod.vis_dec(fr)*255)).resize((96, 96), Image.NEAREST))
        ims[0].save(f"outputs/omni/video{d}.gif", save_all=True, append_images=ims[1:], duration=350, loop=0)
    print('  "video 3" / "video 7"  -> outputs/omni/video3.gif, video7.gif')
    print(f"\ndone in {time.time()-t0:.0f}s")


def grid(imgs, path, cols=10, scale=6, gap=1):
    rows = (len(imgs)+cols-1)//cols; h = w = imgs[0].shape[0]
    cv = np.ones((rows*(h+gap)-gap, cols*(w+gap)-gap), np.float32)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols); cv[r*(h+gap):r*(h+gap)+h, c*(w+gap):c*(w+gap)+w] = im
    Image.fromarray(np.uint8(np.clip(cv,0,1)*255)).resize((cv.shape[1]*scale, cv.shape[0]*scale), Image.NEAREST).save(path)


if __name__ == "__main__":
    main()
