"""
PSC-Omni-Real: unified multimodal model on REAL data (no backprop).
-------------------------------------------------------------------
Real tri-modal digits: MNIST handwriting (vision) + Free Spoken Digit Dataset
(REAL human speech, audio) + text. One autoregressive predictive-state model over
an interleaved token stream, prompted LLM-style. Audio is real mel-VQ tokens, not
synthetic tones. Same universal learner as psc_omni; only the codecs are real.

Tasks in ONE model:
  text->image, text->text, image->text,
  spoken-audio->text (recognize a REAL spoken digit),
  spoken-audio->image (hear a spoken digit, draw it),
  text->video.

    python psc_omni_real.py --scale 25000 --aug 2      # large run
"""
from __future__ import annotations
import argparse, io, os, sys, time, numpy as np
from PIL import Image
np.seterr(over="ignore", invalid="ignore", divide="ignore")
os.makedirs("outputs/omni_real", exist_ok=True)
RNG = np.random.default_rng(0)
from psc_studio import UniversalPSC, _sample

KV, KA, GC, T_AUD = 256, 256, 32, 16
VIS0, AUD0, CON0 = 256, 256+KV, 256+KV+KA
BOS, EOS, SEP, FRAME = CON0+GC, CON0+GC+1, CON0+GC+2, CON0+GC+3
VOCAB = FRAME+1
WORD = ["zero","one","two","three","four","five","six","seven","eight","nine"]
def txt(s): return [b for b in s.encode()]
LOG = open("outputs/omni_real/train.log", "a")
def log(m): print(m); LOG.write(m+"\n"); LOG.flush()


def kmeans(X, G, iters=12, cap=200000):
    if len(X) > cap: X = X[RNG.choice(len(X), cap, replace=False)]
    C = X[RNG.choice(len(X), G, replace=False)].copy(); xn = (X**2).sum(1)[:, None]
    for _ in range(iters):
        a = (xn - 2*X@C.T + (C**2).sum(1)[None]).argmin(1)
        for g in range(G):
            if (a == g).any(): C[g] = X[a == g].mean(0)
    return C
def assign(x, C): return int(np.argmin(((C-x)**2).sum(1)))


# ---------------- real data ----------------
def img_patches(x32):
    return np.stack([x32[y*8:y*8+8, c*8:c*8+8].ravel() for y in range(4) for c in range(4)]).astype(np.float32)

def mel(w, T=T_AUD, nm=32):
    import librosa
    S = librosa.power_to_db(librosa.feature.melspectrogram(y=w, sr=8000, n_mels=nm, n_fft=256, hop_length=96)+1e-9).T
    S = (S-S.mean())/(S.std()+1e-6)
    if len(S) < T: S = np.vstack([S, np.zeros((T-len(S), nm), np.float32)])
    return S[:T].astype(np.float32)

def load_fsdd():
    from huggingface_hub import hf_hub_download, HfApi
    import pyarrow.parquet as pq, soundfile as sf
    fs = [f for f in HfApi().list_repo_files("mteb/free-spoken-digit-dataset", repo_type="dataset") if f.endswith(".parquet")]
    def split(key):
        p = hf_hub_download("mteb/free-spoken-digit-dataset", [f for f in fs if key in f][0], repo_type="dataset")
        t = pq.read_table(p); A = t.column("audio").to_pylist(); L = t.column("label").to_pylist()
        out = []
        for a, l in zip(A, L):
            w, _ = sf.read(io.BytesIO(a["bytes"])); out.append((mel(w.astype(np.float32)), int(l)))
        return out
    return split("train"), split("test")

def aug_audio(M):  # small spectral jitter for a real-data augmentation
    return np.clip(M + 0.15*RNG.standard_normal(M.shape).astype(np.float32), -5, 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=25000)   # MNIST images
    ap.add_argument("--aug", type=int, default=2)         # audio pairings per image
    ap.add_argument("--order", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()

    log(f"\n==== PSC-Omni-Real run scale={args.scale} aug={args.aug} order={args.order} {time.ctime()} ====")
    from torchvision.datasets import MNIST
    mn = MNIST(root="./data", train=True, download=True)
    n = min(args.scale, 60000)
    Xi = mn.data.numpy().astype(np.float32)[:n]/255.0; yi = mn.targets.numpy()[:n]
    X32 = np.stack([np.asarray(Image.fromarray(np.uint8(x*255)).resize((32, 32)), np.float32)/255.0 for x in Xi])
    log(f"MNIST loaded n={n}  ({time.time()-t0:.0f}s)")
    fsdd_tr, fsdd_te = load_fsdd()
    by_digit = {d: [M for M, l in fsdd_tr if l == d] for d in range(10)}
    log(f"FSDD real speech: train={len(fsdd_tr)} test={len(fsdd_te)}  ({time.time()-t0:.0f}s)")

    # ---- fit codecs ----
    Pv = np.concatenate([img_patches(x) for x in X32[:8000]])
    muv, sdv = Pv.mean(0), Pv.std(0)+1e-6; Cv = kmeans((Pv-muv)/sdv, KV)
    Pa = np.concatenate([M for M, _ in fsdd_tr]); mua, sda = Pa.mean(0), Pa.std(0)+1e-6
    Ca = kmeans((Pa-mua)/sda, KA)
    def vis_enc(x32):
        z = (img_patches(x32)-muv)/sdv; return (VIS0+np.argmin(((z[:, None]-Cv[None])**2).sum(2), 1)).tolist()
    def vis_dec(toks):
        F = Cv[np.array(toks)-VIS0]*sdv+muv; out = np.zeros((32, 32), np.float32)
        for i, f in enumerate(F):
            y, x = divmod(i, 4); out[y*8:y*8+8, x*8:x*8+8] = f.reshape(8, 8)
        return np.clip(out, 0, 1)
    def aud_enc(M):
        z = (M-mua)/sda; return (AUD0+np.argmin(((z[:, None]-Ca[None])**2).sum(2), 1)).tolist()
    log(f"codecs fit (vision {KV}, audio {KA})  ({time.time()-t0:.0f}s)")

    # ---- precompute vision tokens + joint concept ----
    Vt = [vis_enc(X32[i]) for i in range(n)]
    def hist(toks, lo, K):
        h = np.bincount(np.array(toks)-lo, minlength=K).astype(np.float32); return h/(np.linalg.norm(h)+1e-8)
    VH = np.stack([hist(Vt[i], VIS0, KV) for i in range(n)])
    AH = np.stack([hist(aud_enc(by_digit[int(yi[i])][RNG.integers(len(by_digit[int(yi[i])]))]), AUD0, KA) for i in range(n)])
    Cg = kmeans(np.concatenate([VH, 6.0*AH], 1), GC); Cvc, Cac = Cg[:, :KV], Cg[:, KV:]
    gist = [assign(np.concatenate([VH[i], 6.0*AH[i]]), Cg) for i in range(n)]
    log(f"concepts fit  ({time.time()-t0:.0f}s)")

    # ---- build interleaved corpus with REAL audio ----
    seqs = []
    for i in range(n):
        d = int(yi[i]); v = Vt[i]; c = [CON0+gist[i]]
        seqs += [
            [BOS]+txt(f"draw {d}")+[SEP]+v+[EOS],
            [BOS]+txt(f"name {d}")+[SEP]+txt(WORD[d])+[EOS],
            [BOS]+v+txt("what digit")+c+[SEP]+txt(str(d))+[EOS],
        ]
        for _ in range(args.aug):                              # real spoken clips of this digit
            a = aud_enc(aug_audio(by_digit[d][RNG.integers(len(by_digit[d]))]))
            seqs += [
                [BOS]+a+txt("what digit")+c+[SEP]+txt(str(d))+[EOS],     # spoken -> text
                [BOS]+a+txt("show it")+c+[SEP]+v+[EOS],                  # spoken -> image
                [BOS]+txt(f"say {d}")+[SEP]+a+[EOS],                     # text -> spoken
            ]
        if i % 5000 == 0: log(f"  built corpus {i}/{n} seqs={len(seqs)}  ({time.time()-t0:.0f}s)")
    log(f"corpus: {len(seqs)} sequences  ({time.time()-t0:.0f}s)")

    psc = UniversalPSC(VOCAB, [(-k,) for k in range(1, args.order+1)], ())
    psc.fit([((len(s),), {(t,): int(x) for t, x in enumerate(s)}, ()) for s in seqs])
    log(f"OMNI MODEL FIT on {len(seqs)} seqs, states={psc.n_states() if hasattr(psc,'n_states') else sum(len(d) for d in psc.t)}  ({time.time()-t0:.0f}s)")

    def roll(prefix, maxlen, temp=0.4, top_p=0.92):
        E = {(t,): int(v) for t, v in enumerate(prefix)}; seq = list(prefix)
        for t in range(len(prefix), maxlen):
            v = _sample(psc.predict(E, (), (t,)), temp, top_p); E[(t,)] = v; seq.append(v)
            if v == EOS: break
        return [x for x in seq[len(prefix):] if x != EOS]

    # ---- evaluate on REAL held-out speech ----
    sp_ok = 0
    for M, lab in fsdd_te:
        a = aud_enc(M); ga = assign(6.0*hist(a, AUD0, KA), Cac)
        o = bytes([x for x in roll([BOS]+a+txt("what digit")+[CON0+ga]+[SEP], len(a)+24) if 48 <= x <= 57])
        sp_ok += int(o[:1] == str(lab).encode())
    log(f"\n=== RESULTS (no backprop) ===")
    log(f"  REAL spoken-digit -> text (held-out FSDD, 300 clips): {100*sp_ok/len(fsdd_te):.1f}%")

    # text->image, spoken->image grids; text->audio is for completeness (not rendered to wav here)
    gi = []
    for d in range(10):
        tk = [x for x in roll([BOS]+txt(f"draw {d}")+[SEP], 40) if VIS0 <= x < VIS0+KV][:16]
        gi.append(vis_dec((tk+[VIS0]*16)[:16]))
    grid(gi, "outputs/omni_real/text2image.png")
    gi = []
    for M, lab in fsdd_te[:10]:
        a = aud_enc(M); ga = assign(6.0*hist(a, AUD0, KA), Cac)
        tk = [x for x in roll([BOS]+a+txt("show it")+[CON0+ga]+[SEP], len(a)+24) if VIS0 <= x < VIS0+KV][:16]
        gi.append(vis_dec((tk+[VIS0]*16)[:16]))
    grid(gi, "outputs/omni_real/spoken2image.png")
    log(f"  wrote outputs/omni_real/text2image.png, spoken2image.png")
    log(f"==== DONE in {time.time()-t0:.0f}s ====")


def grid(imgs, path, cols=10, scale=6, gap=1):
    rows = (len(imgs)+cols-1)//cols; h = w = imgs[0].shape[0]
    cv = np.ones((rows*(h+gap)-gap, cols*(w+gap)-gap), np.float32)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols); cv[r*(h+gap):r*(h+gap)+h, c*(w+gap):c*(w+gap)+w] = im
    Image.fromarray(np.uint8(np.clip(cv,0,1)*255)).resize((cv.shape[1]*scale, cv.shape[0]*scale), Image.NEAREST).save(path)


if __name__ == "__main__":
    main()
