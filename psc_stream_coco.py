"""
Stream COCO captions (image+text) fully through the unified PSC (no backprop).
------------------------------------------------------------------------------
Real vision+language: sayakpaul/coco-30-val-2014 (30k real images + captions),
streamed end-to-end. Codec warmed on a prefix, then the ENTIRE dataset is streamed
once, encoding each (image,caption) into interleaved tokens and updating the count
model incrementally (the dataset is never held in memory). One vocabulary:
    text bytes | vision codes | BOS/EOS/SEP
Two task views per row: caption->image and image->caption. Honest: COCO is hard
for this architecture; this demonstrates full streaming + scaling, with metrics.
"""
from __future__ import annotations
import argparse, os, time, numpy as np
from PIL import Image
np.seterr(over="ignore", invalid="ignore", divide="ignore")
os.makedirs("outputs/coco", exist_ok=True)
RNG = np.random.default_rng(0)
from psc_studio import UniversalPSC, _sample

KV = 512
VIS0 = 256
BOS, EOS, SEP = VIS0+KV, VIS0+KV+1, VIS0+KV+2
VOCAB = SEP+1
SZ, P, G = 32, 8, 4           # 32x32 RGB, 8x8 patches -> 16 tokens/image
LOG = open("outputs/coco/stream.log", "a");
def log(m): print(m); LOG.write(m+"\n"); LOG.flush()


def to32(pil):
    return np.asarray(pil.convert("RGB").resize((SZ, SZ)), np.float32) / 255.0

def patches(x):
    return np.stack([x[y*P:y*P+P, c*P:c*P+P].ravel() for y in range(G) for c in range(G)]).astype(np.float32)

def clean(cap, maxlen):
    return cap.strip().lower().encode("ascii", "ignore")[:maxlen]


class VisCodec:
    def fit(self, imgs):
        Pm = np.concatenate([patches(x) for x in imgs])
        self.mu, self.sd = Pm.mean(0), Pm.std(0)+1e-6
        Z = (Pm-self.mu)/self.sd
        C = Z[RNG.choice(len(Z), KV, replace=False)].copy(); xn = (Z**2).sum(1)[:, None]
        for _ in range(12):
            a = (xn - 2*Z@C.T + (C**2).sum(1)[None]).argmin(1)
            for g in range(KV):
                if (a == g).any(): C[g] = Z[a == g].mean(0)
        self.C = C
    def enc(self, x):
        z = (patches(x)-self.mu)/self.sd
        return (VIS0 + np.argmin(((z[:, None]-self.C[None])**2).sum(2), 1)).tolist()
    def dec(self, toks):
        F = self.C[np.array(toks)-VIS0]*self.sd+self.mu; out = np.zeros((SZ, SZ, 3), np.float32)
        for i, f in enumerate(F):
            y, x = divmod(i, G); out[y*P:y*P+P, x*P:x*P+P] = f.reshape(P, P, 3)
        return np.clip(out, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30000)
    ap.add_argument("--warm", type=int, default=2500)
    ap.add_argument("--order", type=int, default=6)
    ap.add_argument("--caplen", type=int, default=48)
    args = ap.parse_args()
    t0 = time.time()
    from datasets import load_dataset
    ds = load_dataset("sayakpaul/coco-30-val-2014", split="train", streaming=True)
    log(f"\n==== stream COCO captions limit={args.limit} order={args.order} {time.ctime()} ====")

    psc = UniversalPSC(VOCAB, [(-k,) for k in range(1, args.order+1)], ())
    codec = VisCodec()
    warm_imgs, warm_rows, held = [], [], []
    buf = []
    seen = 0
    bpc_hist = []

    def flush():
        if buf:
            psc.fit([((len(s),), {(t,): int(x) for t, x in enumerate(s)}, ()) for s in buf]); buf.clear()

    def make_seqs(toks, cap):
        cb = [b for b in cap]
        return [[BOS]+cb+[SEP]+toks+[EOS], [BOS]+toks+[SEP]+cb+[EOS]]

    def heldout_bpc():
        if not held: return float("nan")
        bits = nt = 0
        for toks, cap in held[:120]:
            E = {(i,): v for i, v in enumerate([BOS]+toks+[SEP])}; base = len(E)
            for j, b in enumerate(cap):
                p = psc.predict(E, (), (base+j,)); bits += -np.log2(max(p[b], 1e-12)); nt += 1
                E[(base+j,)] = b
        return bits/max(1, nt)

    for row in ds:
        if seen >= args.limit: break
        try:
            x = to32(row["image"]); cap = clean(row["caption"], args.caplen)
        except Exception:
            continue
        if len(cap) < 4:
            seen += 1; continue
        if seen < args.warm:
            warm_imgs.append(x); warm_rows.append((x, cap))
            if seen == args.warm-1:
                codec.fit(warm_imgs); log(f"  codec fit on {len(warm_imgs)} imgs ({time.time()-t0:.0f}s)")
                for xx, cc in warm_rows: buf += make_seqs(codec.enc(xx), cc)
                flush(); warm_imgs = []
        else:
            toks = codec.enc(x)
            if seen % 50 == 0 and len(held) < 400:
                held.append((toks, cap))
            else:
                buf += make_seqs(toks, cap)
                if len(buf) >= 2000: flush()
        seen += 1
        if seen % 5000 == 0:
            flush(); b = heldout_bpc(); bpc_hist.append((seen, b))
            states = sum(len(d) for d in psc.t)
            log(f"  streamed {seen}/{args.limit}  states={states}  heldout_caption_bits/char={b:.3f}  ({time.time()-t0:.0f}s)")
    flush()
    states = sum(len(d) for d in psc.t)
    log(f"STREAM COMPLETE: {seen} rows, states={states}, final caption bits/char={heldout_bpc():.3f}  ({time.time()-t0:.0f}s)")

    # ---- image -> caption (a few held-out) ----
    log("\n=== image -> caption samples (held-out) ===")
    for toks, cap in held[:6]:
        E = {(i,): v for i, v in enumerate([BOS]+toks+[SEP])}; gen = []
        for t in range(len(E), len(E)+40):
            v = _sample(psc.predict(E, (), (t,)), 0.5, 0.9); E[(t,)] = v
            if v == EOS or not (0 <= v < 256): break
            gen.append(v)
        log(f"  REAL: {cap.decode('latin1')[:50]!r}   ->  GEN: {bytes(gen).decode('latin1')[:50]!r}")

    # ---- caption -> image (real held-out captions) + reconstructions ----
    gi = []
    for toks, cap in held[:10]:
        E = {(i,): v for i, v in enumerate([BOS]+[b for b in cap]+[SEP])}; vis = []
        for t in range(len(E), len(E)+16):
            p = psc.predict(E, (), (t,)).copy(); p[:VIS0] = 0; p[VIS0+KV:] = 0
            if p.sum() < 1e-9: p[VIS0:VIS0+KV] = 1
            v = _sample(p/p.sum(), 0.7, 0.95); E[(t,)] = v; vis.append(v)
        gi.append(codec.dec(vis))
    grid_rgb(gi, "outputs/coco/caption2image.png")
    grid_rgb([codec.dec(t) for t, _ in held[:10]], "outputs/coco/recon.png")
    log("  wrote outputs/coco/caption2image.png, recon.png")
    log(f"  scaling (rows, bits/char): {bpc_hist}")
    log(f"==== DONE {time.time()-t0:.0f}s ====")


def grid_rgb(imgs, path, cols=10, scale=5, gap=1):
    rows = (len(imgs)+cols-1)//cols; h = w = imgs[0].shape[0]
    cv = np.ones((rows*(h+gap)-gap, cols*(w+gap)-gap, 3), np.float32)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols); cv[r*(h+gap):r*(h+gap)+h, c*(w+gap):c*(w+gap)+w] = im
    Image.fromarray(np.uint8(cv*255)).resize((cv.shape[1]*scale, cv.shape[0]*scale), Image.NEAREST).save(path)


if __name__ == "__main__":
    main()
