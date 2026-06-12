"""
CIFAR-10 color generation with a Predictive State Column (no backprop).
Same recipe as psc_image_gen.py, extended to 32x32 RGB. Codec = online k-means
on 4x4x3 patches; PSC = backoff count model predicting patch codes from
(class, position, causal neighbours); raster top-p sampling -> decode -> PNG.
"""
from __future__ import annotations
import argparse, os
import numpy as np
from PIL import Image

np.seterr(over="ignore", invalid="ignore", divide="ignore")


def load_cifar(n_train, n_test):
    from datasets import load_dataset
    ds = load_dataset("uoft-cs/cifar10")
    def grab(split, n):
        rows = ds[split].select(range(min(n, len(ds[split]))))
        X = np.stack([np.asarray(r["img"], np.float32) / 255.0 for r in rows])
        y = np.array([r["label"] for r in rows], np.int64)
        return X, y
    Xtr, ytr = grab("train", n_train); Xte, yte = grab("test", n_test)
    return Xtr, ytr, Xte, yte


class RGBPatchCodec:
    def __init__(self, size=32, patch=4, codes=512, seed=0):
        self.S, self.p, self.K, self.rng = size, patch, codes, np.random.default_rng(seed)
        self.g = size // patch

    def _patches(self, img):
        p, g = self.p, self.g
        return np.stack([img[y*p:(y+1)*p, x*p:(x+1)*p, :].ravel()
                         for y in range(g) for x in range(g)])

    def fit(self, X, lr0=0.25):
        allp = np.concatenate([self._patches(x) for x in X[:3000]])
        self.C = allp[self.rng.choice(len(allp), self.K, replace=False)].astype(np.float32)
        seen = np.full(self.K, 1e-3, np.float32)
        for x in X:
            for p in self._patches(x):
                d2 = np.sum((self.C - p) ** 2, 1)
                k = int(np.argmin(d2 / (1.0 / np.sqrt(seen))))
                lr = lr0 / (1.0 + 0.000003 * seen.sum())
                self.C[k] += lr * (p - self.C[k]); seen[k] += 1
        self.seen = seen

    def encode(self, img):
        P = self._patches(img)
        return np.argmin(((P[:, None] - self.C[None]) ** 2).sum(2), 1).reshape(self.g, self.g).astype(np.int32)

    def decode(self, grid):
        p, g = self.p, self.g
        out = np.zeros((self.S, self.S, 3), np.float32)
        for y in range(g):
            for x in range(g):
                out[y*p:(y+1)*p, x*p:(x+1)*p, :] = self.C[grid[y, x]].reshape(p, p, 3)
        return np.clip(out, 0, 1)


class PSCImage:
    def __init__(self, K, g, alpha=0.02, ab=4.0):
        self.K, self.g, self.alpha, self.ab = K, g, alpha, ab
        self.t = [{} for _ in range(4)]

    def _keys(self, lab, y, x, L, U, UL):
        return [(lab, y, x, L, U, UL), (lab, y, x, L, U), (lab, y, x), (y, x)]

    def fit(self, grids, labels):
        for grid, lab in zip(grids, labels):
            for y in range(self.g):
                for x in range(self.g):
                    L = int(grid[y, x-1]) if x else -1
                    U = int(grid[y-1, x]) if y else -1
                    UL = int(grid[y-1, x-1]) if (y and x) else -1
                    code = int(grid[y, x])
                    for lvl, k in enumerate(self._keys(int(lab), y, x, L, U, UL)):
                        d = self.t[lvl]; a = d.get(k)
                        if a is None: a = np.zeros(self.K); d[k] = a
                        a[code] += 1.0

    def dist(self, lab, y, x, L, U, UL):
        p = np.full(self.K, 1.0 / self.K)
        for lvl, k in reversed(list(enumerate(self._keys(lab, y, x, L, U, UL)))):
            a = self.t[lvl].get(k)
            if a is not None:
                c = a.sum(); lam = c / (c + self.ab)
                p = lam * ((a + self.alpha) / (c + self.alpha * self.K)) + (1 - lam) * p
        return p

    def n_states(self): return sum(len(d) for d in self.t)


def sample_topp(p, temp, top_p, rng):
    logp = np.log(p + 1e-12) / max(temp, 1e-3); q = np.exp(logp - logp.max()); q /= q.sum()
    o = np.argsort(-q); cut = o[:max(1, int(np.searchsorted(np.cumsum(q[o]), top_p)) + 1)]
    return int(rng.choice(cut, p=q[cut] / q[cut].sum()))


def generate(psc, codec, lab, rng, temp=0.9, top_p=0.95):
    g = psc.g; grid = np.zeros((g, g), np.int32)
    for y in range(g):
        for x in range(g):
            L = int(grid[y, x-1]) if x else -1; U = int(grid[y-1, x]) if y else -1
            UL = int(grid[y-1, x-1]) if (y and x) else -1
            grid[y, x] = sample_topp(psc.dist(lab, y, x, L, U, UL), temp, top_p, rng)
    return codec.decode(grid)


def grid_png(imgs, path, cols, scale=4, gap=1):
    rows = (len(imgs) + cols - 1) // cols; h = w = imgs[0].shape[0]
    cv = np.ones((rows*(h+gap)-gap, cols*(w+gap)-gap, 3), np.float32)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        cv[r*(h+gap):r*(h+gap)+h, c*(w+gap):c*(w+gap)+w] = im
    Image.fromarray(np.uint8(cv*255)).resize((cv.shape[1]*scale, cv.shape[0]*scale), Image.NEAREST).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=15000); ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--patch", type=int, default=4); ap.add_argument("--codes", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.9); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs("outputs/image", exist_ok=True); rng = np.random.default_rng(args.seed)
    CLASSES = ["plane","auto","bird","cat","deer","dog","frog","horse","ship","truck"]

    Xtr, ytr, Xte, yte = load_cifar(args.n_train, args.n_test)
    print(f"CIFAR-10 train={len(Xtr)} test={len(Xte)}")
    codec = RGBPatchCodec(size=32, patch=args.patch, codes=args.codes, seed=args.seed)
    codec.fit(Xtr)
    print(f"codec: grid={codec.g}x{codec.g} codes={args.codes} dead={int(np.sum(codec.seen<1.5))}")
    Gtr = [codec.encode(x) for x in Xtr]; Gte = [codec.encode(x) for x in Xte]
    mse = np.mean([(codec.decode(codec.encode(x)) - x) ** 2 for x in Xte[:300]])
    print(f"codec recon PSNR={10*np.log10(1.0/(mse+1e-12)):.2f} dB")

    psc = PSCImage(args.codes, codec.g); psc.fit(Gtr, ytr)
    bits, corr, n = 0.0, 0, 0
    for grid, lab in zip(Gte, yte):
        for y in range(codec.g):
            for x in range(codec.g):
                L = int(grid[y,x-1]) if x else -1; U = int(grid[y-1,x]) if y else -1
                UL = int(grid[y-1,x-1]) if (y and x) else -1
                p = psc.dist(int(lab), y, x, L, U, UL); t = int(grid[y, x])
                bits += -np.log2(max(p[t],1e-12)); corr += int(p.argmax()==t); n += 1
    print(f"PSC: states={psc.n_states()} heldout bits/token={bits/n:.3f} token_acc={corr/n:.3f}")

    recon = [Xte[i] for i in range(10)] + [codec.decode(Gte[i]) for i in range(10)]
    grid_png(recon, "outputs/image/cifar_recon.png", cols=10, scale=5)
    samples = [generate(psc, codec, lab, rng, temp=args.temp) for lab in range(10) for _ in range(8)]
    grid_png(samples, "outputs/image/cifar_samples.png", cols=8, scale=5)
    print("classes (rows):", CLASSES)
    print("wrote outputs/image/cifar_{recon,samples}.png")


if __name__ == "__main__":
    main()
