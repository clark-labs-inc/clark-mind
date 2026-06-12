"""
Generative perception with a Predictive State Column (no backprop)
------------------------------------------------------------------
Image-0 smoke test of the PSC-L generation plan: same predictive-state learning
rule used for dynamics/language, now turned into an image generator. Pipeline:

    MNIST image
      -> no-backprop patch codec (online k-means w/ homeostasis)  = sensory organ
      -> sequence of patch-code events on a grid
      -> PSC: predict each patch code from (class, position, causal neighbours)
              via a backoff count model (predictive states; merge=backoff)
      -> raster-sample codes, decode patches -> PNG

No loss.backward(), no optimizer, no GAN/diffusion/transformer. Outputs:
    outputs/image/codebook.png      learned patch atoms (the V1-like dictionary)
    outputs/image/recon.png         originals (top) vs codec round-trip (bottom)
    outputs/image/samples.png       class-conditioned generated digits
"""

from __future__ import annotations
import argparse
import os
import numpy as np
from PIL import Image

np.seterr(over="ignore", invalid="ignore", divide="ignore")


# -----------------------------------------------------------------------------
# Data (local MNIST via torchvision; no download if already present)
# -----------------------------------------------------------------------------
def load_mnist(n_train, n_test):
    from torchvision.datasets import MNIST
    tr = MNIST(root="./data", train=True, download=True)
    te = MNIST(root="./data", train=False, download=True)
    Xtr = tr.data.numpy().astype(np.float32)[:n_train] / 255.0
    ytr = tr.targets.numpy().astype(np.int64)[:n_train]
    Xte = te.data.numpy().astype(np.float32)[:n_test] / 255.0
    yte = te.targets.numpy().astype(np.int64)[:n_test]
    return Xtr, ytr, Xte, yte


# -----------------------------------------------------------------------------
# No-backprop patch codec: online k-means with homeostatic boost
# -----------------------------------------------------------------------------
class PatchCodec:
    def __init__(self, patch=4, codes=256, seed=0):
        self.p, self.K, self.rng = patch, codes, np.random.default_rng(seed)
        self.C = None

    def _patches(self, img):                       # non-overlapping -> trivial decode
        p, g = self.p, 28 // self.p
        return np.stack([img[y * p:(y + 1) * p, x * p:(x + 1) * p].ravel()
                         for y in range(g) for x in range(g)]), g

    def fit(self, X, epochs=1, lr0=0.25):
        allp = np.concatenate([self._patches(x)[0] for x in X[:4000]])
        self.C = allp[self.rng.choice(len(allp), self.K, replace=False)].astype(np.float32)
        seen = np.full(self.K, 1e-3, np.float32)
        for _ in range(epochs):
            for x in X:
                P, _ = self._patches(x)
                for p in P:
                    d2 = np.sum((self.C - p) ** 2, 1)
                    boost = 1.0 / np.sqrt(seen)            # homeostasis: revive dead codes
                    k = int(np.argmin(d2 / boost))
                    lr = lr0 / (1.0 + 0.000003 * seen.sum())
                    self.C[k] += lr * (p - self.C[k]); seen[k] += 1
        self.seen = seen

    def encode(self, img):
        P, g = self._patches(img)
        codes = np.argmin(((P[:, None, :] - self.C[None]) ** 2).sum(2), 1)
        return codes.reshape(g, g).astype(np.int32)

    def decode(self, grid):
        g, p = grid.shape[0], self.p
        out = np.zeros((28, 28), np.float32)
        for y in range(g):
            for x in range(g):
                out[y * p:(y + 1) * p, x * p:(x + 1) * p] = self.C[grid[y, x]].reshape(p, p)
        return np.clip(out, 0, 1)


# -----------------------------------------------------------------------------
# PSC image model: predictive-state backoff count model over patch codes
# context for code[y,x] = (label, y, x, left, up, upleft); backs off when unseen
# -----------------------------------------------------------------------------
class PSCImage:
    def __init__(self, K, grid, n_class=10, alpha=0.02, ab=4.0):
        self.K, self.g, self.alpha, self.ab = K, grid, alpha, ab
        self.t = [{} for _ in range(4)]            # backoff levels (high->low order)

    def _keys(self, lab, y, x, L, U, UL):
        return [(3, lab, y, x, L, U, UL), (2, lab, y, x, L, U), (1, lab, y, x), (0, y, x)]

    def _bump(self, lvl, key, code):
        d = self.t[lvl]
        a = d.get(key)
        if a is None:
            a = np.zeros(self.K, np.float64); d[key] = a
        a[code] += 1.0

    def fit(self, grids, labels):
        for grid, lab in zip(grids, labels):
            for y in range(self.g):
                for x in range(self.g):
                    L = grid[y, x - 1] if x else -1
                    U = grid[y - 1, x] if y else -1
                    UL = grid[y - 1, x - 1] if (y and x) else -1
                    keys = self._keys(int(lab), y, x, int(L), int(U), int(UL))
                    code = int(grid[y, x])
                    for lvl, k in enumerate(keys):
                        self._bump(lvl, k, code)

    def dist(self, lab, y, x, L, U, UL):
        p = np.full(self.K, 1.0 / self.K)          # uniform prior (deepest backoff)
        for lvl, k in reversed(list(enumerate(self._keys(lab, y, x, L, U, UL)))):
            a = self.t[lvl].get(k)
            if a is not None:
                c = a.sum(); lam = c / (c + self.ab)
                p = lam * ((a + self.alpha) / (c + self.alpha * self.K)) + (1 - lam) * p
        return p

    def n_states(self):
        return sum(len(d) for d in self.t)


def sample_topp(p, temp, top_p, rng):
    if temp <= 0:
        return int(p.argmax())
    logp = np.log(p + 1e-12) / temp
    q = np.exp(logp - logp.max()); q /= q.sum()
    order = np.argsort(-q); cum = np.cumsum(q[order])
    cut = order[:max(1, int(np.searchsorted(cum, top_p)) + 1)]
    qq = q[cut] / q[cut].sum()
    return int(rng.choice(cut, p=qq))


def generate(psc, codec, label, rng, temp=0.9, top_p=0.95):
    g = psc.g
    grid = np.zeros((g, g), np.int32)
    for y in range(g):
        for x in range(g):
            L = grid[y, x - 1] if x else -1
            U = grid[y - 1, x] if y else -1
            UL = grid[y - 1, x - 1] if (y and x) else -1
            grid[y, x] = sample_topp(psc.dist(label, y, x, int(L), int(U), int(UL)),
                                     temp, top_p, rng)
    return codec.decode(grid)


def grid_png(imgs, path, cols, scale=3, gap=1):
    rows = (len(imgs) + cols - 1) // cols
    h = w = 28
    canvas = np.ones((rows * (h + gap) - gap, cols * (w + gap) - gap), np.float32)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        canvas[r * (h + gap):r * (h + gap) + h, c * (w + gap):c * (w + gap) + w] = im
    img = Image.fromarray(np.uint8(canvas * 255)).resize(
        (canvas.shape[1] * scale, canvas.shape[0] * scale), Image.NEAREST)
    img.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--n_test", type=int, default=3000)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--codes", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs("outputs/image", exist_ok=True)
    rng = np.random.default_rng(args.seed)

    Xtr, ytr, Xte, yte = load_mnist(args.n_train, args.n_test)
    print(f"MNIST train={len(Xtr)} test={len(Xte)}")

    codec = PatchCodec(patch=args.patch, codes=args.codes, seed=args.seed)
    codec.fit(Xtr, epochs=1)
    g = 28 // args.patch
    dead = int(np.mean(codec.seen < 1.5))
    print(f"codec: patch={args.patch} grid={g}x{g} codes={args.codes} dead={dead}")

    Gtr = [codec.encode(x) for x in Xtr]
    Gte = [codec.encode(x) for x in Xte]
    # codec reconstruction PSNR on held-out
    mse = np.mean([(codec.decode(codec.encode(x)) - x) ** 2 for x in Xte[:500]])
    print(f"codec recon PSNR={10*np.log10(1.0/(mse+1e-12)):.2f} dB")

    psc = PSCImage(args.codes, g)
    psc.fit(Gtr, ytr)
    # held-out bits/token + token accuracy
    bits, correct, ntok = 0.0, 0, 0
    for grid, lab in zip(Gte, yte):
        for y in range(g):
            for x in range(g):
                L = grid[y, x-1] if x else -1; U = grid[y-1, x] if y else -1
                UL = grid[y-1, x-1] if (y and x) else -1
                p = psc.dist(int(lab), y, x, int(L), int(U), int(UL))
                t = int(grid[y, x]); bits += -np.log2(max(p[t], 1e-12))
                correct += int(p.argmax() == t); ntok += 1
    print(f"PSC: states={psc.n_states()} heldout bits/token={bits/ntok:.3f} "
          f"token_acc={correct/ntok:.3f}")

    # ---- human-inspectable outputs ----
    atoms = [codec.C[i].reshape(args.patch, args.patch) for i in range(args.codes)]
    atoms = [(a - a.min()) / (a.max() - a.min() + 1e-8) for a in atoms]
    # upscale atoms to 28 for the grid helper
    atoms28 = [np.kron(a, np.ones((28 // args.patch, 28 // args.patch))) for a in atoms]
    grid_png(atoms28[:256], "outputs/image/codebook.png", cols=16, scale=2)

    recon = []
    for i in range(10):
        recon.append(Xte[i]);
    for i in range(10):
        recon.append(codec.decode(Gte[i]))
    grid_png(recon, "outputs/image/recon.png", cols=10, scale=4)

    samples = []
    for lab in range(10):
        for _ in range(8):
            samples.append(generate(psc, codec, lab, rng, temp=args.temp))
    grid_png(samples, "outputs/image/samples.png", cols=8, scale=4)
    print("wrote outputs/image/{codebook,recon,samples}.png")


if __name__ == "__main__":
    main()
