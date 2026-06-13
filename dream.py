"""clark-mind dreams: generative art from pure group theory (no backprop).
-------------------------------------------------------------------------------
The symmetry primitive's mathematical heart -- the ORBIT of a motif under a
symmetry group -- is also how nature and art make mandalas. So the brain
"dreams": scribble a random motif into one angular wedge, then let a symmetry
group (cyclic C_n + optional mirror = dihedral D_n) act on it. Structure
emerges from noise with no training, no diffusion, no gradients -- only the
group orbit (Klein's Erlangen program, turned into pictures).

Each tile = a different random seed x a different group. Same idea that makes
the brain recognize "a door" up to symmetry, run backwards to CREATE.

Run:  python3 dream.py           -> outputs/dream/mandala_gallery.png
"""
from __future__ import annotations
import os, colorsys, numpy as np
from PIL import Image, ImageDraw

os.makedirs("outputs/dream", exist_ok=True)
RNG = np.random.default_rng()
SZ = 240


def _palette(n):
    h0 = RNG.random()
    out = []
    for i in range(n):
        h = (h0 + i / n + 0.12 * RNG.standard_normal()) % 1.0
        s = 0.55 + 0.4 * RNG.random()
        v = 0.65 + 0.35 * RNG.random()
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


def _motif(n_fold):
    """Draw a random motif inside one wedge of angle 2pi/n_fold near center."""
    im = Image.new("RGB", (SZ, SZ), (8, 8, 14))
    d = ImageDraw.Draw(im, "RGBA")
    cx = cy = SZ / 2
    wedge = 2 * np.pi / n_fold
    pal = _palette(RNG.integers(3, 6))
    for _ in range(RNG.integers(5, 11)):
        col = pal[RNG.integers(len(pal))] + (int(RNG.integers(90, 200)),)
        r0 = RNG.uniform(8, SZ * 0.46)
        a0 = RNG.uniform(0, wedge)
        kind = RNG.integers(3)
        if kind == 0:                                    # blob
            rr = RNG.uniform(6, 26)
            x, y = cx + r0 * np.cos(a0), cy + r0 * np.sin(a0)
            d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=col)
        elif kind == 1:                                  # spoke (radial bar)
            r1 = min(SZ * 0.48, r0 + RNG.uniform(20, 80))
            a1 = a0 + RNG.uniform(-0.15, 0.15)
            d.line([cx + r0*np.cos(a0), cy + r0*np.sin(a0),
                    cx + r1*np.cos(a1), cy + r1*np.sin(a1)],
                   fill=col, width=int(RNG.integers(2, 7)))
        else:                                            # arc petal
            rr = RNG.uniform(15, 60)
            x, y = cx + r0*np.cos(a0), cy + r0*np.sin(a0)
            d.pieslice([x - rr, y - rr, x + rr, y + rr],
                       np.degrees(a0), np.degrees(a0) + RNG.integers(40, 140),
                       fill=col)
    return im


def mandala():
    n = int(RNG.integers(4, 10))                         # rotational order
    mirror = bool(RNG.integers(0, 2))                    # cyclic C_n vs dihedral D_n
    base = _motif(n)
    arr = np.asarray(base, np.float32)
    acc = arr.copy()
    for k in range(1, n):                                # apply the rotation group
        rot = base.rotate(360.0 * k / n, resample=Image.BILINEAR, expand=False)
        acc = np.maximum(acc, np.asarray(rot, np.float32))
        if mirror:
            acc = np.maximum(acc, np.asarray(
                rot.transpose(Image.FLIP_LEFT_RIGHT), np.float32))
    if mirror:
        acc = np.maximum(acc, np.asarray(
            base.transpose(Image.FLIP_LEFT_RIGHT), np.float32))
    return Image.fromarray(acc.astype(np.uint8)), n, mirror


def gallery(rows=3, cols=4, gap=6):
    tiles = [mandala() for _ in range(rows * cols)]
    W = cols * SZ + (cols + 1) * gap
    H = rows * SZ + (rows + 1) * gap
    cv = Image.new("RGB", (W, H), (8, 8, 14))
    for i, (im, n, mir) in enumerate(tiles):
        r, c = divmod(i, cols)
        cv.paste(im, (gap + c * (SZ + gap), gap + r * (SZ + gap)))
    path = "outputs/dream/mandala_gallery.png"
    cv.save(path)
    groups = ", ".join(f"{'D' if m else 'C'}{n}" for _, n, m in tiles)
    print(f"clark-mind dreamed {rows*cols} mandalas (no backprop, no training).")
    print(f"groups: {groups}")
    print(f"-> {path}")
    return path


if __name__ == "__main__":
    gallery()
