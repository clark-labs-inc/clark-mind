"""Visual CAPTCHA skills: blurry / noisy / rotated character recognition.
-------------------------------------------------------------------------------
SCOPE / ETHICS: synthetic, self-generated, verifiable character puzzles to
develop and MEASURE the brain's perception -- a cognitive benchmark, like a
distorted-MNIST. NOT a tool against deployed CAPTCHA services.

These play to the architecture's STRENGTH (vision/perception). Recognition is
no-backprop: render many distorted samples, keep class PROTOTYPES (mean
templates) + nearest-prototype match. Two ideas worth noting:

  ROTATION is handled by SEARCH, not by invariance -- you cannot canonicalize
  text by the rotation group (a rotated 6 is a 9, rotated N is Z). So the brain
  DESKEWS: try un-rotating by candidate angles, keep the angle whose result
  best matches an upright prototype. Reasoning-as-search again, on pixels.

Tasks: clean / blurry / noisy / rotated single char, and a multi-char CAPTCHA
string (segment + recognize).

Run:  python3 captcha_vision.py
"""
from __future__ import annotations
import os, numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

np.seterr(over="ignore", invalid="ignore", divide="ignore")
RNG = np.random.default_rng(0)
CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"           # omit easily-confused I,O,0,1
_FCANDS = ["/System/Library/Fonts/Supplemental/Arial.ttf",
           "/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf",
           "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
_FPATH = next((c for c in _FCANDS if os.path.exists(c)), None)
S = 24                                                 # tile size


def _font(sz):
    return ImageFont.truetype(_FPATH, sz) if _FPATH else ImageFont.load_default()


def render(ch, rot=0.0, blur=0.0, noise=0.0, jitter=0, sz=18):
    im = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(im)
    fx = 3 + int(jitter * (RNG.random() - 0.5) * 4)
    fy = 1 + int(jitter * (RNG.random() - 0.5) * 4)
    d.text((fx, fy), ch, fill=255, font=_font(sz))
    if rot:
        im = im.rotate(rot, resample=Image.BILINEAR, expand=False)
    if blur:
        im = im.filter(ImageFilter.GaussianBlur(blur))
    a = np.asarray(im, np.float32) / 255.0
    if noise:
        a = np.clip(a + noise * RNG.standard_normal(a.shape), 0, 1)
    return a


def feat(a):
    return a.ravel()                                   # raw pixels (vision strength)


# ---- no-backprop recognizer: class prototypes from distorted samples ----
class CharRecognizer:
    def __init__(self, distort):
        self.protos = {}
        for ch in CHARS:
            samples = [feat(render(ch, **distort())) for _ in range(40)]
            self.protos[ch] = np.mean(samples, 0)
        self.M = np.stack([self.protos[c] for c in CHARS])

    def classify(self, a):
        f = feat(a)
        d = ((self.M - f) ** 2).sum(1)
        return CHARS[int(np.argmin(d))]

    def deskew_classify(self, a_img_grid_char, angles=range(-40, 41, 5)):
        """ROTATION via SEARCH: a is a rendered grid; we don't have the source
        char, so we match against upright prototypes after un-rotating by each
        candidate angle, taking the best-matching (angle, class)."""
        best, bd = None, 1e18
        base = Image.fromarray(np.uint8(a_img_grid_char * 255))
        for ang in angles:
            r = np.asarray(base.rotate(-ang, resample=Image.BILINEAR), np.float32) / 255.0
            f = r.ravel()
            d = ((self.M - f) ** 2).sum(1)
            j = int(np.argmin(d))
            if d[j] < bd:
                bd, best = d[j], CHARS[j]
        return best


def _acc(distort, rec, n=400, deskew=False):
    ok = 0
    for _ in range(n):
        ch = CHARS[int(RNG.integers(len(CHARS)))]
        a = render(ch, **distort())
        pred = rec.deskew_classify(a) if deskew else rec.classify(a)
        ok += (pred == ch)
    return 100 * ok / n


# ---- multi-char CAPTCHA string: segment by columns, recognize each ----
PITCH = 16                                             # px between chars


def render_string(s, blur=0.6, noise=0.05):
    w = PITCH * len(s) + 8
    im = Image.new("L", (w, S), 0); d = ImageDraw.Draw(im)
    for i, ch in enumerate(s):
        d.text((4 + i * PITCH, 1 + int(RNG.integers(-2, 3))), ch, fill=255,
               font=_font(18))
    im = im.filter(ImageFilter.GaussianBlur(blur))
    a = np.asarray(im, np.float32) / 255.0
    return np.clip(a + noise * RNG.standard_normal(a.shape), 0, 1)


def solve_string(a, rec, k):
    """Extract each char into an S x S window aligned to how single chars were
    rendered for the prototypes (char near x=3). Fixed pitch segmentation."""
    out = []
    for i in range(k):
        x0 = i * PITCH + 1                              # align char to ~x=3
        tile = np.zeros((S, S), np.float32)
        seg = a[:, x0:x0 + S]
        tile[:, :seg.shape[1]] = seg
        out.append(rec.classify(tile))
    return "".join(out)


def main():
    print("VISUAL CAPTCHA -- distorted character recognition (no backprop, "
          "verifiable)\n")
    print(f"   font: {os.path.basename(_FPATH) if _FPATH else 'PIL default'}, "
          f"{len(CHARS)} classes (chance {100/len(CHARS):.0f}%)\n")

    clean = CharRecognizer(lambda: dict(sz=18))
    print(f"   clean characters          {_acc(lambda: dict(sz=18), clean):4.0f}%")

    blurry = CharRecognizer(lambda: dict(blur=float(RNG.uniform(0.4, 1.4)), sz=18))
    print(f"   BLURRY (gaussian 0.4-1.4) {_acc(lambda: dict(blur=float(RNG.uniform(0.4,1.4)), sz=18), blurry):4.0f}%")

    noisy = CharRecognizer(lambda: dict(noise=float(RNG.uniform(0.1, 0.35)), sz=18))
    print(f"   NOISY (sigma 0.1-0.35)    {_acc(lambda: dict(noise=float(RNG.uniform(0.1,0.35)), sz=18), noisy):4.0f}%")

    jit = CharRecognizer(lambda: dict(blur=0.6, noise=0.12, jitter=1, sz=18))
    print(f"   BLUR+NOISE+JITTER (combo) {_acc(lambda: dict(blur=0.6, noise=0.12, jitter=1, sz=18), jit):4.0f}%")

    # ROTATION: invariance is WRONG (6<->9), so deskew-by-search instead
    upr = CharRecognizer(lambda: dict(sz=18))           # upright prototypes only
    rot_naive = _acc(lambda: dict(rot=float(RNG.uniform(-35, 35)), sz=18), upr)
    rot_search = _acc(lambda: dict(rot=float(RNG.uniform(-35, 35)), sz=18), upr, deskew=True)
    print(f"   ROTATED +-35 naive match  {rot_naive:4.0f}%   "
          f"(upright prototypes fail on tilt)")
    print(f"   ROTATED +-35 deskew SEARCH {rot_search:4.0f}%   "
          f"(try-angles, best match -- reasoning as search)")

    # multi-char CAPTCHA string
    sok = 0; cok = 0; ct = 0
    for _ in range(150):
        k = int(RNG.integers(3, 6))
        s = "".join(CHARS[int(RNG.integers(len(CHARS)))] for _ in range(k))
        pred = solve_string(render_string(s), jit, k)
        sok += (pred == s)
        cok += sum(a == b for a, b in zip(pred, s)); ct += k
    print(f"\n   CAPTCHA STRING (3-5 chars) {100*sok/150:4.0f}% exact   "
          f"{100*cok/ct:4.0f}% per-char  (segment + recognize)")
    print("\nperception is the strength: blur/noise barely dent it. rotation needs")
    print("SEARCH not invariance (a rotated 6 is a 9). all synthetic + verifiable.")


if __name__ == "__main__":
    main()
