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


# ---- HARD adversarial distortions (warp, occlusion, clutter) ----
def warp(a, amp=2.0, period=10.0):
    H, W = a.shape
    ph = RNG.uniform(0, 6.28)
    out = np.zeros_like(a)
    for y in range(H):
        sh = int(round(amp * np.sin(2 * np.pi * y / period + ph)))
        out[y] = np.roll(a[y], sh)
    ph2 = RNG.uniform(0, 6.28)
    for x in range(W):
        sh = int(round(amp * np.sin(2 * np.pi * x / period + ph2)))
        out[:, x] = np.roll(out[:, x], sh)
    return out


def occlude(a, n=2):
    im = Image.fromarray(np.uint8(a * 255)); d = ImageDraw.Draw(im)
    H, W = a.shape
    for _ in range(n):
        d.line([(RNG.integers(0, W), RNG.integers(0, H)),
                (RNG.integers(0, W), RNG.integers(0, H))],
               fill=int(RNG.integers(120, 255)), width=1)
    return np.asarray(im, np.float32) / 255.0


def clutter(a, dots=18):
    a = a.copy(); H, W = a.shape
    for _ in range(dots):
        a[RNG.integers(0, H), RNG.integers(0, W)] = RNG.uniform(0.4, 1.0)
    return a


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


def render_char_hard(ch):
    a = render(ch, rot=float(RNG.uniform(-25, 25)),
               blur=float(RNG.uniform(0.3, 1.0)), jitter=1, sz=18)
    a = warp(a, amp=1.6); a = occlude(a, 1); a = clutter(a, 10)
    return a


def hard_distort():
    return None  # marker; CharRecognizer can't call render(**); handled below


class HardRecognizer:
    """Prototypes built from the HARD adversarial distortions; deskew at test."""
    def __init__(self):
        self.protos = {}
        for ch in CHARS:
            self.protos[ch] = np.mean([feat(render_char_hard(ch))
                                       for _ in range(60)], 0)
        self.M = np.stack([self.protos[c] for c in CHARS])

    def best(self, tileSxS):
        f = tileSxS.ravel(); d = ((self.M - f) ** 2).sum(1)
        j = int(np.argmin(d))
        return CHARS[j], float(d[j])

    def deskew(self, a, angles=range(-30, 31, 6)):
        base = Image.fromarray(np.uint8(np.clip(a, 0, 1) * 255))
        best, bd = "?", 1e18
        for ang in angles:
            r = np.asarray(base.rotate(-ang, resample=Image.BILINEAR),
                           np.float32) / 255.0
            ch, d = self.best(r)
            if d < bd:
                bd, best = d, ch
        return best, bd


# ---- HARD string: overlapping glued chars + warp + occlusion + clutter ----
def render_string_hard(s, pitch=12):
    """Hardest-still-LEGIBLE: overlap (pitch 12), per-char size + rotation
    jitter, mild warp, one occlusion line, clutter. (Cranking past this makes
    the string illegible to humans too -- not a fair test; the real lever for
    'harder' beyond here is a segmentation primitive, not more noise.)"""
    w = pitch * len(s) + 12
    im = Image.new("L", (w, S), 0)
    for i, ch in enumerate(s):
        sz = int(RNG.integers(16, 20))
        ci = Image.new("L", (S, S), 0)
        ImageDraw.Draw(ci).text((2, 1), ch, fill=255, font=_font(sz))
        ci = ci.rotate(float(RNG.uniform(-9, 9)), resample=Image.BILINEAR)
        x0 = 4 + i * pitch
        reg = np.maximum(np.asarray(im.crop((x0, 0, x0 + S, S))), np.asarray(ci))
        im.paste(Image.fromarray(reg.astype(np.uint8)), (x0, 0))
    a = np.asarray(im.filter(ImageFilter.GaussianBlur(0.6)), np.float32) / 255.0
    a = warp(a, amp=0.9); a = occlude(a, 1); a = clutter(a, 12)
    return a


def _window_tile(a, x, w):
    """Pad the window into an S x S tile (char left-aligned, like training) --
    do NOT resize: stretching a narrow window distorts the aspect away from the
    upright single-char prototypes."""
    seg = a[:, max(0, x):min(a.shape[1], x + w)]
    if seg.shape[1] == 0:
        return None
    tile = np.zeros((S, S), np.float32)
    ww = min(S, seg.shape[1])
    tile[:, :ww] = seg[:, :ww]
    return tile


def _score_window(a, x, w, rec, angles=(-12, 0, 12)):
    """Best (char, confidence) for the window [x:x+w], trying small deskews."""
    seg = a[:, max(0, x):min(a.shape[1], x + w)]
    if seg.shape[1] == 0:
        return "?", -1e18
    base = Image.fromarray(np.uint8(np.clip(seg, 0, 1) * 255))
    best_ch, best_d = "?", 1e18
    for ang in angles:
        r = base.rotate(-ang, resample=Image.BILINEAR) if ang else base
        tile = np.zeros((S, S), np.float32)
        rr = np.asarray(r, np.float32) / 255.0
        ww = min(S, rr.shape[1]); tile[:, :ww] = rr[:, :ww]
        ch, d = rec.best(tile)
        if d < best_d:
            best_d, best_ch = d, ch
    return best_ch, -best_d                              # higher = better


def solve_string_search(a, rec, kmin=3, kmax=6, widths=range(9, 16, 2)):
    """RECOGNITION-DRIVEN SEGMENTATION: DP over cut positions x and char-count
    j; each candidate window is scored by the recognizer's best confidence
    across small deskews. Pick, over k in [kmin,kmax], the full-width covering
    with best AVERAGE confidence (so char count is chosen, not biased)."""
    W = a.shape[1]
    dp = {(0, 0): (0.0, [])}
    for x in range(0, W):
        for j in range(0, kmax):
            if (x, j) not in dp:
                continue
            base_s, base_p = dp[(x, j)]
            for w in widths:
                nx = min(x + w, W)
                if x + w > W + 3:
                    continue
                ch, conf = _score_window(a, x, w, rec)
                ns = base_s + conf
                key = (nx, j + 1)
                if key not in dp or ns > dp[key][0]:
                    dp[key] = (ns, base_p + [ch])
    best = None
    for k in range(kmin, kmax + 1):
        if (W, k) in dp:
            avg = dp[(W, k)][0] / k
            if best is None or avg > best[0]:
                best = (avg, "".join(dp[(W, k)][1]))
    return best[1] if best else ""


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
          f"{100*cok/ct:4.0f}% per-char  (fixed-pitch segment + recognize)")

    # ===================== HARD MODE =====================
    print("\n   --- HARD MODE: warp + occlusion lines + clutter + overlap ---")
    hard = HardRecognizer()
    h_ok = sum(hard.deskew(render_char_hard(
        ch := CHARS[int(RNG.integers(len(CHARS)))]))[0] == ch for _ in range(300))
    print(f"   HARD single char           {100*h_ok/300:4.0f}%   "
          f"(warp+occlude+clutter+rot, deskew search)")
    # hard overlapping string via segmentation-by-search
    sok = cok = ct = 0
    for _ in range(120):
        k = int(RNG.integers(3, 6))
        s = "".join(CHARS[int(RNG.integers(len(CHARS)))] for _ in range(k))
        pred = solve_string_search(render_string_hard(s), hard)
        sok += (pred == s)
        cok += sum(a == b for a, b in zip(pred, s)); ct += k
    print(f"   HARD overlapping string    {100*sok/120:4.0f}% exact   "
          f"{100*cok/ct:4.0f}% per-char  (segmentation BY SEARCH)")
    print("\nperception is the strength: blur/noise barely dent it. rotation needs")
    print("SEARCH not invariance (a rotated 6 is a 9). hard mode shows the real")
    print("wall -- glued/warped strings, where SEGMENTATION (not recognition) is")
    print("the bottleneck, attacked by search. all synthetic + verifiable.")


if __name__ == "__main__":
    main()
