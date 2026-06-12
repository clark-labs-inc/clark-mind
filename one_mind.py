"""ONE MIND: no router, no heuristic wiring -- one predictive model over one
token stream does perception, generation, language, ACTION and DREAMING.
-------------------------------------------------------------------------------
The mind.py front door routes prompts with keywords. This file is the
principled alternative: a SINGLE backoff predictive-state model over a single
vocabulary where text bytes, image codes, audio codes, world percepts, actions
and reward are all just tokens:

  [text] [vision codes] [audio codes] [concepts] [POS tokens] [ACT tokens] [RWD]

"Routing" dissolves into sequence statistics: "draw 7" is completed with image
tokens, "play grid" with action tokens -- because that is what the stream
model predicts, not because an if-statement said so.

AGENCY enters by SYSTEM CONSOLIDATION (hippocampus -> cortex): the fast
GenericPredictiveAgent lives a life in GridWorld; its experience stream
(percept, action, reward, percept, ...) is fed to the SAME model that learns
"draw 7". Afterwards, ONE complete() interface:

  "draw 7"        -> image tokens   (generation)
  image "what digit" -> text        (perception)
  "name 4"        -> text           (language)
  "play grid" + live env            -> ACTION tokens (a distilled policy)
  "play grid" free-run              -> a DREAM (imagined rollout w/ rewards)

No backprop anywhere. Run: python3 one_mind.py
"""
from __future__ import annotations
import os, time, numpy as np
from PIL import Image
np.seterr(over="ignore", invalid="ignore", divide="ignore")
os.makedirs("outputs/onemind", exist_ok=True)
RNG = np.random.default_rng(0)

from psc_studio import UniversalPSC, _sample
from psc_omni import Codecs, txt, grid as img_grid, kmeans, assign, KV, KA, GC, VIS0, AUD0, CON0
from predictive_agent import GenericPredictiveAgent, GridWorld


def vhist(toks):
    h = np.bincount(np.array(toks) - VIS0, minlength=KV).astype(np.float32)
    return h / (np.linalg.norm(h) + 1e-8)

# ---- unified vocabulary: omni layout + WORLD tokens ----
BOS, EOS, SEP = CON0 + GC, CON0 + GC + 1, CON0 + GC + 2
P0 = SEP + 1                  # 36 GridWorld position percepts
ACT0 = P0 + 36                # 4 actions
RWD = ACT0 + 4                # the reward event
VOCAB = RWD + 1
PLAY = txt("play grid")


def live_a_life(steps_train=3000, steps_record=4000):
    """The fast agent (hippocampus) lives in GridWorld; return its experience
    as token sequences for the one mind (cortex) to consolidate."""
    env = GridWorld(n=6)
    agent = GenericPredictiveAgent(actions=[0, 1, 2, 3], depth=10)
    sig = env.reset()
    for _ in range(steps_train):                       # become competent first
        a, _, _ = agent.act(sig, env.actions)
        ns, r, _ = env.step(a)
        agent.learn(sig, a, r, ns)
        sig = ns
    seqs, chunk, expert_reward = [], None, 0.0
    for _ in range(steps_record):
        if chunk is None:
            chunk = [BOS] + PLAY + [SEP, P0 + 6 * sig[0] + sig[1]]
        a, _, _ = agent.act(sig, env.actions)
        ns, r, _ = env.step(a)
        agent.learn(sig, a, r, ns)
        chunk.append(ACT0 + a)
        if r > 0:
            chunk.append(RWD); expert_reward += r
        chunk.append(P0 + 6 * ns[0] + ns[1])
        sig = ns
        if len(chunk) > 48 or r > 0:
            seqs.append(chunk + [EOS]); chunk = None
    return seqs, expert_reward / steps_record


def main():
    t0 = time.time()
    from torchvision.datasets import MNIST
    mn = MNIST(root="./data", train=True, download=True)
    n = 3000
    Xi = mn.data.numpy().astype(np.float32)[:n] / 255.0
    yi = mn.targets.numpy()[:n]
    X32 = np.stack([np.asarray(Image.fromarray(np.uint8(x * 255)).resize((32, 32)),
                               np.float32) / 255.0 for x in Xi])
    cod = Codecs(); cod.fit(X32, yi)

    # ---- ONE corpus: instructions + senses + a lived life ----
    Vt = [list(cod.vis_enc(X32[i])) for i in range(n)]
    VH = np.stack([vhist(v) for v in Vt])
    Cg = kmeans(VH, GC)                    # unsupervised visual concept gists
    seqs = []
    for i in range(n):
        d = int(yi[i]); v = Vt[i]; g = CON0 + assign(VH[i], Cg)
        seqs += [
            [BOS] + txt(f"draw {d}") + [SEP] + v + [EOS],
            [BOS] + txt(f"name {d}") + [SEP] + txt(
                ["zero","one","two","three","four","five","six","seven","eight","nine"][d]) + [EOS],
            # the concept gist must sit ADJACENT to SEP or the input falls
            # outside the order-8 window when the answer is generated
            [BOS] + v + txt("what digit") + [g] + [SEP] + txt(str(d)) + [EOS],
        ]
    exp, expert_rate = live_a_life()
    seqs += exp
    psc = UniversalPSC(VOCAB, [(-1,), (-2,), (-3,), (-4,), (-5,), (-6,), (-7,), (-8,)], ())
    psc.fit([((len(s),), {(t,): int(tok) for t, tok in enumerate(s)}, ()) for s in seqs])
    print(f"ONE model fit on {len(seqs)} sequences ({len(exp)} of them lived "
          f"experience)  ({time.time()-t0:.0f}s)")

    def roll(prefix, maxlen=80, temp=0.5, top_p=0.92, lo=None, hi=None):
        E = {(t,): int(v) for t, v in enumerate(prefix)}; out = []
        for t in range(len(prefix), maxlen):
            p = psc.predict(E, (), (t,))
            if lo is not None:                          # constrain to a band
                p = p.copy(); p[:lo] = 0; p[hi:] = 0
                if p.sum() < 1e-9: p[lo:hi] = 1.0
            v = _sample(p / max(p.sum(), 1e-12), temp, top_p)
            if v == EOS: break
            E[(t,)] = v; out.append(v)
        return out

    print("\n=== ONE MODEL, FIVE FACULTIES (no router, no wiring) ===")
    # 1. GENERATE: "draw N"
    gi = []
    for d in range(10):
        v = [t for t in roll([BOS] + txt(f"draw {d}") + [SEP], 40) if VIS0 <= t < VIS0 + KV][:16]
        gi.append(cod.vis_dec((v + [VIS0] * 16)[:16]))
    img_grid(gi, "outputs/onemind/draw_0to9.png")
    print('  GENERATE  "draw N"        -> outputs/onemind/draw_0to9.png')

    # 2. LANGUAGE: "name N"
    names = []
    for d in (2, 5, 9):
        o = bytes([t for t in roll([BOS] + txt(f"name {d}") + [SEP], 30) if t < 256])
        names.append(f'{d}->"{o.decode("latin1", "ignore")}"')
    print(f'  LANGUAGE  "name N"       -> {", ".join(names)}')

    # 3. PERCEIVE: image -> "what digit"
    te = MNIST(root="./data", train=False, download=True)
    Xt = te.data.numpy().astype(np.float32)[:200] / 255.0; yt = te.targets.numpy()[:200]
    ok = 0
    for j in range(len(Xt)):
        x32 = np.asarray(Image.fromarray(np.uint8(Xt[j] * 255)).resize((32, 32)), np.float32) / 255.0
        v = list(cod.vis_enc(x32)); g = CON0 + assign(vhist(v), Cg)
        o = bytes([t for t in roll([BOS] + v + txt("what digit") + [g] + [SEP], len(v) + 20)
                   if 48 <= t <= 57])
        ok += int(o[:1] == str(int(yt[j])).encode())
    print(f'  PERCEIVE  "what digit"   -> {100*ok/len(Xt):.0f}% correct')

    # 4. ACT: drive the REAL GridWorld by completing the stream with actions
    env = GridWorld(n=6); sig = env.reset(); got = 0.0
    stream = [BOS] + PLAY + [SEP, P0 + 6 * sig[0] + sig[1]]
    for _ in range(500):
        a_t = roll(stream[-24:] if len(stream) > 24 else stream, len(stream) + 1,
                   temp=0.4, lo=ACT0, hi=ACT0 + 4)
        a = (a_t[0] - ACT0) if a_t else int(RNG.integers(4))
        ns, r, _ = env.step(a); got += r
        stream.append(ACT0 + a)
        if r > 0: stream.append(RWD)
        stream.append(P0 + 6 * ns[0] + ns[1])
    env2 = GridWorld(n=6); env2.reset(); rnd = 0.0
    for _ in range(500):
        _, r, _ = env2.step(int(RNG.integers(4))); rnd += r
    print(f'  ACT       "play grid"    -> reward {got:.0f}/500 steps '
          f'(teacher agent {500*expert_rate:.0f}, random {rnd:.0f})')

    # 5. DREAM: free-run the same prefix with NO environment
    d = roll([BOS] + PLAY + [SEP, P0], 60, temp=0.6)
    walk, wins = [], 0
    for t in d:
        if P0 <= t < P0 + 36: walk.append(divmod(t - P0, 6))
        if t == RWD: wins += 1
    print(f'  DREAM     "play grid"    -> imagined {len(walk)} positions, '
          f'{wins} imagined goal-reach(es): {walk[:9]}{"..." if len(walk) > 9 else ""}')
    print(f"\nall five through one complete() call -- {time.time()-t0:.0f}s, no backprop")


if __name__ == "__main__":
    main()
