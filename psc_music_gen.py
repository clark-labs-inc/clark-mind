"""
Generative music with a Predictive State Column (no backprop)
-------------------------------------------------------------
Same predictive-state learning rule as the image generator and PSC-2 dynamics,
now on symbolic music. Pipeline:

    MAESTRO piano MIDI
      -> MIDI event codec (note_on / note_off / time_shift / velocity)  = codec
      -> event-id stream
      -> PSC: variable-order backoff count model predicts the next event id
              (predictive states; backoff = merge of equivalent-future contexts)
      -> top-p sampling with a loop-collapse guard
      -> events -> MIDI -> piano-roll PNG + WAV

No backprop / optimizer / transformer. Human-inspectable outputs:
    outputs/music/sample_*.mid / .wav / .pianoroll.png
    outputs/music/continuation_*.mid  (prompt-conditioned)
"""

from __future__ import annotations
import argparse
import os
from collections import defaultdict
from pathlib import Path
import numpy as np
import pretty_midi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.seterr(over="ignore", invalid="ignore", divide="ignore")
K_VOCAB = 512


class MidiEventCodec:
    def __init__(self, time_step=0.02, max_shift=100):
        self.time_step, self.max_shift = time_step, max_shift

    def midi_to_events(self, path):
        pm = pretty_midi.PrettyMIDI(str(path))
        notes = []
        for inst in pm.instruments:
            if inst.is_drum:
                continue
            for n in inst.notes:
                notes.append((n.start, "on", n.pitch, n.velocity))
                notes.append((n.end, "off", n.pitch, 0))
        notes.sort(key=lambda z: z[0])
        ev = [("bos", 0)]
        tprev = 0.0
        for t, typ, pitch, vel in notes:
            shift = int(round((t - tprev) / self.time_step))
            while shift > 0:
                s = min(shift, self.max_shift); ev.append(("time_shift", s)); shift -= s
            if typ == "on":
                ev.append(("velocity", min(vel // 8, 15))); ev.append(("note_on", pitch))
            else:
                ev.append(("note_off", pitch))
            tprev = t
        ev.append(("eos", 0))
        return [self.eid(e) for e in ev]

    def eid(self, e):
        typ, v = e
        return {"bos": 0, "eos": 1}.get(typ, None) if typ in ("bos", "eos") else \
            (2 + v if typ == "time_shift" else 128 + v if typ == "velocity"
             else 256 + v if typ == "note_on" else 384 + v)

    def id_to_event(self, i):
        if i == 0: return ("bos", 0)
        if i == 1: return ("eos", 0)
        if 2 <= i < 128: return ("time_shift", i - 2)
        if 128 <= i < 256: return ("velocity", i - 128)
        if 256 <= i < 384: return ("note_on", i - 256)
        if 384 <= i < 512: return ("note_off", i - 384)
        return ("eos", 0)

    def events_to_midi(self, ids, out_path, max_dur=2.0):
        pm = pretty_midi.PrettyMIDI(); inst = pretty_midi.Instrument(program=0)
        t, vel, active = 0.0, 80, {}

        def release(pitch, end):                       # cap duration -> no stuck notes
            st, vv = active.pop(pitch)
            inst.notes.append(pretty_midi.Note(vv, pitch, st, min(end, st + max_dur)))

        for i in ids:
            typ, v = self.id_to_event(int(i))
            if typ == "time_shift":
                t += v * self.time_step
                for pitch in [p for p, (st, _) in active.items() if t - st >= max_dur]:
                    release(pitch, t)                  # auto-release notes older than max_dur
            elif typ == "velocity": vel = max(1, min(127, v * 8))
            elif typ == "note_on":
                if v in active: release(v, t)
                active[v] = (t, vel)
            elif typ == "note_off" and v in active:
                release(v, t)
        for pitch in list(active):
            release(pitch, t)
        pm.instruments.append(inst); pm.write(str(out_path)); return pm


# -----------------------------------------------------------------------------
# PSC music model: variable-order backoff count model over event ids
# -----------------------------------------------------------------------------
class PSCMusic:
    def __init__(self, order=6, alpha=0.02, ab=4.0, n_phase=16, bar=2.0,
                 time_step=0.02, use_phase=True):
        self.Q, self.alpha, self.ab = order, alpha, ab
        self.n_phase, self.bar, self.ts, self.use_phase = n_phase, bar, time_step, use_phase
        self.t = [defaultdict(lambda: defaultdict(float)) for _ in range(order + 1)]
        self.tph1 = defaultdict(lambda: defaultdict(float))   # (phase, last_id) -> next
        self.tph0 = defaultdict(lambda: defaultdict(float))   # phase -> next

    def shift_secs(self, idv):
        return (idv - 2) * self.ts if 2 <= idv < 128 else 0.0   # time_shift events

    def phase(self, cum):
        return int((cum % self.bar) / self.bar * self.n_phase)

    def fit(self, seqs):
        for s in seqs:
            cum = 0.0
            for i in range(len(s)):
                ph, nxt = self.phase(cum), s[i]
                for q in range(self.Q + 1):
                    if i - q < 0: break
                    self.t[q][tuple(s[i - q:i])][nxt] += 1.0
                if self.use_phase:
                    self.tph0[ph][nxt] += 1.0
                    self.tph1[(ph, s[i - 1] if i else -1)][nxt] += 1.0
                cum += self.shift_secs(s[i])

    def _interp(self, p, d):
        a = np.full(K_VOCAB, self.alpha)
        for k, v in d.items(): a[k] += v
        c = sum(d.values()); lam = c / (c + self.ab)
        return lam * (a / a.sum()) + (1 - lam) * p

    def dist(self, ctx, ph=None):
        p = np.full(K_VOCAB, 1.0 / K_VOCAB)
        if self.use_phase and ph is not None:        # metric/rhythmic prior (coarse first)
            d0 = self.tph0.get(ph);  p = self._interp(p, d0) if d0 else p
            d1 = self.tph1.get((ph, ctx[-1] if ctx else -1)); p = self._interp(p, d1) if d1 else p
        for q in range(1, self.Q + 1):               # event-id n-gram (specific last)
            if len(ctx) < q: break
            d = self.t[q].get(tuple(ctx[-q:]))
            if d: p = self._interp(p, d)
        return p

    def n_states(self):
        return sum(len(d) for d in self.t) + len(self.tph0) + len(self.tph1)


def sample_topp(p, temp, top_p, rng):
    logp = np.log(p + 1e-12) / max(temp, 1e-3)
    q = np.exp(logp - logp.max()); q /= q.sum()
    o = np.argsort(-q); cut = o[:max(1, int(np.searchsorted(np.cumsum(q[o]), top_p)) + 1)]
    return int(rng.choice(cut, p=q[cut] / q[cut].sum()))


def generate(psc, n_events, rng, temp=0.95, top_p=0.95, prompt=None):
    seq = list(prompt) if prompt else [0]              # 0 = BOS
    cum = sum(psc.shift_secs(i) for i in seq)
    for _ in range(n_events):
        nxt = sample_topp(psc.dist(seq, psc.phase(cum)), temp, top_p, rng)
        if nxt == 1 and len(seq) > 64: break           # EOS
        seq.append(nxt); cum += psc.shift_secs(nxt)
        if len(seq) > 128 and seq[-32:] == seq[-64:-32]:  # loop-collapse guard
            jolt = sample_topp(psc.dist(seq[:-16], psc.phase(cum)), temp + 0.3, 0.99, rng)
            seq.append(jolt); cum += psc.shift_secs(jolt)
    return seq


def loop_collapse(seq, w=64):
    if len(seq) < 2 * w: return 0.0
    a, b = np.array(seq[-w:]), np.array(seq[-2 * w:-w])
    return float(np.mean(a == b))


def save_pianoroll(pm, png):
    roll = pm.get_piano_roll(fs=50)
    plt.figure(figsize=(13, 4.5))
    plt.imshow(roll, aspect="auto", origin="lower", interpolation="nearest", cmap="magma")
    plt.xlabel("time (frames)"); plt.ylabel("MIDI pitch"); plt.tight_layout()
    plt.savefig(png, dpi=130); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi_dir", default="data/maestro_midi")
    ap.add_argument("--n_train", type=int, default=250)
    ap.add_argument("--n_heldout", type=int, default=40)
    ap.add_argument("--max_events", type=int, default=4000)
    ap.add_argument("--order", type=int, default=6)
    ap.add_argument("--n_phase", type=int, default=16)
    ap.add_argument("--bar", type=float, default=2.0)
    ap.add_argument("--no_phase", action="store_true")
    ap.add_argument("--gen_events", type=int, default=1500)
    ap.add_argument("--n_samples", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs("outputs/music", exist_ok=True)
    rng = np.random.default_rng(args.seed)
    codec = MidiEventCodec()

    files = sorted(Path(args.midi_dir).rglob("*.mid*"))
    rng.shuffle(files)
    tr_files, he_files = files[:args.n_train], files[args.n_train:args.n_train + args.n_heldout]

    def load(fs):
        out = []
        for f in fs:
            try:
                e = codec.midi_to_events(f)[:args.max_events]
                if len(e) > 50: out.append(e)
            except Exception:
                pass
        return out
    tr = load(tr_files); he = load(he_files)
    print(f"MAESTRO: train_seqs={len(tr)} heldout_seqs={len(he)} "
          f"avg_events={int(np.mean([len(s) for s in tr]))}")

    psc = PSCMusic(order=args.order, n_phase=args.n_phase, bar=args.bar,
                   use_phase=not args.no_phase)
    psc.fit(tr)

    bits, correct, ntok = 0.0, 0, 0
    for s in he:
        cum = 0.0
        for i in range(1, min(len(s), 1500)):
            cum += psc.shift_secs(s[i - 1])
            p = psc.dist(s[:i], psc.phase(cum)); t = s[i]
            bits += -np.log2(max(p[t], 1e-12)); correct += int(p.argmax() == t); ntok += 1
    print(f"PSC music [phase={not args.no_phase}]: states={psc.n_states()} "
          f"heldout bits/event={bits/ntok:.3f} next_event_acc={correct/ntok:.3f}")

    # unconditional samples
    for i in range(args.n_samples):
        seq = generate(psc, args.gen_events, rng, temp=args.temp)
        pm = codec.events_to_midi(seq, f"outputs/music/sample_{i:02d}.mid")
        save_pianoroll(pm, f"outputs/music/sample_{i:02d}.pianoroll.png")
        try:
            import soundfile as sf
            sf.write(f"outputs/music/sample_{i:02d}.wav", pm.synthesize(fs=16000), 16000)
        except Exception as e:
            print("  wav skip:", repr(e)[:80])
        print(f"  sample {i}: {len(seq)} events, {len(pm.instruments[0].notes)} notes, "
              f"loop_collapse={loop_collapse(seq):.2f}")

    # prompt continuation from a held-out piece
    if he:
        prompt = he[0][:200]
        seq = generate(psc, args.gen_events, rng, temp=args.temp, prompt=prompt)
        pm = codec.events_to_midi(seq, "outputs/music/continuation_00.mid")
        save_pianoroll(pm, "outputs/music/continuation_00.pianoroll.png")
        print("  wrote prompt continuation")
    print("wrote outputs/music/*.mid / .wav / .pianoroll.png")


if __name__ == "__main__":
    main()
