"""
MicroCortex: brain-inspired MICRO-learning (no backprop).
---------------------------------------------------------
The generic agent's count tables are EPISODIC memory (hippocampus): exact,
one-shot, zero generalization -- a state hash either recurs or it doesn't.
This module is the CORTICAL half of a complementary-learning-systems pair:
a population of micro-learners over sparse LOCAL features, so value, credit
and novelty attach to re-usable local patterns instead of whole-state hashes.

Mechanisms (each a local rule, nothing global, no gradients through anything):
  - value:      r_hat(features) = sum of feature weights, learned by the
                delta rule (Widrow-Hoff) -- the cerebellar/CMAC micro-zone idea
  - credit:     three-factor plasticity: every active feature leaves a decaying
                ELIGIBILITY TRACE; when a reward scalar arrives (dopamine), all
                traced features are potentiated proportionally to their trace.
                Sparse delayed reward is assigned to the recent local patterns
                that caused it -- no backprop through time.
  - curiosity:  micro-novelty 1/(1+n_f): a never-seen PATTERN is novel; a
                re-randomized board full of known patterns is not.

The cortex is task/modality-agnostic: features are opaque hashable ids supplied
by an adapter (the adapter is the retina/cochlea -- receptive-field wiring is
allowed to be modality-specific; the learning rules are not).

Self-test below: an environment where episodic counts PROVABLY fail (every
observation carries fresh noise, so no state hash ever recurs) and the micro
cortex still learns to act. Run:  python micro_cortex.py
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np


class MicroCortex:
    """SARSA(lambda) over sparse micro-features: the textbook three-factor rule.
    delta (dopamine RPE) = r + gamma*Q(next pattern) - Q(pattern); every synapse
    updates by lr * delta * its own eligibility trace. Entirely local."""
    def __init__(self, lr=0.25, gamma=0.9, lam=0.9, beta=0.5):
        self.lr, self.gamma, self.lam, self.beta = lr, gamma, lam, beta
        self.v = defaultdict(float)      # feature -> value weight
        self.n = defaultdict(int)        # feature -> times active (for novelty)
        self.e = {}                      # feature -> eligibility trace
        self._pending = None             # (features_t, reward_t) awaiting bootstrap

    def _val(self, fs):
        return sum(self.v.get(f, 0.0) for f in fs)

    # ---- evaluate a candidate action's feature set ----
    def score(self, feats):
        fs = list(feats)
        if not fs:
            return 0.0
        nov = sum(1.0 / (1.0 + self.n.get(f, 0)) for f in fs) / len(fs)
        return self._val(fs) + self.beta * nov

    # ---- one experience step: features of the action just taken + its reward ----
    def learn(self, active, reward):
        fs = tuple(active)
        # finish the PREVIOUS step's TD update, bootstrapping on this step's Q
        if self._pending is not None:
            pf, pr = self._pending
            delta = pr + self.gamma * self._val(fs) - self._val(pf)
            g = self.lr * delta / max(len(pf), 1)
            for f, e in self.e.items():
                self.v[f] += g * e                   # three-factor: lr*RPE*trace
        # advance traces to include the current pattern
        for f in list(self.e):
            self.e[f] *= self.gamma * self.lam
            if self.e[f] < 1e-3:
                del self.e[f]
        for f in fs:
            self.e[f] = 1.0
            self.n[f] += 1
        self._pending = (fs, reward)

    # ---- persistence (joins the agent's brain file) ----
    def state(self):
        return {"v": dict(self.v), "n": dict(self.n)}

    def restore(self, d):
        self.v.update(d["v"]); self.n.update(d["n"])


# =============================================================================
# Self-test: episodic memory CANNOT solve this; micro-learning can.
# NoisyGridWorld: the observation is (y, x, noise) with fresh noise every step,
# so no signature ever recurs -- count tables stay at one visit per state and
# the hash agent is permanently blind. Micro features expose (y, x, action)
# as a local reusable pattern. Reward is DELAYED by 3 steps after reaching the
# goal, so credit must travel through eligibility traces.
# =============================================================================
class NoisyGridWorld:
    actions = [0, 1, 2, 3]
    def __init__(self, n=6, delay=3, seed=0):
        self.n, self.delay = n, delay
        self.rng = np.random.default_rng(seed)
        self.goal = (n - 1, n - 1); self.pos = (0, 0); self.pending = []
    def obs(self):
        return (*self.pos, int(self.rng.integers(1 << 30)))
    def step(self, a):
        y, x = self.pos
        y += (a == 1) - (a == 0); x += (a == 3) - (a == 2)
        self.pos = (min(max(y, 0), self.n - 1), min(max(x, 0), self.n - 1))
        r = 0.0
        self.pending = [t - 1 for t in self.pending]
        if self.pending and self.pending[0] <= 0:
            self.pending.pop(0); r = 1.0
        if self.pos == self.goal:
            self.pending.append(self.delay)              # reward arrives later
            self.pos = (0, 0)
        return self.obs(), r


def demo():
    from predictive_agent import GenericPredictiveAgent
    print("NoisyGridWorld: no observation ever repeats; reward delayed 3 steps.")
    for use_cortex in (False, True):
        env = NoisyGridWorld()
        cortex = MicroCortex(beta=0.5) if use_cortex else None
        agent = GenericPredictiveAgent(actions=env.actions, depth=6, cortex=cortex)
        sig = env.obs(); total = 0.0; hist = []
        for step in range(1, 4001):
            feats = {a: (("pos-act", sig[0], sig[1], a),) for a in env.actions}
            a, _, _ = agent.act(sig, env.actions, feats=feats)
            nsig, r = env.step(a)
            agent.learn(sig, a, r, nsig, feats=feats)
            sig = nsig; total += r
            if step % 1000 == 0:
                hist.append(total); total = 0.0
        name = "episodic+MICRO-CORTEX" if use_cortex else "episodic only       "
        print(f"  {name}: reward/1000-step block: {[f'{h:.0f}' for h in hist]}")
    print("(episodic stays near zero -- every state is new to it; "
          "the cortex learns reusable (pos,action) micro-patterns through traces)")


if __name__ == "__main__":
    demo()
