"""Heuristic-free generic agent: posterior-sampling RL over a hierarchical
Bayesian world model (no backprop, and no hand-tuned exploration constants).
--------------------------------------------------------------------------------
ONE objective -- maximize expected discounted environment score under the
posterior over worlds -- and every behavior is derived from it:

  WORLD MODEL   per (state, action): Dirichlet posterior over next states with
                an escape mass to the UNKNOWN (Chinese-restaurant style), and a
                Beta posterior over P(score increase), whose PRIOR is the
                posterior of the same action under the next-coarser state
                abstraction (hierarchical Beta-Binomial chain over the
                SigStack levels; the root is the global action posterior).
                Backoff is not a rule here -- it is what a hierarchical prior
                does when local evidence is scarce.

  EXPLORATION   posterior SAMPLING (PSRL): sample one world from the
                posterior, value-iterate it, act greedily. Uncertain pairs
                sometimes sample as good -- that IS directed exploration.
                Before any score evidence exists the sampled reward landscape
                is flat ~0, and the only term left in expected utility is
                INFORMATION: a second value iteration on the same sampled
                world with information gain (1/(n+1), the expected entropy
                reduction of a Dirichlet observation) as reward yields the
                optimal info-seeking policy. No epsilon, no bonus weights:
                reward value decides when it speaks; information decides when
                reward is silent (their Q-gap below sampling noise).

  RESAMPLING    UCRL2's parameter-free episode rule: resample the world when
                a KNOWN pair's within-episode visits double its history
                (2*nu >= n, snapshot-free). Novel states do NOT trigger a
                replan -- a never-seen state needs one posterior draw per
                action against the current values, not a new world; that
                single change is what makes PSRL fast enough to deploy.
                A bounded-staleness heartbeat (256 steps) caps how old a
                sampled world can get. The fine level is solved as flat
                numpy arrays with warm-started value iteration.

  PERCEPTION    Habituator (clock masking) decided by BAYES FACTOR between
                "changes depend on cause" and "changes independent of cause",
                masked at Jeffreys' decisive-evidence level -- no count or
                share thresholds.

The single fixed quantity is gamma (temporal preference = problem definition)
and the unit Dirichlet/Beta concentration alpha=1 (uninformative).
"""
from __future__ import annotations
from collections import Counter, defaultdict, deque
import math
import pickle
import numpy as np

from predictive_agent import SigStack

ALPHA = 1.0                                  # unit (uninformative) concentration
UNKNOWN = ("?",)                             # the escape pseudo-state


class _Lvl:
    """Counts at one resolution of the state abstraction."""
    def __init__(self):
        self.trans = {}                      # (s,a) -> Counter(next)
        self.n = {}                          # (s,a) -> visits
        self.pos = {}                        # (s,a) -> score-increase events

    def learn(self, s, a, scored, ns):
        k = (s, a)
        self.n[k] = self.n.get(k, 0) + 1
        if scored:
            self.pos[k] = self.pos.get(k, 0) + 1
        if ns is not None:
            self.trans.setdefault(k, Counter())[ns] += 1


class BayesAgent:
    def __init__(self, actions, gamma=0.95, seed=0, names=None,
                 max_pairs=100000, hab=None):
        self.actions = list(actions)
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.names = names or {}
        self.levels = [_Lvl()]               # fine first; grown by SigStacks
        self.sighier = {}                    # fine sig -> full stack
        self.avail = {}                      # sig -> offered actions
        self.states = set()
        self.mag = 0.0; self.magn = 0        # observed positive score magnitudes
        self.gpos = defaultdict(int)         # action -> score events (root prior)
        self.gn = defaultdict(int)
        self.last = {}                       # recency for the resource bound
        self.last_state = {}                 # sig -> step last visited
        self.plan_cap = 1500                 # states value-iterated per sample
        self._mc = {}                        # per-pass _mean_chain memo
        self._cV = {}                        # warm-start coarse values per level
        self.min_episode = 20                # min steps between world samples
        self.max_pairs = max_pairs
        self.t = 0
        # episode state (PSRL)
        self.policy = {}                     # state -> action (greedy in sample)
        self.ep_nu = Counter()               # within-episode visit counts
        self.plan_t = -(10 ** 9)             # step of the last world sample
        self._parents = [{}]                 # per-level child -> parent links
        self.Q1r, self.Q1i = {}, {}          # level-1 sampled Q (escape targets)
        self.vu_r = self.vu_i = 0.0          # UNKNOWN closure values
        self._V_store = {"r": {}, "i": {}}   # warm-start values across plans
        self.hab = hab                       # optional Habituator (adapter-fed)

    # ----------------------------------------------------------- learning
    def _intern(self, s):
        if isinstance(s, SigStack):
            while len(self.levels) < len(s):
                self.levels.append(_Lvl())
            self.sighier[s[0]] = tuple(s)
            return s[0]
        return s

    def learn(self, sig, a, reward, next_sig, available=None):
        sig = self._intern(sig)
        next_sig = self._intern(next_sig) if next_sig is not None else None
        scored = reward > 0
        if scored:
            self.mag += reward; self.magn += 1
        h = self.sighier.get(sig, (sig,))
        nh = self.sighier.get(next_sig, (next_sig,)) if next_sig is not None else None
        for i, lv in enumerate(self.levels):
            if i < len(h):
                lv.learn(h[i], a, scored,
                         nh[i] if nh is not None and i < len(nh) else None)
        self.gpos[a] += int(scored); self.gn[a] += 1
        if available is not None:
            self.avail[sig] = tuple(available)
        self.states.add(sig)
        if next_sig is not None:
            self.states.add(next_sig)
        self.t += 1
        self.last[(sig, a)] = self.t
        self.last_state[sig] = self.t        # state recency for bounded planning
        # UCRL2 episode rule, snapshot-free: with nu = visits since the last
        # world sample, pre-sample history is n - nu and the episode ends
        # when nu doubles it. A pair's FIRST visit never ends the episode
        # (sweeps through novelty stay replan-free; act() handles new states
        # with a one-step draw) -- but its SECOND within-episode visit can:
        # revisiting under one frozen sample means the policy is looping.
        # ...subject to a minimum episode length: realizing a sampled world
        # costs ~1s at scale, and the doubling rule thrashes when counts are
        # small (resampling every few steps). UCRL permits ending an episode
        # LATER than the doubling point (it stays a valid optimistic schedule);
        # a min length bounds worst-case replans/session without changing the
        # asymptotic regret regime. min_episode=0 recovers textbook UCRL2.
        k = (sig, a)
        self.ep_nu[k] += 1
        nu, n = self.ep_nu[k], self.levels[0].n.get(k, 0)
        if nu >= 2 and 2 * nu >= n and self.t - self.plan_t >= self.min_episode:
            self.policy = {}
        if self.max_pairs and len(self.levels[0].n) > self.max_pairs:
            self.sleep()

    def _acts(self, s):
        return self.avail.get(s) or self.actions

    # ------------------------------------------------------- PSRL sampling
    def _sample_p(self, level_i, s, a):
        """One Beta draw of P(score | (s,a) at level i); the prior mean is
        the next-coarser key's posterior mean (hierarchical Beta-Binomial),
        bottoming out at the global per-action posterior."""
        lv = self.levels[level_i]
        k = (s, a)
        n, p = lv.n.get(k, 0), lv.pos.get(k, 0)
        up = self._parents[level_i].get(s) if level_i < len(self._parents) else None
        if up is not None and level_i + 1 < len(self.levels):
            m = self._mean_chain(level_i + 1, up, a)
        else:
            m = (self.gpos[a] + ALPHA) / (self.gn[a] + 2 * ALPHA)
        a1 = p + ALPHA * m
        b1 = (n - p) + ALPHA * (1 - m)
        return float(self.rng.beta(max(a1, 1e-6), max(b1, 1e-6)))

    def _mean_chain(self, level_i, s, a):
        """Posterior mean of P(score) at (s,a,level i), prior chained up.
        Memoized within a planning pass: counts don't change during planning,
        and the coarse levels query the same (level,sig,a) thousands of times
        (886k -> a few k calls; the dominant cost on big-state games)."""
        ck = (level_i, s, a)
        c = self._mc.get(ck)
        if c is not None:
            return c
        m = (self.gpos[a] + ALPHA) / (self.gn[a] + 2 * ALPHA)
        keys = []
        cur = s
        for j in range(level_i, len(self.levels)):
            keys.append((j, cur))
            cur = self._parents[j].get(cur) if j < len(self._parents) else None
            if cur is None:
                break
        for j, ks in reversed(keys):
            lv = self.levels[j]
            n, p = lv.n.get((ks, a), 0), lv.pos.get((ks, a), 0)
            m = (p + ALPHA * m) / (n + ALPHA)
        self._mc[ck] = m
        return m

    def _sample_world(self):
        """Hierarchical PSRL, engineered for deployment: coarse levels are
        tiny and solved in dicts; the FINE level is solved as flat numpy
        arrays -- one vectorized Gamma call realizes the whole Dirichlet
        world, and value iteration is warm-started from the previous plan's
        values. Each level's escape mass resolves to the coarser level's
        sampled Q (knowledge transfers down the hierarchy as the prior);
        the root escapes to the UNKNOWN closure."""
        gamma = self.gamma
        L = len(self.levels)
        self._mc = {}                        # fresh memo: counts frozen this pass
        # PSRL plans the MDP it can actually traverse this episode: value-
        # iterate only the RECENTLY-VISITED component (warm-start retains the
        # rest, and the coarse levels carry long-range value as escape Q).
        # This bounds per-sample cost by plan_cap regardless of total brain
        # size -- the difference between 35-min and seconds on big-state games.
        fine_states = [s for s in self.states if s in self.avail]
        if len(fine_states) > self.plan_cap:
            fine_states.sort(key=lambda s: self.last_state.get(s, 0), reverse=True)
            fine_states = fine_states[:self.plan_cap]
        if not fine_states:
            return
        lvl_states = [set() for _ in range(L)]
        parents = [{} for _ in range(L)]
        lvl_acts = [defaultdict(set) for _ in range(L)]
        for s in fine_states:
            h = self.sighier.get(s, (s,))
            acts = self.avail[s]
            for i, hi in enumerate(h):
                if i:
                    lvl_states[i].add(hi)
                    lvl_acts[i][hi].update(acts)
                if i + 1 < len(h):
                    parents[i][hi] = h[i + 1]
        self._parents = parents
        mag = (self.mag / self.magn) if self.magn else 0.0
        self.vu_r = float(self.rng.beta(ALPHA, ALPHA + self.t)) * mag / (1 - gamma)
        self.vu_i = 1.0 / (1 - gamma)
        vu_r, vu_i = self.vu_r, self.vu_i
        # ---- coarse levels: small, dict value iteration as before ----
        Qr_up, Qi_up = {}, {}
        for i in range(L - 1, 0, -1):
            lv = self.levels[i]
            states = lvl_states[i]
            Tr, Rw, Ig, Esc = {}, {}, {}, {}
            for s in states:
                up = parents[i].get(s)
                for a in lvl_acts[i][s]:
                    k = (s, a)
                    tc = lv.trans.get(k)
                    outs, w = [None], [self.rng.gamma(ALPHA)]
                    if tc:
                        for ns, c in tc.items():
                            if ns in states:
                                outs.append(ns); w.append(self.rng.gamma(c))
                    w = np.asarray(w); w /= w.sum()
                    Tr[k] = (outs, w)
                    Rw[k] = self._sample_p(i, s, a) * mag
                    Ig[k] = 1.0 / (lv.n.get(k, 0) + 1.0)
                    Esc[k] = ((Qr_up.get((up, a), vu_r), Qi_up.get((up, a), vu_i))
                              if up is not None else (vu_r, vu_i))
            # warm-start coarse value iteration from the previous pass: the
            # sampled world barely moves between samples, so a few sweeps
            # suffice instead of 40-from-zero (coarse VI is the dominant cost
            # on big-state games once the fine level is vectorized).
            cV = self._cV.setdefault(i, ({}, {}))
            Vr = {s: cV[0].get(s, 0.0) for s in states}
            Vi = {s: cV[1].get(s, 0.0) for s in states}
            for which, (V, R) in enumerate(((Vr, Rw), (Vi, Ig))):
                for _ in range(40):
                    delta = 0.0
                    for s in states:
                        best = 0.0
                        for a in lvl_acts[i][s]:
                            k = (s, a)
                            outs, w = Tr[k]
                            ev = Esc[k][which]
                            q = R[k] + gamma * sum(
                                wi * (ev if o is None else V.get(o, ev))
                                for o, wi in zip(outs, w))
                            if q > best:
                                best = q
                        delta = max(delta, abs(best - V.get(s, 0.0)))
                        V[s] = best
                    if delta < 1e-4:
                        break
            self._cV[i] = (Vr, Vi)           # carry to the next pass
            Qr, Qi = {}, {}
            for s in states:
                for a in lvl_acts[i][s]:
                    k = (s, a)
                    outs, w = Tr[k]
                    er, ei = Esc[k]
                    Qr[k] = Rw[k] + gamma * sum(
                        wi * (er if o is None else Vr.get(o, er))
                        for o, wi in zip(outs, w))
                    Qi[k] = Ig[k] + gamma * sum(
                        wi * (ei if o is None else Vi.get(o, ei))
                        for o, wi in zip(outs, w))
            Qr_up, Qi_up = Qr, Qi
        self.Q1r, self.Q1i = Qr_up, Qi_up    # escape targets for the fine level
        # ---- fine level: flat arrays ----
        lv = self.levels[0]
        sidx = {s: j for j, s in enumerate(fine_states)}
        p_state, p_act = [], []
        a1, b1, nv, e_r, e_i = [], [], [], [], []
        out_off, out_idx, out_cnt = [0], [], []
        state_pairs = [[] for _ in fine_states]
        mc = {}
        for s in fine_states:
            h = self.sighier.get(s, (s,))
            up = h[1] if len(h) > 1 else None
            si = sidx[s]
            for a in self.avail[s]:
                k = (s, a)
                n = lv.n.get(k, 0); pos = lv.pos.get(k, 0)
                key = (up, a)
                if key not in mc:
                    mc[key] = (self._mean_chain(1, up, a)
                               if up is not None and L > 1 else
                               (self.gpos[a] + ALPHA) / (self.gn[a] + 2 * ALPHA))
                m = mc[key]
                state_pairs[si].append(len(p_state))
                p_state.append(si); p_act.append(a)
                a1.append(pos + ALPHA * m); b1.append((n - pos) + ALPHA * (1 - m))
                nv.append(n)
                e_r.append(Qr_up.get(key, vu_r) if up is not None else vu_r)
                e_i.append(Qi_up.get(key, vu_i) if up is not None else vu_i)
                tc = lv.trans.get(k)
                if tc:
                    for ns, c in tc.items():
                        j = sidx.get(ns)
                        if j is not None:
                            out_idx.append(j); out_cnt.append(c)
                out_off.append(len(out_idx))
        ps = np.asarray(p_state)
        a1 = np.maximum(np.asarray(a1), 1e-6); b1 = np.maximum(np.asarray(b1), 1e-6)
        Rw = self.rng.beta(a1, b1) * mag
        noise = np.abs(self.rng.beta(a1, b1) * mag - Rw)   # the posterior's own spread
        Ig = 1.0 / (np.asarray(nv, dtype=np.float64) + 1.0)
        e_r = np.asarray(e_r); e_i = np.asarray(e_i)
        off = np.asarray(out_off)
        out_idx = np.asarray(out_idx, dtype=np.int64)
        w = self.rng.gamma(np.asarray(out_cnt, dtype=np.float64)) \
            if out_cnt else np.zeros(0)
        we = self.rng.gamma(ALPHA, size=len(ps))
        cs = np.concatenate(([0.0], np.cumsum(w)))
        norm = (cs[off[1:]] - cs[off[:-1]]) + we
        Vfin = {}
        for tag, R, ESC in (("r", Rw, e_r), ("i", Ig, e_i)):
            store = self._V_store[tag]
            V = np.asarray([store.get(s, 0.0) for s in fine_states])
            for _ in range(60):
                val = w * V[out_idx] if len(out_idx) else w
                cv = np.concatenate(([0.0], np.cumsum(val)))
                rowv = cv[off[1:]] - cv[off[:-1]]
                q = R + gamma * (we * ESC + rowv) / norm
                newV = np.zeros(len(fine_states))
                np.maximum.at(newV, ps, q)
                if np.max(np.abs(newV - V)) < 1e-3:
                    V = newV; break
                V = newV
            self._V_store[tag] = {s: float(V[j]) for j, s in enumerate(fine_states)}
            Vfin[tag] = (V, q)
        qr, qi = Vfin["r"][1], Vfin["i"][1]
        pol = {}
        for s in fine_states:
            ids = state_pairs[sidx[s]]
            qrs = qr[ids]
            top = qrs.max(); nz = noise[ids].max()
            cand = [ids[j] for j in range(len(ids)) if top - qrs[j] <= nz]
            ti = max(qi[c] for c in cand)
            cand = [c for c in cand if qi[c] >= ti - 1e-12]
            pol[s] = p_act[cand[int(self.rng.integers(len(cand)))]]
        self.policy = pol
        self.plan_t = self.t
        self.ep_nu = Counter()

    def _one_step(self, sig, acts):
        """A novel or off-policy state does not need a fresh WORLD: one
        posterior draw per action against the current plan's values is the
        same decision rule at depth one (escape resolves to the coarser
        level's sampled Q, exactly as in the full plan)."""
        self._mc = {}
        h = self.sighier.get(sig, (sig,))
        up = h[1] if len(h) > 1 else None
        while len(self._parents) < max(1, len(h)):
            self._parents.append({})
        for i in range(len(h) - 1):
            self._parents[i][h[i]] = h[i + 1]
        mag = (self.mag / self.magn) if self.magn else 0.0
        lv = self.levels[0]
        qr, qi, noise = {}, {}, 0.0
        for a in acts:
            k = (h[0], a)
            n = lv.n.get(k, 0); pos = lv.pos.get(k, 0)
            m = (self._mean_chain(1, up, a)
                 if up is not None and len(self.levels) > 1 else
                 (self.gpos[a] + ALPHA) / (self.gn[a] + 2 * ALPHA))
            aa = max(pos + ALPHA * m, 1e-6); bb = max((n - pos) + ALPHA * (1 - m), 1e-6)
            d1 = float(self.rng.beta(aa, bb)); d2 = float(self.rng.beta(aa, bb))
            qr[a] = d1 * mag + self.gamma * self.Q1r.get((up, a), self.vu_r)
            qi[a] = 1.0 / (n + 1.0) + self.gamma * self.Q1i.get((up, a), self.vu_i)
            noise = max(noise, abs(d2 - d1) * mag)
        top = max(qr.values())
        cand = [a for a in acts if top - qr[a] <= noise]
        ti = max(qi[a] for a in cand)
        cand = [a for a in cand if qi[a] >= ti - 1e-12]
        return cand[int(self.rng.integers(len(cand)))]

    def act(self, sig, available, feats=None):
        sig = self._intern(sig)
        if available is not None:
            self.avail[sig] = tuple(available)
        if not self.policy or self.t - self.plan_t >= 256:
            self._sample_world()             # bounded staleness (lazy PSRL)
        a = self.policy.get(sig)
        if a is None or (available is not None and a not in available):
            a = self._one_step(sig, list(available or self.actions))
        return a, {}, []

    # --------------------------------------------------- bounded memory
    def sleep(self):
        """Resource bound (not a decision rule): evict the stalest pairs that
        carry neither score evidence nor recent use; coarser levels keep the
        gist."""
        fine = self.levels[0]
        keep = int(self.max_pairs * 0.7)
        cand = [k for k in fine.n if not fine.pos.get(k)]
        cand.sort(key=lambda k: self.last.get(k, 0))
        for k in cand[:max(0, len(fine.n) - keep)]:
            fine.n.pop(k, None); fine.pos.pop(k, None)
            fine.trans.pop(k, None); self.last.pop(k, None)
        alive = {s for s, _ in fine.n}
        self.states &= alive
        self.policy = {}

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"levels": [lv.__dict__ for lv in self.levels],
                         "sighier": self.sighier, "avail": self.avail,
                         "states": self.states, "mag": self.mag,
                         "magn": self.magn, "gpos": dict(self.gpos),
                         "gn": dict(self.gn), "t": self.t,
                         "last": self.last}, f)

    def load(self, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        while len(self.levels) < len(d["levels"]):
            self.levels.append(_Lvl())
        for lv, ld in zip(self.levels, d["levels"]):
            lv.__dict__.update(ld)
        self.sighier.update(d["sighier"]); self.avail.update(d["avail"])
        self.states |= d["states"]; self.mag = d["mag"]; self.magn = d["magn"]
        self.gpos.update(d["gpos"]); self.gn.update(d["gn"])
        self.t = d["t"]; self.last.update(d["last"])
        for (s, _a), tt in self.last.items():    # rebuild state recency
            if tt > self.last_state.get(s, 0):
                self.last_state[s] = tt


class BayesHabituator:
    """Clock masking by Bayes factor, no thresholds: for each element compare
    the marginal likelihood of its change-causes under DEPENDENT (own
    Dirichlet-multinomial) vs INDEPENDENT (global cause distribution) models;
    mask at Jeffreys' decisive evidence for independence (log10 BF < -2)."""
    def __init__(self):
        self.cnt = {}                        # element -> Counter(cause)
        self.glob = Counter()                # global cause usage
        self.mask = set()
        self.version = 0
        self.frozen = False

    def observe(self, changed, cause):
        self.glob[cause] += 1
        if self.frozen:
            return
        gtot = sum(self.glob.values())
        for k, _m in changed:
            if k in self.mask:
                continue
            c = self.cnt.setdefault(k, Counter())
            c[cause] += 1
            n = sum(c.values())
            if n < 2:
                continue
            # log marginal likelihood, dependent: Dirichlet-multinomial(alpha=1)
            K = max(len(self.glob), 2)
            dep = (math.lgamma(K) - math.lgamma(n + K)
                   + sum(math.lgamma(ci + 1) for ci in c.values()))
            # independent: causes drawn from the global empirical distribution
            ind = sum(ci * math.log(self.glob[cc] / gtot) for cc, ci in c.items())
            if (dep - ind) / math.log(10) < -2:          # decisive for H0
                self.mask.add(k)
                self.version += 1

    def state(self):
        return {"cnt": {k: dict(v) for k, v in self.cnt.items()},
                "glob": dict(self.glob), "mask": tuple(self.mask),
                "version": self.version}

    def restore(self, d):
        self.cnt = {k: Counter(v) for k, v in d["cnt"].items()}
        self.glob = Counter(d["glob"]); self.mask = set(d["mask"])
        self.version = d["version"]; self.frozen = True


# ================================ gates =====================================
def gates():
    from predictive_agent import GridWorld, SkinnedGridWorld, ClockedGridWorld
    print("BAYES AGENT (PSRL, heuristic-free) on the standard gates:")

    env = GridWorld(n=6)
    ag = BayesAgent(actions=[0, 1, 2, 3])
    sig = env.reset(); tot = 0.0; hist = []
    for step in range(1, 3001):
        a, _, _ = ag.act(sig, env.actions)
        ns, r, _ = env.step(a)
        ag.learn(sig, a, r, ns, env.actions)
        sig = ns; tot += r
        if step % 500 == 0:
            hist.append(tot); tot = 0.0
    print(f"  GridWorld     reward/500: {[f'{h:.0f}' for h in hist]} "
          f"(old agent: 23,41,47,47,50,50)")

    for stacked in (False, True):
        env = SkinnedGridWorld()
        ag = BayesAgent(actions=env.actions)
        wrap = (lambda o: SigStack((o, o[:2]))) if stacked else (lambda o: o)
        sig = wrap(env.obs()); tot = 0.0; hist = []
        for step in range(1, 5001):
            a, _, _ = ag.act(sig, env.actions)
            nobs, r = env.step(a)
            ag.learn(sig, a, r, wrap(nobs), env.actions)
            sig = wrap(nobs); tot += r
            if step % 1000 == 0:
                hist.append(tot); tot = 0.0
        print(f"  Skinned {'multi' if stacked else 'exact'}  reward/1000: "
              f"{[f'{h:.0f}' for h in hist]}")

    for habit in (False, True):
        env = ClockedGridWorld()
        ag = BayesAgent(actions=env.actions)
        hab = BayesHabituator()
        prev = env.obs()
        enc = lambda o: tuple("·" if i in hab.mask else v for i, v in enumerate(o))
        sig = enc(prev); tot = 0.0; hist = []
        for step in range(1, 5001):
            a, _, _ = ag.act(sig, env.actions)
            nobs, r = env.step(a)
            if habit:
                hab.observe({(i, (prev[i], nobs[i]))
                             for i in range(3) if nobs[i] != prev[i]}, a)
            ag.learn(sig, a, r, enc(nobs), env.actions)
            sig, prev = enc(nobs), nobs; tot += r
            if step % 1000 == 0:
                hist.append(tot); tot = 0.0
        print(f"  Clocked {'hab  ' if habit else 'exact'}  reward/1000: "
              f"{[f'{h:.0f}' for h in hist]} (masked: {sorted(hab.mask)})")


if __name__ == "__main__":
    gates()
