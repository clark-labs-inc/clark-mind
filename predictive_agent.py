"""
Generic predictive-state world-model + planning agent (no backprop).
--------------------------------------------------------------------
Task/modality-agnostic. The agent only ever sees:
    observation  -> (an adapter encodes it to a hashable STATE SIGNATURE)
    a set of available discrete actions (any hashable ids, incl. tuples)
    a scalar reward
It learns an action-conditioned predictive-state WORLD MODEL from interaction
(counts: (state,action) -> next-state distribution, reward, visits, effect) and
PLANS by bounded-depth value lookahead over that learned model -- imagining
rollouts and scoring them (+ a curiosity bonus for the unknown).

The INNER MONOLOGUE is now real thinking, not just value printout:
  [situation]  am I somewhere new? how big is the unexplored frontier?
               how long since I discovered anything?
  per-action   value + what the model predicts the action DOES here
  [plan]       an imagined greedy rollout through its own world model
  [flag]       stuck (long no-effect streak) / local exhaustion -> the agent
               searches its world-model graph (BFS) for the nearest state
               with an untried action and heads there.

Nothing here knows about ARC, grids, pixels, words, or sound. Any environment
that yields (obs, actions, reward) + an encode(obs)->signature plugs in. Same
agent below solves a GridWorld and (in clark_arc_agent.py) plays ARC-AGI-3.
"""
from __future__ import annotations
from collections import defaultdict, Counter, deque
import pickle
import numpy as np


class SigStack(tuple):
    """A MULTI-RESOLUTION state signature: (fine, coarser, coarser, ...).
    The adapter supplies candidate abstraction levels (it is the retina and may
    know e.g. 'pixels' vs 'object layout' vs 'object inventory'); the agent
    learns at EVERY level and backs off fine->coarse exactly like an n-gram
    model: use the most specific level that has data. Plain hashables remain
    valid signatures (zero levels, behavior unchanged)."""


class _Level:
    """One coarser resolution of the world: the same count machinery as the
    main model, keyed by that level's signature. Pattern completion for the
    hippocampus: when the exact state is novel, these stats say what acting
    here USUALLY does/earns -- and V (its own replayed value) transfers
    long-range goal knowledge to never-seen exact states."""
    def __init__(self, gamma):
        self.gamma = gamma
        self.vis = {}; self.rew = {}; self.eff = {}
        self.trans = {}; self.acts = {}; self.V = {}
        self.svis = {}                       # cs -> total visits (territory familiarity)

    def learn(self, cs, a, r, ncs):
        k = (cs, a)
        self.vis[k] = self.vis.get(k, 0) + 1
        self.svis[cs] = self.svis.get(cs, 0) + 1
        self.rew[k] = self.rew.get(k, 0.0) + r
        self.acts.setdefault(cs, set()).add(a)
        if ncs is not None:
            tc = self.trans.setdefault(k, Counter())
            tc[ncs] += 1
            if ncs != cs:
                self.eff[k] = self.eff.get(k, 0) + 1

    def q(self, cs, a):
        """Mean reward + discounted replayed value of where this usually leads."""
        v = self.vis.get((cs, a), 0)
        if not v:
            return None
        q = self.rew.get((cs, a), 0.0) / v
        tc = self.trans.get((cs, a))
        if tc:
            tot = sum(tc.values())
            q += self.gamma * sum((c / tot) * self.V.get(n, 0.0) for n, c in tc.items())
        return q

    def replay(self, sweeps=30):
        if not any(r > 0 for r in self.rew.values()):
            self.V = {}
            return
        for _ in range(sweeps):
            delta = 0.0
            for s, acts in self.acts.items():
                best = 0.0                       # don't propagate negative tails
                for a in acts:
                    q = self.q(s, a)
                    if q is not None and q > best:
                        best = q
                delta = max(delta, abs(best - self.V.get(s, 0.0)))
                self.V[s] = best
            if delta < 1e-4:
                break


class Habituator:
    """Perceptual habituation by ACTION-INDEPENDENCE -- fully generic.
    Observations are made of ELEMENTS (any hashable ids: grid cells, tuple
    components, sensor channels). An element whose changes are NOT explained
    by any particular action (no single cause accounts for more than
    `top_share` of its changes) is a CLOCK -- a counter, HUD, timer or
    animation: its value encodes time, not consequence. Clocks are masked out
    of the state signature so that states RECUR; without this, one ticking
    element makes every state forever novel and no world-model graph can form.
    The adapter decides what an 'element' and a 'cause' are (retina wiring);
    the rule itself knows nothing about pixels or grids."""
    def __init__(self, min_changes=6, top_share=0.45):
        self.min_changes, self.top_share = min_changes, top_share
        self.cnt = {}                        # element -> Counter(cause)
        self.modes = {}                      # element -> {mode: Counter(cause)}
        self.mask = set()                    # elements declared clocks (permanent)
        self.version = 0                     # bumps when the mask grows
        # CRITICAL PERIOD: the mask may only grow while the perceptual system
        # is young (first session). Restored habituators are FROZEN: every
        # mask change re-keys every stored state signature, so a mask that
        # keeps drifting across sessions orphans the whole consolidated brain
        # (v8: games with big growing masks lost all pass-2 re-completions;
        # games with small converged masks consolidated perfectly).
        self.frozen = False

    def observe(self, changed, cause):
        """Record that elements changed under `cause`. `changed` is an
        iterable of (element, mode) -- mode = HOW it changed (e.g. old->new
        value), opaque to this class. Masking needs POSITIVE evidence of
        independence on one of two paths:
          element path: many changes, no dominant cause, AND no action-locked
            mode (a player cell's bg->player mode is locked to the arriving
            action, so players never mask even when crossed from many sides);
          mode path: any single mode with many multi-cause changes (a counter
            tick or blinking animation masks even though its reset-restore
            mode is action-locked)."""
        if self.frozen:
            return
        for k, m in changed:
            if k in self.mask:
                continue
            c = self.cnt.setdefault(k, Counter())
            if len(c) < 32 or cause in c:
                c[cause] += 1
            km = self.modes.setdefault(k, {})
            if len(km) < 16 or m in km:
                mc = km.setdefault(m, Counter())
                if len(mc) < 32 or cause in mc:
                    mc[cause] += 1
            n = sum(c.values())
            if n < self.min_changes:
                continue
            indep_mode = locked = False
            for mc in km.values():
                nm = sum(mc.values())
                if nm >= 3 and max(mc.values()) / nm > 0.8:
                    locked = True
                # mode path needs STRONG evidence: a board region repainted
                # from a few scattered sources is consequence, not clockwork
                if nm >= 2 * self.min_changes and max(mc.values()) / nm <= 0.3:
                    indep_mode = True
            if indep_mode or (max(c.values()) / n <= self.top_share and not locked):
                self.mask.add(k)
                self.version += 1

    def state(self):
        return {"cnt": {k: dict(v) for k, v in self.cnt.items()},
                "modes": {k: {m: dict(v) for m, v in km.items()}
                          for k, km in self.modes.items()},
                "mask": tuple(self.mask), "version": self.version}

    def restore(self, d):
        self.cnt = {k: Counter(v) for k, v in d["cnt"].items()}
        self.modes = {k: {m: Counter(v) for m, v in km.items()}
                      for k, km in d.get("modes", {}).items()}
        self.mask = set(d["mask"]); self.version = d["version"]
        self.frozen = True                   # critical period is over


class GenericPredictiveAgent:
    def __init__(self, actions, gamma=0.95, depth=8, beta=0.6, seed=0, names=None,
                 replay_every=250, cortex=None, max_pairs=100000, sleep_every=2000):
        self.actions = list(actions)                 # any hashable action ids
        self.gamma, self.depth, self.beta = gamma, depth, beta
        self.replay_every = replay_every             # "micro-sleep" cadence (0=off)
        self.tailV = {}                              # replay-consolidated state values
        # optional MicroCortex: generalizing micro-learner over local features
        # (complementary learning systems: counts = hippocampus, this = cortex)
        self.cortex = cortex
        self.rng = np.random.default_rng(seed)
        self.names = names or {}                     # optional pretty names for monologue
        self.trans = defaultdict(Counter)   # (sig,a) -> Counter(next_sig)
        self.rew = defaultdict(float)        # (sig,a) -> summed reward
        self.vis = defaultdict(int)          # (sig,a) -> visits
        self.effect = defaultdict(int)       # (sig,a) -> times the state CHANGED
        self.avail = {}                      # sig -> actions the world offered there
        # BACKOFF over actions (same idea as n-gram backoff): pooled per-action
        # stats across ALL states, used as a prior wherever (sig,a) is untried.
        self.gvis = defaultdict(int)         # a -> global visits
        self.grew = defaultdict(float)       # a -> global summed reward
        self.geff = defaultdict(int)         # a -> global state-change count
        self.states = set()
        # BACKOFF over states (multi-resolution): one count-model per coarser
        # signature level the adapter supplies via SigStack. sighier remembers
        # each fine sig's full stack so planning can back off for ANY state.
        self.levels = []                     # _Level per coarser resolution
        self.sighier = {}                    # fine sig -> full stack tuple
        self.no_change_streak = 0            # consecutive steps where nothing changed
        self.steps_since_new = 0             # steps since a never-seen state appeared
        self.reward_seen = False             # has ANY positive reward ever occurred?
        self.plan = deque()                  # committed frontier path: (state, action)
        # LIFELONG MEMORY (complementary learning systems across time):
        # the exact count tables are SHORT-TERM memory -- fast-binding, high
        # resolution, BOUNDED at max_pairs and recency/frequency-evicted in
        # sleep(). LONG-TERM memory is what survives indefinitely: the coarse
        # _Level models (learned in parallel every step = continuous
        # distillation), the cortex weights, replayed values, and PROTECTED
        # SKILLS (rewarding pairs + anything on a value path) which are never
        # evicted. A year of running stays bounded; competence compounds.
        self.max_pairs = max_pairs           # episodic bound (0 = unbounded)
        self.sleep_every = sleep_every
        self.last = {}                       # (sig,a) -> step of last use
        self.t = 0

    def _intern(self, s):
        """Accept plain signatures or SigStacks; return the fine signature."""
        if isinstance(s, SigStack):
            while len(self.levels) < len(s) - 1:
                self.levels.append(_Level(self.gamma))
            fine = s[0]
            self.sighier[fine] = tuple(s)
            return fine
        return s

    # ---- learning (local, count-based; no backprop) ----
    def learn(self, sig, a, reward, next_sig, available=None, feats=None):
        if self.cortex is not None and feats is not None:
            self.cortex.learn(feats.get(a) or (), reward)
        sig = self._intern(sig)
        next_sig = self._intern(next_sig) if next_sig is not None else None
        h = self.sighier.get(sig)
        nh = self.sighier.get(next_sig) if next_sig is not None else None
        for i, lv in enumerate(self.levels):
            if h is not None and len(h) > i + 1:
                lv.learn(h[i + 1], a, reward,
                         nh[i + 1] if nh is not None and len(nh) > i + 1 else None)
        self.t += 1
        if reward > 0:
            self.reward_seen = True
        self.vis[(sig, a)] += 1
        self.rew[(sig, a)] += reward
        self.last[(sig, a)] = self.t
        self.gvis[a] += 1
        self.grew[a] += reward
        if available is not None:
            self.avail[sig] = tuple(available)
        if next_sig is not None:
            self.trans[(sig, a)][next_sig] += 1
            if next_sig != sig:
                self.effect[(sig, a)] += 1
                self.geff[a] += 1
                self.no_change_streak = 0
            else:
                self.no_change_streak += 1
            self.steps_since_new = 0 if next_sig not in self.states else self.steps_since_new + 1
            self.states.add(next_sig)
        else:
            self.no_change_streak += 1
            self.steps_since_new += 1
        self.states.add(sig)
        if self.replay_every and self.t % self.replay_every == 0:
            self.replay()
        if self.max_pairs and self.t % self.sleep_every == 0:
            self.sleep()

    def _acts(self, sig):
        """Actions the world actually offers at sig (learned), else all."""
        return self.avail.get(sig) or self.actions

    # ---- replay ("sleep"): value-iterate the learned count model so known
    # rewards pull on EVERY state, far beyond the lookahead horizon. Pure
    # dynamic programming over counts -- no gradients, no backprop. ----
    def replay(self, sweeps=30):
        if not any(r > 0 for r in self.rew.values()):
            self.tailV = {}
            return
        V = self.tailV
        for _ in range(sweeps):
            delta = 0.0
            for s in self.states:
                best = 0.0                           # don't propagate negative tails
                for a in self._acts(s):
                    v = self.vis.get((s, a), 0)
                    if not v:
                        continue
                    q = self.rew.get((s, a), 0.0) / v
                    tc = self.trans.get((s, a))
                    if tc:
                        tot = sum(tc.values())
                        q += self.gamma * sum((c / tot) * V.get(ns, 0.0) for ns, c in tc.items())
                    if q > best:
                        best = q
                delta = max(delta, abs(best - V.get(s, 0.0)))
                V[s] = best
            if delta < 1e-4:
                break
        self.tailV = V
        for lv in self.levels:                       # coarse worlds sleep too
            lv.replay()

    # ---- sleep: bounded short-term memory for LIFELONG operation ----
    def sleep(self):
        """Memory consolidation beyond replay: when the episodic store
        exceeds its bound, evict the stalest, rarest, valueless pairs.
        PROTECTED FOR LIFE: rewarding pairs and any pair whose state lies on
        a value path (tailV > 0) -- a skill learned long ago survives any
        amount of junk exploration, because its replayed value keeps it
        alive. What is evicted is not entirely lost: the coarse _Level models
        absorbed every transition as it happened (continuous distillation),
        so the gist persists at lower resolution after the exact trace goes."""
        if not self.max_pairs or len(self.vis) <= self.max_pairs:
            return
        self.replay()                        # settle values before deciding
        evict_n = len(self.vis) - int(self.max_pairs * 0.7)
        cand = [k for k in self.vis
                if self.rew.get(k, 0.0) <= 0.0 and self.tailV.get(k[0], 0.0) <= 1e-6]
        cand.sort(key=lambda k: (self.last.get(k, 0), self.vis[k]))
        for k in cand[:evict_n]:
            for d in (self.vis, self.rew, self.trans, self.effect, self.last):
                d.pop(k, None)
        alive = {s for s, _ in self.vis}
        alive |= {ns for tc in self.trans.values() for ns in tc}
        self.states &= alive
        for d in (self.avail, self.tailV, self.sighier):
            for s in [s for s in d if s not in alive]:
                del d[s]
        for lv in self.levels:               # long-term store: prune only
            if len(lv.vis) > self.max_pairs:  # single-visit, never-rewarding gist
                for k in [k for k, v in lv.vis.items()
                          if v <= 1 and lv.rew.get(k, 0.0) <= 0.0]:
                    lv.vis.pop(k, None); lv.rew.pop(k, None)
                    lv.trans.pop(k, None); lv.eff.pop(k, None)

    # ---- consolidation: persist/restore the world model across sessions ----
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"trans": dict(self.trans), "rew": dict(self.rew),
                         "vis": dict(self.vis), "effect": dict(self.effect),
                         "avail": self.avail, "states": self.states,
                         "gvis": dict(self.gvis), "grew": dict(self.grew),
                         "geff": dict(self.geff), "t": self.t,
                         "last": dict(self.last),
                         "levels": [lv.__dict__ for lv in self.levels],
                         "sighier": self.sighier,
                         "cortex": self.cortex.state() if self.cortex else None}, f)

    def load(self, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.trans.update(d["trans"]); self.rew.update(d["rew"])
        self.vis.update(d["vis"]); self.effect.update(d["effect"])
        self.avail.update(d["avail"]); self.states |= d["states"]
        self.gvis.update(d["gvis"]); self.grew.update(d["grew"])
        self.geff.update(d["geff"]); self.t = d["t"]
        self.last.update(d.get("last") or {})
        self.sighier.update(d.get("sighier") or {})
        for i, ld in enumerate(d.get("levels") or []):
            if i >= len(self.levels):
                self.levels.append(_Level(self.gamma))
            self.levels[i].__dict__.update(ld)
        self.reward_seen = any(r > 0 for r in self.rew.values())
        if self.cortex is not None and d.get("cortex"):
            self.cortex.restore(d["cortex"])
        self.replay()                                # consolidate on wake-up

    # ---- planning: bounded-depth value lookahead over the learned model ----
    def _hor(self, depth):
        """Discounted horizon weight: value of earning beta every step for `depth`."""
        return (1.0 - self.gamma ** max(depth, 1)) / (1.0 - self.gamma)

    def _Q(self, sig, a, depth, memo):
        v = self.vis.get((sig, a), 0)
        if v == 0:
            # OPTIMISM in the face of uncertainty, weighted by BACKOFF: an
            # untried action is imagined to be worth beta per step over the
            # horizon -- scaled by its pooled effect rate across all states
            # (Laplace). An action that has done nothing in 400 states keeps
            # almost no optimism here; one that always changes the world keeps
            # it all. Must still dominate curiosity chains of known no-ops.
            # STATE BACKOFF (Katz): before the global pool, ask the most
            # specific COARSE resolution that has tried `a` -- "I've never seen
            # this exact state, but in states with this object layout, `a`
            # usually does X and leads somewhere worth Y." The level's q()
            # includes its own replayed value, so long-range goal knowledge
            # transfers to never-seen exact states. ONLY once reward exists:
            # pre-reward the levels can only transfer PESSIMISM (low effect
            # rates), which suppresses the exhaustive trying that finding the
            # first reward requires (tu93 collapsed 700->28 states without
            # this gate).
            gv, ge, gr = self.gvis.get(a, 0), self.geff.get(a, 0), None
            h = self.sighier.get(sig)
            if h and self.reward_seen:
                for i, lv in enumerate(self.levels):
                    if i + 1 >= len(h):
                        break
                    cv = lv.vis.get((h[i + 1], a), 0)
                    if cv:
                        gv, ge = cv, lv.eff.get((h[i + 1], a), 0)
                        gr = lv.q(h[i + 1], a)
                        break
            # THOMPSON-sampled effect-rate (Beta posterior), floored: discount
            # dead actions but never write a class off entirely. The posterior
            # MEAN here is a trap: for equally-effective actions it rises with
            # sample count, so argmax locks onto the most-used action forever
            # (rich-get-richer). Sampling keeps choice stochastic in proportion
            # to uncertainty -- neural-noise exploration.
            promise = max(0.15, float(self.rng.beta(ge + 1.0, (gv - ge) + 1.0)))
            if gr is None:
                gr = self.grew.get(a, 0.0) / gv if gv else 0.0
            # TERRITORY NOVELTY: novelty is measured at the most abstract
            # level that is actually novel. An untried exact pair whose
            # MID-LEVEL territory has been ground down thousands of times is
            # barely novel -- the coarse model already knows what acting there
            # does. Scale optimism by territory familiarity (post-reward only:
            # pre-reward the sweep must stay exhaustive). Without this, a game
            # that randomizes its early levels mints endless 'novel' exact
            # states and the agent grinds them forever instead of climbing to
            # genuinely new ground (vc33: 629 states swept, level 2 never).
            return self.beta * self._hor(depth) * promise * self._terr_nov(sig) + gr
        rbar = self.rew.get((sig, a), 0.0) / v
        curiosity = self.beta / (1.0 + v)                # residual explore bonus
        tc = self.trans.get((sig, a))
        if depth <= 0 or not tc:
            return rbar + curiosity
        tot = sum(tc.values())
        nv = sum((c / tot) * self._V(ns, depth - 1, memo) for ns, c in tc.items())
        return rbar + curiosity + self.gamma * nv

    def _V(self, sig, depth, memo):
        if depth <= 0:
            if sig in self.tailV:
                return self.tailV[sig]               # replay value beyond horizon
            h = self.sighier.get(sig)                # novel state: back off to the
            if h:                                    # coarse worlds' replayed value
                for i, lv in enumerate(self.levels):
                    if i + 1 < len(h) and h[i + 1] in lv.V:
                        return lv.V[h[i + 1]]
            return 0.0
        key = (sig, depth)
        if key in memo:
            return memo[key]
        memo[key] = 0.0                                  # guard cycles
        q = max(self._Q(sig, a, depth, memo) for a in self._acts(sig))
        memo[key] = q
        return q

    # ---- frontier search: BFS through the LEARNED model graph ----
    def _has_untried(self, s):
        return any(self.vis.get((s, a), 0) == 0 for a in self._acts(s))

    def _terr(self, s):
        """Familiarity of s's TERRITORY: visits of its mid-level signature
        (0 for plain signatures -- behavior then reduces to the exact model)."""
        h = self.sighier.get(s)
        if h and len(h) > 1 and self.levels:
            return self.levels[0].svis.get(h[1], 0)
        return 0

    def _terr_nov(self, s):
        """Optimism multiplier in [0,1]: 1 in never-seen territory, ->0 as the
        territory's mid-level signature accumulates visits. Active only once
        reward exists; pre-reward everything stays maximally promising."""
        if not self.reward_seen:
            return 1.0
        return 50.0 / (50.0 + self._terr(s))

    def frontier_path(self, sig, max_nodes=30000, max_targets=64):
        """FULL action path (through learned transitions) to the best state
        offering an untried action: [(state to act in, action), ...].
        Returns ([], 0) if the frontier is right here; (None, None) if the
        reachable world is closed. The caller COMMITS to the whole path and
        walks it while reality keeps matching the model (graph-exploration
        style), instead of re-planning every step.
        Pre-reward: NEAREST untried (exhaustive systematic sweep). Post-reward:
        collect candidates and prefer the least-familiar TERRITORY (then
        nearest) -- the 600th cosmetic variant of a ground-down region loses
        to one untried action on genuinely new ground."""
        if self._has_untried(sig):
            return [], 0
        seen = {sig}; parent = {}; dist = {sig: 0}; dq = deque([sig])
        targets = []
        while dq and len(seen) < max_nodes and len(targets) < max_targets:
            s = dq.popleft()
            for a in self._acts(s):
                tc = self.trans.get((s, a))
                if not tc:
                    continue
                for ns in tc:
                    if ns in seen:
                        continue
                    seen.add(ns); parent[ns] = (s, a); dist[ns] = dist[s] + 1
                    if self._has_untried(ns):
                        if not self.reward_seen:     # nearest wins: go now
                            targets = [ns]; dq.clear(); break
                        targets.append(ns)
                    else:
                        dq.append(ns)
                else:
                    continue
                break
        if not targets:
            return None, None
        t = min(targets, key=lambda s: (self._terr(s), dist[s]))
        # STAY IN FRESH TERRITORY: if even the best frontier target sits in
        # MORE familiar territory than here, re-trying here beats walking
        # home. In a region so new the model graph is still sparse (every
        # state novel), BFS can only walk known edges -- which all lead back
        # to the dense, well-trodden component. Without this the agent kept
        # abandoning a just-reached new level ~14 steps in, planning RESET
        # back to the start because that's where the recheckable frontier was.
        if self.reward_seen and self._terr(sig) < self._terr(t):
            return [], 0
        path = []
        while t != sig:
            ps, pa = parent[t]
            path.append((ps, pa)); t = ps
        path.reverse()
        return path, len(path)

    def frontier_size(self):
        return sum(1 for s in self.states for a in self._acts(s) if self.vis.get((s, a), 0) == 0)

    # ---- acting + thinking ----
    def act(self, sig, available, feats=None):
        sig = self._intern(sig)
        # COMMITTED PLAN: while reality keeps matching the model's predictions,
        # walk the precomputed frontier path -- no per-step replanning, no
        # lookahead cost. Any surprise (unexpected state, action no longer
        # offered) aborts the plan and falls through to full thinking.
        if self.plan:
            ps, pa = self.plan[0]
            if ps == sig and pa in available:
                self.plan.popleft()
                return pa, {pa: 0.0}, [
                    f"    [plan] walking frontier path: {len(self.plan) + 1} "
                    f"step(s) left, next {self._aname(pa)}"]
            self.plan.clear()                # model was wrong here: rethink
        memo = {}
        qs = {a: self._Q(sig, a, self.depth, memo) for a in available}
        if self.cortex is not None and feats is not None:
            for a in available:
                fa = feats.get(a)
                if fa:
                    qs[a] += self.cortex.score(fa)   # cortical pattern value+novelty
        mx, mn = max(qs.values()), min(qs.values())
        ties = [a for a in available if qs[a] >= mx - 1e-9]
        untried = [a for a in ties if self.vis.get((sig, a), 0) == 0]
        pool = untried or ties
        best = pool[int(self.rng.integers(len(pool)))]    # random tie-break
        # SYSTEMATIC EXPLORATION: everything here is tried, and either the
        # values are flat OR the model sees NO value reachable from HERE
        # (_V at horizon 0 = replay value w/ coarse backoff; covers both
        # "no reward ever seen" and "this level/region is uncharted") -> commit
        # to the full BFS path to the nearest untried state-action pair.
        # Value-gating (rather than pre-first-reward only) keeps the sweep
        # running on NEW levels after old ones become exploitable: that is
        # where v2 stalled. Systematic sweeping provably dominates noisy
        # argmax when no value gradient exists: it visits every reachable
        # state-action pair exactly once.
        # gate on the FINE replay value only: coarse backoff value is for
        # guiding action choice (re-climb knowledge), but as a gate it is too
        # optimistic -- aliased coarse value leaks into brand-new levels and
        # silences the sweep exactly where it is needed most.
        fa = fd = None; searched = False
        if all(self.vis.get((sig, a), 0) > 0 for a in available) and \
                ((mx - mn) < 1e-3 or self.tailV.get(sig, 0.0) <= 1e-9):
            searched = True
            path, fd = self.frontier_path(sig)
            if path:
                self.plan = deque(path)
                _, pa0 = self.plan.popleft()
                if pa0 in available:
                    best = fa = pa0
                else:
                    self.plan.clear()
        lines = self._think(sig, available, qs, memo, fa, fd, searched)
        return best, qs, lines

    def _aname(self, a):
        if a in self.names:
            return self.names[a]
        return f"ACTION{a}" if not isinstance(a, tuple) else f"ACTION{a[0]}@{a[1]}"

    def _think(self, sig, available, qs, memo, fa, fd, searched):
        lines = []
        nvis = sum(self.vis.get((sig, a), 0) for a in available)
        where = "somewhere NEW" if nvis == 0 else f"a known place (acted here {nvis}x)"
        tv = self.tailV.get(sig, 0.0)
        goal = f" | replay says this place is worth {tv:.2f}" if tv > 0.01 else ""
        lines.append(f"    [situation] I am {where} | world model: {len(self.states)} states, "
                     f"frontier={self.frontier_size()} untried acts | "
                     f"last discovery {self.steps_since_new} steps ago{goal}")
        for a in sorted(available, key=lambda x: -qs[x])[:5]:
            v = self.vis.get((sig, a), 0)
            r = self.rew.get((sig, a), 0.0) / v if v else 0.0
            tc = self.trans.get((sig, a))
            if v == 0:
                gv = self.gvis.get(a, 0)
                pred = ("never tried anywhere -- unknown" if gv == 0 else
                        f"new here; elsewhere does something "
                        f"{self.geff.get(a, 0) / gv:.0%} of {gv} tries")
            elif tc:
                eff = self.effect[(sig, a)] / v
                pred = ("does nothing here" if eff == 0
                        else f"changes the world {eff:.0%} of the time, {len(tc)} outcome(s)")
            else:
                pred = "rejected by the world here"
            tag = "novel" if v == 0 else ("REWARDING" if r > 0 else "known")
            lines.append(f"    {self._aname(a)}: value={qs[a]:.3f} "
                         f"(tried {v}x, r̄={r:+.2f}, {tag}; {pred})")
        # imagined plan: greedy rollout through my own model
        plan, s = [], sig
        for d in range(min(self.depth, 6)):
            acts = available if d == 0 else self._acts(s)
            qa = {a: self._Q(s, a, max(self.depth - d, 1), memo) for a in acts}
            a2 = max(qa, key=qa.get)
            plan.append(self._aname(a2))
            tc = self.trans.get((s, a2))
            if not tc:
                plan.append("?(unknown beyond)"); break
            s = tc.most_common(1)[0][0]
        lines.append(f"    [plan] imagine: {' -> '.join(plan)}")
        if self.no_change_streak >= 10:
            lines.append(f"    [flag] feels STUCK: {self.no_change_streak} steps with no effect")
        if fa is not None:
            lines.append(f"    [flag] exhausted here -- heading to frontier "
                         f"{fd} step(s) away via {self._aname(fa)}")
        elif searched and fd is None:
            lines.append("    [flag] every reachable state is exhausted -- "
                         "the world I can reach is closed; need a reset/new door")
        return lines


# =============================================================================
# A generic toy task to prove the agent plans/solves (NOT ARC-specific):
# navigate a gridworld to a hidden goal that gives reward, then keep scoring.
# =============================================================================
class GridWorld:
    def __init__(self, n=6, seed=0):
        self.n = n; self.rng = np.random.default_rng(seed)
        self.goal = (n-1, n-1); self.start = (0, 0); self.pos = self.start
    def reset(self):
        self.pos = self.start; return self.pos
    actions = [0, 1, 2, 3]   # up, down, left, right
    def step(self, a):
        y, x = self.pos
        if a == 0: y -= 1
        elif a == 1: y += 1
        elif a == 2: x -= 1
        elif a == 3: x += 1
        y = min(max(y, 0), self.n-1); x = min(max(x, 0), self.n-1)
        self.pos = (y, x)
        if self.pos == self.goal:
            self.pos = self.start                        # teleport home, +1 reward
            return self.start, 1.0, False
        return self.pos, 0.0, False


class SkinnedGridWorld:
    """GridWorld whose observation carries a cosmetic SKIN code that changes
    every `flip` steps (think: level 2 looks different but works the same).
    Exact-state memory is structurally blind to the equivalence -- every flip
    voids its entire model. Multi-resolution backoff (SigStack with a
    skin-free coarse level) transfers the whole policy across flips."""
    actions = [0, 1, 2, 3]
    def __init__(self, n=6, flip=250, seed=0):
        self.n, self.flip = n, flip
        self.rng = np.random.default_rng(seed)
        self.goal = (n - 1, n - 1); self.pos = (0, 0)
        self.skin = 1; self.tk = 0
    def obs(self):
        return (*self.pos, self.skin)
    def step(self, a):
        self.tk += 1
        if self.tk % self.flip == 0:
            self.skin = int(self.rng.integers(1 << 30))
        y, x = self.pos
        y += (a == 1) - (a == 0); x += (a == 3) - (a == 2)
        self.pos = (min(max(y, 0), self.n - 1), min(max(x, 0), self.n - 1))
        if self.pos == self.goal:
            self.pos = (0, 0)                        # teleport home, +1 reward
            return self.obs(), 1.0
        return self.obs(), 0.0


def demo_skinned():
    print("\nSkinnedGridWorld: cosmetic skin code changes every 250 steps;")
    print("exact-state memory must relearn the whole maze after every flip.")
    for stacked in (False, True):
        env = SkinnedGridWorld()
        agent = GenericPredictiveAgent(actions=env.actions, depth=10)
        wrap = (lambda o: SigStack((o, o[:2]))) if stacked else (lambda o: o)
        sig = wrap(env.obs()); total = 0.0; hist = []
        for step in range(1, 5001):
            a, _, _ = agent.act(sig, env.actions)
            nobs, r = env.step(a)
            nsig = wrap(nobs)
            agent.learn(sig, a, r, nsig)
            sig = nsig; total += r
            if step % 1000 == 0:
                hist.append(total); total = 0.0
        name = "multi-res backoff" if stacked else "exact-state only "
        print(f"  {name}: reward/1000-step block: {[f'{h:.0f}' for h in hist]}")


class ClockedGridWorld:
    """GridWorld whose observation carries a step COUNTER component -- the
    generic analogue of a HUD move-counter: it changes every step no matter
    what you do, so exact-state memory never sees the same state twice."""
    actions = [0, 1, 2, 3]
    def __init__(self, n=6):
        self.n = n; self.goal = (n - 1, n - 1); self.pos = (0, 0); self.t = 0
    def obs(self):
        return (*self.pos, self.t)
    def step(self, a):
        self.t += 1
        y, x = self.pos
        y += (a == 1) - (a == 0); x += (a == 3) - (a == 2)
        self.pos = (min(max(y, 0), self.n - 1), min(max(x, 0), self.n - 1))
        if self.pos == self.goal:
            self.pos = (0, 0)
            return self.obs(), 1.0
        return self.obs(), 0.0


def demo_clocked():
    print("\nClockedGridWorld: a step-counter element ticks in every observation;")
    print("exact-state memory never sees the same state twice (the vc33 HUD trap).")
    for habit in (False, True):
        env = ClockedGridWorld()
        agent = GenericPredictiveAgent(actions=env.actions, depth=10)
        hab = Habituator()
        prev = env.obs()
        def enc(o):                          # adapter wiring: mask clock elements
            return tuple("·" if i in hab.mask else v for i, v in enumerate(o))
        sig = enc(prev); total = 0.0; hist = []
        for step in range(1, 5001):
            a, _, _ = agent.act(sig, env.actions)
            nobs, r = env.step(a)
            if habit:
                hab.observe({(i, (prev[i], nobs[i]))
                             for i in range(len(nobs)) if nobs[i] != prev[i]}, a)
            nsig = enc(nobs)
            agent.learn(sig, a, r, nsig)
            sig, prev = nsig, nobs; total += r
            if step % 1000 == 0:
                hist.append(total); total = 0.0
        name = "with habituation   " if habit else "exact-state only   "
        print(f"  {name}: reward/1000-step block: {[f'{h:.0f}' for h in hist]} "
              f"(masked elements: {sorted(hab.mask)})")


def demo_lifelong():
    print("\nLIFELONG MEMORY: learn a skill, then wander a junk world for 6000")
    print("steps (floods episodic memory with valueless states), then return.")
    print("Sleep must keep the brain bounded AND keep the old skill alive.")
    rng = np.random.default_rng(1)
    for bounded in (False, True):
        env = GridWorld(n=6)
        agent = GenericPredictiveAgent(actions=[0, 1, 2, 3], depth=10,
                                       max_pairs=800 if bounded else 0,
                                       sleep_every=500)
        sig = env.reset(); learned = 0.0
        for _ in range(2000):                            # phase 1: learn the skill
            a, _, _ = agent.act(sig, env.actions)
            nobs, r, _ = env.step(a)
            agent.learn(sig, a, r, nobs); sig = nobs; learned += r
        for _ in range(6000):                            # phase 2: junk world
            jsig = ("junk", int(rng.integers(1 << 30)))
            a, _, _ = agent.act(jsig, env.actions)
            agent.learn(jsig, a, 0.0, ("junk", int(rng.integers(1 << 30))))
        sig = env.reset(); back = 0.0
        for _ in range(1000):                            # phase 3: return to the skill
            a, _, _ = agent.act(sig, env.actions)
            nobs, r, _ = env.step(a)
            agent.learn(sig, a, r, nobs); sig = nobs; back += r
        name = "bounded + sleep " if bounded else "unbounded       "
        print(f"  {name}: skill learned={learned:.0f}, after junk year={back:.0f}/1000 steps, "
              f"brain={len(agent.vis)} pairs")


def demo():
    env = GridWorld(n=6)
    agent = GenericPredictiveAgent(actions=[0, 1, 2, 3], depth=10)
    sig = env.reset(); total = 0.0; block = 0.0; hist = []
    for step in range(1, 3001):
        a, qs, _ = agent.act(sig, env.actions)
        nobs, r, _ = env.step(a)
        nsig = nobs                                      # gridworld: obs IS the signature
        agent.learn(sig, a, r, nsig)
        sig = nsig; total += r; block += r
        if step % 500 == 0:
            hist.append((step, block)); print(f"  steps {step-499}-{step}: reward={block:.0f}"); block = 0.0
    print(f"GENERIC AGENT on GridWorld: total reward={total:.0f} over 3000 steps, "
          f"states learned={len(agent.states)}")
    print(f"  reward/500-step block: {[b for _,b in hist]}  (should rise as it learns to plan to the goal)")


if __name__ == "__main__":
    demo()
    demo_skinned()
    demo_clocked()
    demo_lifelong()
