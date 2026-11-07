"""Simulated teleoperator for Teleop-Fix.

A leader-follower teleop demo is modelled as the scripted expert (tf.sim.Expert) driven through a human "operator
profile". The recorded pair at step t is (follower observation x_t, operator command a_t), as in leader-follower
recording. Artifacts, each a known habit of human teleoperation:

- lag:        the operator reacts to what the follower showed `lag` steps ago (stale observation -> late, overshooting
              commands; the follower trails the leader).
- jitter:     Gaussian hand tremor on the xyz command (std `jitter`, in action units).
- hesitate_p / hesitate_steps: with probability hesitate_p a demo contains one pause of U(hesitate_steps) steps once
              the gripper reaches the hover pose above the block (hand held still with tremor `hesitate_tremor`).
- early_grip_p / early_grip_dz: with probability early_grip_p the operator closes the gripper while still
              early_grip_dz metres above the block, misses, notices after `notice` steps, reopens, rises and retries.
- comfort:    table region the operator demonstrates in (block start); "whole" = everywhere. A "near" comfort zone
              leaves the far side (x offset > 0.05 m) undemonstrated.
- keep_failed: failed demos are kept in the dataset (operators rarely delete them).
- desync:     the recording pipeline's leader and follower streams are misaligned: observation t is stored with the
              command of step t - desync (the demo itself was executed correctly).

Every artifact the profile applies is written to the demo's metadata, which the agent never sees (ground truth for
the diagnosis score and for the heuristic-filter baseline's evaluation, not for its decisions).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from tf import sim

SECONDS_PER_DEMO = 10.0     # 100 steps at 10 Hz
RESET_SECONDS = 20.0        # operator resets the scene between demos
OPERATOR_MIN_PER_DEMO = (SECONDS_PER_DEMO + RESET_SECONDS) / 60


@dataclass(frozen=True)
class Profile:
    name: str = "clean"
    lag: int = 0
    gain: float = 1.0            # scale of the xyz command (a laggy operator slows down)
    jitter: float = 0.05
    hesitate_p: float = 0.0
    hesitate_steps: tuple = (15, 30)
    hesitate_tremor: float = 0.05
    early_grip_p: float = 0.0
    early_grip_dz: float = 0.025
    notice: int = 4
    comfort: str = "whole"
    desync: int = 0              # recording pipeline pairs observation t with the command of step t - desync
    keep_failed: bool = True


PROFILES = {
    "clean": Profile(),
    "hesitant": Profile(name="hesitant", hesitate_p=0.5),
    "laggy": Profile(name="laggy", lag=3, gain=0.5),      # decision 3: does not hurt the policy; not used
    "desync": Profile(name="desync", desync=2),
    "comfort": Profile(name="comfort", comfort="near"),
    "early_grip": Profile(name="early_grip", early_grip_p=0.4),
}


class Operator:
    """Callable policy x -> a that also records which artifacts it produced."""

    def __init__(self, prof: Profile, rng: np.random.Generator):
        self.p, self.rng = prof, rng
        self.expert = sim.Expert()
        self.hist = []
        self.hes_left = int(rng.integers(prof.hesitate_steps[0], prof.hesitate_steps[1] + 1)) if rng.random() < prof.hesitate_p else 0
        self.early = rng.random() < prof.early_grip_p
        self.meta = dict(hesitation_steps=self.hes_left, early_grip=self.early, lag=prof.lag, retry=False)
        self.state, self.k = None, 0       # early-grip retry state machine
        self.hover_hold = None

    def __call__(self, x):
        p = self.p
        self.hist.append(x.copy())
        xs = self.hist[max(0, len(self.hist) - 1 - p.lag)]          # what the operator perceives
        grip, obj = xs[0:3], xs[3:6]
        # hesitation at the hover pose: hold still (with tremor), expert state untouched
        if self.hes_left and self.expert.phase == "descend" and self.hover_hold is None:
            self.hover_hold = xs[0:3].copy()
        if self.hes_left and self.hover_hold is not None:
            self.hes_left -= 1
            a = sim.toward(x, self.hover_hold, sim.OPEN)
            a[:3] += self.rng.normal(0, p.hesitate_tremor, 3)
            return np.clip(a, -1, 1)
        # early grip: close while still above the block, then reopen, rise and retry
        if self.early and self.state is None and self.expert.phase == "descend" and 0.005 < grip[2] - obj[2] < p.early_grip_dz:
            self.state, self.k = "closing", 0
            self.meta["retry"] = True
        if self.state == "closing":
            self.k += 1
            if self.k > 5 + p.notice:
                self.state, self.k = "reopen", 0
            return np.array([0, 0, 0, sim.CLOSED]) + np.r_[self.rng.normal(0, p.jitter, 3), 0]
        if self.state == "reopen":
            self.k += 1
            if self.k > 4:
                self.state = "done"
                self.expert = sim.Expert()          # start the approach again from the hover pose
            return np.r_[0, 0, 0.6, sim.OPEN]
        a = self.expert(xs).copy()
        a[:3] = a[:3] * p.gain + self.rng.normal(0, p.jitter, 3)
        return np.clip(a, -1, 1)


def demo(prof: Profile, seed: int):
    """One recorded teleop demo. Returns X (100 x 28), A (100 x 4), success, metadata."""
    rng = np.random.default_rng(seed + 31337)
    op = Operator(prof, rng)
    spec = sim.Spec(seed, prof.comfort)
    ok, X, A, _ = sim.rollout(op, spec, record=True)
    if prof.desync:                       # logging misalignment: the executed demo is fine, the recorded pairs are not
        rest = np.tile(np.array([0, 0, 0, sim.OPEN], np.float64), (prof.desync, 1))   # leader not yet moving
        A = np.concatenate([rest, A[:-prof.desync]])
    meta = dict(seed=seed, success=bool(ok), profile=prof.name, region=prof.comfort, desync=prof.desync, **op.meta)
    return X.astype(np.float32), A.astype(np.float32), ok, meta


def _demo(args):
    return demo(*args)


def record(prof: Profile, seeds, workers=6):
    """Record demos in parallel; failed demos are dropped only if the profile says so."""
    from multiprocessing import get_context
    with get_context("spawn").Pool(workers) as pool:
        out = pool.map(_demo, [(prof, int(s)) for s in seeds])
    keep = [o for o in out if o[2] or prof.keep_failed]
    return dict(X=np.stack([o[0] for o in keep]), A=np.stack([o[1] for o in keep]),
                meta=[o[3] for o in keep], profile=asdict(prof))
