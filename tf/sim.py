"""Fetch pick-and-place simulator for Teleop-Fix (copied from weightspace/ws/fetch.py, unchanged API).

Environment set-up follows RoboChunk (FetchPickAndPlace-v4, goals on the table, object placed by region), with the
protocol's 100-step episodes and table regions measured from the object range's centre. Success: object within 5 cm of
the goal at the last step. Policy input = the 25-dim observation then the 3-dim goal; action = (dx, dy, dz, gripper).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

ENV_ID = "FetchPickAndPlace-v4"
STEPS = 100
SUCCESS_M = 0.05
STEP_M = 0.05          # Fetch moves the mocap at most 5 cm per action
HOVER = 0.05           # expert hovers 5 cm above the object before descending
RANGE = 0.15           # object spawn half-width around the centre
OBS = dict(grip_pos=slice(0, 3), object_pos=slice(3, 6), object_rel=slice(6, 9), fingers=slice(9, 11))
OPEN, CLOSED = 1.0, -1.0


@dataclass(frozen=True)
class Spec:
    """One episode: seed fixes object and goal; region restricts the object start; mass multiplies the object mass."""
    seed: int
    region: str = "whole"        # whole | left | far | near (x <= -0.02) | right (y <= -0.03) | core
    mass_mult: float = 1.0


def in_region(d, region):
    x, y = d
    return {"whole": True, "left": y > 0.03, "far": x > 0.05, "near": x <= -0.02, "right": y <= -0.03,
            "core": abs(x) <= RANGE / 2 and abs(y) <= RANGE / 2}[region]


_ENV = None


def env():
    global _ENV
    if _ENV is None:
        warnings.filterwarnings("ignore")
        import gymnasium as gym
        import gymnasium_robotics
        import mujoco
        gym.register_envs(gymnasium_robotics)
        e = gym.make(ENV_ID).unwrapped
        m = e.model
        e._ws = dict(body=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object0"))
        e._ws["mass0"] = float(m.body_mass[e._ws["body"]])
        e.target_in_the_air = False
        _ENV = e
    return _ENV


def valid_start(obj_d, goal_d, region="whole"):
    obj_d, goal_d = np.asarray(obj_d, float), np.asarray(goal_d, float)
    return bool(np.all(np.abs(obj_d) <= RANGE) and np.linalg.norm(obj_d) >= 0.03 and in_region(obj_d, region)
                and np.all(np.abs(goal_d) <= RANGE) and np.linalg.norm(goal_d - obj_d) > 0.10)


def reset(spec: Spec, obj_d=None, goal_d=None):
    """Place object and goal from the spec (or from explicit offsets from the range centre). Returns the policy input."""
    import mujoco
    e = env()
    e.model.body_mass[e._ws["body"]] = e._ws["mass0"] * spec.mass_mult
    e.reset(seed=int(spec.seed))
    rng = np.random.default_rng(spec.seed + 7919)
    center = e.initial_gripper_xpos[:2]
    while True:
        d = rng.uniform(-RANGE, RANGE, 2)
        if np.linalg.norm(d) >= 0.03 and in_region(d, spec.region):
            break
    while True:
        gd = rng.uniform(-RANGE, RANGE, 2)
        if np.linalg.norm(gd - d) > 0.10:
            break
    if obj_d is not None:
        d = np.asarray(obj_d, float)
    if goal_d is not None:
        gd = np.asarray(goal_d, float)
    obj, g = center + d, center + gd
    q = e._utils.get_joint_qpos(e.model, e.data, "object0:joint").copy()
    q[:2] = obj
    e._utils.set_joint_qpos(e.model, e.data, "object0:joint", q)
    e.goal = np.array([g[0], g[1], e.height_offset])
    mujoco.mj_forward(e.model, e.data)
    return observe()


def observe():
    e = env()
    return np.concatenate([e._get_obs()["observation"], e.goal]).astype(np.float64)


def step(a):
    env().step(np.clip(np.asarray(a, dtype=np.float64), -1, 1))
    return observe()


def success(x):
    return bool(np.linalg.norm(x[3:6] - x[25:28]) < SUCCESS_M)


def toward(x, target, grip):
    a = np.empty(4)
    a[:3] = np.clip((np.asarray(target) - x[OBS["grip_pos"]]) / STEP_M, -1, 1)
    a[3] = grip
    return a


class Expert:
    """Four-phase proportional controller: hover 5 cm above the object, descend, close for 5 steps, carry to the goal.
    `pause` inserts that many zero-motion steps (gripper held open) once the gripper has settled at the hover pose."""

    def __init__(self, pause=0):
        self.pause, self.phase, self.k = pause, "approach", 0
        self.hover = None

    def __call__(self, x):
        obj, goal, grip = x[3:6], x[25:28], x[0:3]
        if self.phase == "approach":
            hover = obj + [0, 0, HOVER]
            if np.linalg.norm(hover - grip) < 0.01:
                self.phase, self.k, self.hover = ("pause" if self.pause else "descend"), 0, hover
            else:
                return toward(x, hover, OPEN)
        if self.phase == "pause":
            if self.k < self.pause:
                self.k += 1
                return toward(x, self.hover, OPEN)   # hold the hover pose (zero motion when on it)
            self.phase = "descend"
        if self.phase == "descend":
            if abs(grip[2] - obj[2]) < 0.005 and np.linalg.norm(grip[:2] - obj[:2]) < 0.01:
                self.phase, self.k = "close", 0
            else:
                return toward(x, obj, OPEN)
        if self.phase == "close":
            if self.k < 5:
                self.k += 1
                return toward(x, obj, CLOSED)
            self.phase = "carry"
        lift = goal[2] + 0.06          # fixed carry height above the table (the held object rises with the gripper)
        if self.phase == "carry":
            above = np.array([goal[0], goal[1], lift])
            if np.linalg.norm(grip[:2] - goal[:2]) < 0.01:
                self.phase = "lower"
            elif grip[2] < lift - 0.01 and np.linalg.norm(grip[:2] - goal[:2]) > 0.03:
                return toward(x, [grip[0], grip[1], lift], CLOSED)   # lift straight up first
            else:
                return toward(x, above, CLOSED)
        return toward(x, [goal[0], goal[1], goal[2] + 0.01], CLOSED)


def rollout(policy, spec: Spec, steps=STEPS, record=False, obj_d=None, goal_d=None):
    """policy(x) -> action. Returns success at the last step (and the trajectory if record)."""
    x = reset(spec, obj_d, goal_d)
    xs, acts = [], []
    for _ in range(steps):
        a = np.clip(np.asarray(policy(x), dtype=np.float64), -1, 1)
        if record:
            xs.append(x.copy()); acts.append(a.copy())
        x = step(a)
    ok = success(x)
    return (ok, np.array(xs), np.array(acts), x) if record else ok
