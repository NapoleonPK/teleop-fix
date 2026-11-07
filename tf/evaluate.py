"""Closed-loop evaluation of an ACT-lite policy on fixed seeds and table regions."""
from __future__ import annotations

import numpy as np

from tf import policy, sim


def _ep(args):
    path, seed, region = args
    import torch
    torch.set_num_threads(1)
    m = policy.load(path)
    return bool(sim.rollout(policy.Runner(m), sim.Spec(int(seed), region)))


def success(path, seeds, region="whole", workers=8):
    from multiprocessing import get_context
    with get_context("spawn").Pool(workers) as pool:
        r = pool.map(_ep, [(str(path), s, region) for s in seeds])
    return float(np.mean(r))
