"""ACT-lite: a small action-chunking transformer policy for state input (no images, no CVAE).

Input: the 28-dim Fetch state, normalised with the dataset mean/std (stored with the weights), split into 8 field
tokens (gripper pos, block pos, block-relative pos, fingers, block rotation, block velocity, gripper velocity + finger
velocity, goal). C learned query tokens attend jointly with the state tokens through L encoder layers; each query
token predicts one action of the next-C-steps chunk. Training loss: L1 on the chunk (ACT). Inference: temporal
ensembling over overlapping chunks with weights exp(-m * age), m = 0.1 (ACT).
Weights travel as an .npz of the state dict plus "norm_mean"/"norm_std"; tf/policy_np.py is a numpy forward pass of
the same network (for agents that inspect the weights without torch).
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

FIELDS = [(0, 3), (3, 6), (6, 9), (9, 11), (11, 14), (14, 17), (17, 25), (25, 28)]
ARCH = dict(d=128, layers=3, heads=4, ff=512, chunk=10, dropout=0.0)


class ACTLite(nn.Module):
    def __init__(self, d=128, layers=3, heads=4, ff=512, chunk=10, dropout=0.0):
        super().__init__()
        self.chunk = chunk
        self.embed = nn.ModuleList([nn.Linear(b - a, d) for a, b in FIELDS])
        self.pos = nn.Parameter(torch.zeros(len(FIELDS) + chunk, d))
        self.query = nn.Parameter(torch.randn(chunk, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 4)
        self.register_buffer("norm_mean", torch.zeros(28))
        self.register_buffer("norm_std", torch.ones(28))

    def forward(self, x):                     # x: (B, 28) raw state -> (B, chunk, 4)
        z = (x - self.norm_mean) / self.norm_std
        toks = [e(z[:, a:b]) for e, (a, b) in zip(self.embed, FIELDS)]
        q = self.query.unsqueeze(0).expand(len(x), -1, -1)
        h = torch.cat([torch.stack(toks, 1), q], 1) + self.pos
        h = self.encoder(h)
        return self.head(self.norm(h[:, len(FIELDS):]))


def new(seed=0, **arch):
    torch.manual_seed(seed)
    return ACTLite(**{**ARCH, **arch})


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def save(m, path):
    sd = {k: v.detach().cpu().numpy() for k, v in m.state_dict().items()}
    np.savez(path, **sd)


def load(path, **arch):
    m = ACTLite(**{**ARCH, **arch})
    with np.load(path) as z:
        sd = {k: torch.from_numpy(z[k]) for k in z.files}
    m.load_state_dict(sd)
    m.eval()
    return m


def load_bytes(b, **arch):
    return load(io.BytesIO(b), **arch)


class Runner:
    """Stateful closed-loop policy with ACT temporal ensembling."""

    def __init__(self, m, m_ens=0.1):
        self.m, self.m_ens, self.buf, self.t = m, m_ens, [], 0

    @torch.no_grad()
    def __call__(self, x):
        chunk = self.m(torch.as_tensor(x, dtype=torch.float32)[None])[0].numpy()
        self.buf.append((self.t, chunk))
        self.buf = [(t0, c) for t0, c in self.buf if self.t - t0 < len(c)]
        acts = np.array([c[self.t - t0] for t0, c in self.buf])
        ages = np.array([self.t - t0 for t0, _ in self.buf])
        w = np.exp(-self.m_ens * (ages.max() - ages))         # oldest prediction weighted most (ACT)
        self.t += 1
        return np.clip((w[:, None] * acts).sum(0) / w.sum(), -1, 1)


def chunks(X, A, C):
    """Per-step training targets: the next C actions, padded with the last action; mask marks real steps."""
    N, T, _ = A.shape
    idx = np.arange(T)[:, None] + np.arange(C)[None]
    mask = idx < T
    idx = np.minimum(idx, T - 1)
    Y = A[:, idx]                                   # N, T, C, 4
    return X.reshape(-1, X.shape[-1]), Y.reshape(-1, C, 4), np.broadcast_to(mask, (N, T, C)).reshape(-1, C)


def train(X, A, seed=0, steps=4000, batch=256, lr=3e-4, init=None, threads=4, log=None):
    """Behaviour cloning on demos X (N, T, 28), A (N, T, 4). init = model to fine-tune (normalisation kept)."""
    torch.set_num_threads(threads)
    rng = np.random.default_rng(seed)
    m = init if init is not None else new(seed)
    if init is None:
        flat = X.reshape(-1, 28)
        m.norm_mean.copy_(torch.as_tensor(flat.mean(0)))
        m.norm_std.copy_(torch.as_tensor(np.maximum(flat.std(0), 1e-3)))
    Xs, Ys, Ms = chunks(X, A, m.chunk)
    Xs, Ys, Ms = (torch.as_tensor(v, dtype=torch.float32) for v in (Xs, Ys, Ms))
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    m.train()
    t0 = time.time()
    for s in range(steps):
        i = torch.as_tensor(rng.integers(0, len(Xs), batch))
        pred = m(Xs[i])
        loss = ((pred - Ys[i]).abs().mean(-1) * Ms[i]).sum() / Ms[i].sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if log and (s % 500 == 0 or s == steps - 1):
            log(f"step {s} loss {loss.item():.4f} {time.time() - t0:.0f}s")
    m.eval()
    return m
