"""Numpy forward pass of ACT-lite (same numbers as tf/policy.py, no torch needed).
usage: w = load("policy.npz"); chunk = forward(w, x)  # x: (28,) or (B, 28) raw state -> (B, chunk, 4) actions
Runner(w) gives the closed-loop policy with temporal ensembling (m = 0.1), as used in every rollout."""
import math

import numpy as np

FIELDS = [(0, 3), (3, 6), (6, 9), (9, 11), (11, 14), (14, 17), (17, 25), (25, 28)]
HEADS = 4


def load(path):
    with np.load(path) as z:
        return {k: z[k].astype(np.float64) for k in z.files}


def layer_norm(h, g, b, eps=1e-5):
    mu = h.mean(-1, keepdims=True)
    var = h.var(-1, keepdims=True)
    return (h - mu) / np.sqrt(var + eps) * g + b


def gelu(x):
    return 0.5 * x * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))


def attention(h, w, p):
    B, T, d = h.shape
    qkv = h @ w[p + "self_attn.in_proj_weight"].T + w[p + "self_attn.in_proj_bias"]
    q, k, v = np.split(qkv, 3, -1)
    dh = d // HEADS
    q, k, v = (t.reshape(B, T, HEADS, dh).transpose(0, 2, 1, 3) for t in (q, k, v))
    s = q @ k.transpose(0, 1, 3, 2) / math.sqrt(dh)
    s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True)
    o = (s @ v).transpose(0, 2, 1, 3).reshape(B, T, d)
    return o @ w[p + "self_attn.out_proj.weight"].T + w[p + "self_attn.out_proj.bias"]


def forward(w, x):
    x = np.atleast_2d(np.asarray(x, np.float64))
    z = (x - w["norm_mean"]) / w["norm_std"]
    toks = [z[:, a:b] @ w[f"embed.{i}.weight"].T + w[f"embed.{i}.bias"] for i, (a, b) in enumerate(FIELDS)]
    q = np.broadcast_to(w["query"], (len(x),) + w["query"].shape)
    h = np.concatenate([np.stack(toks, 1), q], 1) + w["pos"]
    n_layers = len({k.split(".")[2] for k in w if k.startswith("encoder.layers.")})
    for i in range(n_layers):
        p = f"encoder.layers.{i}."
        h = h + attention(layer_norm(h, w[p + "norm1.weight"], w[p + "norm1.bias"]), w, p)
        f = layer_norm(h, w[p + "norm2.weight"], w[p + "norm2.bias"])
        h = h + gelu(f @ w[p + "linear1.weight"].T + w[p + "linear1.bias"]) @ w[p + "linear2.weight"].T + w[p + "linear2.bias"]
    h = layer_norm(h[:, len(FIELDS):], w["norm.weight"], w["norm.bias"])
    return h @ w["head.weight"].T + w["head.bias"]


class Runner:
    def __init__(self, w, m=0.1):
        self.w, self.m, self.buf, self.t = w, m, [], 0

    def __call__(self, x):
        self.buf.append((self.t, forward(self.w, x)[0]))
        self.buf = [(t0, c) for t0, c in self.buf if self.t - t0 < len(c)]
        acts = np.array([c[self.t - t0] for t0, c in self.buf])
        ages = np.array([self.t - t0 for t0, _ in self.buf])
        wt = np.exp(-self.m * (ages.max() - ages))
        self.t += 1
        return np.clip((wt[:, None] * acts).sum(0) / wt.sum(), -1, 1)
