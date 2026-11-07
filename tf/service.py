"""Host-side services for one Teleop-Fix agent session: rollouts and training, each with a hard budget.
One process per session, bound to 127.0.0.1. Every call is logged with a timestamp and the SHA-256 of the request.

POST /rollout {"weights_npz_b64": ..., "object_offset": [dx, dy]?, "goal_offset": [dx, dy]?}
     -> one 100-step closed-loop episode of the submitted ACT-lite weights (same arrays and shapes as the given
        policy): success, final distance, observations, actions. Uses 1 of the rollout budget.
POST /train   {"demos": [int, ...] | null, "weights": [float, ...] | null, "relabel_npz_b64": ... | null,
               "init": "given" | "scratch" | {"weights_npz_b64": ...}, "steps": int <= RECIPE_STEPS}
     -> trains with the frozen recipe on the chosen demos of the session's dataset (indices into dataset.npz; repeats
        allowed; null = all), optional per-demo sampling weights, optional replacement actions
        ({"idx": [...], "A": array (n, 100, 4)} as npz), from the given policy, from scratch, or from submitted
        weights; optional extra demos ({"X": (n, 100, 28), "A": (n, 100, 4)}, n <= 500, appended after the
        dataset's demos: synthetic or augmented data). Returns the trained weights (npz, base64) and the final training loss. Uses 1 of the training budget.
POST /status  -> budgets used and left.
usage: python -m tf.service CONFIG_JSON  (port, rollout_budget, train_budget, dataset, policy, session_index, log)
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from tf import policy, seeds, sim

RECIPE = dict(steps=8000, batch=256, lr=3e-4, finetune_steps=2000, finetune_lr=1e-4)
CFG = json.load(open(sys.argv[1])) if __name__ == "__main__" else {}
LOCK = threading.Lock()
STATE = dict(rollouts=0, trains=0)


def log(rec):
    with open(CFG["log"], "a") as f:
        f.write(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), **rec)) + "\n")


def weights_from(b64):
    with np.load(io.BytesIO(base64.b64decode(b64))) as z:
        w = {k: z[k] for k in z.files}
    want = {k: list(v.shape) for k, v in GIVEN.items()}
    if {k: list(v.shape) for k, v in w.items()} != want:
        raise ValueError(f"weights must have exactly the arrays and shapes of task/policy.npz")
    if not all(np.all(np.isfinite(v)) for v in w.values()):
        raise ValueError("weights contain NaN or inf")
    buf = io.BytesIO(); np.savez(buf, **w)
    return policy.load_bytes(buf.getvalue())


def params_sha(b64):
    """Hash of the parameter values only (independent of npz container bytes)."""
    with np.load(io.BytesIO(base64.b64decode(b64))) as z:
        h = hashlib.sha256()
        for k in sorted(z.files):
            h.update(k.encode()); h.update(np.ascontiguousarray(z[k], np.float32).tobytes())
    return h.hexdigest()


def to_b64(m):
    buf = io.BytesIO(); policy.save(m, buf)
    return base64.b64encode(buf.getvalue()).decode()


def do_rollout(req):
    m = weights_from(req["weights_npz_b64"])
    obj_d, goal_d = req.get("object_offset"), req.get("goal_offset")
    if (obj_d is None) != (goal_d is None):
        raise ValueError("give both object_offset and goal_offset, or neither")
    if obj_d is not None and not sim.valid_start(obj_d, goal_d):
        raise ValueError("start outside the table range: offsets within +-0.15 m, object at least 0.03 m from the "
                         "centre, goal more than 0.10 m from the object")
    spec = sim.Spec(seeds.agent_rollout(CFG["session_index"], STATE["rollouts"]))
    ok, xs, acts, xe = sim.rollout(policy.Runner(m), spec, record=True, obj_d=obj_d, goal_d=goal_d)
    return dict(success=ok, final_object_to_goal_m=float(np.linalg.norm(xe[3:6] - xe[25:28])),
                observations=np.round(xs, 5).tolist(), actions=np.round(acts, 5).tolist())


def do_train(req):
    X, A = DATA["X"], DATA["A"].copy()
    n0 = len(X)
    if req.get("relabel_npz_b64"):
        with np.load(io.BytesIO(base64.b64decode(req["relabel_npz_b64"]))) as z:
            idx, newA = z["idx"].astype(int), z["A"].astype(np.float32)
        if newA.shape != (len(idx),) + A.shape[1:] or not np.all(np.isfinite(newA)) or idx.min() < 0 or idx.max() >= n0:
            raise ValueError(f"relabel: idx must index dataset demos and A must have shape (len(idx), {A.shape[1]}, 4)")
        A[idx] = np.clip(newA, -1, 1)
    if req.get("extra_npz_b64"):                  # synthetic or augmented demos appended to the dataset
        with np.load(io.BytesIO(base64.b64decode(req["extra_npz_b64"]))) as z:
            eX, eA = z["X"].astype(np.float32), z["A"].astype(np.float32)
        if eX.ndim != 3 or eX.shape[1:] != X.shape[1:] or eA.shape != eX.shape[:2] + (4,) or not 1 <= len(eX) <= 500 \
                or not (np.all(np.isfinite(eX)) and np.all(np.isfinite(eA))):
            raise ValueError(f"extra: X (n, {X.shape[1]}, 28) and A (n, {X.shape[1]}, 4), 1 <= n <= 500, finite")
        X, A = np.concatenate([X, eX]), np.concatenate([A, np.clip(eA, -1, 1)])
    sel = np.arange(len(X)) if req.get("demos") is None else np.asarray(req["demos"], int)
    if sel.size == 0 or sel.min() < 0 or sel.max() >= len(X):
        raise ValueError("demos must be indices into dataset.npz (extra demos follow it: index 500 = first extra)")
    w = req.get("weights")
    if w is not None:
        w = np.asarray(w, float)
        if w.shape != sel.shape or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("weights must be one non-negative number per entry of demos")
        rng = np.random.default_rng(CFG["session_index"] * 100 + STATE["trains"])
        sel = rng.choice(sel, size=len(sel), p=w / w.sum())        # weighted resample of the demo list
    init = req.get("init", "given")
    if init == "given":
        m0, steps_max, lr = policy.load(CFG["policy"]), RECIPE["finetune_steps"], RECIPE["finetune_lr"]
    elif init == "scratch":
        m0, steps_max, lr = None, RECIPE["steps"], RECIPE["lr"]
    else:
        m0, steps_max, lr = weights_from(init["weights_npz_b64"]), RECIPE["finetune_steps"], RECIPE["finetune_lr"]
    steps = int(req.get("steps", steps_max))
    if not 1 <= steps <= steps_max:
        raise ValueError(f"steps must be between 1 and {steps_max} for this init")
    losses = []
    m = policy.train(X[sel], A[sel], seed=CFG["session_index"] * 100 + STATE["trains"], steps=steps,
                     lr=lr, init=m0, threads=4, log=losses.append)
    b64 = to_b64(m)
    return dict(weights_npz_b64=b64, log=losses[-3:], n_demos=int(len(sel)), steps=steps,
                weights_sha256=hashlib.sha256(base64.b64decode(b64)).hexdigest())


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        digest = hashlib.sha256(raw).hexdigest()
        with LOCK:                                  # one request at a time (training uses the CPU)
            if self.path == "/status":
                return self.reply(200, dict(rollouts_used=STATE["rollouts"], rollout_budget=CFG["rollout_budget"],
                                            trains_used=STATE["trains"], train_budget=CFG["train_budget"]))
            kind = {"/rollout": "rollouts", "/train": "trains"}.get(self.path)
            if kind is None:
                return self.reply(404, dict(error="endpoints: /rollout, /train, /status"))
            budget = CFG["rollout_budget" if kind == "rollouts" else "train_budget"]
            if STATE[kind] >= budget:
                log(dict(endpoint=self.path, sha256=digest, refused="budget exhausted"))
                return self.reply(403, dict(error=f"{kind} budget of {budget} exhausted"))
            try:
                req = json.loads(raw)
                out = do_rollout(req) if kind == "rollouts" else do_train(req)
            except Exception as e:                 # invalid requests use no budget
                log(dict(endpoint=self.path, sha256=digest, invalid=str(e)[:300]))
                return self.reply(400, dict(error=str(e)[:500]))
            STATE[kind] += 1
            out["budget_left"] = budget - STATE[kind]
            rec = dict(endpoint=self.path, sha256=digest, k=STATE[kind])
            if kind == "rollouts":
                rec.update(success=out["success"], object_offset=req.get("object_offset"), goal_offset=req.get("goal_offset"))
            else:
                rec.update(n_demos=out["n_demos"], steps=out["steps"],
                           init=req.get("init") if isinstance(req.get("init"), str) else "submitted",
                           relabel=bool(req.get("relabel_npz_b64")), reweighted=req.get("weights") is not None,
                           extra=bool(req.get("extra_npz_b64")), out_sha256=out["weights_sha256"],
                           out_params_sha=params_sha(out["weights_npz_b64"]),
                           demos_sha=hashlib.sha256(json.dumps(req.get("demos")).encode()).hexdigest()[:16])
            log(rec)
            return self.reply(200, out)


if __name__ == "__main__":
    with np.load(CFG["dataset"]) as z:
        DATA = dict(X=z["X"], A=z["A"])
    with np.load(CFG["policy"]) as z:
        GIVEN = {k: z[k] for k in z.files}
    sim.env()
    log(dict(event="start", rollout_budget=CFG["rollout_budget"], train_budget=CFG["train_budget"], port=CFG["port"]))
    ThreadingHTTPServer(("127.0.0.1", CFG["port"]), H).serve_forever()
