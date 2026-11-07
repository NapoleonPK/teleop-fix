"""Client for this session's robot services (rollouts and training). Python standard library only.

  python task/svc.py status
  python task/svc.py rollout WEIGHTS.npz [--object DX DY --goal DX DY] [--out FILE.json]
      One 100-step closed-loop episode of the weights (same arrays/shapes as task/policy.npz) in the simulator.
      Offsets are metres from the centre of the block start square (x toward the far side of the table, y to the
      robot's left); without them the start is random over the whole table. Uses 1 rollout of the budget.
  python task/svc.py train --out NEW.npz [--init given|scratch|START.npz] [--demos IDX.json] [--weights W.json]
                           [--relabel RELABEL.npz] [--extra EXTRA.npz] [--steps N]
      Trains with the fixed recipe on demos of task/dataset.npz and writes the trained weights to NEW.npz.
      --init given (default): fine-tune task/policy.npz (at most 2000 steps, lr 1e-4)
      --init scratch: train a new network (at most 8000 steps, lr 3e-4; the recipe that produced task/policy.npz)
      --init START.npz: fine-tune your own weights (at most 2000 steps)
      --demos IDX.json: a JSON list of demo indices (repeats allowed); default all demos
      --weights W.json: a JSON list of sampling weights, one per entry of --demos (the list is resampled)
      --relabel RELABEL.npz: arrays idx (n,) and A (n, 100, 4): replacement actions for those demos
      --extra EXTRA.npz: arrays X (n, 100, 28) and A (n, 100, 4), n <= 500: extra demos you made (augmented or
          synthetic), appended after the dataset (index 500 = first extra demo; --demos may refer to them)
      Uses 1 training job of the budget and takes a few minutes; the call waits for it.
"""
import base64, json, sys, urllib.error, urllib.request

PORT = 49829


def post(path, body, timeout=3600):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return json.load(e)


def b64(path):
    return base64.b64encode(open(path, "rb").read()).decode()


def opts(a, flags):
    out, i = {}, 0
    while i < len(a):
        if a[i] not in flags:
            sys.exit(f"unknown argument {a[i]}")
        n = flags[a[i]]
        out[a[i]] = a[i + 1:i + 1 + n] if n > 1 else a[i + 1]
        i += 1 + n
    return out


def main(a):
    if not a or a[0] not in ("status", "rollout", "train"):
        sys.exit(__doc__)
    if a[0] == "status":
        print(json.dumps(post("/status", {}))); return
    if a[0] == "rollout":
        o = opts(a[2:], {"--object": 2, "--goal": 2, "--out": 1})
        body = {"weights_npz_b64": b64(a[1])}
        if "--object" in o:
            body["object_offset"] = [float(v) for v in o["--object"]]
        if "--goal" in o:
            body["goal_offset"] = [float(v) for v in o["--goal"]]
        r = post("/rollout", body)
        if "--out" in o:
            json.dump(r, open(o["--out"], "w"))
            print(json.dumps({k: v for k, v in r.items() if k in ("success", "final_object_to_goal_m", "budget_left", "error")}))
        else:
            print(json.dumps(r))
        return
    o = opts(a[1:], {"--out": 1, "--init": 1, "--demos": 1, "--weights": 1, "--relabel": 1, "--extra": 1, "--steps": 1})
    if "--out" not in o:
        sys.exit("train needs --out NEW.npz")
    init = o.get("--init", "given")
    body = {"init": init if init in ("given", "scratch") else {"weights_npz_b64": b64(init)},
            "demos": json.load(open(o["--demos"])) if "--demos" in o else None,
            "weights": json.load(open(o["--weights"])) if "--weights" in o else None,
            "relabel_npz_b64": b64(o["--relabel"]) if "--relabel" in o else None,
            "extra_npz_b64": b64(o["--extra"]) if "--extra" in o else None}
    if "--steps" in o:
        body["steps"] = int(o["--steps"])
    r = post("/train", body)
    if "weights_npz_b64" in r:
        open(o["--out"], "wb").write(base64.b64decode(r.pop("weights_npz_b64")))
        r["written"] = o["--out"]
    print(json.dumps(r))


if __name__ == "__main__":
    main(sys.argv[1:])
