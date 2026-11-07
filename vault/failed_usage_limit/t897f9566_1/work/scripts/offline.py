import sys, numpy as np
sys.path.insert(0, '../task'); import policy_np as P
d = np.load('../task/dataset.npz'); X, A = d['X'], d['A']; e = np.load('extra_mirror.npz')
rng = np.random.default_rng(0)
def l1(w, X, A, n=300):
    idx = [(rng.integers(len(X)), rng.integers(0, 90)) for _ in range(n)]
    xs = np.array([X[i, t] for i, t in idx]); ys = np.array([A[i, t:t+10] for i, t in idx])
    return np.abs(P.forward(w, xs) - ys).mean()
for f in sys.argv[1:]:
    w = P.load(f); print(f, 'L1 orig', l1(w, X, A).round(4), 'L1 mirrored', l1(w, e['X'], e['A']).round(4))
