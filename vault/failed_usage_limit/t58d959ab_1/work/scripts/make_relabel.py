import numpy as np
d = np.load('../task/dataset.npz'); X, A = d['X'], d['A']
dp = X[:, 1:, 0:3] - X[:, :-1, 0:3]
g = np.array([0.0312, 0.0317, 0.0307])
B = np.empty_like(A)
B[:, :98] = A[:, 2:]                                   # true command executed at t was logged at t+2
B[:, 98, :3] = np.clip(dp[:, 98] / g, -1, 1)           # last executed command, recovered from motion
B[:, 98, 3] = A[:, 99, 3]
B[:, 99] = B[:, 98]
np.savez('relabel_shift2.npz', idx=np.arange(500), A=B.astype(np.float32))
# check consistency
print(np.abs(B[:, :99, :3] * g - dp).max())
