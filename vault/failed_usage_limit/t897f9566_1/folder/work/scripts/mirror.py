import numpy as np, json
d = np.load('../task/dataset.npz'); X, A = d['X'].astype(np.float64), d['A'].astype(np.float64)
cx = X[:, 0, 0].mean()
Xm, Am = X.copy(), A.copy()
for i in (0, 3, 25): Xm[..., i] = 2 * cx - X[..., i]          # absolute x positions
for i in (6, 14, 20): Xm[..., i] = -X[..., i]                 # x of relative pos / block lin vel / gripper vel
for i in (12, 13, 18, 19): Xm[..., i] = -X[..., i]            # rotation (ry, rz) and ang vel (wy, wz) under x-reflection
Am[..., 0] = -A[..., 0]
print('cx', cx, 'mirrored obj x range', Xm[:, 0, 3].min(), Xm[:, 0, 3].max())
np.savez('extra_mirror.npz', X=Xm.astype(np.float32), A=Am.astype(np.float32))
json.dump(list(range(1000)), open('demos_all1000.json', 'w'))
