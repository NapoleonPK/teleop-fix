import numpy as np
CX = 1.3419348001480103
def mirror_obs(X):
    X = np.array(X, np.float64).copy()
    for i in (0, 3, 25): X[..., i] = 2 * CX - X[..., i]
    for i in (6, 14, 20, 12, 13, 18, 19): X[..., i] = -X[..., i]
    return X
