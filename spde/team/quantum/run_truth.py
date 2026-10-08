"""Coupled truth run: M parallel trajectories, save X, U, B every 5 steps (0.005 MTU)."""
import numpy as np, time, sys
sys.path.insert(0, '.')
from l96 import *
rng = np.random.default_rng(1)
M = 8
T = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
X = F/4 + rng.standard_normal((M, K))
Y = 0.1 * rng.standard_normal((M, K*J))
t0 = time.time()
for _ in range(int(5/DT)):
    X, Y = rk4(X, Y)
nsave = int(T/0.005)
Xs = np.empty((nsave, M, K)); Us = np.empty_like(Xs); Bs = np.empty_like(Xs)
Ss = np.empty_like(Xs); varY = np.empty_like(Xs)
for i in range(nsave):
    for _ in range(5):
        X, Y = rk4(X, Y)
    Xs[i] = X; Us[i] = U_of(Y); Bs[i] = edge_flux(Y)
    varY[i] = Y.reshape(M, K, J).var(-1)
print('time', time.time()-t0)
np.savez('truth.npz', X=Xs, U=Us, B=Bs, varY=varY, Xend=X, Yend=Y)
