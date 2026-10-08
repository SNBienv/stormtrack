"""Fast subsystem with X frozen: static response d<U_0>/dX_0 by direct perturbation."""
import numpy as np, time, sys
sys.path.insert(0, '.')
from l96 import *
rng = np.random.default_rng(2)
x0s = [float(v) for v in sys.argv[2].split(",")]
d = float(sys.argv[3])
rows = []
for x0 in x0s:
    for dx in (0.0, +d, -d):
        X = np.full(K, x0); X[0] += dx; rows.append(X)
X = np.array(rows); M = len(rows)
Y = 0.5 * rng.standard_normal((M, K*J)) + 1.0
T = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
for _ in range(int(3/DT)):
    X, Y = rk4(X, Y, Xfrozen=True)
nsave = int(T/0.005)
Us = np.empty((nsave, M, K), np.float32)
Yun = np.empty((nsave, len(x0s), K*J), np.float32)
t0 = time.time()
for i in range(nsave):
    for _ in range(5):
        X, Y = rk4(X, Y, Xfrozen=True)
    Us[i] = U_of(Y); Yun[i] = Y[0::3]
print('time', time.time()-t0)
np.savez(sys.argv[4], U=Us, Yun=Yun, x0s=x0s, d=d)
