"""Two-scale Lorenz-96: system-size scaling of the subgrid tendency U_k = -(h_X c/b) sum_j Y_jk.

K=8, F=20, h=1, b=c=10.  J varied.  "Fixed mean forcing" scaling:
   X eq coupling  h_X = h * J0 / J   (J0 = 32, the Wilks 2005 reference)
   Y eq coupling  h_Y = h            (each fast 'molecule' feels the same slow forcing)
so E[U|X] is J-independent if the Y statistics are; prediction Var[U|X] ~ 1/J (van Kampen, Omega = J/l_c).
Second scaling ("naive", h fixed in both eqs) also reported: Var[U|X] ~ J, mean ~ J.
"""
import sys
import numpy as np

K, F, h, b, c = 8, 20.0, 1.0, 10.0, 10.0
J0 = 32


def run(J, hX, hY, T=30.0, spin=5.0, dt=0.001, sample=0.01, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, K) + 5
    Y = rng.normal(0, 0.1, K * J)

    def rhs(X, Y):
        Ysum = Y.reshape(K, J).sum(1)
        dX = -np.roll(X, 1) * (np.roll(X, 2) - np.roll(X, -1)) - X + F - (hX * c / b) * Ysum
        Xr = np.repeat(X, J)
        dY = -c * b * np.roll(Y, -1) * (np.roll(Y, -2) - np.roll(Y, 1)) - c * Y + (hY * c / b) * Xr
        return dX, dY

    nsteps = int((T + spin) / dt)
    every = int(sample / dt)
    nspin = int(spin / dt)
    Xs, Us = [], []
    for n in range(nsteps):
        k1x, k1y = rhs(X, Y)
        k2x, k2y = rhs(X + 0.5 * dt * k1x, Y + 0.5 * dt * k1y)
        k3x, k3y = rhs(X + 0.5 * dt * k2x, Y + 0.5 * dt * k2y)
        k4x, k4y = rhs(X + dt * k3x, Y + dt * k3y)
        X = X + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        Y = Y + dt / 6 * (k1y + 2 * k2y + 2 * k3y + k4y)
        if n >= nspin and n % every == 0:
            Xs.append(X.copy())
            Us.append(-(hX * c / b) * Y.reshape(K, J).sum(1))
    return np.array(Xs).ravel(), np.array(Us).ravel()


def cond_stats(X, U, nb=20):
    # residual variance about a cubic fit (Wilks-style deterministic closure)  + binned variance
    p = np.polyfit(X, U, 3)
    res = U - np.polyval(p, X)
    edges = np.quantile(X, np.linspace(0, 1, nb + 1))
    idx = np.clip(np.digitize(X, edges) - 1, 0, nb - 1)
    bv = np.array([U[idx == i].var() for i in range(nb)])
    return p, res.var(), bv.mean(), U.mean(), U.var()


if __name__ == "__main__":
    mode = sys.argv[1]
    T = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    for J in [8, 16, 32, 64, 128]:
        hX, hY = (h * J0 / J, h) if mode == "scaled" else (h, h)
        X, U = run(J, hX, hY, T=T)
        p, rv, bv, um, uv = cond_stats(X, U)
        # mean closure evaluated at X = 0, 5, 10
        mc = np.polyval(p, [0, 5, 10])
        print(f"{mode} J={J:4d} hX={hX:.3f}  <X>={X.mean():6.3f} var(X)={X.var():6.2f}  <U>={um:7.3f} "
              f"var(U)={uv:8.4f}  Var[U|X]cubic={rv:8.5f}  Var[U|X]bins={bv:8.5f}  "
              f"E[U|X=0,5,10]={mc.round(2)}", flush=True)
