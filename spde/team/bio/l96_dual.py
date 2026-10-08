"""Dual-reporter decomposition of the L96 subgrid tendency, scaled coupling h_X = h*J0/J.
Reporter 1 = first half of the fast ring in sector k, reporter 2 = second half.
U = U1 + U2.  Residuals e_i of U_i after regression on resolved predictors.
  intrinsic Var(U|.) = Var(e1 - e2),   extrinsic = 4 Cov(e1, e2)     (Elowitz/Swain, exchangeable halves)
Predictors: P0 = cubic in X_k ; P1 = P0 + X_{k-1}, X_{k+1}, X_{k-2} (local resolved state) ;
            P2 = P1 + lagged X_k at 0.05, 0.1, 0.2, 0.4 time units (resolved history).
"""
import sys
import numpy as np

K, F, h, b, c, J0 = 8, 20.0, 1.0, 10.0, 10.0, 32


def run(J, T, spin=5.0, dt=0.001, sample=0.01, seed=0):
    hX, hY = h * J0 / J, h
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, K) + 5
    Y = rng.normal(0, 0.1, K * J)

    def rhs(X, Y):
        Ysum = Y.reshape(K, J).sum(1)
        dX = -np.roll(X, 1) * (np.roll(X, 2) - np.roll(X, -1)) - X + F - (hX * c / b) * Ysum
        dY = -c * b * np.roll(Y, -1) * (np.roll(Y, -2) - np.roll(Y, 1)) - c * Y + (hY * c / b) * np.repeat(X, J)
        return dX, dY

    nsteps, every, nspin = int((T + spin) / dt), int(sample / dt), int(spin / dt)
    Xs, U1s, U2s = [], [], []
    for n in range(nsteps):
        k1x, k1y = rhs(X, Y)
        k2x, k2y = rhs(X + 0.5 * dt * k1x, Y + 0.5 * dt * k1y)
        k3x, k3y = rhs(X + 0.5 * dt * k2x, Y + 0.5 * dt * k2y)
        k4x, k4y = rhs(X + dt * k3x, Y + dt * k3y)
        X = X + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        Y = Y + dt / 6 * (k1y + 2 * k2y + 2 * k3y + k4y)
        if n >= nspin and n % every == 0:
            Yr = Y.reshape(K, J)
            Xs.append(X.copy())
            U1s.append(-(hX * c / b) * Yr[:, : J // 2].sum(1))
            U2s.append(-(hX * c / b) * Yr[:, J // 2:].sum(1))
    return np.array(Xs), np.array(U1s), np.array(U2s)


def design(X, level):
    lags = [5, 10, 20, 40]          # samples (0.01 each)
    L = max(lags)
    Xk = X[L:]
    cols = [np.ones_like(Xk), Xk, Xk**2, Xk**3]
    if level >= 1:
        for s in (1, -1, 2):
            Xn = np.roll(X, s, axis=1)[L:]
            cols += [Xn, Xn * Xk]
    if level >= 2:
        for l in lags:
            Xl = X[L - l: X.shape[0] - l]
            cols += [Xl, Xl * Xk]
    return np.stack([cc.ravel() for cc in cols], 1), L


if __name__ == "__main__":
    T = float(sys.argv[1])
    Js = [int(j) for j in sys.argv[2:]]
    for J in Js:
        X, U1, U2 = run(J, T)
        out = [f"J={J:4d}"]
        for lev in (0, 1, 2):
            A, L = design(X, lev)
            y1, y2 = U1[L:].ravel(), U2[L:].ravel()
            e1 = y1 - A @ np.linalg.lstsq(A, y1, rcond=None)[0]
            e2 = y2 - A @ np.linalg.lstsq(A, y2, rcond=None)[0]
            tot = np.var(e1 + e2)
            intr = np.var(e1 - e2)
            ext = 4 * np.mean((e1 - e1.mean()) * (e2 - e2.mean()))
            out.append(f"P{lev}: tot={tot:.3f} int={intr:.3f} ext={ext:.3f}")
        print("  ".join(out), flush=True)
