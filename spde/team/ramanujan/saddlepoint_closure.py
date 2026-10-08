#!/usr/bin/env python3
"""
Family-free closure for linearly mixed Weibull wind fields: saddlepoint / Lugannani-Rice.

Setting.  S(t) = P S(0) with S_j(0) ~ Weibull(k_j, lam_j) independent, P >= 0 (advection + diffusion + scaling).
The CGF of S_i is EXACT:      K_i(s) = sum_j K_j(P_ij s),   K_j = log E exp(s X_j).
We evaluate the Weibull CGF and its derivatives on a fixed log-grid (spectrally accurate trapezoid rule):

    M(s) = int_R exp( phi(y) ) dy,   phi(y) = s lam e^y + log k + k y - e^{k y},   (x = lam e^y)

and the tilted moments of x under exp(phi) give K', K'', K''', K''''.  Then Lugannani-Rice (1980):

    K_i'(s^) = q,  w = sgn(s^) sqrt(2(s^ q - K_i(s^))),  u = s^ sqrt(K_i''(s^))
    P(S_i > q) ~= 1 - Phi(w) + phi(w) (1/u - 1/w)                                    [LR]
    + phi(w) [ (1/u)(l4/8 - 5 l3^2/24) - l3/(2u^2) - 1/u^3 + 1/w^3 ]  (Daniels 1987)  [LR2]

Quantiles: solve LR(s^) = 1-p for s^ by safeguarded Newton (dP/ds^ ~= -phi(w) sqrt(K'')), then q = K'(s^).

Reference ("exact"):  Gil-Pelaez inversion of the exact characteristic function prod_j phi_j(P_ij t),
with the Weibull CF computed on a rotated contour x = lam e^{y + i beta}, beta = pi/(4k) (no oscillation problem).
Also: 100,000-member Monte Carlo, and the exp-Weibull 3-cumulant closure for comparison.

Run:  python3 saddlepoint_closure.py          (prints all tables; ~1-2 min, dominated by the MC + exact reference)
"""
import numpy as np, time, sys
from scipy.special import ndtr, ndtri, gamma as Gfun, gammaln
from scipy.optimize import brentq, least_squares

SQ2PI = np.sqrt(2 * np.pi)

# ----------------------------------------------------------------------------------------------
# 1. Weibull CGF and derivatives (real s), vectorised over arrays of (k, lam, s)
# ----------------------------------------------------------------------------------------------
def _peak(k, sig):
    """Mode y* of phi(y) = sig e^y + log k + k y - e^{ky} (Newton from the right; phi concave there)."""
    y = np.maximum(np.log(np.maximum(sig, k) * 2 / k) / np.maximum(k - 1, 1e-3), 0.0) + 1.0
    for _ in range(60):
        g = sig * np.exp(y) + k - k * np.exp(k * y)
        H = sig * np.exp(y) - k * k * np.exp(k * y)
        y = y - np.clip(g / H, -2, 2)
    return y, -(sig * np.exp(y) - k * k * np.exp(k * y))

def _ygrid(k, sig, h):
    """Fixed grid covering all rows; step refined if some saddle is very sharp (very large lam*s)."""
    kmin = float(np.min(k))
    ys, curv = _peak(k, sig)
    h = min(h, 0.5 / np.sqrt(curv.max()))
    ylo = min(-32.0 / kmin, ys.min() - 32.0 / kmin)
    yhi = ys.max() + max(8.0 / np.sqrt(curv.min()), 3.0)
    return np.arange(ylo, yhi + h, h), h

def weibull_cgf(k, lam, s, h=0.08, nder=4):
    """Return (K, K1, K2, K3, K4) of Weibull(k, lam) CGF at real s (arrays broadcast, k > 1 for s > 0)."""
    k, lam, s = (np.asarray(v, float) for v in np.broadcast_arrays(k, lam, s))
    shp = k.shape
    k, lam, s = k.ravel(), lam.ravel(), s.ravel()
    sig = s * lam
    y, h = _ygrid(k, sig, h)
    ey = np.exp(y)
    phi = sig[:, None] * ey[None, :] + np.log(k)[:, None] + k[:, None] * y[None, :] - np.exp(k[:, None] * y[None, :])
    pm = phi.max(axis=1, keepdims=True)
    E = np.exp(phi - pm)
    Z = E.sum(axis=1)
    K = np.log(h * Z) + pm[:, 0]
    p = E / Z[:, None]
    x = lam[:, None] * ey[None, :]
    m1 = (p * x).sum(1)
    out = [K, m1]
    if nder >= 2:
        d = x - m1[:, None]
        d2 = d * d
        c2 = (p * d2).sum(1); out.append(c2)
        if nder >= 3:
            c3 = (p * d2 * d).sum(1); out.append(c3)
            if nder >= 4:
                c4 = (p * d2 * d2).sum(1); out.append(c4 - 3 * c2 ** 2)
    return [o.reshape(shp) for o in out]

def weibull_mgf_series(k, lam, s, nmax=200):
    """M(s) = sum_n (lam s)^n Gamma(1+n/k)/n!  -- entire for k>1; fine for |lam s| <~ 5 (cancellation for s<0)."""
    n = np.arange(nmax)
    lt = n * np.log(abs(lam * s) + 1e-300) + gammaln(1 + n / k) - gammaln(n + 1)
    sgn = np.sign(lam * s) ** n
    return np.sum(sgn * np.exp(lt))

def weibull_cf(k, lam, tau, h=0.04):
    """Characteristic function E exp(i tau X), tau >= 0, on rotated contour x = lam e^{y+i beta}."""
    k, lam, tau = (np.asarray(v, float) for v in np.broadcast_arrays(k, lam, tau))
    shp = k.shape; k, lam, tau = k.ravel(), lam.ravel(), tau.ravel()
    beta = np.pi / (4 * k)
    y = np.arange(-38.0 / k.min(), 4.0 + h, h)
    z = y[None, :] + 1j * beta[:, None]
    ez = np.exp(z)
    phi = 1j * (tau * lam)[:, None] * ez + np.log(k)[:, None] + k[:, None] * z - np.exp(k[:, None] * z)
    return (h * np.exp(phi).sum(1)).reshape(shp)

# ----------------------------------------------------------------------------------------------
# 2. CGF of mixed variables  S_i = sum_j W_ij X_j  for many targets at once (sparse pairs)
# ----------------------------------------------------------------------------------------------
class MixedCGF:
    def __init__(self, W, k0, lam0, tol=1e-13):
        self.nt = W.shape[0]
        ti, sj = np.nonzero(W > tol * W.max(axis=1, keepdims=True))
        self.ti, self.w = ti, W[ti, sj]
        self.k, self.lam = k0[sj], lam0[sj]
        # exact cumulants kappa_1..4 from Weibull raw moments
        g = np.array([Gfun(1 + r / self.k) for r in range(5)]) * self.lam[None, :] ** np.arange(5)[:, None]
        m1, m2, m3, m4 = g[1], g[2], g[3], g[4]
        kap = [m1, m2 - m1**2, m3 - 3*m2*m1 + 2*m1**3, m4 - 4*m3*m1 - 3*m2**2 + 12*m2*m1**2 - 6*m1**4]
        self.kappa = np.array([np.bincount(ti, self.w ** (n + 1) * kap[n], minlength=self.nt) for n in range(4)])

    def cgf(self, s, nder=4):
        """s: array (nt, m) of saddle values -> list of (nt, m) arrays K, K1..K4."""
        s = np.atleast_2d(s)
        m = s.shape[1]
        sp = s[self.ti, :] * self.w[:, None]                         # (npairs, m)
        res = weibull_cgf(np.repeat(self.k[:, None], m, 1), np.repeat(self.lam[:, None], m, 1), sp, nder=nder)
        out = []
        for d, r in enumerate(res):
            v = r * self.w[:, None] ** d
            out.append(np.stack([np.bincount(self.ti, v[:, c], minlength=self.nt) for c in range(m)], 1))
        return out

    def cf(self, t):
        """Characteristic function of each target at t (nt, nt_nodes) via product of Weibull CFs."""
        t = np.atleast_2d(t)
        m = t.shape[1]
        tau = t[self.ti, :] * self.w[:, None]
        phi = weibull_cf(np.repeat(self.k[:, None], m, 1), np.repeat(self.lam[:, None], m, 1), tau)
        logphi = np.log(phi)
        out = np.zeros((self.nt, m), complex)
        np.add.at(out, self.ti, logphi)
        return np.exp(out)

# ----------------------------------------------------------------------------------------------
# 3. Lugannani-Rice tail and quantile inversion
# ----------------------------------------------------------------------------------------------
def lr_tail(s, q, K, K2, K3=None, K4=None, order=1):
    """P(S>q) from saddle s (K'(s)=q). Handles s ~ 0 by the limit formula."""
    w2 = np.maximum(2 * (s * q - K), 0.0)
    w = np.sign(s) * np.sqrt(w2)
    u = s * np.sqrt(K2)
    small = np.abs(w) < 1e-5
    ws = np.where(small, 1.0, w); us = np.where(small, 1.0, u)
    P = ndtr(-w) + np.exp(-w2 / 2) / SQ2PI * (1 / us - 1 / ws)
    if order == 2:
        l3 = K3 / K2 ** 1.5; l4 = K4 / K2 ** 2
        corr = np.exp(-w2 / 2) / SQ2PI * ((1 / us) * (l4 / 8 - 5 * l3 ** 2 / 24) - l3 / (2 * us ** 2) - 1 / us ** 3 + 1 / ws ** 3)
        # the 1/u^3 - 1/w^3 - l3/(2u^2) terms cancel catastrophically as s^ -> 0: use them only for |w| > 0.5
        P = P + np.where(np.abs(w) > 0.5, corr, 0.0)
    lim = 0.5 - (K3 / K2 ** 1.5 if K3 is not None else 0.0) / (6 * SQ2PI)   # s^ -> 0 limit
    return np.where(small, lim, P), w

def lr_quantiles(mc: MixedCGF, probs, order=1, maxit=30, qtol=1e-10):
    """Quantiles (nt, len(probs)).  Newton on the normal score z(s^) = -Phi^{-1}(P_LR(s^)), which is ~linear in s^:
    dz/ds^ ~= dw/ds^ = s^ K''(s^)/w  (-> sqrt(K'') as s^ -> 0).  Converges in ~4-6 iterations."""
    probs = np.asarray(probs, float)
    nt, m = mc.nt, len(probs)
    sd = np.sqrt(mc.kappa[1])[:, None]
    zt = ndtri(probs)[None, :]
    s = zt / sd * np.ones((nt, 1))
    for it in range(maxit):
        K, K1, K2, K3, K4 = mc.cgf(s, nder=4 if order == 2 else 3) + ([None] if order == 1 else [])
        P, w = lr_tail(s, K1, K, K2, K3, K4, order)
        z = -ndtri(np.clip(P, 1e-300, 1 - 1e-16))
        dz = np.where(np.abs(w) > 1e-6, s * K2 / np.where(np.abs(w) > 1e-6, w, 1.0), np.sqrt(K2))
        step = np.clip((zt - z) / dz, -1.0 / sd, 1.0 / sd)
        s = s + step
        if np.max(np.abs(step) * K2 / K1) < qtol:        # |dq/q| < qtol
            break
    K, K1 = mc.cgf(s, nder=1)
    return K1, s, it + 1

# ----------------------------------------------------------------------------------------------
# 4. Exact reference by Gil-Pelaez inversion of the exact CF
# ----------------------------------------------------------------------------------------------
def exact_quantiles(mc: MixedCGF, probs, eps=1e-15):
    mu, sd = mc.kappa[0], np.sqrt(mc.kappa[1])
    out = np.zeros((mc.nt, len(probs)))
    for i in range(mc.nt):
        sub = MixedCGF.__new__(MixedCGF)
        sel = mc.ti == i
        sub.nt, sub.ti, sub.w, sub.k, sub.lam = 1, np.zeros(sel.sum(), int), mc.w[sel], mc.k[sel], mc.lam[sel]
        ht = np.pi / (mu[i] + 20 * sd[i])                     # aliasing period 2pi/ht >> support
        t, ph = [], []
        n0 = 0
        while True:
            tt = (np.arange(n0, n0 + 64) + 0.5) * ht
            pp = sub.cf(tt[None, :])[0]
            t.append(tt); ph.append(pp); n0 += 64
            if np.max(np.abs(pp)) < eps or n0 > 20000:
                break
        t = np.concatenate(t); ph = np.concatenate(ph)
        F = lambda q: 0.5 - ht / np.pi * np.sum(np.imag(np.exp(-1j * t * q) * ph) / t)
        for c, p in enumerate(probs):
            a, b = mu[i] - 6 * sd[i], mu[i] + 12 * sd[i]
            out[i, c] = brentq(lambda q: F(q) - p, max(a, 0.0), b, xtol=1e-12, rtol=1e-13)
    return out

# ----------------------------------------------------------------------------------------------
# 5. Exponentiated-Weibull 3-cumulant closure (for comparison)
# ----------------------------------------------------------------------------------------------
_Z = np.arange(-60.0, 4.5, 0.03)
def ew_raw_moments(k, a, rmax=3):
    """E[Y^r], Y ~ expWeibull(k, lam=1, a): a int exp((r/k+1) z - e^z) (1-e^{-e^z})^{a-1} dz (trapezoid, spectral)."""
    u = np.exp(_Z)
    base = -u + (a - 1) * np.log(-np.expm1(-u))
    return np.array([a * 0.03 * np.sum(np.exp((r / k + 1) * _Z + base)) for r in range(1, rmax + 1)])

def ew_fit_quantiles(kappa, probs):
    out = np.zeros((kappa.shape[1], len(probs))); params = []
    x0 = np.array([np.log(2.0), 0.0])
    for i in range(kappa.shape[1]):
        k1, k2, k3 = kappa[0, i], kappa[1, i], kappa[2, i]
        cv, sk = np.sqrt(k2) / k1, k3 / k2 ** 1.5
        def res(x):
            k, a = np.exp(x)
            m1, m2, m3 = ew_raw_moments(k, a)
            v = m2 - m1 ** 2; c3 = m3 - 3 * m2 * m1 + 2 * m1 ** 3
            return [np.sqrt(v) / m1 - cv, c3 / v ** 1.5 - sk]
        sol = least_squares(res, x0, xtol=1e-14, ftol=1e-14, gtol=1e-14)
        x0 = sol.x
        k, a = np.exp(sol.x)
        lam = k1 / ew_raw_moments(k, a)[0]
        params.append((k, lam, a, np.max(np.abs(sol.fun))))
        out[i] = lam * (-np.log1p(-np.asarray(probs) ** (1 / a))) ** (1 / k)
    return out, np.array(params)

# ----------------------------------------------------------------------------------------------
# 6. The test problem
# ----------------------------------------------------------------------------------------------
N, NU = 200, 0.25
X = np.arange(N) / N
K0 = 2 + 0.3 * np.sin(2 * np.pi * X)
L0 = 7 + 2 * np.cos(2 * np.pi * X)
AX = 1 + 0.004 * np.sin(4 * np.pi * X)

def step(S):
    S = np.roll(S, 1, axis=0)
    S = S + NU * (np.roll(S, -1, axis=0) - 2 * S + np.roll(S, 1, axis=0))
    return S * (AX[:, None] if S.ndim == 2 else AX)

def self_tests():
    import mpmath as mp
    print("== self-tests: Weibull CGF / CF evaluation ==")
    worst = 0
    for k, lam, s in [(1.7, 9.0, 0.8), (2.0, 7.0, -1.3), (2.3, 5.0, 2.5), (2.0, 7.0, 0.01), (1.2, 3.0, 4.0), (2.0, 7.0, 10.0), (1.2, 3.0, -20.0)]:
        mp.mp.dps = 30
        f = lambda y: mp.exp(s * lam * mp.exp(y) + mp.log(k) + k * y - mp.exp(k * y))
        Mq = mp.quad(f, mp.linspace(-60 / k, 25, 200))
        K = weibull_cgf(k, lam, s)[0]
        worst = max(worst, abs(float(mp.log(Mq)) - float(K)))
    print(f"  K(s) quadrature-grid vs mpmath (incl. lam*s = 70, 12, -60): worst abs err {worst:.1e}")
    print(f"  K(s) vs power series (k=2,lam=7,s=0.3): {abs(weibull_cgf(2.0,7.0,0.3)[0]-np.log(weibull_mgf_series(2.0,7.0,0.3))):.1e}")
    k, lam = 2.0, 7.0
    g = [Gfun(1 + r / k) * lam ** r for r in range(5)]
    c = weibull_cgf(k, lam, 0.0)
    ex = [g[1], g[2] - g[1] ** 2, g[3] - 3 * g[2] * g[1] + 2 * g[1] ** 3]
    print(f"  K',K'',K''' at 0 vs exact cumulants: rel errs {[f'{abs(c[i+1]/ex[i]-1):.1e}' for i in range(3)]}")
    worst = 0
    for k, lam, tau in [(1.7, 9.0, 0.3), (2.0, 7.0, 1.5), (2.3, 5.0, 6.0), (2.0, 7.0, 25.0)]:
        re = mp.quad(lambda x: mp.cos(tau * x) * (k / lam) * (x / lam) ** (k - 1) * mp.exp(-(x / lam) ** k), mp.linspace(0, 8 * lam, 200))
        im = mp.quad(lambda x: mp.sin(tau * x) * (k / lam) * (x / lam) ** (k - 1) * mp.exp(-(x / lam) ** k), mp.linspace(0, 8 * lam, 200))
        worst = max(worst, abs(complex(re, im) - weibull_cf(k, lam, tau)))
    print(f"  CF on rotated contour vs mpmath: worst abs err {worst:.1e}")
    # Gil-Pelaez + LR on a single Weibull (W = 1x1)
    mc = MixedCGF(np.ones((1, 1)), np.array([1.8]), np.array([6.0]))
    pr = [0.5, 0.9, 0.99, 0.999]
    qe = 6.0 * (-np.log1p(-np.array(pr))) ** (1 / 1.8)
    print(f"  Gil-Pelaez on single Weibull: max rel quantile err {np.max(np.abs(exact_quantiles(mc, pr)[0]/qe-1)):.1e}")
    print(f"  LR on single Weibull (worst case for LR): rel errs {np.round(100*(lr_quantiles(mc, pr)[0][0]/qe-1),3)} %")

def main(nmc=100_000, seed=12345):
    t_all = time.perf_counter()
    self_tests()
    probs = np.array([0.5, 0.9, 0.99, 0.999])
    steps_out = [5, 10, 20, 40]
    idx = np.arange(0, N, 5)
    # transfer matrix P^t
    P1 = step(np.eye(N))
    # Monte Carlo
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    S = L0[:, None] * (-np.log(rng.random((N, nmc)))) ** (1 / K0[:, None])
    mcq, mcf = {}, {}
    W = np.eye(N)
    Ws = {}
    for n in range(1, max(steps_out) + 1):
        S = step(S); W = P1 @ W
        if n in steps_out:
            mcq[n] = np.quantile(S[idx], probs, axis=1).T
            Ws[n] = W[idx]
    del S
    t_mc = time.perf_counter() - t0
    print(f"\nMonte Carlo: {nmc} members x {N} points x {max(steps_out)} steps: {t_mc:.1f} s")
    hdr = "  ".join(f"{int(p*1000)/10:>5}%" for p in probs)
    results = {}
    timing = {"LR": 0, "LR2": 0, "EW": 0, "exact": 0}
    for n in steps_out:
        mc = MixedCGF(Ws[n], K0, L0)
        t0 = time.perf_counter(); q_lr, shat, its = lr_quantiles(mc, probs, order=1); timing["LR"] += time.perf_counter() - t0
        t0 = time.perf_counter(); q_lr2, _, _ = lr_quantiles(mc, probs, order=2); timing["LR2"] += time.perf_counter() - t0
        t0 = time.perf_counter(); q_ew, par = ew_fit_quantiles(mc.kappa, probs); timing["EW"] += time.perf_counter() - t0
        t0 = time.perf_counter(); q_ex = exact_quantiles(mc, probs); timing["exact"] += time.perf_counter() - t0
        # MC sampling standard error of quantiles: sqrt(p(1-p)/n)/f(q), f from the exact CF (numerical derivative)
        results[n] = dict(lr=q_lr, lr2=q_lr2, ew=q_ew, ex=q_ex, mc=mcq[n], par=par, its=its, npairs=len(mc.ti) // len(idx))
    for ref in ["mc", "ex"]:
        print(f"\n=== % errors of quantiles vs {'Monte Carlo (1e5)' if ref=='mc' else 'EXACT (Gil-Pelaez inversion of exact CF)'}: mean (worst |.|) over {len(idx)} points ===")
        for meth in ["lr", "lr2", "ew"] + (["mc"] if ref == "ex" else []):
            name = {"lr": "Lugannani-Rice", "lr2": "LR + Daniels 2nd order", "ew": "exp-Weibull 3-cumulant", "mc": "MC 1e5 itself"}[meth]
            print(f"  {name}")
            print(f"    step  {hdr}")
            for n in steps_out:
                e = 100 * (results[n][meth] / results[n][ref] - 1)
                cells = "  ".join(f"{e[:,c].mean():+6.3f}({np.abs(e[:,c]).max():5.3f})" for c in range(len(probs)))
                print(f"    {n:4d}  {cells}")
    print("\nexp-Weibull fitted parameters (range over points):")
    for n in steps_out:
        par = results[n]["par"]
        print(f"   step {n:2d}: k in [{par[:,0].min():.2f},{par[:,0].max():.2f}], a in [{par[:,2].min():.2f},{par[:,2].max():.2f}], max fit residual {par[:,3].max():.1e}; sources/point {results[n]['npairs']}")
    print("\nCPU time (all 40 points x 4 quantiles x 4 times):")
    for k_, v in timing.items():
        print(f"   {k_:6s}: {v:7.2f} s  ({1e3*v/(4*len(idx)):.1f} ms per point-time)")
    print(f"   MC    : {t_mc:7.2f} s (whole field, 1e5 members)")
    print(f"total wall {time.perf_counter()-t_all:.1f} s")
    return results

if __name__ == "__main__":
    main()
