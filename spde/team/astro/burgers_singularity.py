#!/usr/bin/env python3
"""
burgers_singularity.py -- complex-singularity tracking (Sulem, Sulem & Frisch 1983) as a
diagnostic of where a distribution-family closure for wind speed must break.

Self-contained (numpy + scipy). Prints its tables. Parts:
  A. 1-D viscous Burgers u_t + u u_x = nu u_xx, u0 = sin x, 2pi-periodic, pseudo-spectral
     (2/3 dealiasing, integrating-factor RK4). SSF fit  |u_k| ~ C k^-n exp(-delta k)  every 0.05.
     Benchmarks: inviscid  delta_0(t) = arccosh(1/t) - sqrt(1-t^2)  (t<1, ~ (2(1-t))^{3/2}/3),
     exact viscous delta from Cole-Hopf (first zero of phi(pi + i eta, t)), and the tanh-shock
     asymptote delta = pi nu / U(t) (t>1), U = half-jump.
  B. The same run seen as "resolved + unresolved": sharp filter at K_c; the unresolved residual
     u' = u - P_{K_c} u plays the role of the intrinsic noise. Its rms is predicted from delta fitted
     on the RESOLVED band only; its flatness and the KS distance of Weibull / exp-Weibull fits to |u|
     are reported vs delta.
  C. Parameter-space singularities of the exp-Weibull moment map (k, a) -> (CV, skewness).
  D. One-point wind-speed distribution seen by a fixed mast as a Holland (1980) vortex passes at a
     random miss distance (fold caustic at Vmax); KS of Weibull / exp-Weibull fits.
"""
import time as _time
import warnings

import numpy as np
from scipy import stats, optimize, integrate

warnings.filterwarnings("ignore")
rng = np.random.default_rng(1)


# ----------------------------------------------------------------------------------------------
# A. Burgers solver + SSF fit
# ----------------------------------------------------------------------------------------------
def burgers_run(nu, N=8192, dt=2.5e-4, t_end=2.5, every=0.05):
    x = 2 * np.pi * np.arange(N) / N
    k = np.arange(N // 2 + 1).astype(float)
    kc = N // 3
    mask = (k <= kc).astype(float)
    uh = np.fft.rfft(np.sin(x))
    L = -nu * k ** 2
    E, E2 = np.exp(L * dt), np.exp(L * dt / 2)

    def nl(vh):
        v = np.fft.irfft(vh * mask, n=N)
        return -0.5j * k * np.fft.rfft(v * v) * mask

    out = []
    nsteps = int(round(t_end / dt))
    nev = int(round(every / dt))
    out.append((0.0, uh.copy()))
    for s in range(1, nsteps + 1):
        a = nl(uh)
        b = nl(E2 * (uh + dt / 2 * a))
        c = nl(E2 * uh + dt / 2 * b)
        d = nl(E * uh + dt * E2 * c)
        uh = E * uh + dt / 6 * (E * a + 2 * E2 * (b + c) + d)
        if s % nev == 0:
            out.append((s * dt, uh.copy()))
    return x, k, kc, out


def ssf_fit(amp, k, kmin, kmax, floor=1e-13, n_fixed=None):
    """least squares  log|u_k| = log C - n log k - delta k  on kmin<=k<=kmax above the roundoff floor"""
    sel = (k >= kmin) & (k <= kmax) & (amp > floor * amp.max())
    kk, la = k[sel], np.log(amp[sel])
    if sel.sum() < 6:
        return np.nan, np.nan, 0
    if n_fixed is None:
        A = np.c_[np.ones_like(kk), -np.log(kk), -kk]
        (c0, n, dl), *_ = np.linalg.lstsq(A, la, rcond=None)
    else:
        A = np.c_[np.ones_like(kk), -kk]
        (c0, dl), *_ = np.linalg.lstsq(A, la + n_fixed * np.log(kk), rcond=None)
        n = n_fixed
    return dl, n, int(kk.max())


def delta_inviscid(t):
    return np.arccosh(1 / t) - np.sqrt(1 - t * t) if 0 < t < 1 else np.nan


def half_jump(t):
    """inviscid half-jump U(t) at the stationary shock x=pi (t>1): U = y/t with sin y = y/t"""
    if t <= 1:
        return np.nan
    y = optimize.brentq(lambda y: np.sin(y) - y / t, 1e-9, np.pi - 1e-12)
    return y / t


def delta_colehopf(t, nu, eta_max=0.6, ny=40001):
    """first zero eta>0 of phi(pi+i eta,t) ~ int exp(-(y^2/(2t)+cos y)/(2nu)) cos(eta y/(2 nu t)) dy.
    u = -2 nu phi_x/phi has its nearest pole at pi + i*eta (Cole-Hopf; exact for viscous Burgers).
    Returns nan if double precision is insufficient (|phi| below 1e-11 of the integrand mass)."""
    y = np.linspace(-9, 9, 400001)
    g = -(y * y / (2 * t) + np.cos(y)) / (2 * nu)
    keep = g - g.max() > -60                      # restrict to the support of the weight
    y = np.linspace(y[keep].min(), y[keep].max(), ny)
    g = -(y * y / (2 * t) + np.cos(y)) / (2 * nu)
    w = np.exp(g - g.max())
    scale = np.trapezoid(w, y)
    f = lambda e: np.trapezoid(w * np.cos(np.multiply.outer(np.atleast_1d(e), y) / (2 * nu * t)), y, axis=-1)
    etas = np.linspace(1e-4, eta_max, 400)
    vals = np.concatenate([f(c) for c in np.array_split(etas, 20)])
    sgn = np.where(np.sign(vals[1:]) != np.sign(vals[:-1]))[0]
    if len(sgn) == 0:
        return np.nan
    i = sgn[0]
    if max(abs(vals[i]), abs(vals[i + 1])) < 1e-11 * scale:
        return np.nan
    return optimize.brentq(lambda e: f(e)[0], etas[i], etas[i + 1], xtol=1e-7)


def part_A_B():
    results = {}
    Kc = 32  # "resolved" cutoff (grid of a coarse model: dx_eff = pi/Kc ~ 0.1)
    for nu in (0.02, 0.01, 0.005):
        t0 = _time.time()
        x, k, kc, snaps = burgers_run(nu)
        N = len(x)
        rows = []
        for t, uh in snaps:
            if t < 0.2:
                continue
            amp = np.abs(uh) / (N / 2)
            dl, n, kup = ssf_fit(amp, k, 4, kc)
            dl43, _, _ = ssf_fit(amp, k, 4, kc, n_fixed=4.0 / 3.0)
            # resolved-band-only fit (what a coarse model could measure)
            dlr, nr, _ = ssf_fit(amp, k, 3, Kc)
            u = np.fft.irfft(uh, n=N)
            ures = np.fft.irfft(np.where(k <= Kc, uh, 0), n=N)
            up = u - ures
            var_true = np.mean(up ** 2)
            # predicted unresolved variance from extrapolating the resolved-band SSF fit
            if np.isfinite(dlr):
                sel = (k >= 3) & (k <= Kc)
                A = np.c_[np.ones(sel.sum()), -np.log(k[sel]), -k[sel]]
                (c0, nn, dd), *_ = np.linalg.lstsq(A, np.log(amp[sel]), rcond=None)
                kt = np.arange(Kc + 1, kc + 1)
                var_pred = 0.5 * np.sum(np.exp(2 * (c0 - nn * np.log(kt) - dd * kt)))
            else:
                var_pred = np.nan
            flat = np.mean(up ** 4) / var_true ** 2 if var_true > 0 else np.nan
            ux = np.fft.irfft(1j * k * uh, n=N)
            flat_ux = np.mean(ux ** 4) / np.mean(ux ** 2) ** 2
            # family fits to the one-point distribution of |u| over the domain
            s = np.abs(u)
            s = s[s > 1e-6]
            sub = s[:: max(1, len(s) // 4096)]
            pw = stats.weibull_min.fit(sub, floc=0)
            ks_w = stats.kstest(sub, "weibull_min", args=pw).statistic
            try:
                pe = stats.exponweib.fit(sub, floc=0)
                ks_e = stats.kstest(sub, "exponweib", args=pe).statistic
            except Exception:
                ks_e = np.nan
            rows.append(dict(t=t, d=dl, n=n, d43=dl43, kup=kup, dinv=delta_inviscid(t),
                             dch=np.nan, dtanh=(np.pi * nu / half_jump(t)) if t > 1 else np.nan,
                             dlr=dlr, nr=nr, rms=np.sqrt(var_true), rms_pred=np.sqrt(var_pred),
                             flat=flat, flat_ux=flat_ux, ks_w=ks_w, ks_e=ks_e, umax=np.abs(u).max()))
        for r in rows:  # Cole-Hopf only where delta is small enough for double precision
            if r["t"] >= 0.7 and abs(round(r["t"] / 0.1) * 0.1 - r["t"]) < 1e-6 or 0.9 < r["t"] < 1.1:
                r["dch"] = delta_colehopf(r["t"], nu)
        results[nu] = rows
        print(f"[nu={nu}] N={N}, run+diagnostics {(_time.time()-t0):.1f}s")

    print("\n=== A. SSF fit |u_k| ~ C k^-n exp(-delta k)  (fit 4<=k<=k_up, roundoff floor 1e-13) ===")
    for nu, rows in results.items():
        print(f"\nnu = {nu}:  pi*nu = {np.pi*nu:.4f},  nu^(3/4) = {nu**0.75:.4f}")
        print("   t    delta_fit   n_fit  delta(n=4/3)  k_up  delta_inviscid  delta_ColeHopf  pi*nu/U(t)")
        for r in rows:
            if abs(round(r["t"] / 0.1) * 0.1 - r["t"]) > 1e-6 and not (0.9 < r["t"] < 1.1):
                continue
            print(f"  {r['t']:4.2f}  {r['d']:9.4f}  {r['n']:6.2f}  {r['d43']:10.4f}  {r['kup']:5d}"
                  f"  {r['dinv']:12.4f}  {r['dch']:13.4f}  {r['dtanh']:11.4f}")
        ds = np.array([r["d"] for r in rows]); ts = np.array([r["t"] for r in rows])
        i = np.nanargmin(ds)
        r1 = [r for r in rows if abs(r["t"] - 1.0) < 1e-6][0]
        print(f"  -> delta bottoms out at t = {ts[i]:.2f} with delta_min = {ds[i]:.4f}"
              f"  (delta_min/(pi nu) = {ds[i]/(np.pi*nu):.3f}); delta(t*=1) = {r1['d']:.4f}"
              f" = {r1['d']/nu**0.75:.3f} nu^(3/4)")
        sel = (ts >= 0.5) & (ts <= 0.8)
        p = np.polyfit(np.log(1 - ts[sel]), np.log(ds[sel]), 1)
        pi_ = np.polyfit(np.log(1 - ts[sel]), np.log([delta_inviscid(t) for t in ts[sel]]), 1)
        print(f"  -> log-log slope of delta vs (1-t) on 0.5<=t<=0.8: fit {p[0]:.3f} "
              f"(inviscid exact on same window {pi_[0]:.3f}; asymptotic 3/2)")
        sel = (ts >= 0.5 - 1e-9) & (ts <= 0.7 + 1e-9)
        q = np.polyfit(ts[sel], ds[sel] ** (2 / 3), 1)
        print(f"  -> collapse-time forecast from delta^(2/3) linear on 0.5<=t<=0.7: t* = {-q[1]/q[0]:.3f} (true 1)")

    print("\n=== B. resolved (k<=Kc=32) vs unresolved residual u' = u - P_Kc u ; family fits to |u| ===")
    print("  delta_res = SSF fit on the RESOLVED band 3<=k<=32 only; rms_pred = sqrt(sum_{k>Kc} |C k^-n e^-dk|^2/2)")
    for nu, rows in results.items():
        print(f"\nnu = {nu}")
        print("   t   delta  delta_res  delta*Kc   rms(u')    rms_pred   F(u')   F(u_x)   KS_Weib  KS_expW")
        for r in rows:
            if abs(round(r["t"] / 0.1) * 0.1 - r["t"]) > 1e-6:
                continue
            print(f"  {r['t']:4.2f} {r['d']:6.3f}  {r['dlr']:7.3f}  {r['d']*32:7.2f}  {r['rms']:9.2e}  "
                  f"{r['rms_pred']:9.2e}  {r['flat']:6.1f}  {r['flat_ux']:7.1f}  {r['ks_w']:7.3f}  {r['ks_e']:7.3f}")
    return results


# ----------------------------------------------------------------------------------------------
# C. exp-Weibull moment map singularities
# ----------------------------------------------------------------------------------------------
def ew_raw(k, a, n):
    """E[(X/lambda)^n] for F=(1-exp(-(x/l)^k))^a : int z^{n/k} a(1-e^-z)^{a-1} e^-z dz"""
    f = lambda z: z ** (n / k) * a * (-np.expm1(-z)) ** (a - 1) * np.exp(-z)
    v1, _ = integrate.quad(f, 0, 1, limit=200)
    v2, _ = integrate.quad(f, 1, np.inf, limit=200)
    return v1 + v2


def ew_cv_skew(k, a):
    m1, m2, m3 = (ew_raw(k, a, n) for n in (1, 2, 3))
    var = m2 - m1 ** 2
    return np.sqrt(var) / m1, (m3 - 3 * m1 * var - m1 ** 3) / var ** 1.5


def part_C():
    print("\n=== C. exp-Weibull moment map (k,a) -> (CV, skewness): Jacobian det on log-grid ===")
    lk = np.linspace(np.log(0.6), np.log(12), 22)
    la = np.linspace(np.log(0.15), np.log(40), 22)
    h = 1e-3
    det = np.full((len(lk), len(la)), np.nan)
    for i, a_ in enumerate(lk):
        for j, b_ in enumerate(la):
            k, a = np.exp(a_), np.exp(b_)
            try:
                c0, s0 = ew_cv_skew(k, a)
                c1, s1 = ew_cv_skew(k * np.exp(h), a)
                c2, s2 = ew_cv_skew(k, a * np.exp(h))
                det[i, j] = ((c1 - c0) * (s2 - s0) - (c2 - c0) * (s1 - s0)) / h ** 2
            except Exception:
                pass
    sgn = np.sign(det)
    flips = np.argwhere((sgn[:, 1:] * sgn[:, :-1]) < 0)
    print(f"  grid k in [0.6,12], a in [0.15,40] (22x22): det range [{np.nanmin(det):.3e}, {np.nanmax(det):.3e}],"
          f" sign flips along a: {len(flips)}")
    for i, j in flips[:12]:
        print(f"    fold between a={np.exp(la[j]):.3f} and {np.exp(la[j+1]):.3f} at k={np.exp(lk[i]):.3f}")
    print("  |det| along the a-direction at k=2 (decay => unidentifiable combination):")
    i2 = np.argmin(abs(lk - np.log(2)))
    for j in range(0, len(la), 3):
        cv, sk = ew_cv_skew(np.exp(lk[i2]), np.exp(la[j]))
        print(f"    a={np.exp(la[j]):7.3f}  CV={cv:.3f}  skew={sk:+.3f}  det={det[i2, j]:+.3e}")
    tgt = np.array([np.sqrt(0.5 - 4 / np.pi ** 2) / (2 / np.pi), np.nan])
    m, v_ = 2 / np.pi, 0.5 - 4 / np.pi ** 2
    tgt[1] = (4 / (3 * np.pi) - 3 * m * v_ - m ** 3) / v_ ** 1.5
    res = lambda p: np.array(ew_cv_skew(np.exp(p[0]), np.exp(p[1]))) - tgt
    sol = optimize.least_squares(res, (np.log(8), np.log(0.3)),
                                 bounds=([np.log(0.3), np.log(1e-3)], [np.log(400), np.log(100)]))
    kk, aa = np.exp(sol.x)
    print(f"  moment-matching the arcsine law of |sin x| (CV={tgt[0]:.3f}, skew={tgt[1]:+.3f}):"
          f" optimiser runs to k={kk:.0f}, a={aa:.4f} (k*a={kk*aa:.2f}), residual {np.round(sol.fun,3)}")
    print("    -> corner k->inf, a->0, k*a=c fixed: EW -> power-function law F=(x/l)^c on [0,l] (rank-1 moment map)")
    print("  Weibull (a=1) and Gumbel limits: skew(Weibull,k->inf) -> -1.1395, skew(EW, a->inf) -> Gumbel(max) +1.1395")
    for k in (3.0, 3.602, 10.0, 50.0):
        print(f"    Weibull k={k:6.3f}: CV={ew_cv_skew(k,1)[0]:.4f}, skew={ew_cv_skew(k,1)[1]:+.4f}")


# ----------------------------------------------------------------------------------------------
# D. Holland vortex seen by a fixed mast
# ----------------------------------------------------------------------------------------------
def holland(r, vmax, rm, B):
    s = (rm / np.maximum(r, 1e-9)) ** B
    return vmax * np.sqrt(s * np.exp(1 - s))


def part_D():
    print("\n=== D. Holland (1980) vortex passing a fixed mast at random miss distance ===")
    vmax, rm, B = 50.0, 30.0, 1.5        # m/s, km
    D, L = 300.0, 500.0                    # miss distance ~ U[0,D], along-track window |x|<=L
    n = 400_000
    d = rng.uniform(0, D, n)
    xs = rng.uniform(-L, L, n)             # uniform in time at constant translation speed
    r = np.hypot(d, xs)
    v = holland(r, vmax, rm, B)
    print(f"  Vmax={vmax} m/s, Rm={rm} km, B={B}, miss distance U[0,{D}] km, window +-{L} km, n={n}")
    # caustic exponent: P(V > Vmax - e) ~ e^{1/2}
    eps = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.6])
    P = np.array([(v > vmax - e).mean() for e in eps])
    slope = np.polyfit(np.log(eps), np.log(P), 1)[0]
    # analytic prefactor: p(r) = int_0^min(r,D) r/sqrt(r^2-d^2) dd /(2 D L) ; p(v) ~ 2 p(Rm)/|V'| summed
    pr_rm = rm * np.arcsin(min(rm, D) / rm) / (D * 2 * L)
    Vpp = vmax * B ** 2 / (2 * rm ** 2)
    pred = [2 * pr_rm * 2 * np.sqrt(2 * e / Vpp) for e in eps]   # two branches r<Rm, r>Rm
    print("   eps(m/s)  P(V>Vmax-eps) MC   analytic 4 p_r(Rm) sqrt(2 eps/|V''|)")
    for e, p_, q in zip(eps, P, pred):
        print(f"   {e:6.2f}   {p_:.5f}          {q:.5f}")
    print(f"  log-log slope of P(V>Vmax-eps) vs eps = {slope:.3f}  (fold caustic: 1/2)")
    for amb in (0.0, 5.0):
        if amb > 0:  # add an ambient Weibull wind + translation asymmetry (vector sum, random direction)
            sa = stats.weibull_min.rvs(2.0, scale=amb, size=n, random_state=2)
            th = rng.uniform(0, 2 * np.pi, n)
            s = np.hypot(v + sa * np.cos(th), sa * np.sin(th))
        else:
            s = v
        sub = s[rng.choice(n, 20000, replace=False)]
        sub = sub[sub > 1e-3]
        pw = stats.weibull_min.fit(sub, floc=0)
        pe = stats.exponweib.fit(sub, floc=0)
        ksw = stats.kstest(sub, "weibull_min", args=pw).statistic
        kse = stats.kstest(sub, "exponweib", args=pe).statistic
        h, e = np.histogram(s, bins=50, range=(0, vmax + 3 * amb + 1), density=True)
        peaks = [0.5 * (e[i] + e[i + 1]) for i in range(1, len(h) - 1) if h[i] > h[i - 1] and h[i] >= h[i + 1] and h[i] > 0.2 * h.max()]
        q = np.quantile(s, [0.5, 0.9, 0.99, 0.999])
        qw = stats.weibull_min.ppf([0.5, 0.9, 0.99, 0.999], *pw)
        qe = stats.exponweib.ppf([0.5, 0.9, 0.99, 0.999], *pe)
        print(f"  ambient Weibull scale {amb} m/s: KS Weibull={ksw:.3f}, KS expW={kse:.3f}; histogram modes at {np.round(peaks,1)} m/s")
        print(f"     quantiles 50/90/99/99.9%: MC {np.round(q,2)}  Weib {np.round(qw,2)}  expW {np.round(qe,2)}")


if __name__ == "__main__":
    T0 = _time.time()
    part_A_B()
    part_C()
    part_D()
    print(f"\ntotal {(_time.time()-T0):.0f}s")
