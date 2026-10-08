"""Task 1: raw moments of the exponentiated Weibull F(x) = (1-exp(-(x/lam)^k))^a.

E[X^r] = a lam^r Gamma(1+r/k) * S(a, s),  s = 1 + r/k,
S(a,s) = sum_{m>=0} c_m (m+1)^{-s},  c_m = (-1)^m C(a-1,m) = Gamma(m+1-a)/(Gamma(1-a) m!)

Schemes:
 (A) integer a: finite sum (m = 0..a-1).
 (B) non-integer a: head sum m<N + Hurwitz-zeta tail
     c_m = (1/Gamma(1-a)) z^{-a} sum_j g_j z^{-j},  z=m+1   (Gamma-ratio Stirling expansion)
     tail = (1/Gamma(1-a)) sum_j g_j zeta(s+a+j, N+1)
 (C) generalized Gauss-Laguerre:  E[X^r] = a lam^r int_0^inf u^{r/k+a-1} e^{-u} [(1-e^{-u})/u]^{a-1} du
"""
import numpy as np, mpmath as mp, time
from scipy.special import gammaln, gamma, zeta, roots_genlaguerre, comb, bernoulli

def _bernpoly(n, h):
    B = bernoulli(n)
    return sum(comb(n, i, exact=False) * B[i] * h ** (n - i) for i in range(n + 1))

import functools
@functools.lru_cache(None)
def gamma_ratio_coeffs(a, J=14):
    """g_j with Gamma(z-a)/Gamma(z) ~ z^{-a} sum_j g_j z^{-j}."""
    # log ratio = -a log z + sum_n d_n z^{-n},
    # d_n = (-1)^{n+1} (B_{n+1}(-a) - B_{n+1}(0)) / (n(n+1))
    d = np.zeros(J + 1)
    for n in range(1, J + 1):
        d[n] = (-1) ** (n + 1) * (_bernpoly(n + 1, -a) - _bernpoly(n + 1, 0.0)) / (n * (n + 1))
    # exp of power series: g' = (sum n d_n w^{n-1}) g  (in w=1/z)
    g = np.zeros(J + 1); g[0] = 1.0
    for n in range(1, J + 1):
        g[n] = sum(i * d[i] * g[n - i] for i in range(1, n + 1)) / n
    return g

def S_series(a, s, N=None, J=12):
    if N is None: N = int(24 + 3 * abs(a))
    if abs(a - round(a)) < 1e-14 and a >= 1:
        n = int(round(a)); m = np.arange(n)
        return np.sum((-1.0) ** m * comb(n - 1, m) * (m + 1.0) ** (-s))
    m = np.arange(N)
    # c_m via log-gamma with sign
    c = np.empty(N); c[0] = 1.0
    for i in range(1, N):
        c[i] = c[i - 1] * (i - a) / i          # c_m = c_{m-1} (m-a)/m
    head = np.sum(c * (m + 1.0) ** (-s))
    g = gamma_ratio_coeffs(a, J)
    tail = sum(g[j] * zeta(s + a + j, N + 1) for j in range(J + 1)) / gamma(1 - a)
    return head + tail

def ew_moment_series(r, k, lam, a, **kw):
    s = 1.0 + r / k
    return a * lam ** r * gamma(s) * S_series(a, s, **kw)

_GL = {}
def ew_moment_gl(r, k, lam, a, n=80):
    alpha = r / k + a - 1.0
    key = (n, round(alpha, 15))
    if key not in _GL:
        _GL[key] = roots_genlaguerre(n, alpha)
    x, w = _GL[key]
    f = np.where(x < 1e-8, 1.0 - x / 2, -np.expm1(-x) / x) ** (a - 1.0)
    return a * lam ** r * np.sum(w * f)

def ew_moment_quad(r, k, lam, a, dps=30):
    mp.mp.dps = dps
    k, lam, a = mp.mpf(k), mp.mpf(lam), mp.mpf(a)
    # in w = 1-e^{-u}:  a lam^r int_0^1 (-log(1-w))^{r/k} w^{a-1} dw
    f = lambda u: u ** (r / k) * mp.exp(-u) * (-mp.expm1(-u)) ** (a - 1)
    return float(a * lam ** r * mp.quad(f, [0, 1, 5, 20, 60, mp.inf]))

def raw_to_cumulants(m1, m2, m3, m4=None):
    k2 = m2 - m1 ** 2
    k3 = m3 - 3 * m2 * m1 + 2 * m1 ** 3
    out = [m1, k2, k3]
    if m4 is not None:
        out.append(m4 - 4 * m3 * m1 - 3 * m2 ** 2 + 12 * m2 * m1 ** 2 - 6 * m1 ** 4)
    return out

if __name__ == "__main__":
    print("naive partial sums, a=0.5,k=2,r=1 (terms ~ m^-2):")
    s = 1.5; ref = S_series(0.5, s)
    for N in [10, 100, 1000, 10000]:
        m = np.arange(N); c = np.cumprod(np.r_[1.0, (np.arange(1, N) - 0.5) / np.arange(1, N)])
        print(f"   N={N:6d}  rel err {abs(np.sum(c*(m+1.0)**-s)-ref)/ref:.2e}")
    worst_s = worst_g = 0.0
    rows = []
    for k in [0.7, 1.3, 2.0, 3.5]:
        for a in [0.3, 0.5, 0.9, 1.0, 1.7, 2.0, 3.0, 4.6, 12.3]:
            for r in [1, 2, 3, 4, 0.5]:
                q = ew_moment_quad(r, k, 7.0, a)
                es = abs(ew_moment_series(r, k, 7.0, a) / q - 1)
                eg = abs(ew_moment_gl(r, k, 7.0, a) / q - 1)
                worst_s = max(worst_s, es); worst_g = max(worst_g, eg)
                rows.append((k, a, r, es, eg))
    print(f"Hurwitz-accelerated series: worst rel err vs mpmath quad over 180 cases = {worst_s:.2e}")
    print(f"gen. Gauss-Laguerre n=80  : worst rel err                               = {worst_g:.2e}")
    bad = [r for r in rows if r[4] > 1e-10]
    print("GL cases worse than 1e-10:", bad[:6])
    t = time.perf_counter()
    for _ in range(1000): ew_moment_series(2, 2.0, 7.0, 1.7)
    print(f"series eval time: {(time.perf_counter()-t)/1000*1e6:.0f} us/moment")
