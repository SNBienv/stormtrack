"""
Family solve vs brute-force ensemble, for the two families the user asked for:
  Champernowne (Ndeba form on wind speed, truncated below at L): params alpha, lam, v0, L
  Weibull 3-parameter: params k, lam, gamma (location)
Dynamics per step: advection (1 cell), affine physics S -> a S + d, Markov gust step
(EAR(1) applied to Y = -log(1 - F(S)), which is Exp(1) for ANY continuous family).
Both families are exactly closed under these operations:
  Champernowne: alpha -> alpha / a, v0 -> a v0 + d, L -> a L + d, lam unchanged
  Weibull 3p:   lam -> a lam, gamma -> a gamma + d, k unchanged
"""
import time
import numpy as np
from scipy import stats
import champ_lin

rng = np.random.default_rng(11)
N, M, steps, rho = 100, 20_000, 100, 0.85
x = np.arange(N) / N
a_phys = 1.0 + 0.004 * np.sin(4 * np.pi * x)
d_phys = 0.02 * np.cos(2 * np.pi * x)


def ear1(Y):
    return rho * Y + (rng.random(Y.shape) > rho) * rng.exponential(1.0, Y.shape)


def run(name, p0, update, cdf, ppf, sample):
    t0 = time.perf_counter()
    p = [q.copy() for q in p0]
    for _ in range(steps):
        p = update([np.roll(q, 1) for q in p])
    t_fam = time.perf_counter() - t0

    t0 = time.perf_counter()
    S = sample(p0)
    pc = [q.copy() for q in p0]
    for _ in range(steps):
        S = np.roll(S, 1, axis=1)
        pc = update([np.roll(q, 1) for q in pc])
        S = a_phys * S + d_phys
        U = np.clip(cdf(S, pc), 1e-15, 1 - 1e-15)
        Y = ear1(-np.log1p(-U))
        S = ppf(-np.expm1(-Y), pc)
    t_ens = time.perf_counter() - t0

    pv = np.array([stats.kstest(S[:, i], lambda v, i=i: cdf(v, [q[i] for q in p])).pvalue for i in range(N)])
    print(f"{name:<14} KS rejected at {np.mean(pv < 0.05)*100:4.1f}% of {N} points (≈5% expected if exact), "
          f"p-value uniformity p = {stats.kstest(pv, 'uniform').pvalue:.2f};  "
          f"family {t_fam*1e3:.1f} ms vs ensemble {t_ens:.0f} s")


# Champernowne (Ndeba form), L = 0 initially
ch0 = [0.6 + 0.1 * np.sin(2 * np.pi * x),            # alpha
       0.3 + 0.5 * np.cos(2 * np.pi * x),            # lam
       5.0 + 1.5 * np.sin(2 * np.pi * x),            # v0
       np.zeros(N)]                                  # L


def ch_update(p):
    a, lam, v0, L = p
    return [a / a_phys, lam, a_phys * v0 + d_phys, a_phys * L + d_phys]


run("Champernowne", ch0, ch_update,
    cdf=lambda v, p: champ_lin.cdf(v, *p),
    ppf=lambda u, p: champ_lin.ppf(u, *p),
    sample=lambda p: champ_lin.ppf(rng.random((M, N)), *p))

# Weibull 3-parameter
w0 = [2.0 + 0.3 * np.sin(2 * np.pi * x),             # k
      7.0 + 2.0 * np.cos(2 * np.pi * x),             # lam
      0.5 + 0.3 * np.sin(2 * np.pi * x)]             # gamma


def w_update(p):
    k, lam, g = p
    return [k, a_phys * lam, a_phys * g + d_phys]


run("Weibull 3p", w0, w_update,
    cdf=lambda v, p: stats.weibull_min.cdf(v, p[0], loc=p[2], scale=p[1]),
    ppf=lambda u, p: stats.weibull_min.ppf(u, p[0], loc=p[2], scale=p[1]),
    sample=lambda p: stats.weibull_min.ppf(rng.random((M, N)), p[0], loc=p[2], scale=p[1]))
