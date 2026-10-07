"""
Mixing closure: keep the family solve ensemble-free when diffusion is present.

Diffusion (mixing) is the one operation the wind families are not closed under (weibull_family.py,
Part 3). Here the family solve gets a closure step. For linear dynamics (advection, scaling,
diffusion) the marginal cumulants of the wind speed at every grid point evolve EXACTLY.
If the initial values are independent between grid points and the run so far is S(t) = P S(0),
with P the product of the step matrices, then

    mean(t)  = P mean(0)
    kappa_n(t)_i = sum_j P_ij^n kappa_n(0)_j        (n >= 2, cumulants add over independent terms)

So the family solve tracks (mean, kappa_2, kappa_3, kappa_4) fields and re-projects onto a family
by matching the first 2 (two-parameter family) or 3 (three-parameter family) cumulants.
kappa_4 is not used for the fit. It is kept as an independent check of the tail.

Families:
  Weibull 2p          k, lam            matched to mean, variance
  Weibull 3p          k, lam, gamma     matched to mean, variance, skewness
  Exp-Weibull 3p      k, a, lam         matched to mean, variance, skewness
  Champernowne 3p     alpha, lam, v0    Ndeba-form reconstruction (champ_lin.py), truncated at 0

Compared against a 100,000-member ensemble that carries out the same dynamics on every sample.
Also shown: "no closure" (family only advected and scaled, mixing ignored) and, at the last
checkpoint, the best fit each family can achieve (maximum likelihood on the ensemble itself),
which separates the error of the family from the error of the closure.
"""
import time
import numpy as np
from scipy import optimize, special, stats
import champ_lin

rng = np.random.default_rng(2026)
N, M, nu = 200, 100_000, 0.25
CHECKPOINTS = (5, 10, 20, 40)
PROBS = (0.5, 0.9, 0.99, 0.999)
TEST = np.arange(0, N, 5)                                   # 40 grid points scored
x = np.arange(N) / N
k0 = 2.0 + 0.3 * np.sin(2 * np.pi * x)
lam0 = 7.0 + 2.0 * np.cos(2 * np.pi * x)
a_phys = 1.0 + 0.004 * np.sin(4 * np.pi * x)                # terrain speed-up


# ---------------------------------------------------------------- cumulant algebra
def cumulants_from_raw(m1, m2, m3, m4):
    k2 = m2 - m1 ** 2
    k3 = m3 - 3 * m2 * m1 + 2 * m1 ** 3
    k4 = m4 - 4 * m3 * m1 - 3 * m2 ** 2 + 12 * m2 * m1 ** 2 - 6 * m1 ** 4
    return m1, k2, k3, k4


def weibull_raw(k, lam, r):
    return lam ** r * special.gamma(1 + r / k)


def shape_stats(mean, k2, k3, k4):
    """CV, skewness, excess kurtosis."""
    sd = np.sqrt(k2)
    return sd / mean, k3 / sd ** 3, k4 / k2 ** 2


# ---------------------------------------------------------------- moment-matched families
# Each returns (ppf, implied excess kurtosis, residual of the moment match).
def match_weibull2(mean, k2, k3):
    cv = np.sqrt(k2) / mean
    f = lambda k: np.sqrt(special.gamma(1 + 2 / k) / special.gamma(1 + 1 / k) ** 2 - 1) - cv
    k = optimize.brentq(f, 0.2, 200)
    lam = mean / special.gamma(1 + 1 / k)
    m = [weibull_raw(k, lam, r) for r in (1, 2, 3, 4)]
    _, kk2, kk3, kk4 = cumulants_from_raw(*m)
    d = stats.weibull_min(k, scale=lam)
    return d.ppf, kk4 / kk2 ** 2, abs(kk3 / kk2 ** 1.5 - k3 / k2 ** 1.5)


def match_weibull3(mean, k2, k3):
    skew = k3 / k2 ** 1.5
    g = lambda k, r: special.gamma(1 + r / k)

    def wskew(k):
        g1, g2, g3 = g(k, 1), g(k, 2), g(k, 3)
        return (g3 - 3 * g1 * g2 + 2 * g1 ** 3) / (g2 - g1 ** 2) ** 1.5
    lo, hi = 0.2, 500.0
    skew_c = np.clip(skew, wskew(hi) + 1e-9, wskew(lo) - 1e-9)   # skewness range of the family
    k = optimize.brentq(lambda k: wskew(k) - skew_c, lo, hi)
    lam = np.sqrt(k2 / (g(k, 2) - g(k, 1) ** 2))
    gamma = mean - lam * g(k, 1)
    g1, g2, g3, g4 = (g(k, r) for r in (1, 2, 3, 4))
    exk = (g4 - 4 * g1 * g3 + 6 * g1 ** 2 * g2 - 3 * g1 ** 4) / (g2 - g1 ** 2) ** 2 - 3
    d = stats.weibull_min(k, loc=gamma, scale=lam)
    return d.ppf, exk, abs(skew_c - skew)


# Moments from the quantile function: E[X^r] = int_0^1 Q(u)^r du, with u = 1 - exp(-s) so that the
# upper tail is resolved; fixed Gauss-Legendre nodes on s in [0, 32] (tail mass beyond: 1e-14).
_GL_S, _GL_W = np.polynomial.legendre.leggauss(400)
_GL_S = 16.0 * (_GL_S + 1)
_GL_W = 16.0 * _GL_W * np.exp(-_GL_S)
_U = -np.expm1(-_GL_S)


def quantile_raw(q, rs=(1, 2, 3, 4)):
    return [np.dot(_GL_W, q ** r) for r in rs]


def ew_quantile(k, a):
    """Exponentiated-Weibull quantile at the nodes, lam = 1, written to keep the upper tail accurate."""
    v = -np.expm1(np.log1p(-np.exp(-_GL_S)) / a)        # 1 - u^(1/a)
    return (-np.log(v)) ** (1 / k)


def match_expweib(mean, k2, k3):
    cv, skew = np.sqrt(k2) / mean, k3 / k2 ** 1.5

    def stats_of(t):
        k, a = np.exp(t)
        m1, c2, c3, c4 = cumulants_from_raw(*quantile_raw(ew_quantile(k, a)))
        return m1, np.sqrt(c2) / m1, c3 / c2 ** 1.5, c4 / c2 ** 2

    def resid(t):
        _, c, s, _ = stats_of(t)
        return [(c - cv) / cv, s - skew]
    k_start = optimize.brentq(lambda k: np.sqrt(special.gamma(1 + 2 / k) / special.gamma(1 + 1 / k) ** 2 - 1) - cv,
                              0.2, 200)
    r = optimize.least_squares(resid, [np.log(k_start), 0.0], xtol=1e-10, ftol=1e-10)
    k, a = np.exp(r.x)
    m1, _, _, exk = stats_of(r.x)
    lam = mean / m1
    d = stats.exponweib(a, k, scale=lam)
    return d.ppf, exk, float(np.max(np.abs(r.fun)))


def champ_moments(a, lam, v0):
    return cumulants_from_raw(*quantile_raw(champ_lin.ppf(_U, a, lam, v0)))


def match_champ(mean, k2, k3):
    sd, skew = np.sqrt(k2), k3 / k2 ** 1.5

    def unpack(t):
        return np.exp(t[0]), -1 + np.exp(t[1]), t[2]

    def resid(t):
        m1, c2, c3, _ = champ_moments(*unpack(t))
        return [(m1 - mean) / sd, (np.sqrt(c2) - sd) / sd, c3 / c2 ** 1.5 - skew]
    a0 = np.pi / (np.sqrt(3) * sd)                          # logistic (lam = 1) start
    r = optimize.least_squares(resid, [np.log(a0), np.log(2.0), mean], xtol=1e-10, ftol=1e-10)
    p = unpack(r.x)
    _, c2, _, c4 = champ_moments(*p)
    return (lambda q, p=p: champ_lin.ppf(np.asarray(q), *p)), c4 / c2 ** 2, float(np.max(np.abs(r.fun)))


FAMILIES = {
    "Weibull 2p": match_weibull2,
    "Weibull 3p": match_weibull3,
    "Exp-Weibull 3p": match_expweib,
    "Champernowne 3p": match_champ,
}


# ---------------------------------------------------------------- maximum-likelihood fits (oracle)
def mle_fit(name, s):
    if name == "Weibull 2p":
        k, _, lam = stats.weibull_min.fit(s, floc=0)
        return stats.weibull_min(k, scale=lam).ppf
    if name == "Weibull 3p":
        k, loc, lam = stats.weibull_min.fit(s)
        return stats.weibull_min(k, loc=loc, scale=lam).ppf
    if name == "Exp-Weibull 3p":
        k0_, _, l0_ = stats.weibull_min.fit(s, floc=0)
        nll = lambda t: -stats.exponweib.logpdf(s, np.exp(t[0]), np.exp(t[1]), scale=np.exp(t[2])).sum()
        r = min((optimize.minimize(nll, [np.log(a), np.log(k0_), np.log(l0_)], method="Nelder-Mead",
                                   options=dict(xatol=1e-6, fatol=1e-4, maxiter=4000)) for a in (0.5, 1, 2)),
                key=lambda r: r.fun)
        a, k, lam = np.exp(r.x)
        return stats.exponweib(a, k, scale=lam).ppf
    # Champernowne
    med, sd = np.median(s), s.std()
    unpack = lambda t: (np.exp(t[0]), -1 + np.exp(t[1]), t[2])

    def nll(t):
        v = -champ_lin.logpdf(s, *unpack(t)).sum()
        return v if np.isfinite(v) else 1e300
    starts = [[np.log(c / sd), np.log(l1), med] for c in (1.0, 2.0, 4.0) for l1 in (0.2, 1.0, 4.0)]
    r = min((optimize.minimize(nll, st, method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-6, maxiter=20000))
             for st in sorted(starts, key=nll)[:3]), key=lambda r: r.fun)
    p = unpack(r.x)
    return lambda q, p=p: champ_lin.ppf(np.asarray(q), *p)


# ---------------------------------------------------------------- dynamics
R = np.roll(np.eye(N), 1, axis=0)                           # advection by one cell
Lap = np.roll(np.eye(N), 1, axis=0) + np.roll(np.eye(N), -1, axis=0) - 2 * np.eye(N)
K = np.diag(a_phys) @ (np.eye(N) + nu * Lap) @ R            # one step: advect, mix, scale


def ens_step(S):
    S = np.roll(S, 1, axis=1)
    S = S + nu * (np.roll(S, 1, axis=1) - 2 * S + np.roll(S, -1, axis=1))
    return a_phys * S


# initial cumulant fields (Weibull k0, lam0, independent between grid points)
raw0 = [weibull_raw(k0, lam0, r) for r in (1, 2, 3, 4)]
cum0 = cumulants_from_raw(*raw0)

S = stats.weibull_min(k0, scale=lam0).rvs(size=(M, N), random_state=rng)
P = np.eye(N)
kn, ln = k0.copy(), lam0.copy()                             # "no closure" family solve
step = 0
results = {}
t_cum, t_match = 0.0, 0.0
for cp in CHECKPOINTS:
    while step < cp:
        S = ens_step(S)
        P = K @ P
        kn, ln = np.roll(kn, 1), a_phys * np.roll(ln, 1)    # mixing ignored
        step += 1

    t0 = time.perf_counter()
    mean = P @ cum0[0]
    k2, k3, k4 = ((P ** n) @ cum0[n - 1] for n in (2, 3, 4))
    t_cum += time.perf_counter() - t0

    emp_q = np.quantile(S[:, TEST], PROBS, axis=0)          # (4, 40)
    e_mean, e_var = S[:, TEST].mean(0), S[:, TEST].var(0)
    e_skew = stats.skew(S[:, TEST], axis=0)
    e_exk = stats.kurtosis(S[:, TEST], axis=0)
    cv, sk, exk = shape_stats(mean[TEST], k2[TEST], k3[TEST], k4[TEST])

    res = dict(check=dict(mean=np.max(np.abs(mean[TEST] / e_mean - 1)),
                          var=np.max(np.abs(k2[TEST] / e_var - 1)),
                          skew=np.max(np.abs(sk - e_skew)), exk=np.max(np.abs(exk - e_exk))),
               skew=sk, exk=exk, fam={})
    nc_q = np.array([stats.weibull_min(kn[i], scale=ln[i]).ppf(PROBS) for i in TEST]).T
    res["fam"]["no closure (W2p)"] = dict(qerr=(nc_q - emp_q) / emp_q * 100)
    for name, match in FAMILIES.items():
        t0 = time.perf_counter()
        qs, exks, resid = [], [], []
        for j, i in enumerate(TEST):
            ppf, fam_exk, r = match(mean[i], k2[i], k3[i])
            qs.append(ppf(np.array(PROBS)))
            exks.append(fam_exk)
            resid.append(r)
        t_match += time.perf_counter() - t0
        res["fam"][name] = dict(qerr=(np.array(qs).T - emp_q) / emp_q * 100,
                                exk=np.array(exks), resid=np.array(resid))
    if cp == CHECKPOINTS[-1]:
        for name in FAMILIES:
            qs = [mle_fit(name, S[:, i])(np.array(PROBS)) for i in TEST]
            res["fam"][name]["mle_qerr"] = (np.array(qs).T - emp_q) / emp_q * 100
    results[cp] = res
    print(f"checkpoint {cp:>2} steps done", flush=True)


# ---------------------------------------------------------------- report
def mw(e):
    """mean (worst |.|) over grid points"""
    return f"{np.mean(e):+6.1f} ({np.max(np.abs(e)):4.1f})"


print(f"\nEnsemble {M:,} members x {N} points; per step: advect 1 cell, diffuse (nu = {nu}), scale by a(x).")
print("Initial wind independent between grid points, Weibull(k0(x), lam0(x)).  Errors are fitted quantile")
print(f"minus ensemble quantile, in %, over {TEST.size} grid points: mean (worst).\n")
print("Exactness of the cumulant equations (max over points, closure vs ensemble):")
for cp, r in results.items():
    c = r["check"]
    print(f"  step {cp:>2}: mean {c['mean']*100:.2f}%   variance {c['var']*100:.2f}%   "
          f"skewness {c['skew']:.3f} (abs)   excess kurtosis {c['exk']:.3f} (abs)")
print("\nShape of the true distribution (from the cumulant equations, median over points):")
for cp, r in results.items():
    print(f"  step {cp:>2}: skewness {np.median(r['skew']):.3f}   excess kurtosis {np.median(r['exk']):.3f}")

for cp, r in results.items():
    print(f"\n### after {cp} steps")
    print(f"{'family solve':<22}" + "".join(f"{p*100:>16g}%" for p in PROBS) + "   kurtosis gap  match resid")
    for name, f in r["fam"].items():
        line = f"{name:<22}" + "".join(f"{mw(f['qerr'][j]):>17}" for j in range(len(PROBS)))
        if "exk" in f:
            line += f"   {np.median(f['exk'] - r['exk']):+8.3f}   {np.max(f['resid']):.1e}"
        print(line)

last = results[CHECKPOINTS[-1]]
print(f"\n### after {CHECKPOINTS[-1]} steps: closure vs best possible fit in the family (MLE on the ensemble)")
print(f"{'family':<18}{'':>9}" + "".join(f"{p*100:>16g}%" for p in PROBS))
for name in FAMILIES:
    f = last["fam"][name]
    print(f"{name:<18}{'closure':>9}" + "".join(f"{mw(f['qerr'][j]):>17}" for j in range(len(PROBS))))
    print(f"{'':<18}{'MLE':>9}" + "".join(f"{mw(f['mle_qerr'][j]):>17}" for j in range(len(PROBS))))
print(f"\ncost over all checkpoints: cumulant fields {t_cum*1e3:.0f} ms (all {N} points); "
      f"moment matching {t_match:.1f} s ({len(FAMILIES)} families x {TEST.size} points x {len(CHECKPOINTS)} times)")
print("kurtosis gap: family's implied excess kurtosis minus true, median over points (not used in the fit).")
