"""
Mixing case: does a Champernowne family fit mixed (diffused) Weibull wind better than Weibull?

Families compared at each grid point (all fitted by maximum likelihood):
  Weibull                  2 params (k, lambda)                 tail ~ exp(-x^k)
  Champernowne, c = 0      2 params (alpha, M)  [log-logistic]   tail ~ x^-alpha
  Modified Champernowne    3 params (alpha, M, c)                tail ~ x^-alpha
      F(x) = ((x+c)^a - c^a) / ((x+c)^a + (M+c)^a - 2 c^a),   x >= 0
"""
import numpy as np
from scipy import stats, optimize

rng = np.random.default_rng(7)

N, M_ENS, mix_steps, nu = 200, 100_000, 20, 0.25
x = np.arange(N) / N
k0 = 2.0 + 0.3 * np.sin(2 * np.pi * x)
lam0 = 7.0 + 2.0 * np.cos(2 * np.pi * x)


# ------------------------------------------------ modified Champernowne
def champ_logpdf(x, a, m, c):
    A, C, D = (x + c) ** a, c ** a, (m + c) ** a
    return np.log(a) + (a - 1) * np.log(x + c) + np.log(D - C) - 2 * np.log(A + D - 2 * C)


def champ_cdf(x, a, m, c):
    A, C, D = (x + c) ** a, c ** a, (m + c) ** a
    return (A - C) / (A + D - 2 * C)


def champ_ppf(p, a, m, c):
    C, D = c ** a, (m + c) ** a
    return ((p * D + C * (1 - 2 * p)) / (1 - p)) ** (1 / a) - c


def champ_fit(s, c_free):
    a0, _, m0 = stats.fisk.fit(s, floc=0)           # log-logistic start
    if not c_free:
        return a0, m0, 0.0
    best = None
    for c_init in (0.5 * m0, 2 * m0, 8 * m0):       # several starts: likelihood is flat in c
        def nll(th):
            a, m, c = np.exp(th)
            return -champ_logpdf(s, a, m, c).sum()
        r = optimize.minimize(nll, np.log([a0, m0, c_init]), method="Nelder-Mead",
                              options=dict(xatol=1e-6, fatol=1e-4, maxiter=4000))
        if best is None or r.fun < best.fun:
            best = r
    return tuple(np.exp(best.x))


# ------------------------------------------------ build mixed ensemble
S = stats.weibull_min(k0, scale=lam0).rvs(size=(M_ENS, N), random_state=rng)
for _ in range(mix_steps):
    S = S + nu * (np.roll(S, 1, axis=1) - 2 * S + np.roll(S, -1, axis=1))

probs = (0.5, 0.9, 0.99, 0.999)
fams = ("Weibull", "Champernowne c=0", "Champernowne 3-par")
res = {f: dict(rej=[], aic=[], qerr={p: [] for p in probs}) for f in fams}
cvals = []

for i in range(0, N, 5):
    s = S[:, i]
    emp_q = np.quantile(s, probs)

    k, _, lam = stats.weibull_min.fit(s, floc=0)
    w = stats.weibull_min(k, scale=lam)
    fitted = {
        "Weibull": (w.cdf, w.ppf, w.logpdf(s).sum(), 2),
    }
    for name, c_free in (("Champernowne c=0", False), ("Champernowne 3-par", True)):
        a, m, c = champ_fit(s, c_free)
        if c_free:
            cvals.append(c / m)
        fitted[name] = (lambda v, a=a, m=m, c=c: champ_cdf(v, a, m, c),
                        lambda p, a=a, m=m, c=c: champ_ppf(np.asarray(p), a, m, c),
                        champ_logpdf(s, a, m, c).sum(), 3 if c_free else 2)

    for f, (cdf, ppf, ll, npar) in fitted.items():
        res[f]["rej"].append(stats.kstest(s, cdf).pvalue < 0.05)
        res[f]["aic"].append(2 * npar - 2 * ll)
        for p, q in zip(probs, emp_q):
            res[f]["qerr"][p].append((ppf(p) - q) / q * 100)

npts = len(res["Weibull"]["rej"])
print(f"Mixed ensemble: {M_ENS:,} members, {mix_steps} diffusion steps (nu={nu}), {npts} grid points tested\n")
print(f"{'family':<22}{'KS reject':>10}{'mean dAIC':>12}{'  quantile error, mean (worst)':>32}")
print(f"{'':<22}{'':>10}{'vs Weibull':>12}   " + "".join(f"{p*100:>11g}%" for p in probs))
aic_w = np.array(res["Weibull"]["aic"])
for f in fams:
    r = res[f]
    daic = np.mean(np.array(r["aic"]) - aic_w)
    q = "".join(f"{np.mean(r['qerr'][p]):>+6.1f} ({np.max(np.abs(r['qerr'][p])):>4.1f})" for p in probs)
    print(f"{f:<22}{np.mean(r['rej'])*100:>9.0f}%{daic:>12.0f}   {q}")
print(f"\n(dAIC < 0 means a better fit than Weibull; quantile error = fitted minus empirical, in %)")
print(f"Fitted c/M for 3-parameter Champernowne: median {np.median(cvals):.2f}, range {min(cvals):.2f} .. {max(cvals):.2f}")
