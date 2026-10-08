"""Bursty-gust (Friedman-Cai-Xie) model for kinetic energy -> law of speed S = sqrt(2E).

Shot noise: dE/dt = -gamma E + sum_i B_i delta(t - t_i),  t_i ~ Poisson(alpha),  B_i ~ Exp(mean beta)
=> stationary E ~ Gamma(shape m = alpha/gamma, scale beta)  =>  S = sqrt(2E) ~ Nakagami(m).
m = 1  <=>  Rayleigh = Weibull(k=2)  <=>  the repo's EAR(1) (Gaver-Lewis) step.
"""
import numpy as np
from scipy import stats, special

rng = np.random.default_rng(1)


def simulate_shot(m, n, dt=0.05, gamma=1.0, beta=1.0):
    """Exact discretisation of the shot-noise ODE: decay over dt + Poisson(alpha dt) bursts,
    each decayed by a uniform fraction of the step."""
    alpha = m * gamma
    E = np.empty(n)
    e = m * beta
    dec = np.exp(-gamma * dt)
    N = rng.poisson(alpha * dt, n)
    for t in range(n):
        e *= dec
        if N[t]:
            u = rng.random(N[t])
            e += (rng.exponential(beta, N[t]) * np.exp(-gamma * dt * u)).sum()
        E[t] = e
    return E


def fits(S, label):
    S = S[S > 0]
    k, _, lam = stats.weibull_min.fit(S, floc=0)
    llw = stats.weibull_min.logpdf(S, k, 0, lam).sum()
    a, c, _, sc = stats.exponweib.fit(S, floc=0)
    lle = stats.exponweib.logpdf(S, a, c, 0, sc).sum()
    q = 0.999
    emp = np.quantile(S, q)
    qw = stats.weibull_min.ppf(q, k, 0, lam)
    qe = stats.exponweib.ppf(q, a, c, 0, sc)
    dAIC = (2 * 2 - 2 * llw) - (2 * 3 - 2 * lle)
    Dw = stats.kstest(S, stats.weibull_min(k, 0, lam).cdf).statistic
    De = stats.kstest(S, stats.exponweib(a, c, 0, sc).cdf).statistic
    print(f"{label:42s} Weib k={k:5.2f} | EW a={a:5.2f} k={c:5.2f} | dAIC(W-EW)={dAIC:9.1f} "
          f"| KS W={Dw:.4f} EW={De:.4f} | q99.9 err W={100*(qw/emp-1):+5.1f}% EW={100*(qe/emp-1):+5.1f}%")


n = 200_000
print("-- check: simulated shot noise vs Gamma(m) (KS on thinned series)")
for m in [0.5, 1.0, 2.0]:
    E = simulate_shot(m, 400_000)[20_000::40]
    print(f"m={m}: KS vs Gamma(m,1): D={stats.kstest(E, stats.gamma(m).cdf).statistic:.4f}  n={E.size}")

print("\n-- A. intrinsic only: E ~ Gamma(m), S = sqrt(2E) (Nakagami-m)")
for m in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]:
    S = np.sqrt(2 * rng.gamma(m, 1 / m, n))
    fits(S, f"Nakagami m={m}")

print("\n-- B. intrinsic (m=1 bursts) x extrinsic slow modulation of burst SIZE beta ~ Gamma(nu)")
for nu in [1, 2, 4, 8, 16]:
    beta = rng.gamma(nu, 1 / nu, n)
    S = np.sqrt(2 * beta * rng.exponential(1, n))
    fits(S, f"m=1, beta~Gamma(nu={nu})  (K-distribution)")

print("\n-- C. intrinsic (m=2) x extrinsic lognormal modulation of mean energy, sigma")
for sig in [0.2, 0.4, 0.6]:
    beta = np.exp(sig * rng.normal(size=n) - sig**2 / 2)
    S = np.sqrt(2 * beta * rng.gamma(2, 0.5, n))
    fits(S, f"m=2, ln beta ~ N(0,{sig}^2)")

print("\n-- D. extrinsic modulation of burst RATE: m ~ Gamma-distributed around m0 (beta fixed)")
for m0, cv in [(1, 0.5), (2, 0.5), (2, 1.0)]:
    sh = 1 / cv**2
    mm = rng.gamma(sh, m0 / sh, n)
    S = np.sqrt(2 * rng.gamma(mm, 1.0))
    fits(S, f"m ~ Gamma(mean {m0}, cv {cv})")

print("\n-- E. 'mast' block average: S(t)=sqrt(2E(t)) shot noise m=1, averaged over T (units of 1/gamma)")
E = simulate_shot(1.0, 2_000_000, dt=0.05)
S = np.sqrt(2 * E)
for T in [1, 5, 20]:
    w = int(T / 0.05)
    Sb = S[: (S.size // w) * w].reshape(-1, w).mean(1)
    fits(Sb, f"m=1 block mean T={T}/gamma (n={Sb.size})")
for T in [5, 20]:
    w = int(T / 0.05)
    Sb = S[: (S.size // w) * w].reshape(-1, w).max(1)
    fits(Sb, f"m=1 block MAX (gust) T={T}/gamma (n={Sb.size})")
