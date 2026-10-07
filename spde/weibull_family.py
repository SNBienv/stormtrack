"""
Weibull-family wind prototype.

Part 1  EAR(1) exponential Markov process -> Weibull wind speed (no Gaussian).
Part 2  1-D grid: advection + scaling/power-law physics + Markov gust step.
        "Family solve" evolves only k(x,t), lambda(x,t).
        "Brute force" evolves a 20,000-member ensemble of wind speeds.
        Check: is the ensemble exactly Weibull(k, lambda) at every grid point?
Part 3  Mixing (diffusion) -- an operation the Weibull family is NOT closed under.
        How far does the ensemble drift from the best-fitting Weibull?
"""
import time
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)


def ear1_step(Y, rho, rng):
    """Gaver-Lewis EAR(1): keeps Y ~ Exp(1) exactly, Markov in time."""
    B = rng.random(Y.shape) > rho          # Bernoulli(1 - rho)
    E = rng.exponential(1.0, Y.shape)
    return rho * Y + B * E


# ---------------------------------------------------------------- Part 1
print("=" * 70)
print("PART 1  EAR(1) -> Weibull time series")
print("=" * 70)
k, lam, rho, T = 2.0, 8.0, 0.9, 200_000
Y = np.empty(T)
Y[0] = rng.exponential()
for t in range(1, T):
    Y[t] = ear1_step(np.array(Y[t - 1]), rho, rng)
S = lam * Y ** (1 / k)

# Thin the series so samples are ~independent for a valid KS p-value
thin = S[::60]                                   # rho^60 ~ 0.002
D, p = stats.kstest(thin, stats.weibull_min(k, scale=lam).cdf)
k_fit, _, lam_fit = stats.weibull_min.fit(S, floc=0)
print(f"target          k = {k:.3f}   lambda = {lam:.3f}")
print(f"fitted (MLE)    k = {k_fit:.3f}   lambda = {lam_fit:.3f}")
print(f"KS test vs target Weibull (n={thin.size}): D = {D:.4f}, p = {p:.3f}")
Yc = Y - Y.mean()
print("autocorrelation of Y   lag:  empirical   theory rho^lag")
for lag in (1, 2, 5, 10, 20):
    ac = np.dot(Yc[:-lag], Yc[lag:]) / np.dot(Yc, Yc)
    print(f"                       {lag:>3}:    {ac:.4f}      {rho**lag:.4f}")

# ---------------------------------------------------------------- Part 2
print()
print("=" * 70)
print("PART 2  Family solve vs 20,000-member ensemble on a 1-D grid")
print("=" * 70)
N, M, steps = 200, 20_000, 150
x = np.arange(N) / N
k0 = 2.0 + 0.3 * np.sin(2 * np.pi * x)
lam0 = 7.0 + 2.0 * np.cos(2 * np.pi * x)
a = 1.0 + 0.004 * np.sin(4 * np.pi * x)          # speed-up factor (terrain)
b = 1.0 + 0.003 * np.cos(2 * np.pi * x)          # power-law exponent
rho_x = 0.85 + 0.1 * np.sin(2 * np.pi * x) ** 2  # Markov persistence field


def physics_family(k, lam):
    # S -> a S^b  maps Weibull(k, lam) to Weibull(k/b, a lam^b)  (exact)
    return k / b, a * lam ** b


# Family solve: only 2 x N numbers
t0 = time.perf_counter()
kf, lf = k0.copy(), lam0.copy()
for _ in range(steps):
    kf, lf = np.roll(kf, 1), np.roll(lf, 1)        # advection, Courant number 1
    kf, lf = physics_family(kf, lf)
    # Markov gust step leaves the family unchanged (EAR(1) keeps Exp(1))
t_family = time.perf_counter() - t0

# Brute force ensemble: M x N numbers
t0 = time.perf_counter()
Se = stats.weibull_min(k0, scale=lam0).rvs(size=(M, N), random_state=rng)
kc, lc = k0.copy(), lam0.copy()                    # needed to define the gust step
for _ in range(steps):
    Se = np.roll(Se, 1, axis=1)
    kc, lc = np.roll(kc, 1), np.roll(lc, 1)
    Se = a * Se ** b                               # physics on every sample
    kc, lc = physics_family(kc, lc)
    Ye = (Se / lc) ** kc                           # to exponential space
    Ye = ear1_step(Ye, rho_x, rng)                 # Markov gust process
    Se = lc * Ye ** (1 / kc)                       # back to wind speed
t_ens = time.perf_counter() - t0

pvals = np.empty(N)
Ds = np.empty(N)
for i in range(N):
    Ds[i], pvals[i] = stats.kstest(Se[:, i], stats.weibull_min(kf[i], scale=lf[i]).cdf)
idx = np.linspace(0, N - 1, 8).astype(int)
fits = [stats.weibull_min.fit(Se[:, i], floc=0) for i in idx]

print(f"k range after {steps} steps:      {kf.min():.3f} .. {kf.max():.3f}  (started 1.700 .. 2.300)")
print(f"lambda range after {steps} steps: {lf.min():.3f} .. {lf.max():.3f}  (started 5.000 .. 9.000)")
print(f"KS test at all {N} grid points against the FAMILY prediction:")
print(f"   max D = {Ds.max():.4f}   (critical D at 5% for n={M}: {1.358/np.sqrt(M):.4f})")
print(f"   points rejected at 5% level: {np.mean(pvals < 0.05)*100:.1f}%  (expected ~5% if exact)")
print(f"   uniformity of p-values (KS on p-values): p = {stats.kstest(pvals, 'uniform').pvalue:.3f}")
print("   grid pt   family k  ensemble-fit k   family lam  ensemble-fit lam")
for i, (kk, _, ll) in zip(idx, fits):
    print(f"   {i:>6}    {kf[i]:.4f}    {kk:.4f}          {lf[i]:.4f}     {ll:.4f}")
print(f"cost: family solve {t_family*1e3:.1f} ms ({2*N} numbers), "
      f"ensemble {t_ens:.1f} s ({M*N:,} numbers)")

# ---------------------------------------------------------------- Part 3
print()
print("=" * 70)
print("PART 3  Mixing (diffusion): family NOT closed -> how big is the error?")
print("=" * 70)
nu, mix_steps = 0.25, 20
Sm = stats.weibull_min(k0, scale=lam0).rvs(size=(M, N), random_state=rng)
for _ in range(mix_steps):
    Sm = Sm + nu * (np.roll(Sm, 1, axis=1) - 2 * Sm + np.roll(Sm, -1, axis=1))
rej, q99_err, q999_err = [], [], []
for i in range(0, N, 5):
    s = Sm[:, i]
    kk, _, ll = stats.weibull_min.fit(s, floc=0)
    fitted = stats.weibull_min(kk, scale=ll)
    rej.append(stats.kstest(s, fitted.cdf).pvalue < 0.05)
    q99_err.append((fitted.ppf(0.99) - np.quantile(s, 0.99)) / np.quantile(s, 0.99))
    q999_err.append((fitted.ppf(0.999) - np.quantile(s, 0.999)) / np.quantile(s, 0.999))
print(f"after {mix_steps} diffusion steps (nu={nu}), best-fit Weibull at {len(rej)} points:")
print(f"   KS rejects the best-fit Weibull at {np.mean(rej)*100:.0f}% of points")
print(f"   99th percentile error:   mean {np.mean(q99_err)*100:+.1f}%, worst {np.max(np.abs(q99_err))*100:.1f}%")
print(f"   99.9th percentile error: mean {np.mean(q999_err)*100:+.1f}%, worst {np.max(np.abs(q999_err))*100:.1f}%")
