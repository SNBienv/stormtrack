# spde: wind as an evolving distribution family

Prototype for a stochastic wind model that is solved **once**, without an ensemble. The wind speed at
every point and time is assumed to belong to a known distribution family (Weibull, exponentiated
Weibull, Champernowne). Deterministic equations are solved for the **parameter fields** of that
family, for example `k(x,t)` and `λ(x,t)`, instead of for one random realisation of the wind.

```
Usual SPDE:      u(x, t, ω)                      one random value per realisation
Here:            u(x, t) ~ Family(θ(x, t))       solve deterministic PDEs for θ
```

No Gaussian step is used anywhere. Persistence in time (gusts) comes from an exponential Markov
process (EAR(1), Gaver & Lewis 1980), mapped to wind speed through the family's own CDF:
`Y = −log(1 − F(S))` is Exponential(1) for **any** continuous family, so the gust step leaves the
family's parameters unchanged.

## Which physics keeps a family exact

| Operation on wind speed `S` | Weibull 2p `(k, λ)` | Weibull 3p `(k, λ, γ)` | Champernowne `(α, λ, v₀, L)` |
|---|---|---|---|
| Advection by a flow `c` | `∂θ/∂t + c·∇θ = 0` | same | same |
| Scaling `S → aS` | `λ → aλ` | `λ → aλ, γ → aγ` | `α → α/a, v₀ → av₀, L → aL` |
| Shift `S → S + d` | **not closed** | `γ → γ + d` | `v₀ → v₀ + d, L → L + d` |
| Power law `S → aS^b` | `k → k/b, λ → aλ^b` | not closed | not closed |
| Markov gust step (EAR(1) in `−log(1 − F)`) | unchanged | unchanged | unchanged |
| **Mixing / diffusion** (averaging neighbours) | **not closed** | **not closed** | **not closed** |

Where every operation is closed, the family solve reproduces a 20,000-member ensemble
**exactly**: the KS test rejects at 3–5% of grid points, the rate expected by chance. It runs
about 10,000 times faster.

Mixing is the one operation that breaks all of the families. `mixing_closure.py` closes it (below).

## Scripts

Run `python fetch_data.py` once to download the two real-data sets into `data/`. They are taken from
the CRAN GitHub mirrors, because NOAA, Meteostat and the journal sites are blocked from the cloud
container. Every script prints its own tables. Numbers below come from runs on 2026-10-07.

| Script | What it does | Run time |
|---|---|---|
| `weibull_family.py` | EAR(1) → Weibull time series; family solve vs 20k ensemble (advection, scaling, power law, gusts); mixing error of Weibull 2p | ~1 min |
| `family_solve_affine.py` | Same exactness test for Champernowne (Ndeba form) and Weibull 3p under advection + affine physics + gusts | ~1.5 min |
| `champernowne_mixing.py` | Mixed (diffused) Weibull ensemble: does Champernowne fit it better than Weibull? | ~2 min |
| `real_wind_fit.py` | Fits all families to real wind: London hourly 1998–2005 and Ireland 12 stations 1961–1978, including averaged ("mixed") series | ~3 min |
| `real_wind_fit_ndeba.py` | As above, with the Ndeba et al. (2025) Champernowne reconstruction fitted by least squares (as in the paper) and by maximum likelihood | ~3 min |
| `mixing_closure.py` | **New.** Ensemble-free closure for mixing via exact cumulant equations + moment matching, scored against a 100k ensemble | ~10 min |
| `champ_lin.py`, `champ4.py` | Champernowne densities: closed-form CDF/quantile, normalisation (`python champ_lin.py` self-checks) | — |
| `fetch_data.py` | Downloads the datasets | seconds |

Requires `numpy`, `scipy`, and `pyreadr` for the real-data scripts.

## Results so far

### 1. Family solve is exact where the family is closed

- EAR(1) gives Weibull(2, 8) with fitted k = 1.999, λ = 7.975 (KS p = 0.78), and autocorrelation
  ρ^lag as designed.
- Weibull 2p under advection, scaling, power law and gusts: KS rejects at 5.0% of 200 points.
  The shape k drifts from 1.7–2.3 to 1.52–2.58, and the ensemble stays Weibull throughout.
  Cost: 9.5 ms for 400 numbers, against 38 s for 4,000,000.
- Champernowne (4 parameters) and Weibull 3p under advection, affine physics and gusts: KS rejects at
  4% and 3% of 100 points.

### 2. Mixing breaks it, and swapping in a heavier-tailed family does not fix it

After 20 diffusion steps, the best-fit (MLE) Weibull underestimates the 99.9th percentile by 6.8% on
average. Power-law-tailed Champernowne variants (log-logistic, Buch-Larsen 3p) overshoot it by
+13% to +21%. Averaging makes the tail lighter, which puts it between the two.

### 3. Real wind data (London hourly; Ireland, 12 stations)

- The **exponentiated Weibull** `F(x) = (1 − exp(−(x/λ)^k))^a` fits best overall. It is best by AIC
  at 10 of 12 Irish stations, and its 99.9th percentile on London hourly is within 0.2%.
- Weibull 2p underestimates London hourly extremes by 13% at the 99.9th percentile.
- The classic 3-parameter Champernowne of log-speed fits badly (99.9th percentile +79% to +254%).
- The Ndeba form `n / (cosh(α(v − v₀)) + λ)` (same as Wikipedia's 4 parameters) fits worse than Weibull 2p by AIC on
  these windy, skewed sites. Fitted by maximum likelihood, it still gets the extremes to within a
  few percent. The paper's least-squares fit gives an amplitude n within 1% of the normalising
  constant.

### 4. Mixing closure without an ensemble (`mixing_closure.py`)

Setup: 200-point periodic grid; each step advects by one cell, diffuses (ν = 0.25) and scales by a
terrain factor a(x). The initial wind is Weibull(k₀(x), λ₀(x)), independent between grid points.
The reference is a 100,000-member ensemble that does the same to every sample.

**Closure.** The dynamics are linear, so after t steps `S(t) = P S(0)` and every marginal cumulant
evolves exactly:

```
mean(t) = P · mean(0)            κₙ(t)ᵢ = Σⱼ Pᵢⱼⁿ κₙ(0)ⱼ      (n ≥ 2)
```

The family solve carries the fields (mean, κ₂, κ₃, κ₄). At each output time it matches the family's
parameters to the first 2 cumulants (2-parameter families) or the first 3 (3-parameter families).
κ₄ is never used in the fit. It is kept as an independent check of the tail.

- The cumulant equations agree with the ensemble to sampling accuracy: mean 0.1%, variance 1%,
  skewness 0.02. Cost: 7 ms for all 200 points and 4 times, plus 4 s of moment matching in total.
- Mixing drives the distribution towards Gaussian: skewness 0.30 → 0.18 and excess kurtosis
  0.06 → 0.02 between steps 5 and 40.

Quantile error after 40 steps (fitted minus ensemble, %, mean (worst) over 40 points):

| Family solve | 50th | 90th | 99th | 99.9th | Kurtosis gap |
|---|---|---|---|---|---|
| No closure (Weibull 2p, mixing ignored) | −6.0 (9.2) | +46.5 (52.5) | +84.2 (98.7) | +108 (130) | — |
| Weibull 2p, matched to mean and variance | +1.8 (2.1) | −1.2 (1.4) | −5.6 (6.7) | −9.2 (10.6) | +0.45 |
| Weibull 3p, matched to mean, variance and skewness | +0.0 (0.2) | +0.2 (0.3) | −0.7 (1.2) | −1.7 (2.3) | −0.29 |
| **Exp-Weibull 3p**, matched to mean, variance and skewness | −0.0 (0.1) | −0.0 (0.1) | **+0.1 (0.4)** | **+0.2 (0.8)** | +0.04 |
| Champernowne (Ndeba form), matched to mean, variance and skewness | +0.4 (0.7) | −2.3 (4.7) | +3.1 (7.2) | +11.7 (24.4) | +1.16 |

The same holds at steps 5, 10 and 20. Exp-Weibull stays within 0.4% on average (1.6% worst) at
the 99.9th percentile throughout.

Closure compared with the best fit each family can achieve (maximum likelihood on the ensemble
itself), after 40 steps, 99.9th percentile:

| Family | Closure (no ensemble) | Best fit (MLE on the ensemble) |
|---|---|---|
| Weibull 2p | −9.2% | −6.1% |
| Weibull 3p | −1.7% | −4.0% |
| Exp-Weibull 3p | +0.2% | +0.2% |
| Champernowne (Ndeba form) | +11.7% | −0.0% |

What this shows:

- **Mixing is closable without an ensemble.** Exact cumulant equations plus a 3-parameter family
  match the ensemble at the 99.9th percentile. Ignoring mixing overstates it by about 100%, and
  Weibull 2p with moment matching understates it by about 9%.
- **Exp-Weibull is the right family here.** Its closure is as good as its best possible fit. The
  kurtosis it implies, which the fit never uses, is within 0.04 of the truth. This agrees with the
  real-data fits, where it was also the best family.
- **The kurtosis gap predicts which closures will fail in the tail.** Weibull 3p is too light-tailed
  (−0.29) and slightly under-predicts extremes. Champernowne is far too heavy (+1.16) and
  over-predicts them.
- **The Ndeba Champernowne cannot be closed by moments here.** On speed it is symmetric about v₀
  apart from the cut at 0. Once the mean sits well above zero, it cannot produce the skewness of
  0.2–0.3, so the moment match fails (residual 0.24–0.31). Its maximum-likelihood fit is excellent
  (99.9th percentile −0.0%), so it can describe the distribution. It just cannot be reached from
  cumulants. A Champernowne closure would need a different projection, such as matching quantiles
  or minimising KL divergence.
- **Weibull 3p closure beats its own maximum-likelihood fit in the tail.** Maximum likelihood weights
  the bulk of the distribution, while the skewness match weights the tail.

## Is the approach new?

The pieces exist; the combination appears not to (based on a handful of web searches, not a
literature review):

- **Presumed-PDF methods** in combustion and turbulence assume a family (usually beta) and evolve its
  moments. This is the closest precedent. Pope's transported-PDF methods evolve the whole PDF.
- **Bulk cloud microphysics** evolves gamma drop-size parameters inside weather models.
- **Projection / assumed-density filters** (Brigo, Hanzon & Le Gland) project Fokker–Planck onto a
  family, mostly for Gaussian or low-dimensional problems.
- **Weibull-stationary SDEs for wind** (Zárate-Miñano & Milano; arXiv 1511.02345, 2606.12097) model
  one point in time only, with no spatial physics.

No precedent was found for evolving Weibull or Champernowne wind parameters through physics-based
transport equations, for a closure analysis (which operators keep each family exact, and how to
close mixing), or for the probability-transform Markov step that works for any family. Before claiming
novelty, search "presumed PDF" together with wind or the atmospheric boundary layer, gust
parametrisation, and statistical-dynamical downscaling of Weibull parameters.

## Open items

1. **Ndeba et al. (2025) Champernowne: form confirmed.** It is the same 4-parameter density as in
   the Wikipedia article, `f(y) = n / (cosh(α(y − y₀)) + λ)`, with parameters n, α, λ, y₀.
   `champ_lin.py` implements exactly this form on wind speed, truncated at 0. With n free, it is the
   "Champ-Ndeba LS 4p" row of `real_wind_fit_ndeba.py`, which is the paper's least-squares fit.
   With n fixed by normalisation, it is the "Champ-Ndeba MLE" row. The same density on log-speed is
   the "Champernowne 3p (classic)" row of `real_wind_fit.py`.
2. **Spatial dependence.** The mixing test starts from values that are independent between grid
   points. Real fields are correlated. Correlated mixing needs the cross-cumulants (or a copula for
   the dependence) carried alongside the marginal parameters.
3. **Mixing combined with nonlinear physics and gusts.** After a power law or a gust step, the
   cumulants no longer evolve linearly. The closure then has to re-project at every step.
4. **Real-data check of the closure.** For example, predict the distribution of the Irish 12-station
   mean, or of London daily means, from the single-station families plus their correlations.
5. **Link to WRF.** The parameter fields would replace one deterministic wind field with a
   distribution at each grid point. The residual between WRF and the masts would set the noise.
