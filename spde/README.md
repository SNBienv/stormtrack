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
| `mixing_closure.py` | Ensemble-free closure for mixing via exact cumulant equations + moment matching, scored against a 100k ensemble | ~10 min |
| `rectification.py` | Speed bias from a zero-mean unresolved vector (Rice); London demo | seconds |
| `rice_family_ns2d.py` | Rice vector family solve on deterministic 2-D Navier–Stokes, smooth/rough regimes | ~15 / 40 min |
| `singular.py` | Cyclone as a tracked singular component (Holland + centre-error Rice mixture), strip-width flag | ~1 min |
| `rice_closure.py` | Calibration of the Rice law (unresolved part as a Gaussian vector) on London | ~3 min |
| `team/` | Five specialist reports and their scripts | — |
| `l96_intrinsic.py` | Deterministic two-scale Lorenz-96: the measured unresolved term vs deterministic and Gaussian closures | ~7 min |
| `mast_wrf/` | GEP mast / WRF measurement modules (from the user's ns2d session) + `rectification_budget.py` | user's data |
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

> **Correction (team review, `team/ramanujan`, `team/fields`).** A 100,000-member ensemble is
> itself off by up to about 1% at the 99.9th percentile, so the ensemble-based errors below are
> noise-limited. Against an *exact* reference (Gil-Pelaez inversion, or FFT convolution), the
> exp-Weibull closure has a **systematic over-prediction** at 99.9%: +0.41% at 5 steps, falling to
> +0.11% at 40 steps. In the deeper tail it under-predicts (−2% at an exceedance probability of
> 1e−9), because its tail exponent is wrong: 3–4, against the true ≈ 1.7.
>
> Two exact upgrades already exist, both using the cumulants this script tracks:
> - Cornish–Fisher with κ₄: −0.03% error;
> - the saddlepoint (Lugannani–Rice) on the exact cumulant generating function: about 0.001%, at
>   27 ms per point.
>
> The conclusions below about which family is best still hold. For extremes, use the saddlepoint.

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

### 5. The noise is the unresolved part of the system, not something injected

The project axiom: the "noise" is whatever the model cannot resolve or measure. It is part of the
system, so it has to be measured or derived, never added from outside.

**Rectification: how a zero-mean unresolved part creates a speed bias** (`rectification.py`). A
model holds the resolved vector V. A cup measures `|V + v′|`. Speed is convex in the components,
so `E|V + v′| ≈ |V| + σ²_cross / (2|V|)`, exactly the Rice mean for isotropic Gaussian v′. A model
that is perfect for the resolved vector therefore reads low. London hourly data, with "resolved"
meaning the vector mean over a window:

| Window | Actual gap (mean speed − resolved) | Predicted from the unresolved variance only | Explained | r per window |
|---|---|---|---|---|
| 3 h | 0.101 m/s | 0.092 | 91% | 0.995 |
| 6 h | 0.186 | 0.170 | 92% | 0.990 |
| 12 h | 0.316 | 0.296 | 94% | 0.987 |
| 24 h | 0.511 (11% of the mean) | 0.475 | 93% | 0.989 |

With |V| → 0 the Rice law becomes Rayleigh, which is Weibull with k = 2. The Weibull family for
speed is what an unresolved isotropic vector looks like. `mast_wrf/rectification_budget.py`
applies this to the GEP mast and WRF U10/V10. The unresolved v′ is the mast's 1-min deviation
inside each hour; by Taylor's hypothesis, 1 h × 5.5 m/s ≈ 20 km ≈ 7 Δx. The script predicts the
WRF speed bias out of sample, with no tuning. It must be run on the user's machine, where the
data are.

**A fully deterministic test system** (`l96_intrinsic.py`). Two-scale Lorenz-96 (K = 8 resolved X,
32 fast Y per X, F = 20, b = c = 10). Nothing in it is random. The coarse model's missing term is
the sub-grid tendency U = −(hc/b) ΣY. Its conditional law given X and its memory are measured on
a training truth (7.7 M samples) and then scored on an independent truth.

What the unresolved part looks like:

- non-Gaussian: skewness −0.39, excess kurtosis +0.58;
- state-dependent: its sd varies by a factor of 1.6 across X, and its skewness from −0.81 to +0.17;
- oscillating memory: the autocorrelation is +0.98 at lag 0.005, +0.37 at 0.05, −0.14 at 0.15.

The lag-1 fit implies 0.28 time units of memory. The Green–Kubo integral time is only 0.042.

| Coarse model | Climate mean (truth 3.771) | Hellinger | RMSE at 1.0 | CRPS at 1.0 | Spread/error at 1.0 |
|---|---|---|---|---|---|
| C0: no sub-grid term | 3.340 | 0.201 | 8.67 | 7.02 | — |
| C1: E[U\|X] only (deterministic, unbiased at every state) | 3.597 | 0.028 | 2.26 | 1.53 | — |
| C2: C1 + Gaussian AR(1), measured variance and Green–Kubo memory | 3.703 | 0.024 | 2.06 | 1.05 | 1.02 |
| **C3: C1 + measured conditional residual, Green–Kubo memory** | **3.704** | **0.020** | **1.84** | **0.91** | **1.08** |
| C3 with lag-1 memory | 3.709 | 0.027 | 2.02 | 1.04 | 1.36 |

- **The bias comes from the missing S−.** C1's sub-grid term is exactly right on average at every
  state, yet its climate is 0.17 too low. Putting the zero-mean unresolved part back (C2, C3)
  removes about 60% of that bias. The nonlinear dynamics rectify the fluctuations into a mean
  shift, the same mechanism as the speed bias above.
- **The true law of the unresolved part matters.** The measured, state-dependent, non-Gaussian
  residual (C3) beats a Gaussian with the same variance and memory (C2) at every lead. At lead
  1.0, RMSE is 11% lower and CRPS 13% lower. Against the best deterministic closure (C1), RMSE is
  19% lower and CRPS 41% lower.
- **Memory must be the Green–Kubo integral time, not the lag-1 correlation.** With lag-1 memory the
  ensemble is too wide (spread/error 1.4–1.7 at short leads) and loses most of C3's advantage.
  With Green–Kubo memory the spread/error ratio is 1.00–1.11.
- **Every number in C3 is measured on the system itself.** There is no tuning parameter.

### 6. Rice: the unresolved part as a Gaussian vector (`rice_closure.py`)

Feller's theorem rules out closing the mixing of wind *speed* within any non-Gaussian family. The
structural fix (`team/fields`) is to evolve the vector: V = V_res + v′ with v′ ~ N(0, σ²I). The
speed is then exactly **Rice(|V_res|, σ)**, which is closed under mixing plus intrinsic noise and
reduces to Weibull k = 2 at zero mean. It is the same law that explained 91–94% of the London
speed bias in section 5.

Calibration test on London hourly speeds, given each window's resolved vector. A calibrated law
gives 0.90 coverage of the central 90% and 0.010 exceedance above q99:

| Window | Model | KS | Coverage 90% | Above q99 | CRPS | Calm windows (ν < 2σ): KS / above q99 |
|---|---|---|---|---|---|---|
| 12 h | Rice, oracle σ | 0.018 | 0.924 | 0.006 | 0.610 | 0.073 / 0.005 |
| 12 h | Rice, σ predicted from \|V_res\| | 0.041 | 0.905 | 0.024 | 0.651 | 0.077 / 0.032 |
| 12 h | **Rice, compound σ** (log σ scatter) | 0.027 | 0.939 | **0.004** | 0.651 | **0.059 / 0.004** |
| 12 h | Gaussian on speed (Rice mean and sd) | 0.037 | 0.903 | 0.026 | 0.651 | 0.068 / 0.042 |
| 12 h | Gaussian centred on \|V_res\| ("injected") | 0.109 | 0.885 | 0.039 | 0.682 | 0.241 / 0.085 |

The 3, 6 and 24 h windows behave the same way; run the script for all of them.

- **Injected noise centred on the resolved speed is the worst everywhere.** Its CRPS is 5–8% higher
  and calm windows exceed q99 8–19 times too often, because it misses the rectification.
- **Most of Rice's gain is the rectified mean.** With the same mean and spread, a Gaussian on speed
  scores the same CRPS. The Rice shape helps only in calm windows.
- **σ itself is a fluctuating unresolved quantity** (superstatistics, `team/bio`). With σ predicted
  from |V_res| alone, q99 is exceeded 2.4 times too often. Mixing over the measured scatter of
  log σ fixes the tail (0.3–0.6%), but the result is slightly too wide (central coverage
  0.93–0.96).

### 7. Team synthesis: what is established, and what the residual RMSE is

Five specialist agents worked on this independently: a mathematician, a quantum physicist, a
biologist, Ramanujan and a singularity-tracking astrophysicist. Their reports and scripts are in
`team/*/REPORT.md`. Where they agree:

1. **The axiom is a theorem.** By Mori–Zwanzig, the missing term of a coarse model is
   F(t) = e^{tQL}QLx, a deterministic function of the unmeasured fine-scale state. It looks
   random only because that state is not measured. Its memory kernel follows from it through
   fluctuation–dissipation. With chaotic fast scales it converges to a diffusion whose amplitude
   is given by Green–Kubo (references in `team/fields`).
2. **Checked on deterministic L96** (`l96_intrinsic.py`, `team/quantum`, `team/fields`):
   - Einstein / Green–Kubo friction holds within 6–23% when the fast part is chaotic, and fails
     when it is periodic.
   - Green–Kubo predicts the deterministic model's error growth within 4% up to lead 0.1.
   - The 1/c law holds.
   - The measured intrinsic residual removes 60–80% of the climate bias and gives reliable
     ensembles. It has to use the Green–Kubo memory time, not the lag-1 correlation.
3. **Bias from a zero-mean unresolved part is real, but it needs a nonlinearity.** Speed
   rectification explains 91–94% of the London coarse-graining speed gap. In L96 the nonlinear
   dynamics turn the missing term into a climate-mean bias.
4. **Distribution families have a physical meaning:**
   - Weibull k = 2 (Rayleigh) is an unresolved isotropic vector.
   - Exp-Weibull's a is the number of independent eddies or bursts sampled (Nakagami-m /
     max-of-a). The London fits have the "a > 1, k < 2" signature of a fluctuating energy scale.
5. **Where the families stop working:**
   - Mixing of speed (Feller): use Rice on the vector, or a saddlepoint for extremes.
   - Fronts: the spectral strip width δ flags them about 0.3 time units ahead, and the one-point
     goodness of fit is blind to them.
   - A cyclone passing a mast gives a hard edge at Vmax with a (Vmax−V)^−½ spike. That needs a
     mixture whose vortex component is fed by stormtrack's tracked centre (`team/astro`).

**Scorecard on "the residual RMSE between WRF and the masts is exactly the missing S−":**

- **Supported:** S− exists, it can be derived and measured rather than injected, and it creates
  mean bias through nonlinearity.
- **Not yet supported:** that it is *all* of the WRF residual. Mori–Zwanzig splits the residual
  into memory plus noise, plus initial-state error, structural error and instrument error.
  Mesoscale spectra put the intrinsic floor at about 0.5–0.9 m/s rms, against a typical 10-m WRF
  error of 1.5–2.5 m/s (`team/bio`).
- **Testable on the user's data:** `mast_wrf/intrinsic_floor.py` measures that floor from the GEP
  1-min record and splits the WRF MSE into floor and excess. `mast_wrf/rectification_budget.py`
  predicts the speed bias from the unresolved variance, out of sample. Several masts inside one
  WRF cell would give the true dual reporter.

### 8. Rice vector family solve on deterministic 2-D Navier–Stokes (`rice_family_ns2d.py`)

**Truth.** 2-D Navier–Stokes (pseudo-spectral, 256²) with steady Kolmogorov forcing. Nothing is random.

**The coarse model plays WRF.** It carries the resolved cell wind V̄ and a derived unresolved spread σ². The point speed in each cell is then Rice(|V̄|, σ).

σ² comes from two places, neither tuned:

- **Sub-cell spread.** The exact leading term of what a cell average hides is c·(Δ²/24)|∇V̄|². The coefficient c comes from the resolved field itself (Germano's dynamic procedure): at the test scale 2Δ the answer is resolved.
- **Error of the resolved state.** It is measured on a separate training period, as a regression of the true cell wind on the forecast towards climatology, plus the residual variance. Long-lead error is chaos of the resolved scales, not isotropic noise.

Scored on an independent test period against the true point speeds inside every cell. **Smooth regime** (forcing resolved, spectral slope −4.3, 8×8-point cells):

| Lead | Model | KS | Coverage 90% | Above q99 | CRPS | Bias |
|---|---|---|---|---|---|---|
| 0 | Deterministic \|V̄\| | — | — | — | 0.1235 | +0.014 |
| 0 | Injected, uniform σ (same total variance) | 0.050 | 0.906 | 0.019 | 0.0915 | −0.001 |
| 0 | Derived, Taylor term only | 0.050 | 0.820 | 0.030 | 0.0878 | +0.003 |
| 0 | **Derived family (dynamic c + regression)** | **0.030** | 0.919 | **0.008** | **0.0873** | −0.006 |
| 0 | Oracle (true sub-cell variance) | 0.038 | 0.852 | 0.013 | 0.0872 | +0.000 |
| 1 | Injected, uniform σ | 0.051 | 0.901 | 0.016 | 0.1318 | −0.013 |
| 1 | Derived family | 0.033 | 0.922 | 0.008 | 0.1283 | −0.002 |
| 4 | Injected, uniform σ | 0.150 | 0.919 | 0.004 | 0.3154 | **−0.181** |
| 4 | Derived family | **0.058** | 0.926 | 0.007 | **0.2978** | +0.036 |

- **The derived σ tracks the truth cell by cell** (correlation 0.957). The Taylor term alone catches 75% of the variance; the dynamic coefficient gives 110%.
- **At lead 0 the derived family matches the oracle.** CRPS is 0.0873 against the oracle's 0.0872 and 30% below deterministic. Its tails are better calibrated than with uniform injected noise (>q99 0.008 against 0.019).
- **At long leads, isotropic error inflates the speeds.** Adding error variance around the forecast raises the Rice mean by 0.18. Regression to climatology removes that bias and gives the best calibration and CRPS.
- **Rough regime:** the forcing is inside the cells, so the energy is injected at unresolved scales. Results are pending (`python rice_family_ns2d.py --regime rough`).

### 9. Cyclones and extreme events as singularities (`singular.py`)

A cyclone is a moving near-singularity of the wind field. At a mast it produces a hard edge at Vmax with a (Vmax−V)^−½ caustic (`team/astro`). A family solve on a model grid smooths it into the cell. The singular part is therefore **tracked, not closed**.

**How the mast law is built:**

- stormtrack supplies the centre, motion, depth and vmax.
- Holland B = ρe·Vmax²/Δp; Rm from Willoughby et al. (2006) unless measured.
- The mast speed law is a mixture of Rice laws over the centre error (2-D Gauss–Hermite), plus the unresolved gusts. The caustic, eye and tails come out of the geometry.
- `from_track(row, mast_lat, mast_lon)` builds the law from a stormtrack row.

**Test.** 600 synthetic passages of a Holland storm (Vmax 50 m/s, depth 60 hPa, Rm 30 km), with the true miss distance uniform in ±80 km. Forecasts see a 20 km centre error; the model cell is 27 km. The event is the mast exceeding the threshold at least once during the passage.

| Threshold | Observed frequency | Family on cell-averaged field: mean p / Brier skill | **Singular component: mean p / Brier skill** |
|---|---|---|---|
| 45 m/s | 0.902 | 0.908 / 0.05 | 0.792 / −0.07 |
| 50 m/s | 0.632 | 0.782 / 0.25 | **0.609 / 0.48** |
| 55 m/s | 0.320 | 0.649 / −0.48 | **0.316 / 0.28** |
| 60 m/s | 0.012 | 0.474 / **−31.8** (294 false alarms) | **0.008 / 0.01** (0 false alarms) |

- **Treating the singularity as noise invents extremes.** The family turns the vortex's deterministic internal structure into random spread, so it forecasts P(>60 m/s) = 0.47 for winds that occur 1% of the time.
- **The tracked singular component is reliable where extremes matter.**
- **Known weak spot.** At 45 m/s it under-forecasts, because the centre-error mixture puts weight outside the ±80 km range the storms were drawn from. The fix is to truncate the centre-error posterior to the track climatology.

**The singularity flag.** `strip_width()` fits the Sulem–Sulem–Frisch analyticity-strip width δ (it recovers d = 10.0 and 25.1 for 1/(x²+d²)). With k_c = π/cell, it gives δ = 6.6 km and 11.5 km on transects 10 and 25 km from the eye (singular, δk_c < 3). A transect 50 km out and a smooth ambient field read as smooth.

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
2. **Run the two mast/WRF tests on the GEP data** (`mast_wrf/intrinsic_floor.py`,
   `mast_wrf/rectification_budget.py`). They settle what share of the WRF error is S−.
3. **Rice vector family solve.** Carry the resolved vector (from WRF) plus σ²(x, t). σ² should
   evolve under the squared-weight operator plus a Green–Kubo source, multiplicative in the
   resolved state (σ² ≈ 1.5 εΔt, `team/quantum`), and with its own scatter (compound σ).
4. **Saddlepoint quantiles for extremes** (`team/ramanujan/saddlepoint_closure.py`), plus a guard
   that refuses quantiles beyond the cumulant generating function's singularity s* (`team/astro`).
5. **Spatial dependence.** Real fields are correlated. Mixing a correlated field needs the joint
   cumulants Σ P_ij₁⋯P_ijₙ κ(S_j₁, …, S_jₙ), or a copula.
6. **Fronts and cyclones.** Add the strip-width switch (`sstrip.py`, to be written). Add the Holland
   vortex mixture fed by `stormtrack.track_at` (centre, motion, Rm, B, centre error).
7. **exp-Weibull a = N_eff.** Test on masts whether a grows with averaging window × U / L_u.
