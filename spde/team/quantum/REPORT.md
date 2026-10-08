# Open-system view of the intrinsic noise (quantum physicist)

**Bottom line:** the axiom holds in two-scale L96 when the fast part (the "bath") is chaotic.
- **Fluctuation–dissipation:** the noise amplitude follows from the damping (Einstein / Green–Kubo) within 1–2 standard errors.
- **Error growth:** Green–Kubo predicts the short-lead error of the deterministic model within 4%.
- **Bias:** the intrinsic noise removes about 80% of the deterministic model's climate-mean bias.
- **Limits:** the residual RMSE is not exactly the missing term, and fluctuation–dissipation fails when the bath is periodic.

Scripts are in this folder: `run_truth.py`, `run_frozen*.py` (X held fixed), `analyze*.py`, `param.py`, `cumul.py` and `l96.py`. The `.npz` data are not committed; rerun to regenerate them.

## 1. Zwanzig heat-bath model and the atmospheric mapping

Zwanzig (1973), Ford, Kac & Mazur (1965) and Caldeira & Leggett (1983) use

`H = p²/2 + V(q) + Σⱼ [pⱼ²/2 + ωⱼ²/2 (xⱼ − γⱼ q/ωⱼ²)²]`

Eliminating the bath gives a generalized Langevin equation:

`q̈ = −V′(q) − ∫₀ᵗ K(t−s) q̇(s) ds + ξ(t)`, with `K(t) = Σ (γⱼ²/ωⱼ²) cos ωⱼt`

If the bath starts in a Gibbs state, the noise correlation is `⟨ξ(t)ξ(s)⟩ = k_B T K(t−s)` (second FDT, Kubo 1966). Eliminating the bath also adds a deterministic counterterm, `ΔV = −Σ γⱼ² q²/(2ωⱼ²)`, a static shift like a Lamb shift.

| Open system | Atmosphere |
|---|---|
| system q | filtered (resolved) momentum ū |
| bath | sub-grid field u′ |
| memory kernel K | eddy damping, γ_k = ν_t k² (Markov limit) |
| noise ξ | −∇·τ′, the fluctuating sub-grid stress |
| counterterm | conditional mean ⟨τ given ū⟩ beyond linear friction |
| k_B T | energy per sub-grid mode, T_b ≈ (2/3)e, with e the sub-grid TKE |

**FDT noise amplitude at the grid scale** (k_c = π/Δ):

`σ_u² = 2 ν_t k_c² T_b Δt = (4π²/3) ν_t e Δt / Δ²`

With `ν_t = C_k Δ √e` and `ε = C_ε e^{3/2}/Δ`, this becomes `σ_u² ≈ (4π² C_k / 3 C_ε) ε Δt ≈ 1.3–1.9 ε Δt`.

This has the same form as Thomson (1987, C₀εΔt) and as the backscatter of Mason & Thomson (1992). Related work: Leith (1990); Chasnov (1991, EDQNM); Kraichnan (1976); Frederiksen & Davies (1997); Frederiksen & Kepert (2006); Shutts (2005); Berner et al. (2009, SKEB).

**The noise is multiplicative.** In the surface layer, e ∝ S² and ν_t ∝ S, so σ² ∝ S³ Δt. The Itô/Stratonovich drift ½gg′ is a second route from noise to bias.

**Where the analogy breaks.** The turbulent cascade is forced and dissipative, with no detailed balance, so FDT holds only approximately. EDQNM backscatter goes as k⁴, so it is not white. The bath's "temperature" is set by the resolved flow itself.

## 2. Non-Gaussian, finite bath and the choice of distribution family

Take the forcing to be the sum of N independent energetic eddies, `ξ = Σᵢ aᵢ ηᵢ`. Then every cumulant grows like N, κₙ ∝ N, so the normalised cumulants scale as `κₙ/κ₂^{n/2} ∝ N^{1−n/2}`: skewness falls as N^{−1/2} and excess kurtosis as N^{−1}. If gusts arrive as Poisson shot noise, the cumulant generating function is `ln⟨e^{sΞ}⟩ = νt(⟨e^{sA}⟩ − 1)`, as in Levitov–Lesovik full counting statistics.

Write Y = −log(1−F(S)), which equals (S/λ)^k for Weibull. Then the bath size selects the family:

- **Sum route.** An OU process in Y driven by compound-Poisson Exp jumps is stationary Gamma with shape N_eff = ν/θ (Barndorff-Nielsen & Shephard 2001).
  - N_eff = 1 gives Exp, so S is Weibull. The repository's EAR(1) step is exactly this N = 1 shot-noise bath.
  - N_eff > 1 gives S a generalised gamma (Stacy) distribution.
  - Gaussian (u, v) components (N → ∞) give Rayleigh, which is Weibull with k = 2.
- **Max route.** `(1 − e^{−Y})^a` is the CDF of the maximum of a independent Exp(1) variables, so **the exponentiated Weibull's a ≈ N_eff**, the number of independent eddies sampled.
  - Prediction: a grows like window × U / L_u.
  - Rough sizes: 10-min means at U = 10 m/s with L_u = 100–300 m give a ≈ 20–60. A 3-s gust has a ≈ 1.
- **Shape bound (testable).** Correlated additive-multiplicative (CAM) noise gives excess kurtosis ≥ (3/2) skewness² (Sardeshmukh & Sura 2009). See also Monahan (2006) on sea-surface wind speed.

**Measured in L96:**
- For partial sums of the fast variables, the excess kurtosis falls by a factor of 6 from m = 1 to m = 32. Independent variables would give 32, so N_eff ≈ 6: the fast variables are correlated.
- The coupled residual has skewness −0.39 and excess kurtosis 0.55. This satisfies the CAM bound (0.55 ≥ 0.22). The non-Gaussianity comes mostly from the state-dependent variance, not from N being small.

## 3. Verified on two-scale L96

Setup: K = 8, J = 32, F = 20, h = 1, b = c = 10, RK4 with dt = 0.001. The truth run is 8 trajectories × 100 model time units (MTU).

**(a) Exact Mori–Zwanzig budget.** `dU_k/dt = −c U_k − 32 X_k − B_k` holds to 0.4%.
- The memory kernel is exactly K(τ) = 32 e^{−10τ}.
- The conditional mean ⟨B given X⟩ explains R² = 0.72 of B. This is the counterterm: it cuts the static slope from −3.2 to −1.11, matching the cubic fit `U_det = 0.00287X³ − 0.0004X² − 1.110X − 0.695`.
- The fluctuation B′ is heavy-tailed: sd 75, excess kurtosis 4.5.

**(b) Einstein / Green–Kubo friction, X frozen.** The prediction is `γ ≡ −∂⟨U⟩/∂X = Var(U given X) · τ_U / T_b`, with T_b = σ_Y².

| x₀ | T_b | Var(U) | τ_U | Einstein | Direct (±δ) | Quasi-Gaussian (Leith 1975) |
|---|---|---|---|---|---|---|
| 8 | 0.133 | 1.92 | 0.027 | −0.392 | −0.449 ± 0.031 | −0.90 |
| 10 | 0.192 | 3.11 | 0.020 | −0.321 | −0.340 ± 0.036 | −0.65 |
| 12 | 0.256 | 4.47 | 0.015 | −0.257 | −0.334 ± 0.028 | −0.48 |
| 5 | 0.055 | 0.57 | 0.21 | −2.23 | −0.58 | −7.4 |
| 2 | 0.010 | 0.023 | 0.004 | −0.009 | −0.31 | −7.3 |

- With a chaotic bath (x₀ = 8–12), the Einstein value is 6–23% from the direct one.
- With a periodic bath (x₀ = 2, 5), fluctuation–dissipation fails.
- The noise is multiplicative: Var(U given X) ≈ 16 σ_Y².

**(c) Green–Kubo error growth of the deterministic cubic model.** The prediction is `MSE(t) = 2 ∫₀ᵗ (t−s) C_r(s) ds`.

| Lead (MTU) | 0.01 | 0.025 | 0.05 | 0.1 | 0.2 | 0.5 |
|---|---|---|---|---|---|---|
| MSE, deterministic model | 0.0003 | 0.0019 | 0.0067 | 0.0217 | 0.067 | 0.49 |
| Green–Kubo prediction | 0.0003 | 0.0020 | 0.0070 | 0.0213 | 0.049 | 0.10 |

- The prediction is within 4% up to lead 0.1 MTU; after that, Lyapunov growth dominates.
- Residual: σ_r = 1.83, D = σ_r² τ_int ≈ 0.08, τ_int ≈ 0.024.
- The residual is not AR(1). The lag-1 fit gives τ = 0.29, but the autocorrelation crosses zero at 0.1.

**(d) Climate bias of X.** The truth's sampling error on the mean is about ±0.05.

| Model | Mean | Bias | sd | 99.9th pct |
|---|---|---|---|---|
| Truth | 3.763 | — | 5.07 | 15.26 |
| Deterministic | 3.487 | −0.276 | 4.94 | 15.55 |
| + white noise, D = 0.08 from Green–Kubo (no tuning) | 3.698 | −0.065 | 5.06 | 15.79 |
| + AR(1) | 3.714 | −0.049 | 5.08 | 16.06 |

Adding the noise removes about 80% of the mean bias. Gaussian additive noise overfills the upper tail, because the true residual is negatively skewed and state-dependent.

## Caveats on "residual RMSE = S−"

The WRF − mast MSE has five parts:
1. intrinsic noise S−, growing as t² and then as t;
2. representativeness error, an offset already present at t = 0 (point versus cell, about the sub-grid variance (2/3)e);
3. initial-condition error amplified by chaos;
4. structural error of the deterministic closure;
5. instrument error.

Only parts 1 and 2 are S−. Noise turns into a mean bias only through nonlinear rectification (½f″σ², or the Itô drift).

## What to implement

1. In the speed/momentum parameter equations, add an intrinsic term with σ² = 2ν_t(π/Δ)²(2/3)e Δt ≈ 1.5 ε Δt. It should be multiplicative in e(S) and Stratonovich-consistent, with the ½gg′ drift written explicitly.
2. Fit the counterterm ⟨sub-grid tendency given state⟩ separately from the noise.
3. Treat the exponentiated Weibull's a as N_eff and test a ∝ window·U/L_u on the masts. Use generalised gamma for averages and exp-Weibull for maxima. Generalise the EAR(1) step to Gamma(N_eff) shot noise.
4. Diagnostics on the mast–WRF pairs:
   - MSE versus lead: intercept (representativeness), t² term (σ_r²) and slope 2D;
   - the Einstein check;
   - the CAM bound.
5. L96: keep tables (b)–(d) as regression numbers, and replace the AR(1) residual with an oscillatory kernel.
