# Noise as eliminated degrees of freedom, and family closure (Fields medallist)

Tags: [T] theorem with reference, [C] conjecture, [N] checked numerically. Scripts are in this folder.
`exact_q.py` gives the exact quantiles of the mixing test by FFT convolution: run `python exact_q.py`.
`ew_kurt.py` checks the kurtosis ratio. The L96 scripts are `l96*.py`, `frozen*.py`, `coupled*.py`,
`cmp*.py` and `multi.py`.

## 1. Noise = eliminated degrees of freedom

**(a) Mori–Zwanzig [T, exact identity].** Split the state of dz/dt = R(z) into z = (x resolved, y unresolved). Let P be the conditional expectation onto functions of x, and Q = I − P. Then

`dx/dt = PLx + ∫₀ᵗ K(x(t−s), s) ds + F(t)`, with `F(t) = e^{tQL} QLx`.

**F is a deterministic function of the unmeasured y(0).** It is "random" only because y(0) is not measured, which is the project axiom stated exactly. The memory kernel K is fixed by F (the second fluctuation–dissipation relation). References: Zwanzig 1973; Chorin, Hald & Kupferman 2002.

**(b) Averaging and the central limit theorem [T].** Take dx/dt = f(x,y) and dy/dt = ε⁻² g(x,y), with the fast system mixing.

- The slow variable converges to dX̄/dt = F̄(X̄).
- Its fluctuations are ε·ζ, with dζ = DF̄ζ dt + Σ^{½}(X̄) dW.
- Σ is given by Green–Kubo: Σ(x) = ∫₀^∞ E_{μx}[f̃(x,y₀) ⊗ f̃(x,y_s) + transpose] ds.

References: Khasminskii 1966; Kifer 2003; Majda, Timofeyev & Vanden-Eijnden 2001; Pavliotis & Stuart 2008, Ch. 10–11.

**(c) Homogenization [T].** Take dx/dt = ε⁻¹h(x,y) + f, with h = Σ h_α(x) v_α(y) and E^{αβ} = ∫₀^∞ E[v_α(y₀)v_β(y_s)] ds. The limit is

`dX = [F̄ + Σ E^{αβ}(h_α·∇)h_β] dt + Σ h_α(X) dW_α`, with Cov(W) = E + Eᵀ.

- The drift correction is a parameter-free prediction.
- The limit is not the Wong–Zakai equation unless E is symmetric (a Lévy-area drift appears).
- For a Markov fast process this is Papanicolaou & Kohler 1974. For deterministic chaos it is "deterministic noise": Melbourne & Stuart 2011; Gottwald & Melbourne 2013; Kelly & Melbourne 2016–17.
- When the fast system depends on x, a response drift is added (Wouters & Lucarini 2013). With feedback this is still [C] for deterministic fast dynamics.

**(d) Non-Gaussian limits [T].** Intermittent fast dynamics (LSV maps with β ∈ (½, 1)) give an α-stable Lévy limit with α = 1/β. The slow equation is then a Marcus SDE, `dX = F̄ dt + h(X) ⋄ dL_α`. This is the natural form for gusts: `dS = S ⋄ dL` keeps S positive. References: Chevyrev, Friz, Korepanov & Melbourne 2020; Gottwald & Melbourne 2021.

## 2. When a family stays closed

**Orbit theorem [T].** Let a group G act on the variable, and let the family be the orbit of a base law μ₀. If the dynamics commutes with G and maps μ₀ to h_t#μ₀ with h_t ∈ G, the family is invariant. The parameters then evolve by group multiplication.

| Family | Group | Closed under | Not closed under |
|---|---|---|---|
| Weibull 2p | orbit of Exp(1) under {x ↦ a x^b} ≅ Aff(ℝ) in log x; (λ,k)·(a,b) = (aλ^b, k/b) | scaling, power laws | shifts |
| Weibull 3p | Aff₊-orbits {ax + γ} | scaling, shifts | power laws |
| Exp-Weibull | G_pow × the Lehmann group {F ↦ F^a}; the two commute | power laws, max of n copies (a → na) | — |
| Champernowne | location–scale in y | affine maps of y | — |

- **Gust step [T].** Any kernel T⁻¹K₀T with T = −log(1−F_θ), where K₀ leaves Exp(1) invariant, leaves μ_θ invariant.
- **Diffusion of speed breaks every family [T: Feller II, §VI.1].** A location–scale family closed under aX₁ + bX₂ is stable. With finite variance, that leaves only the Gaussian.
- **Fisher projection [T].** For a general generator L (Brigo, Hanzon & Le Gland 1998–99), θ̇ = I(θ)⁻¹ E_θ[L ∂_θ log p_θ]. It is exact iff L*p_θ lies in span{∂_θ p_θ}. Cumulant matching is the moment-closure variant of this projection.
- **An exact intrinsic SDE that keeps the Weibull shape k [T].** Write E = (S/λ)^k and take dE = (1 − βE) dt + √(2E) dB. Then E = |Z|²/2 for a 2-D Ornstein–Uhlenbeck process Z, so S stays Weibull(k, λs(t)^{1/k}) with ṡ = 1 − βs.

**Structural recommendation: evolve the vector, not the speed.** Mixing acts on the velocity vector V.

- A Gaussian V is the unique family closed under all linear dynamics and under the homogenised Ornstein–Uhlenbeck noise.
- With V = V̄ + V′ and V′ ~ N(0, σ²I), the speed is |V| ~ **Rice(|V̄|, σ)**. At V̄ = 0 this is Rayleigh, i.e. Weibull k = 2.
- So **a Rice family solve is exactly closed under mixing plus intrinsic noise.** V̄ comes from WRF. σ² evolves under the squared-weight operator plus a Green–Kubo source.
- Shapes with k ≠ 2 need a non-Gaussian V′, which brings in the Lévy/Marcus case.

## 3. The mixing closure

**Exactness [T].** κ_n(t)_i = Σ_j P_ij^n κ_n(0)_j for initial values independent between points (McCullagh 1987). For a correlated initial field you need the joint cumulants Σ P_ij₁⋯P_ijₙ κ(S_j₁,…,S_jₙ).

**Error of 3-cumulant matching [T].** Let N_eff = (Σw)²/Σw². The standardized cumulants are γ_n = O(N_eff^{1−n/2}). By Cornish–Fisher, matching (μ, κ₂, κ₃) leaves a quantile error of σ·Δγ₂·(z³ − 3z)/24, where Δγ₂ is the kurtosis gap; at p = 0.999 the factor (z³ − 3z)/24 is 0.84.

**Why the exp-Weibull kurtosis looked accurate [T + N].**

- Both the true and the implied kurtosis are O(γ₁²), so their absolute gap is small.
- Exactly, γ₂/γ₁² = (κ₄κ₂/κ₃²)₀ · Σw⁴Σw²/(Σw³)², and the second factor tends to 3/√8 = 1.0607 for a heat kernel.
- Predicted ratio 0.652, observed 0.655 at t = 40.
- At t = 40, γ₁ = 0.183 and the true γ₂ = 0.022. Exp-Weibull implies 0.061, a gap of +0.039, which reproduces the +0.04 in `mixing_closure.py`. Its implied ratio is about 1.8, roughly 2.8 times the truth.

**Exact check [N]** (`exact_q.py`). The exact quantiles come from FFT convolution of the 200 scaled Weibull pmfs, with no Monte Carlo. Error of the 99.9th percentile:

| | t = 10 | t = 40 |
|---|---|---|
| Exp-Weibull, 3 cumulants | +0.26 to +0.27% | +0.07 to +0.08% |
| Cornish–Fisher (κ₂, κ₃) | −1.35 to 0.00% | −0.58 to −0.03% |
| **Cornish–Fisher with the exact κ₄ already tracked** | **about −0.03%** | **about −0.03%** |

The −0.03% is the resolution limit of the convolution grid.

## 4. Testable prediction: the Green–Kubo diffusion law on L96

Hold the X-coupling hc₀/b = 1 fixed and speed up only the fast clock, dY/dt = c·G(Y; X); c = 10 is the Wilks system. The prediction, with no free parameters, is

`Var(∫_t^{t+Δ} r ds | X) / Δ → Σ_GK(X) / c`.

An injected noise calibrated at c = 10 has no reason to follow this law.

- **Frozen X = 10:**
  - Ū is −4.13, −8.27 and −16.51, and σ²_GK ≈ 0.06, 0.115 and 0.24, for J = 16, 32 and 64.
  - For 0 ≤ x ≤ 4 the fast system becomes non-chaotic after a transient, and σ²_GK ≈ 0.001.
- **The 1/c law holds:** c·σ²_∞ = 1.26, 1.30, 1.45 and 2.05 for c = 5, 10, 20 and 40. The c = 40 run is shorter and noisier.
- **Averaging bias shrinks with c:** for X ≥ 6, the rms of E[U|X] − Ū_frozen is 1.14, 0.22, 0.10 and 0.17.
- **Chaotic regime (X ≥ 6):** coupled c·σ² ≈ 1.4–1.6 against the Green–Kubo value 1.18, within about 25% with no fitting.
- **Transient window (0 ≤ X < 5):** coupled c·σ² ≈ 1.5–1.9 against 0.19, about 10 times too high.
  - X crosses the window in about 0.3 time units, faster than the fast transient decays.
  - There the residual is Mori–Zwanzig memory, not white noise.
  - [C] It should collapse to about 0.19/c once c·T_residence ≫ T_transient.

**WRF–mast analogue.** Intrinsic noise predicts that the RMSE of Δ-averaged tendency residuals is diffusive (∝ Σ_GK·Δ), regime-dependent, zero where the unresolved flow is laminar, and non-Markovian near regime transitions. A measurement-noise floor has none of these features.

## What to implement

1. **Rice / vector family solve.** Carry V̄ (from WRF) and σ²(x, t), with σ² evolving under the squared-weight operator plus the Σ_GK source. Benchmark it against exp-Weibull on mast data.
2. **Use κ₄ in the mixing closure** (Cornish–Fisher, or a 4-parameter family).
3. **Fisher projection** as the fallback when the family has no group structure.
4. **L96:** a frozen-X Green–Kubo estimator, a c-scan, regime-conditional c·σ² against Σ_GK, a transient-lifetime diagnostic, and the drift correction.
5. **Marcus SDE** for heavy-tailed multiplicative gust increments.
