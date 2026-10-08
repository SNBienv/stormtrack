# Intrinsic vs extrinsic noise for wind (biologist)

Scripts are in this folder. `burst.py` simulates bursty gusts against the distribution families. `spec.py` computes the representativeness variance from the spectrum. `realfit.py`, `compound.py` and `compound2.py` run on the London and Irish data in `spde/data`. `l96_J.py` and `l96_dual.py` vary J in two-scale Lorenz-96. The `*.txt` files hold the outputs.

## 1a. The dual reporter for wind

Elowitz et al. 2002 and Swain, Elowitz & Siggia 2002 separate noise using two identical reporters in one cell. For wind, the reporters are two identical anemometers inside one WRF cell, at separation d, with speeds s₁ and s₂ over the same window T. Fu & Pachter (2016) give the unbiased estimators:

- η²_int = ⟨(s₁−s₂)²⟩ / (2⟨s₁⟩⟨s₂⟩)
- η²_ext = (⟨s₁s₂⟩ − ⟨s₁⟩⟨s₂⟩) / (⟨s₁⟩⟨s₂⟩)
- η²_tot = η²_int + η²_ext (an identity)

**This is a model-free test of the project axiom.** MSE(WRF − s₁) = ½⟨(s₁−s₂)²⟩ + E[(WRF − shared)²]. The first term is the intrinsic floor at that scale. If WRF's MSE is far above the floor, the excess is ordinary model or phase error, not S−.

What counts as intrinsic depends on scale:

- ⟨(s₁−s₂)²⟩ is the structure function D_s(d), which goes as (εd)^{2/3}. Choose d ≈ 7Δ, the effective resolution (Skamarock 2004), or map D_s(d) from several mast pairs.
- With a single mast, Taylor's hypothesis turns D_t(τ = 7Δ/U) into a pseudo-dual reporter.
- At d ≈ 0, two booms measure only instrument noise; treat that as the background to subtract.
- Two heights are not identical reporters, because shear varies with stability. Use them only as a bound.
- Condition on WRF speed or stability to get η²_int(state).

The test needs the two reporters to be conditionally independent given the shared state (Hilfinger & Paulsson 2011).

## 1b. What plays the system size Ω?

There are two regimes:

- **Van Kampen regime (Δ ≫ integral scale).** Ω = N_eddies ≈ (Δ/L)^d, or Ω = T/(2τ_int) in time (Lenschow, Mann & Kristensen 1994). The noise falls as 1/Ω.
- **Mesoscale k^{-5/3} range (Δ = 1–9 km; Nastrom & Gage 1985).** There is no 1/Ω law. The sub-grid variance grows with grid size: ∫_{k>k_c} E dk ∝ Δ^{2/3}, and in general Δ^{β−1} for E ∝ k^{−β}.

Using Lindborg's (1999) spectrum with k_c = 2π/(7Δ), the instantaneous point-minus-cell spread is:

| Δ | Variance | rms |
|---|---|---|
| 9 km | 0.65 m²/s² | 0.81 m/s |
| 3 km | 0.31 m²/s² | 0.55 m/s |
| 1 km | 0.15 m²/s² | 0.38 m/s |

These are order-of-magnitude values: the spectrum constants are not for the surface layer. Adding the 10-min micro-turbulence gives a floor of about 0.9 / 0.67 / 0.54 m/s.

**Prediction: the floor is only 10–30% of a typical 10-m WRF MSE (rms 1.5–2.5 m/s). So most of the WRF residual would be extrinsic model error, not intrinsic noise.** The dual reporter settles this on real masts.

**The mast averaging window matters.** The residual is ∫E|H_T − H_Δ|² dk, which is smallest when the window length matches the grid, U·T ≈ 7Δ.

- With hourly mast means, refining the grid from 9 to 3 to 1 km *raises* the representativeness variance: 0.08 → 0.21 → 0.39 m²/s².
- With 10-min means it falls: 0.42 → 0.10 → 0.016 m²/s².
- **Test:** plot MSE against T for each Δ. The minimum should sit at T* ≈ 7Δ/U, which also measures the effective resolution. At matched averaging, MSE should be linear in Δ^{2/3}, and the intercept is the model-error part.

The closest atmospheric case of true van Kampen noise is the convective cloud ensemble of Craig & Cohen (2006) and Plant & Craig (2008).

## 2. Bursty gusts (cf. Friedman, Cai & Xie 2006)

Let kinetic energy decay as dE/dt = −γE and receive Poisson bursts of Exp(β) size at rate α. Then E ~ Gamma(m = α/γ), so the speed √(2E) follows a **Nakagami-m** law.

- m = 1 gives Rayleigh, which is Weibull with k = 2. The repository's EAR(1) step is exactly the m = 1 burst process; GAR(1) (Lawrance 1982) is the version for m ≠ 1.
- m works as a system size: CV²(E) = 1/m.
- The exponentiated Weibull matches Nakagami at both ends when **a·k = 2m and k ≈ 2**.

| m | Weibull k | Exp-Weibull (a, k) | ΔAIC (Weibull − EW) | Weibull q99.9 error | EW q99.9 error |
|---|---|---|---|---|---|
| 0.5 | 1.26 | (0.53, 1.89) | 4318 | +20% | −0.7% |
| 1 | 1.99 | (1.01, 1.99) | −1.7 | — | — |
| 2 | 2.97 | (1.76, 2.22) | 2742 | −5.4% | +0.9% |
| 5 | 4.74 | (3.16, 2.77) | 9557 | −7.1% | +0.2% |

A slowly varying energy scale (superstatistics, giving a K-distribution) pushes the fit to k < 2 and a > 1. London hourly gives EW (a = 2.32, k = 1.31), which has that signature.

**Single-reporter split.** Fit S = √(2βG) with G ~ Gamma(m) and β lognormal(σ):

| | m | Intrinsic η² | Extrinsic η² |
|---|---|---|---|
| London hourly | 1.34 | 0.75 | 0.32 (σ = 0.53) |
| London daily | 4.97 | 0.20 | 0.62 |

Daily averaging cuts the intrinsic part about 3.7-fold, about 4 independent bursts per day. The extrinsic part does not average out. This split rests on the parametric model and is only weakly identifiable; the dual reporter is the model-free check.

## 3. Two-scale L96 with varying J

With coupling h_X = h·32/J in the X equation, the mean closure is the same for every J. A dual reporter splits each sector's sub-grid term into its two halves.

| J | Total | Intrinsic | Extrinsic |
|---|---|---|---|
| 8 | 6.73 | 7.44 | −0.71 |
| 16 | 4.11 | 3.30 | 0.82 |
| 32 | 3.46 | 1.46 | 2.00 |
| 64 | 3.37 | 0.72 | 2.65 |
| 128 | 3.57 | 0.33 | 3.24 |

- The **intrinsic part scales as J^{−1.12}**, the van Kampen law.
- The **total saturates near 3.4**: Var ≈ 2.9 + 28/J.
- The negative extrinsic value at J = 8 means the two halves were not conditionally independent there.
- The shared floor is fast and coherent across each sector, with lag-0.05 autocorrelation 0.47. It is a collective Mori–Zwanzig term.
- Conditioning on neighbouring and lagged X removes only about 15% of the shared floor.

**More resolution, i.e. more "molecules", never makes the closure deterministic.** A closure needs a 1/J term plus a J-independent shared term.

## What to implement

1. **`dual_reporter.py`.** Compute η²_int(d) and η²_ext(d) from mast pairs, per regime bin, together with the structure function D_s(d), the Taylor-hypothesis check, and the instrument floor.
2. **WRF budget.** Model MSE(Δ, T) = intrinsic(Δ, T) + model error. Find T* and check linearity in Δ^{2/3}.
3. **Family.** Use the exponentiated Weibull with a ↔ m (the number of bursts). Add a GAR(1) gust step that keeps Gamma(m) exact. Let a grow with averaging.
4. **Compound likelihood.** Fit Gamma(m) × lognormal(σ) as a parametric intrinsic/extrinsic split, validated against item 1.
5. **L96.** Use the closure E[U|X] + √(B/J)ξ + η_shared.

References: Elowitz et al. 2002; Swain, Elowitz & Siggia 2002; Paulsson 2004; Friedman, Cai & Xie 2006; Taniguchi et al. 2010; Hilfinger & Paulsson 2011; Fu & Pachter 2016; van Kampen; Nastrom & Gage 1985; Lindborg 1999; Skamarock 2004; Lenschow, Mann & Kristensen 1994; Craig & Cohen 2006; Plant & Craig 2008; Wilks 2005; Arnold, Moroz & Palmer 2013; Lawrance 1982; Gaver & Lewis 1980.
