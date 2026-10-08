# Where the family closure meets singularities (astrophysicist)

The script is `burgers_singularity.py` (numpy and scipy only, about 2 min); its full output is in
`burgers_singularity_output.txt`.

## 1. Where the closure breaks

### (a) Physical space

**Burgers fronts.** Take u₀ = sin x. The Lagrangian map x = a + t sin a becomes singular where 1 + t cos a = 0.

- Before the shock, the singularities are a complex pair at x = π ± iδ₀(t), with δ₀ = arccosh(1/t) − √(1−t²) ≈ (2(1−t))^{3/2}/3. The shock forms at t* = 1.
- After the shock, the viscous tanh front has poles at ±iπν/U(t). U peaks at t = π/2, so δ bottoms out there, not at t*.

**Vortex.** The Rankine vortex has a kink at Rm. The Holland vortex has an essential singularity at r = 0, which a mast transect at miss distance d sees at x = ±id. The strip width of a mast time series is therefore about d/c, where c is the storm's translation speed. This link holds only approximately, because the essential singularity converges slowly.

### (b) Parameter space

- **Weibull.** As k → ∞, CV → 0 and skewness → −1.1395. Skewness is zero at k = 3.602.
- **Exponentiated Weibull.** The Jacobian det ∂(CV, skew)/∂(ln k, ln a) never changes sign, so the moment map has no interior fold. It degenerates only at two edges:
  - a → ∞ gives the Gumbel(max) limit, with skewness +1.14;
  - k → ∞ and a → 0 at fixed ka give the power-function law (x/λ)^{ka}.
- **Fold caustics push the matcher into that corner.** Matching the arcsine law of |sin x| gives k = 93, a = 0.019, and still misses the CV by 0.094.
- **Champernowne**, in log S:
  - as λ → −1 the law collapses to δ(x − M);
  - the poles at αy = ±i(π − arccos λ) set the strip width;
  - moments E[S^n] exist only for n < α, so there is no skewness when α ≤ 3.

### (c) The complex s-plane of K(s) = ln E[e^{sS}]

**Tail rate.** s* = liminf −ln(1−F(x))/x.

- If K has a log singularity at s*, then κ_n ≈ β (n−1)! s*^{−n}.
- So κ_{n+1}/(nκ_n) → 1/s* shows how far a cumulant truncation can be trusted.
- A saddlepoint or cumulant quantile is impossible once the required ŝ ≥ s*.

**Families:**

| Law | s* | Note |
|---|---|---|
| Weibull or exp-Weibull, k > 1 | ∞ | M is entire |
| Weibull, k = 1 | 1/λ | pole |
| Weibull, k < 1 | none | M diverges; cumulants are only asymptotic |
| Champernowne | none | E[S^s] has poles at ±α |
| Bounded laws (post-shock \|u\|, Holland) | — | M is entire, but the edge makes φ(ω) ~ \|ω\|^{−½} |

## 2. Burgers demonstration

Setup: N = 8192, 2/3 dealiasing, integrating-factor RK4. The fitted strip width δ is checked against the exact viscous δ from Cole–Hopf.

| ν | δ(0.7): fit / Cole–Hopf / inviscid | δ(1): fit / Cole–Hopf | δ_min (time) | δ_min/(πν) | δ(1)/ν^{3/4} |
|---|---|---|---|---|---|
| 0.02 | 0.317 / 0.331 / 0.181 | 0.136 / 0.139 | 0.0642 (1.65) | 1.022 | 2.55 |
| 0.01 | 0.262 / 0.276 / 0.181 | 0.080 / 0.083 | 0.0314 (1.60) | 0.998 | 2.53 |
| 0.005 | 0.227 / 0.241 / 0.181 | 0.047 / 0.049 | 0.0155 (1.60) | 0.986 | 2.52 |

- **Accuracy.** The spectral fit of δ matches Cole–Hopf within 2–4%.
- **Minimum.** δ bottoms out at t ≈ π/2 with δ = πν/U, within 2%.
- **At the shock time.** δ(t*) ≈ 2.53 ν^{3/4}, the scaling of the preshock layer.
- **Early warning.** Extrapolating δ^{2/3} linearly predicts the collapse at t* = 1.146, 1.058 and 1.005 for the three ν. That is a warning about 0.3 time units ahead.

**Predicting the unresolved part from the resolved band.** The spectrum is fitted on k ≤ Kc = 32 only, then extrapolated. The predicted rms of the unresolved part u′ = u − P_Kc u matches the true value within 3–15% over 12 decades. For example, at ν = 0.01, t = 1: 5.4e−3 predicted against 5.0e−3 true.

| δ·Kc | rms(u′) | F(u′) | F(u_x) | Closure |
|---|---|---|---|---|
| > 10 | < 1e−6 | 2–5 | — | family with Gaussian residual |
| 3–10 | 1e−4 to 1e−3 | 8–15 | — | family plus a non-Gaussian residual |
| ≲ 2 | 2–6% of U | 33–57 | 110–317 | switch to a mixture |

When δ·Kc ≲ 1, the fit on the resolved band itself degenerates to δ ≈ 0. That is a clean switch flag.

**Negative result: the one-point distribution is blind to the front.** The KS distance of Weibull and exp-Weibull fits to |u| stays near 0.135 while δ → 0. The shock occupies measure of order δ, so it barely moves the one-point law; the poor KS comes from the arcsine edge of sin x. **The switch must therefore be driven by δ and by the flatness of the residual, not by goodness of fit.**

## 3. A cyclone seen by a fixed mast

**Setup.** A Holland vortex (Vmax, Rm, B) passes the mast at a uniform random miss distance.

- Time spent at radius r: p_r(r) ∝ r·arcsin(min(r, D)/r).
- Speed law: p(V) = Σ p_r(r_i)/|V′(r_i)| over the two branches.

**Shape of the law:**

- **Fold caustic at Vmax.** p(V) ≈ 2p_r(Rm)Rm / (B√(Vmax(Vmax − V))), so P(V > Vmax − ε) ∝ √ε. Monte Carlo gives a log-log slope of 0.525 (theory ½), and the analytic prefactor agrees within 1–9%.
- **Eye.** p ∝ 1/[V (ln(Vmax/V))^{1+2/B}].
- **Outer region.** p ∝ V^{−1−4/B}.

**None of our families has a hard upper edge with a (Vmax − V)^{−½} spike.**

| Quantile (no ambient wind) | Monte Carlo | Exp-Weibull | Weibull |
|---|---|---|---|
| 99% | 45.2 | 38.5 | 35.1 |
| 99.9% | 49.9 | 57.3 (above Vmax: impossible) | 41.7 |

The global KS can still look acceptable (EW 0.049, or 0.032 with an ambient wind), which is why goodness of fit cannot be trusted here.

**What the closure needs:** a mixture p(S) = (1 − w) f_EW(S; θ_amb) + w f_H(S).

- The mast–centre distance comes from stormtrack's centre and its error, as a Rice law: p(r) = (r/σ²) e^{−(r²+ρ²)/2σ²} I₀(rρ/σ²).
- Translation and ambient wind enter by vector addition.
- w = P(mast inside R_out).

## 4. What to implement

1. **`sstrip.py`.** Fit the spectral strip width per tile (about 64×64 points) on 1-D line spectra normal to the dominant gradient, with a Gaussian taper. Output δ(x, t), δ_res and the extrapolated rms of the unresolved part. Calibrate Kc ≈ π/(6–7Δx).
2. **Switch rule.**
   - δKc > 10: family with Gaussian residual.
   - 3–10: family plus a residual with flatness about 10.
   - < 2–3, or δ_res ≈ 0: mixture.
   - Use the δ^{2/3} extrapolation to switch ahead of the front.
3. **Mast check.** Regress the mast − WRF residual variance and flatness on δ·Kc.
4. **stormtrack feed.**
   - `track_at(df, t)` gives the centre, motion and pmin; `track()` gives vmax_ms.
   - Add Rm (radius of the wind-speed maximum), Holland B = ρe·Vmax²/Δp, and a centre error σ from the jitter of the motion fit.
   - Use the mixture inside Rm + 3σ, or wherever w > 0.05.
5. **Bounded "caustic" component.** Add Beta(c, ½) on [0, Vmax]. Guard the moment matcher: if it drives k → ∞ and a → 0, flag an edge or caustic.
6. **Cumulant guard.** Report s*. Refuse cumulant or saddlepoint quantiles beyond s*, and treat Weibull k < 1 and Champernowne α ≤ 3 as having no cumulant closure.

References: Sulem, Sulem & Frisch 1983; Bessis & Fournier 1984; Senouf, Caflisch & Ercolani 1996; Frisch, Matsumoto & Bec 2003; Brachet et al. 1983; E, Khanin, Mazel & Sinai 1997; Holland 1980; Champernowne 1952; Mudholkar & Srivastava 1993; Daniels 1954.
