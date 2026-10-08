# Closed forms, saddlepoint closure, tail asymptotics (Ramanujan)

Every formula below was checked numerically. Scripts:

- `saddlepoint_closure.py`: self-tests and all tables, about 3.5 min. Output in `saddlepoint_closure_output.txt`. The integrable parts are `weibull_cgf`, `MixedCGF`, `lr_tail`, `lr_quantiles` and `exact_quantiles`.
- `t1_expweibull_moments.py`, `t2_champernowne.py`, `t2b_region.py`: closed-form checks.
- `t4_tail.py`: tail asymptotics; output in `t4_tail_output.txt`.

## Correction to the earlier benchmark

**A 100,000-member Monte Carlo is off by up to about 1% at the 99.9% quantile.** The "+0.2% (worst 0.8%)" reported for the exp-Weibull mixing closure therefore mostly measured Monte Carlo noise. The exact reference used here is Gil-Pelaez inversion of the exact characteristic function, ∏_j φ_j(P_ij t).

## 1. Exp-Weibull moments

**Formula.** E[X^r] = aλ^r Γ(1+r/k) Σ_{m≥0} c_m (m+1)^{−(1+r/k)}, with c_m = c_{m−1}(m−a)/m.

**Problem.** For non-integer a the terms decay only like m^{−(a+1+r/k)}. With a = 0.5 and k = 2, about 10¹⁰ terms are needed to reach 1e−10.

**Fast evaluation:**

- **Hurwitz-zeta tail:** sum N = 24 + 3a terms directly, then add the tail as (1/Γ(1−a)) Σ_j g_j ζ(s+a+j, N+1), with the g_j from the Stirling expansion of Γ(z−a)/Γ(z). Worst error 1.5e−13 over 180 cases; about 0.1 ms per moment.
- **Generalised Gauss–Laguerre** with 80 nodes: worst error 6e−10.

## 2. Champernowne

Write λ = cos θ.

- **Characteristic function:** **φ(t) = (π/θ) sinh(θt/α) / sinh(πt/α)**, with normalisation n = α sinθ/(2θ). Checked to 4e−26.
- **Cumulants:** κ_{2n} = (−1)^n 2^{2n} B_{2n}(θ^{2n} − π^{2n})/(2n α^{2n}), so κ₂ = (π² − θ²)/(3α²). **All odd cumulants are zero.**
- **Excess kurtosis:** (6/5)(π² + θ²)/(π² − θ²), with range (−6/5, ∞).
- **Truncated moments (v ≥ 0):** 1/(cosh x + λ) = 2Σ(−1)^{m−1}U_{m−1}(λ)e^{−mx}, which gives closed polylogarithm forms. Checked to 5e−31.

**Why the moment closure fails.** The untruncated law is even, so all of its skewness has to come from the cut at zero. Matching the skewness and kurtosis of a Weibull with k = 2 needs f(0)·sd ≈ 0.29, a wall of density at v = 0. A Weibull with k > 1 has f(0) = 0.

## 3. Saddlepoint closure: no family assumed

The cumulant generating function is exact:

K_i(s) = Σ_j K_j(P_ij s), and K_i^{(m)} = Σ_j P_ij^m K_j^{(m)}(P_ij s).

- **Weibull CGF.** Substitute x = λe^y and integrate with the trapezoid rule (h = 0.08, about 340 nodes). This is accurate to 1.5e−10, including λs = 70 and λs = −60.
- **Tail probabilities.** Use Lugannani–Rice (LR), plus the Daniels second-order term (LR2) where |w| > 0.5.
- **Quantiles.** Newton iteration on the normal score converges in 6–7 iterations.

Mixing test: 200 points; 40 scored points × 4 quantiles. Errors are % against the exact reference, mean with worst |%| in brackets.

| Steps | LR, 99.9% | LR2, 99.9% | Exp-Weibull, 99% | Exp-Weibull, 99.9% | MC 10⁵, 99.9% |
|---|---|---|---|---|---|
| 5 | −0.003 (0.006) | 0.000 (0.001) | +0.21 | **+0.41** | −0.04 (1.00) |
| 10 | −0.001 (0.003) | 0.000 | +0.18 | +0.28 | −0.03 (0.93) |
| 20 | 0.000 (0.002) | 0.000 | +0.15 | +0.18 | −0.04 (0.87) |
| 40 | 0.000 (0.001) | 0.000 | +0.12 | +0.11 | −0.02 (0.61) |

Cost for 40 points × 4 quantiles over all 4 time levels:

| Method | Cost |
|---|---|
| LR saddlepoint | 4.3 s (27 ms per point and time) |
| Exp-Weibull fit | 0.7 s |
| Exact inversion | 120 s |
| Monte Carlo, 10⁵ members | 51 s |

## 4. Tail asymptotics of mixed Weibulls (equal k > 1)

**Leading order:** −log P(S > s) = **(s/Λ)^k** + O(log s), with Λ = (Σ(w_jλ_j)^q)^{1/q} and q = k/(k−1), the Hölder conjugate.

**Sharp form:** P ~ C s^{(n−1)k/2} e^{−(s/Λ)^k}, by Laplace's method.

- With k = 2, the ratio to exact quadrature is 0.9998 at P = 1e−3 and 1.0000000 at 1e−14.
- The polynomial prefactor matters: without it the ratio is still 0.96 at P = 1e−32.
- With unequal k, k_min sets the leading order.

**Exp-Weibull has the wrong tail.** It has no prefactor, and at 40 steps its fitted k_EW is 3.2–4.2, against the true k_min ≈ 1.7–1.8. So it over-predicts at 99.9% and then under-predicts further out:

| Exceedance probability | Point 50, EW vs LR2 | Point 150, EW vs LR2 |
|---|---|---|
| 1e−3 | +0.11% | +0.11% |
| 1e−5 | −0.35% | −0.42% |
| 1e−7 | −1.1% | −1.3% |
| 1e−9 | −2.0% | −2.3% |

**For extremes (return periods), use the saddlepoint closure, not a family fit.**
