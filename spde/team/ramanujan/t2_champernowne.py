"""Task 2: Champernowne f(v) = n/(cosh(alpha(v-v0)) + lam).

Untruncated, lam = cos(theta) (theta in (0,pi) for -1<lam<1; theta = i*eta for lam = cosh(eta) > 1):
  int e^{ity}/(cosh y + cos th) dy = 2 pi sinh(th t)/(sin th sinh(pi t))
  => n = alpha sin(th)/(2 th),  phi(t) = E e^{it(V-v0)} = (pi/th) sinh(th t/alpha)/sinh(pi t/alpha)
  kappa_{2n} = (-1)^n 2^{2n} B_{2n} (th^{2n} - pi^{2n}) / (2n alpha^{2n}),  odd cumulants = 0
  excess kurtosis = (6/5)(pi^2+th^2)/(pi^2-th^2)  in (-6/5, +inf)

Truncated (v>=0): with x>0,  1/(cosh x + lam) = 2 sum_{m>=1} (-1)^{m-1} U_{m-1}(lam) e^{-m x}
  => int_0^inf v^r/(cosh(alpha(v-v0))+lam) dv  in closed form via polylogarithms
  P_s(z) := sum_{m>=1} (-1)^{m-1} U_{m-1}(lam) z^m / m^s = (Li_s(-z e^{-i th}) - Li_s(-z e^{i th}))/(2 i sin th)
  (analytic in th, so valid for lam>=1 by continuation; verified below)
"""
import mpmath as mp
mp.mp.dps = 30

def theta_of(lam):
    return mp.acos(mp.mpf(lam)) if lam < 1 else (0 if lam == 1 else 1j * mp.acosh(mp.mpf(lam)))

def Ps(s, z, th):
    if th == 0:                      # lam = 1: U_{m-1}(1) = m  =>  P_s(z) = -Li_{s-1}(-z)
        return -mp.polylog(s - 1, -z)
    return mp.re((mp.polylog(s, -z * mp.exp(-1j * th)) - mp.polylog(s, -z * mp.exp(1j * th))) / (2j * mp.sin(th)))

def trunc_integral(r, alpha, v0, lam):
    """I_r = int_0^inf v^r /(cosh(alpha(v-v0))+lam) dv  (v0 >= 0), closed form."""
    th = theta_of(lam); alpha = mp.mpf(alpha); v0 = mp.mpf(v0)
    E = mp.exp(-alpha * v0); tot = mp.mpf(0)
    for j in range(r + 1):
        cj = mp.binomial(r, j) * v0 ** (r - j) * mp.factorial(j) / alpha ** (j + 1)
        # upper part v>v0:   sum_m c_m * j!/(m alpha)^{j+1}
        up = 2 * Ps(j + 1, 1, th)
        # lower part 0<v<v0: (-1)^j [ P_{j+1}(1) - sum_{i<=j} (alpha v0)^i/i! P_{j+1-i}(e^{-alpha v0}) ]
        lo = 2 * (Ps(j + 1, 1, th) - sum((alpha * v0) ** i / mp.factorial(i) * Ps(j + 1 - i, E, th)
                                         for i in range(j + 1)))
        tot += cj * (up + (-1) ** j * lo)
    return tot

def trunc_integral_quad(r, alpha, v0, lam):
    f = lambda v: v ** r / (mp.cosh(alpha * (v - v0)) + lam)
    return mp.quad(f, [0, v0, v0 + 5 / alpha, v0 + 40 / alpha, mp.inf])

def cf_closed(t, alpha, lam):
    th = theta_of(lam)
    if th == 0: return (mp.pi * t / alpha) / mp.sinh(mp.pi * t / alpha)
    return mp.re((mp.pi / th) * mp.sinh(th * t / alpha) / mp.sinh(mp.pi * t / alpha))

def cf_quad(t, alpha, lam):
    th = theta_of(lam); n = alpha * mp.sin(th) / (2 * th) if th != 0 else alpha / 2
    return mp.re(n) * 2 * mp.quad(lambda y: mp.cos(t * y) / (mp.cosh(alpha * y) + lam), [0, 5 / alpha, 60 / alpha])

def cumulant_closed(nn, alpha, lam):
    th = theta_of(lam)
    return mp.re((-1) ** nn * 2 ** (2 * nn) * mp.bernoulli(2 * nn) * (th ** (2 * nn) - mp.pi ** (2 * nn)) / (2 * nn * alpha ** (2 * nn)))

if __name__ == "__main__":
    worst = 0
    for lam in [-0.9, -0.5, 0.0, 0.5, 0.999, 1.0, 2.0, 7.0]:
        for alpha in [0.4, 1.3]:
            for t in [0.1, 0.7, 2.5]:
                worst = max(worst, abs(cf_closed(t, alpha, lam) - cf_quad(t, alpha, lam)))
    print(f"CF closed form vs quadrature: worst abs err {mp.nstr(worst,3)}")
    worst = 0
    for lam in [-0.7, 0.3, 1.0, 3.0]:
        alpha = mp.mpf('0.8'); th = theta_of(lam); n = mp.re(alpha * mp.sin(th) / (2 * th)) if th != 0 else alpha / 2
        mom = [n * mp.quad(lambda y: y ** p / (mp.cosh(alpha * y) + lam), [-mp.inf, 0, mp.inf]) for p in (2, 4)]
        k2, k4 = mom[0], mom[1] - 3 * mom[0] ** 2
        worst = max(worst, abs(k2 / cumulant_closed(1, alpha, lam) - 1), abs(k4 / cumulant_closed(2, alpha, lam) - 1))
        ek = mp.re(mp.mpf(6) / 5 * (mp.pi ** 2 + th ** 2) / (mp.pi ** 2 - th ** 2))
        print(f"  lam={lam:5}: excess kurtosis closed {mp.nstr(ek,10)}  quad {mp.nstr(k4/k2**2,10)}")
    print(f"kappa2,kappa4 closed vs quad: worst rel err {mp.nstr(worst,3)}")
    worst = 0
    for lam in [-0.8, 0.0, 0.6, 1.0, 1.5, 6.0]:
        for alpha, v0 in [(0.5, 3.0), (1.2, 0.7), (0.3, 8.0), (2.0, 0.0)]:
            for r in range(0, 5):
                a = trunc_integral(r, alpha, v0, lam); b = trunc_integral_quad(r, alpha, v0, lam)
                worst = max(worst, abs(a / b - 1))
    print(f"truncated moments (polylog closed form) vs quad, r=0..4, 24 cases: worst rel err {mp.nstr(worst,3)}")
    # skewness of truncated family: how large can it get?
    import itertools
    best = 0
    for lam, av0 in itertools.product([-0.95, -0.5, 0, 1, 5, 50], [0, 0.25, 0.5, 1, 2, 4]):
        I = [trunc_integral(r, 1, av0, lam) for r in range(4)]
        m1, m2, m3 = I[1] / I[0], I[2] / I[0], I[3] / I[0]
        k2 = m2 - m1 ** 2; k3 = m3 - 3 * m2 * m1 + 2 * m1 ** 3
        g1 = k3 / k2 ** 1.5
        f0 = (1 / (mp.cosh(av0) + lam) / I[0]) * mp.sqrt(k2)   # density at 0 in sd units
        best = max(best, g1)
        if av0 in (0, 1, 4):
            print(f"  lam={lam:6} alpha*v0={av0}: skew={mp.nstr(g1,4):>8}  f(0)*sd={mp.nstr(f0,3)}")
    print("max skew found", mp.nstr(best, 4))
