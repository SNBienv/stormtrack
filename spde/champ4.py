"""
Candidate 4-parameter Champernowne for wind speed x >= 0.

Classic Champernowne (1952) on y = log(x + c):
    g(y) = n / (cosh(alpha (y - y0)) + lam),   alpha > 0, lam > -1
Speed density, truncated to x >= 0 (y >= log c):
    f(x) = g(log(x + c)) / ((x + c) (1 - G(log c)))
Parameters: alpha, lam, x0 = exp(y0), c >= 0.
c = 0 gives the classic 3-parameter Champernowne of log-speed.
"""
import numpy as np


def _G(y, a, lam, y0):
    """CDF of the classic Champernowne, closed form for every lam > -1."""
    T = np.exp(np.clip(a * (y - y0), -700, 700))
    if abs(lam - 1) < 1e-7:                                  # logistic
        return T / (T + 1)
    if lam < 1:                                              # lam = cos(theta)
        th = np.arccos(lam)
        return (np.arctan((T + lam) / np.sin(th)) - (np.pi / 2 - th)) / th
    eta = np.arccosh(lam)                                    # lam = cosh(eta)
    return 1 + np.log((T + np.exp(-eta)) / (T + np.exp(eta))) / (2 * eta)


def _logn(a, lam):
    """log normalising constant n."""
    if abs(lam - 1) < 1e-7:
        return np.log(a / 2)
    if lam < 1:
        th = np.arccos(lam)
        return np.log(a * np.sin(th) / (2 * th))
    eta = np.arccosh(lam)
    return np.log(a * np.sinh(eta) / (2 * eta))


def _logcosh_plus(u, lam):
    """log(cosh(u) + lam), overflow-safe."""
    au = np.abs(u)
    return au + np.log(0.5 * (1 + np.exp(-2 * au)) + lam * np.exp(-au))


def logpdf(x, a, lam, x0, c):
    y, y0 = np.log(x + c), np.log(x0)
    tail = 1.0 if c == 0 else 1 - _G(np.log(c), a, lam, y0)
    return _logn(a, lam) - _logcosh_plus(a * (y - y0), lam) - np.log(x + c) - np.log(tail)


def cdf(x, a, lam, x0, c):
    y0 = np.log(x0)
    lo = 0.0 if c == 0 else _G(np.log(c), a, lam, y0)
    return (_G(np.log(x + c), a, lam, y0) - lo) / (1 - lo)


def ppf(p, a, lam, x0, c, hi=1e3):
    """Quantile by bisection (monotone CDF)."""
    p = np.atleast_1d(np.asarray(p, float))
    lo_x, hi_x = np.zeros_like(p), np.full_like(p, hi)
    for _ in range(200):
        mid = 0.5 * (lo_x + hi_x)
        below = cdf(mid, a, lam, x0, c) < p
        lo_x, hi_x = np.where(below, mid, lo_x), np.where(below, hi_x, mid)
    return 0.5 * (lo_x + hi_x)


if __name__ == "__main__":
    from scipy import integrate
    for pars in [(3.0, 0.4, 6.0, 0.0), (2.5, -0.7, 5.0, 1.5), (4.0, 1.0, 8.0, 3.0), (2.0, 6.0, 7.0, 0.5)]:
        tot, _ = integrate.quad(lambda v: np.exp(logpdf(v, *pars)), 0, np.inf, limit=500)
        x = 4.3
        num, _ = integrate.quad(lambda v: np.exp(logpdf(v, *pars)), 0, x, limit=500)
        print(f"params {pars}: integral = {tot:.6f}, cdf(4.3) closed = {cdf(x, *pars):.6f}, numeric = {num:.6f}, "
              f"ppf(cdf) = {ppf(cdf(x, *pars), *pars)[0]:.4f}")
