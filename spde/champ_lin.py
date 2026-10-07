"""
Champernowne on wind speed v (reconstruction of the Ndeba et al. 2025 form from its parameter
descriptions: n amplitude, alpha > 0, lambda, y0 = median):

    h(v) = n / (cosh(alpha (v - v0)) + lam)

As a proper distribution it is truncated below at L (L = 0 for wind speed), and n then follows
from normalisation. Vectorised over parameter arrays (lam may differ per grid point).
"""
import numpy as np

_EPS = 1e-7


def _branches(lam):
    lam = np.asarray(lam, float)
    lo, hi = lam < 1 - _EPS, lam > 1 + _EPS
    th = np.arccos(np.clip(lam, -1 + 1e-12, 1 - 1e-12))             # used where lam < 1
    eta = np.arccosh(np.maximum(lam, 1 + 1e-12))                     # used where lam > 1
    return lo, hi, th, eta


def G(y, a, lam, y0):
    """Untruncated CDF."""
    lo, hi, th, eta = _branches(lam)
    T = np.exp(np.clip(a * (y - y0), -700, 700))
    g_lo = (np.arctan((T + np.cos(th)) / np.sin(th)) - (np.pi / 2 - th)) / th
    g_hi = 1 + np.log((T + np.exp(-eta)) / (T + np.exp(eta))) / (2 * eta)
    g_1 = T / (T + 1)
    return np.where(lo, g_lo, np.where(hi, g_hi, g_1))


def Ginv(p, a, lam, y0):
    lo, hi, th, eta = _branches(lam)
    T_lo = np.sin(th) * np.tan(th * p + np.pi / 2 - th) - np.cos(th)
    r = np.exp(2 * eta * (p - 1))
    T_hi = (r * np.exp(eta) - np.exp(-eta)) / (1 - r)
    T_1 = p / (1 - p)
    T = np.where(lo, T_lo, np.where(hi, T_hi, T_1))
    return y0 + np.log(np.maximum(T, 1e-300)) / a


def logn(a, lam):
    lo, hi, th, eta = _branches(lam)
    return np.log(a) + np.where(lo, np.log(np.sin(th) / (2 * th)),
                                np.where(hi, np.log(np.sinh(eta) / (2 * eta)), np.log(0.5)))


def _logcosh_plus(u, lam):
    au = np.abs(u)
    return au + np.log(0.5 * (1 + np.exp(-2 * au)) + lam * np.exp(-au))


def logpdf(v, a, lam, v0, L=0.0):
    return logn(a, lam) - _logcosh_plus(a * (v - v0), lam) - np.log1p(-G(L, a, lam, v0))


def cdf(v, a, lam, v0, L=0.0):
    gL = G(L, a, lam, v0)
    return (G(v, a, lam, v0) - gL) / (1 - gL)


def ppf(p, a, lam, v0, L=0.0):
    gL = G(L, a, lam, v0)
    return Ginv(gL + p * (1 - gL), a, lam, v0)


if __name__ == "__main__":
    from scipy import integrate
    for pars in [(0.6, 0.3, 5.0, 0.0), (0.4, -0.8, 7.0, 0.0), (0.9, 1.0, 4.0, 0.5), (0.5, 5.0, 6.0, 1.0)]:
        a, lam, v0, L = pars
        tot, _ = integrate.quad(lambda v: np.exp(logpdf(v, *pars)), L, np.inf, limit=500)
        x = 6.3
        num, _ = integrate.quad(lambda v: np.exp(logpdf(v, *pars)), L, x, limit=500)
        print(f"{pars}: integral {tot:.6f}  cdf closed {float(cdf(x, *pars)):.6f} numeric {num:.6f}  "
              f"ppf(cdf) {float(ppf(cdf(x, *pars), *pars)):.5f}")
