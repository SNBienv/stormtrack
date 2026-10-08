"""
Speed bias from the unresolved part of the wind: rectification, no injected noise.

A model (WRF) holds the resolved wind vector V = (U, 0) of a cell / hour. A cup anemometer measures
the SPEED of the full wind, |V + v'|, where v' = (a, c) is the part the model does not resolve
(sub-grid eddies, sub-hour gusts). v' has zero mean, but speed is a nonlinear (convex) function of
the components, so

    E|V + v'| = E sqrt((U + a)^2 + c^2)  >=  U          (Jensen)
              ~ U + sigma_c^2 / (2 U)                  for U >> sigma   (cross-wind dominates)
              = sigma sqrt(pi/2) L_{1/2}(-U^2 / 2 sigma^2)   exactly, isotropic Gaussian v' (Rice)

So even a model that is perfect for the resolved vector under-predicts the mean measured speed,
by an amount fixed by the unresolved variance. This is one mechanism by which "the bias comes from
the missing S-": a zero-mean intrinsic term becomes a mean speed bias through the nonlinearity.
With U -> 0 the Rice law becomes Rayleigh = Weibull(k = 2): the Weibull family for speed is what
an unresolved isotropic vector looks like.

Demo on London hourly wind (openair::mydata, 1998-2005, speed + direction):
    "resolved" = the vector mean over a window (1 h ... 24 h), what a coarse model would hold,
    "measured" = mean of the hourly speeds in that window.
Predicted gap (from the within-window component variances only) vs. the actual gap.

For the GEP mast / WRF pairs: call predicted_speed_bias(U_wrf, sigma_along, sigma_cross) with the
mast's within-hour (or within-cell, via Taylor: 1 h x 5.5 m/s ~ 20 km ~ 7 dx of a 3 km WRF)
along/cross standard deviations, and compare with mean(obs_ws - wrf_ws).
"""
from pathlib import Path
import numpy as np
from scipy import special

HERE = Path(__file__).resolve().parent
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(40)       # probabilists' Gauss-Hermite
_GH_W = _GH_W / _GH_W.sum()


def rice_mean(U, sigma):
    """E|V + v'| for isotropic Gaussian v' with per-component sd sigma (exact, Rice)."""
    U, sigma = np.asarray(U, float), np.asarray(sigma, float)
    x = -U ** 2 / (2 * sigma ** 2)
    # L_{1/2}(x) = exp(x/2) [(1 - x) I0(-x/2) - x I1(-x/2)]; use scaled Bessels for large |x|
    lag = (1 - x) * special.i0e(-x / 2) - x * special.i1e(-x / 2)
    return sigma * np.sqrt(np.pi / 2) * lag


def gaussian_speed_mean(U, sigma_a, sigma_c):
    """E sqrt((U + a)^2 + c^2), a ~ N(0, sigma_a^2), c ~ N(0, sigma_c^2) (anisotropic)."""
    U = np.asarray(U, float)[..., None, None]
    a = np.asarray(sigma_a, float)[..., None, None] * _GH_X[:, None]
    c = np.asarray(sigma_c, float)[..., None, None] * _GH_X[None, :]
    return np.sum(np.sqrt((U + a) ** 2 + c ** 2) * _GH_W[:, None] * _GH_W[None, :], axis=(-2, -1))


def predicted_speed_bias(U, sigma_a, sigma_c):
    """Mean measured speed minus resolved speed, from the unresolved variances alone."""
    return gaussian_speed_mean(U, sigma_a, sigma_c) - np.asarray(U, float)


if __name__ == "__main__":
    import pyreadr
    assert abs(rice_mean(3.0, 1.0) - gaussian_speed_mean(3.0, 1.0, 1.0)) < 1e-4
    lon = pyreadr.read_r(HERE / "data" / "openair_mydata.rda")["mydata"][["date", "ws", "wd"]].dropna()
    lon = lon.set_index("date").asfreq("h")
    rad = np.radians(lon.wd.to_numpy(dtype=float))
    lon["ws"] = lon.ws.astype(float)
    lon["u"], lon["v"] = -lon.ws * np.sin(rad), -lon.ws * np.cos(rad)

    print("London hourly 1998-2005. 'Resolved' = vector mean over the window; 'measured' = mean speed.")
    print("Predicted gap uses only the within-window along/cross variances (Gaussian, anisotropic).\n")
    print(f"{'window':>8}{'n':>7}{'resolved':>10}{'measured':>10}{'actual gap':>12}{'predicted':>11}"
          f"{'explained':>11}{'per-window r':>14}")
    for w in (3, 6, 12, 24):
        n = len(lon) // w * w
        ws = lon.ws.to_numpy()[:n].reshape(-1, w)
        uu = lon.u.to_numpy()[:n].reshape(-1, w)
        vv = lon.v.to_numpy()[:n].reshape(-1, w)
        full = np.isfinite(ws).all(1) & np.isfinite(uu).all(1)
        ws, uu, vv = ws[full], uu[full], vv[full]
        mu, mv = uu.mean(1), vv.mean(1)
        U, th = np.hypot(mu, mv), np.arctan2(mv, mu)
        du, dv = uu - mu[:, None], vv - mv[:, None]
        along = du * np.cos(th)[:, None] + dv * np.sin(th)[:, None]
        cross = -du * np.sin(th)[:, None] + dv * np.cos(th)[:, None]
        sa, sc = along.std(1), cross.std(1)
        meas = ws.mean(1)
        pred = predicted_speed_bias(U, sa, sc)
        gap = meas - U
        print(f"{w:>6} h{len(U):>7}{U.mean():>10.3f}{meas.mean():>10.3f}{gap.mean():>12.3f}{pred.mean():>11.3f}"
              f"{pred.mean() / gap.mean() * 100:>10.0f}%{np.corrcoef(gap, pred)[0, 1]:>14.3f}")
