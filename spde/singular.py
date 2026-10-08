"""
Cyclones and extreme events as singularities: a singular component for the family solve.

A smooth distribution family (Weibull, exp-Weibull, Rice of a cell-mean wind) cannot represent a
tropical cyclone at a mast: the vortex is a moving near-singular structure of the wind field. The
mast sees a hard edge at Vmax with an integrable (Vmax - V)^(-1/2) spike (a fold caustic), a
logarithmic spike in the eye, and a power-law outer tail (team/astro/REPORT.md, sec. 3). A cell
average on a model grid smooths the singularity away, and with it the extremes.

So the singular part is not closed by a family; it is TRACKED (stormtrack.py gives the centre, its
motion, pmin, depth and vmax) and its law at a fixed point follows from geometry:

    V_mast = V_vortex(x_mast - x_centre) + f_t U_translation + V_ambient + v'
    x_centre ~ N(track position, sigma_c^2 I)          (centre error)
    v'       ~ N(0, sigma^2 I)                         (unresolved part, Rice)

so the speed law is a MIXTURE OF RICE LAWS over the centre-error distribution, evaluated by 2-D
Gauss-Hermite quadrature. The caustic at Vmax, the eye and the tails come out of the mixture; no
shape is imposed.

Vortex: Holland (1980) profile, V(r) = Vmax [ (Rm/r)^B exp(1 - (Rm/r)^B) ]^(1/2), counter-clockwise
(northern hemisphere; sign flips south), with inflow angle alpha. B from the pressure deficit,
B = rho e Vmax^2 / dp (Holland 1980). Rm, if not measured, from Willoughby, Darling & Rahn (2006),
Rm = 46.4 exp(-0.0155 Vmax + 0.0169 |lat|) km.

Also here: strip_width(), the Sulem-Sulem-Frisch analyticity-strip width delta of a 1-D field from
its spectrum, |u_k| ~ C k^-n exp(-delta k). delta * k_c is the switch flag between the smooth family
closure (delta k_c > 10), family + non-Gaussian residual (3-10), and a singular component (< 3).
Verified against the exact Cole-Hopf solution of Burgers in team/astro.

Demo (python singular.py): a synthetic storm passes a mast. Compared, for the pooled mast speed
over the passage and for the probability of exceeding high thresholds:
  family on the cell-averaged field   Rice(|cell-mean wind|, derived sub-cell sigma), 27 km cells
  singular component (this module)    Rice mixture over the tracked centre with its error
"""
import numpy as np
from scipy import stats

RHO_AIR, E = 1.15, np.e


# ----------------------------------------------------------------------------- vortex
def holland_B(vmax_ms, depth_hpa, rho=RHO_AIR):
    return float(np.clip(rho * E * vmax_ms ** 2 / (depth_hpa * 100.0), 0.5, 2.5))


def rmw_willoughby(vmax_ms, lat_deg):
    return 46.4 * np.exp(-0.0155 * vmax_ms + 0.0169 * abs(lat_deg))          # km


def holland_speed(r_km, vmax, rm_km, B):
    x = (rm_km / np.maximum(r_km, 1e-3)) ** B
    return vmax * np.sqrt(x * np.exp(1 - x))


def vortex_wind(dx_km, dy_km, vmax, rm_km, B, inflow_deg=20.0, south=False):
    """Wind vector of the vortex at offset (dx, dy) km from the centre (east, north)."""
    r = np.hypot(dx_km, dy_km)
    V = holland_speed(r, vmax, rm_km, B)
    th = np.arctan2(dy_km, dx_km)
    sgn = -1.0 if south else 1.0
    a = np.radians(inflow_deg)
    # tangential (counter-clockwise in the north) rotated inward by the inflow angle
    ut = -np.sin(th) * sgn
    vt = np.cos(th) * sgn
    ur = -np.cos(th)
    vr = -np.sin(th)
    return V * (np.cos(a) * ut + np.sin(a) * ur), V * (np.cos(a) * vt + np.sin(a) * vr)


# ----------------------------------------------------------------------------- mixture law
class RiceMixture:
    """Law of |V| for V = mean_i + N(0, sigma^2 I), with weights w_i (one Rice law per node)."""

    def __init__(self, u, v, w, sigma):
        self.nu = np.hypot(u, v).ravel()
        self.w = (w / w.sum()).ravel()
        self.sigma = float(max(sigma, 1e-3))

    def cdf(self, s):
        s = np.asarray(s, float)
        return (stats.rice.cdf(s[..., None], self.nu / self.sigma, scale=self.sigma) * self.w).sum(-1)

    def sf(self, s):
        s = np.asarray(s, float)
        return (stats.rice.sf(s[..., None], self.nu / self.sigma, scale=self.sigma) * self.w).sum(-1)

    def ppf(self, q, lo=0.0, hi=None):
        hi = hi or float(self.nu.max() + 8 * self.sigma)
        grid = np.linspace(lo, hi, 4000)
        return np.interp(q, self.cdf(grid), grid)


def mast_speed_law(mast_xy_km, centre_xy_km, sigma_c_km, vmax, rm_km, B, translation_ms=(0.0, 0.0),
                   trans_frac=0.5, ambient_ms=(0.0, 0.0), sigma_unres=1.0, n_gh=24, south=False):
    """Speed law at a mast given a tracked centre with isotropic position error sigma_c."""
    x, wx = np.polynomial.hermite_e.hermegauss(n_gh)
    X, Y = np.meshgrid(x * sigma_c_km, x * sigma_c_km)
    W = np.outer(wx, wx)
    dx = mast_xy_km[0] - (centre_xy_km[0] + X)
    dy = mast_xy_km[1] - (centre_xy_km[1] + Y)
    u, v = vortex_wind(dx, dy, vmax, rm_km, B, south=south)
    u = u + trans_frac * translation_ms[0] + ambient_ms[0]
    v = v + trans_frac * translation_ms[1] + ambient_ms[1]
    return RiceMixture(u, v, W, sigma_unres)


def from_track(row, mast_lat, mast_lon, sigma_c_km=25.0, rm_km=None, **kw):
    """Build the mast law from one stormtrack.track_at() / track() row (dict-like)."""
    lat0 = 0.5 * (row["lat"] + mast_lat)
    kx = 111.2 * np.cos(np.radians(lat0))
    mast = ((mast_lon - row["lon"]) * kx, (mast_lat - row["lat"]) * 111.2)
    vmax = float(row["vmax_ms"])
    B = holland_B(vmax, float(row["depth_hpa"])) if "depth_hpa" in row and np.isfinite(row["depth_hpa"]) else 1.5
    rm = rm_km if rm_km is not None else rmw_willoughby(vmax, row["lat"])
    trans = (row.get("u_ms", 0.0), row.get("v_ms", 0.0))
    return mast_speed_law(mast, (0.0, 0.0), sigma_c_km, vmax, rm, B, translation_ms=trans,
                          south=row["lat"] < 0, **kw)


# ----------------------------------------------------------------------------- singularity flag
def strip_width(f, dx=1.0, kmin=2, taper_sd=1 / 12):
    """Analyticity-strip width delta from |f_k| ~ C k^-n exp(-delta k) (Sulem, Sulem & Frisch 1983).

    f: 1-D samples (or 2-D, last axis is transformed and spectra are averaged). Returns
    (delta, n) in the units of dx; delta = inf when no singularity is resolvable, i.e. delta is
    beyond about taper width / 3 (checked: 1 / (x^2 + d^2) gives d = 10.0 and 25.1). The Gaussian taper (an entire function, so it adds no
    singularity) must be negligible at the window edges, otherwise the periodic wrap adds an
    algebraic k^-2 tail that hides the exponential decay. The mean is NOT removed: mean x taper
    is a pure Gaussian whose spectrum would swamp a fast exponential (k = 0 is not fitted anyway).
    The fit stops where the spectrum meets
    its floor (edge leakage, algebraic, or round-off): 3x the largest amplitude in the upper half
    of the spectrum."""
    f = np.atleast_2d(np.asarray(f, float))
    m = f.shape[-1]
    i = np.arange(m)
    t = np.exp(-0.5 * ((i - (m - 1) / 2) / (m * taper_sd)) ** 2)
    a = np.abs(np.fft.rfft(f * t, axis=-1)).mean(0)
    k = np.arange(a.size) * 2 * np.pi / (m * dx)
    floor = 3 * a[a.size // 2:].max()
    below = np.nonzero(a[kmin:] <= floor)[0]
    stop = kmin + (below[0] if below.size else a.size - kmin)
    sel = np.arange(kmin, stop)
    if sel.size < 6:
        return np.nan, np.nan
    A = np.column_stack([np.ones(sel.size), -np.log(k[sel]), -k[sel]])
    coef, *_ = np.linalg.lstsq(A, np.log(a[sel]), rcond=None)
    delta, n = float(coef[2]), float(coef[1])
    if delta < 0 or n > 4:          # Gaussian-like decay: entire within the resolvable range
        return np.inf, n
    return delta, n


# ================================================================================ demo
if __name__ == "__main__":
    rng = np.random.default_rng(1980)
    VMAX, DEPTH, LAT = 50.0, 60.0, 20.0
    B = holland_B(VMAX, DEPTH)
    RM = rmw_willoughby(VMAX, LAT)
    U_TR = np.array([0.0, 6.0])                          # moving north at 6 m/s
    AMB = np.array([3.0, 0.0])
    SIG_TURB = 2.5                                       # unresolved gusts at the mast (truth and model)
    SIG_C = 20.0                                         # track centre error, km
    CELL = 27.0                                          # model effective resolution (9 km grid x 3)
    T = np.arange(-12.0, 12.0 + 1e-9, 1.0 / 6)          # passage, 10-min steps, hours
    print(f"storm: Vmax {VMAX} m/s, depth {DEPTH} hPa -> Holland B {B:.2f}; Rm {RM:.0f} km (Willoughby 2006);"
          f" centre error {SIG_C:.0f} km; model cell {CELL:.0f} km; unresolved gusts sd {SIG_TURB} m/s\n")

    def mean_wind(dx, dy):
        u, v = vortex_wind(dx, dy, VMAX, RM, B)
        return u + 0.5 * U_TR[0] + AMB[0], v + 0.5 * U_TR[1] + AMB[1]

    x, wx = np.polynomial.hermite_e.hermegauss(16)
    NX, NY = np.meshgrid(x * SIG_C, x * SIG_C)
    NW = (np.outer(wx, wx) / np.outer(wx, wx).sum()).ravel()
    g = np.linspace(-CELL / 2, CELL / 2, 9)
    GX, GY = np.meshgrid(g, g)
    thresholds = (45.0, 50.0, 55.0, 60.0)

    def p_exceed_once(nu_t, sig, th):
        """P(max over the passage > th) for independent 10-min draws around a mean series."""
        return 1 - np.exp(np.sum(np.log(np.clip(stats.rice.cdf(th, nu_t / sig, scale=sig), 1e-300, 1)), axis=-1))

    # Many storms: the true miss distance is random (both sides of the track); each forecast sees
    # the track with a centre error. A calibrated forecast has mean probability = event frequency.
    n_storms = 600
    yc = U_TR[1] * 3.6 * T
    res = {k: dict(p=[], o=[], crps=[]) for k in ("family", "singular")}
    for st in range(n_storms):
        miss = rng.uniform(-80, 80)
        tu, tv = mean_wind(miss, -yc)
        obs = np.hypot(tu + SIG_TURB * rng.standard_normal(T.size), tv + SIG_TURB * rng.standard_normal(T.size))
        events = [float(obs.max() > th) for th in thresholds]
        err = SIG_C * rng.standard_normal(2)
        cu, cv = mean_wind(miss - err[0] - GX[..., None], -(yc + err[1]) - GY[..., None])
        nu_f = np.hypot(cu.mean((0, 1)), cv.mean((0, 1)))
        sig_f = np.sqrt(0.5 * (cu.var((0, 1)) + cv.var((0, 1))) + SIG_TURB ** 2)
        su, sv = mean_wind(miss - err[0] - NX.ravel()[:, None], -(yc + err[1])[None, :] - NY.ravel()[:, None])
        nu_n = np.hypot(su, sv)
        res["family"]["p"].append([p_exceed_once(nu_f, sig_f, th) for th in thresholds])
        res["singular"]["p"].append([float(np.dot(NW, p_exceed_once(nu_n, SIG_TURB, th))) for th in thresholds])
        for k in res:
            res[k]["o"].append(events)
        z = rng.standard_normal((2, 32))
        df = np.hypot(nu_f[:, None] + sig_f[:, None] * z[0], sig_f[:, None] * z[1])
        pick = rng.choice(NW.size, 32, p=NW)
        ds = np.hypot(nu_n[pick].T + SIG_TURB * z[0], SIG_TURB * z[1])
        for k, d_ in (("family", df), ("singular", ds)):
            res[k]["crps"].append(np.mean(np.abs(d_ - obs[:, None]).mean(1)
                                          - 0.5 * np.abs(d_[:, :, None] - d_[:, None, :]).mean((1, 2))))

    print(f"{n_storms} storm passages, true miss distance uniform in [-80, 80] km, forecast centre error"
          f" {SIG_C:.0f} km.")
    print("Event: the mast exceeds the threshold at least once during the passage.")
    print("Reliability: mean forecast probability vs observed frequency; Brier score (lower = better);")
    print("Brier skill vs climatology. 'p>0.5 & no event' / 'p<0.1 & event' count confident misses.\n")
    print(f"{'threshold':>9}{'observed':>10}   {'model':<10}{'mean p':>8}{'Brier':>8}{'skill':>8}{'false alarm':>13}{'missed':>8}")
    for j, th in enumerate(thresholds):
        o = np.array(res["family"]["o"])[:, j]
        clim = np.mean((o - o.mean()) ** 2)
        for k in ("family", "singular"):
            pr = np.array(res[k]["p"])[:, j]
            bs = np.mean((pr - o) ** 2)
            fa = np.sum((pr > 0.5) & (o == 0))
            ms = np.sum((pr < 0.1) & (o == 1))
            lead = f"{th:>6.0f} m/s{o.mean():>10.3f}" if k == "family" else f"{'':>19}"
            print(f"{lead}   {k:<10}{pr.mean():>8.3f}{bs:>8.4f}{1 - bs / clim:>8.2f}{fa:>13d}{ms:>8d}")
    print(f"\nCRPS of the 10-min speed: family {np.mean(res['family']['crps']):.2f} m/s, singular "
          f"{np.mean(res['singular']['crps']):.2f} m/s\n")

    xs = np.linspace(-400, 400, 2048)
    kc = np.pi / CELL
    print(f"singularity flag on 800 km transects (model cell {CELL:.0f} km, k_c = pi / cell):")
    fields = [("smooth ambient", 3 * np.sin(2 * np.pi * xs / 400) + np.cos(2 * np.pi * xs / 133) + 8)]
    fields += [(f"{d:.0f} km from the eye", np.hypot(*mean_wind(xs, d * np.ones_like(xs)))) for d in (10, 25, 50)]
    for name, fld in fields:
        d_, n_ = strip_width(fld, dx=xs[1] - xs[0])
        flag = "singular component" if d_ * kc < 3 else "family + residual" if d_ * kc < 10 else "smooth family"
        print(f"   {name:<22} delta = {d_:6.1f} km  delta*k_c = {d_ * kc:6.2f}  -> {flag}")
