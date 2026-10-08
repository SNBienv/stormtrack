"""
Is the unresolved part of the wind a Gaussian VECTOR? Test of the Rice closure on real data.

Theory (team/fields/REPORT.md sec. 2): a Gaussian vector is the one family closed under all
linear dynamics (advection, mixing, rotation) and under homogenised intrinsic noise. If the wind
is V = V_res + v' with v' ~ N(0, sigma^2 I), the speed seen by a cup is exactly

    |V| ~ Rice(nu = |V_res|, sigma)          (Rayleigh = Weibull k=2 when nu -> 0)

so the family solve can carry (V_res, sigma^2): V_res from the model, sigma^2 by the squared-weight
operator plus a Green-Kubo source, and the speed law follows with no further closure.

Test on London hourly wind (openair::mydata, 1998-2005). "Resolved" = the vector mean over a window
(3 ... 24 h), as a coarse model would hold it. For every hour in the window, the probability
integral transform PIT = F(s_hour) is computed under:

  Rice, oracle sigma     sigma from the window's own along/cross variances (isotropic mean)
  Rice, predicted sigma  sigma = polynomial in |V_res|, fitted on the other half of the years
  Rice, compound sigma   predicted sigma with its own measured scatter: log sigma ~ N(fit, s_l^2)
                         (sigma itself fluctuates with unresolved state: superstatistics, team/bio)
  Gaussian on speed      N(Rice mean, Rice sd): same mean and spread, scalar noise on speed
  Gaussian, no rectif.   N(|V_res|, Rice sd): noise added to the resolved speed, the usual
                         "injected noise" (it misses the rectification bias)

Results are also split by regime: calm/weak resolved wind (nu < 2 sigma), where the Rice shape and
the rectification matter, and the rest.

A calibrated law gives uniform PIT: reported are the KS distance of the pooled PIT from uniform,
the coverage of the central 90% and the upper 1% (should be 0.90 and 0.01), and the CRPS.
"""
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent


def windows(lon, w):
    n = len(lon) // w * w
    ws = lon.ws.to_numpy(float)[:n].reshape(-1, w)
    uu = lon.u.to_numpy()[:n].reshape(-1, w)
    vv = lon.v.to_numpy()[:n].reshape(-1, w)
    yr = lon.index.year.to_numpy()[:n].reshape(-1, w)[:, 0]
    ok = np.isfinite(ws).all(1) & np.isfinite(uu).all(1)
    ws, uu, vv, yr = ws[ok], uu[ok], vv[ok], yr[ok]
    mu, mv = uu.mean(1), vv.mean(1)
    nu = np.hypot(mu, mv)
    var_iso = 0.5 * (((uu - mu[:, None]) ** 2).mean(1) + ((vv - mv[:, None]) ** 2).mean(1))
    return ws, nu, np.sqrt(var_iso), yr


def crps_from_samples(dist_ppf, obs, n=64):
    """CRPS by quantile quadrature: 2 * mean over levels of the pinball loss."""
    p = (np.arange(n) + 0.5) / n
    q = dist_ppf(p[:, None, None])                                  # (n, windows, hours)
    loss = np.where(obs[None] < q, (1 - p[:, None, None]) * (q - obs[None]), p[:, None, None] * (obs[None] - q))
    return 2 * loss.mean()


def score(pit, crps):
    pit = pit.ravel()
    ks = stats.kstest(pit, "uniform").statistic
    return ks, np.mean((pit > 0.05) & (pit < 0.95)), np.mean(pit > 0.99), crps


if __name__ == "__main__":
    import pyreadr
    lon = pyreadr.read_r(HERE / "data" / "openair_mydata.rda")["mydata"][["date", "ws", "wd"]].dropna()
    lon = lon.set_index("date").asfreq("h")
    rad = np.radians(lon.wd.to_numpy(dtype=float))
    lon["ws"] = lon.ws.astype(float)
    lon["u"], lon["v"] = -lon.ws * np.sin(rad), -lon.ws * np.cos(rad)

    print("London hourly 1998-2005. PIT of hourly speeds given the window's resolved vector.")
    print("Ideal: KS 0, central-90% coverage 0.900, upper-1% exceedance 0.010; CRPS lower = better.\n")
    print(f"{'window':>7} {'model':<26}{'KS':>7}{'cov90':>8}{'>q99':>8}{'CRPS':>8}   | calm windows (nu < 2 sigma):"
          f"{'KS':>7}{'cov90':>8}{'>q99':>8}")
    for w in (3, 6, 12, 24):
        ws, nu, sig, yr = windows(lon, w)
        half = yr % 2
        # out-of-sample sigma(nu): quadratic in nu, fitted on the other parity of years
        sig_hat = np.empty_like(sig)
        for k in (0, 1):
            pfit = np.polyfit(nu[half != k], sig[half != k], 2)
            sig_hat[half == k] = np.clip(np.polyval(pfit, nu[half == k]), 0.05, None)
        lsig_res = np.empty_like(sig)
        for k in (0, 1):
            pfit = np.polyfit(nu[half != k], sig[half != k], 2)
            r = np.log(np.maximum(sig[half != k], 0.05)) - np.log(np.clip(np.polyval(pfit, nu[half != k]), 0.05, None))
            lsig_res[half == k] = r.std()
        rows = []
        for name, s in (("Rice, oracle sigma", sig), ("Rice, predicted sigma", sig_hat)):
            s = np.maximum(s, 0.05)
            d = stats.rice(nu[:, None] / s[:, None], scale=s[:, None])
            rows.append((name, d.cdf(ws), crps_from_samples(d.ppf, ws), nu / s))
        # compound Rice: average the CDF over log-normal sigma scatter (Gauss-Hermite, 9 nodes)
        gx, gw = np.polynomial.hermite_e.hermegauss(9)
        gw = gw / gw.sum()
        sc = np.maximum(sig_hat, 0.05)[:, None, None] * np.exp(lsig_res[:, None, None] * gx[None, None, :])
        cdf_c = (stats.rice(nu[:, None, None] / sc, scale=sc).cdf(ws[:, :, None]) * gw).sum(-1)
        grid = np.linspace(0, ws.max() * 1.5 + 5, 600)
        Fg = (stats.rice(nu[:, None, None] / sc[:, :1, :], scale=sc[:, :1, :]).cdf(grid[None, :, None]) * gw).sum(-1)

        def ppf_c(p, Fg=Fg, grid=grid):
            p = np.broadcast_to(p, (p.shape[0], Fg.shape[0], 1))[:, :, 0]
            return np.array([[np.interp(pp, Fg[i], grid) for i in range(Fg.shape[0])] for pp in p[:, 0]])[:, :, None]
        rows.append(("Rice, compound sigma", cdf_c, crps_from_samples(ppf_c, ws), nu / np.maximum(sig_hat, 0.05)))
        s = np.maximum(sig_hat, 0.05)
        rice_mean = stats.rice(nu / s, scale=s).mean()
        rice_sd = stats.rice(nu / s, scale=s).std()
        g = stats.norm(rice_mean[:, None], rice_sd[:, None])
        rows.append(("Gaussian on speed", g.cdf(ws), crps_from_samples(g.ppf, ws), nu / s))
        g0 = stats.norm(nu[:, None], rice_sd[:, None])
        rows.append(("Gaussian, no rectif.", g0.cdf(ws), crps_from_samples(g0.ppf, ws), nu / s))
        for name, pit, crps, ratio in rows:
            ks, cov, up, cr = score(pit, crps)
            calm = ratio < 2
            ksc, covc, upc, _ = score(pit[calm], 0)
            print(f"{w:>5} h {name:<26}{ks:>7.3f}{cov:>8.3f}{up:>8.3f}{cr:>8.3f}   |{calm.mean() * 100:>4.0f}% calm:"
                  f"{ksc:>7.3f}{covc:>8.3f}{upc:>8.3f}")
        print()
