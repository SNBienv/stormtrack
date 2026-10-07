"""
Fit wind-speed families to real measurements.

Data (downloaded from CRAN package mirrors on GitHub):
  Irish daily mean wind, 12 stations, 1961-1978   (gstat::wind, Haslett & Raftery 1989), knots -> m/s
  London hourly wind, 1998-2005                     (openair::mydata), m/s
Families: Weibull 2p, Weibull 3p (location), exponentiated Weibull 3p,
          Champernowne 3p (classic, c = 0), Champernowne 4p (candidate, offset c).
"""
import sys
from pathlib import Path
import numpy as np
import pyreadr
from scipy import stats, optimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import champ4
import champ_lin

KNOT = 0.514444
PROBS = (0.5, 0.9, 0.99, 0.999)


# ---------------------------------------------------------------- fitting
def nm(f, x0):
    r = optimize.minimize(f, x0, method="Nelder-Mead",
                          options=dict(xatol=1e-7, fatol=1e-6, maxiter=20000, maxfev=20000))
    return r


def fit_weibull2(s):
    k, _, lam = stats.weibull_min.fit(s, floc=0)
    d = stats.weibull_min(k, scale=lam)
    return d.logpdf, d.ppf, 2


def fit_weibull3(s):
    k0, _, l0 = stats.weibull_min.fit(s, floc=0)
    smin = s.min()

    def nll(t):
        k, lam, loc = np.exp(t[0]), np.exp(t[1]), smin - np.exp(t[2])
        return -stats.weibull_min.logpdf(s, k, loc=loc, scale=lam).sum()
    best = min((nm(nll, [np.log(k0), np.log(l0), np.log(g)]) for g in (0.1 * smin + 1e-3, 0.5, 2.0)),
               key=lambda r: r.fun)
    k, lam, loc = np.exp(best.x[0]), np.exp(best.x[1]), smin - np.exp(best.x[2])
    d = stats.weibull_min(k, loc=loc, scale=lam)
    return d.logpdf, d.ppf, 3


def fit_expweib(s):
    k0, _, l0 = stats.weibull_min.fit(s, floc=0)

    def nll(t):
        return -stats.exponweib.logpdf(s, np.exp(t[0]), np.exp(t[1]), scale=np.exp(t[2])).sum()
    best = min((nm(nll, [np.log(a0), np.log(k0), np.log(l0)]) for a0 in (0.5, 1.0, 2.0)), key=lambda r: r.fun)
    a, k, lam = np.exp(best.x)
    d = stats.exponweib(a, k, scale=lam)
    return d.logpdf, d.ppf, 3


def fit_champ(s, with_c):
    med = np.median(s)

    def unpack(t):
        a, lam, x0 = np.exp(t[0]), -1 + np.exp(t[1]), np.exp(t[2])
        c = np.exp(t[3]) if with_c else 0.0
        return a, lam, x0, c

    def nll(t):
        v = -champ4.logpdf(s, *unpack(t)).sum()
        return v if np.isfinite(v) else 1e300
    starts = []
    for a0 in (1.5, 3.0, 6.0):
        for l0 in (0.2, 1.0, 4.0):                       # lam + 1
            base = [np.log(a0), np.log(l0), np.log(med)]
            if with_c:
                for c0 in (0.1 * med, med, 3 * med):
                    starts.append(base + [np.log(c0)])
            else:
                starts.append(base)
    # quick screen, then polish the best 3
    screened = sorted(starts, key=nll)[:3]
    best = min((nm(nll, st) for st in screened), key=lambda r: r.fun)
    p = unpack(best.x)
    return (lambda v, p=p: champ4.logpdf(v, *p)), (lambda q, p=p: champ4.ppf(q, *p)), (4 if with_c else 3)


LS_RATIO = []


def _ndeba_unpack(t):
    return np.exp(t[0]), -1 + np.exp(t[1]), t[2]          # alpha, lam, v0


def fit_ndeba_mle(s):
    med, sd = np.median(s), s.std()

    def nll(t):
        v = -champ_lin.logpdf(s, *_ndeba_unpack(t)).sum()
        return v if np.isfinite(v) else 1e300
    starts = [[np.log(a0 / sd), np.log(l1), med] for a0 in (1.0, 2.0, 4.0) for l1 in (0.2, 1.0, 4.0)]
    best = min((nm(nll, st) for st in sorted(starts, key=nll)[:3]), key=lambda r: r.fun)
    p = _ndeba_unpack(best.x)
    return (lambda v, p=p: champ_lin.logpdf(v, *p)), (lambda q, p=p: champ_lin.ppf(q, *p)), 3


def fit_ndeba_ls(s):
    """As in the paper: least squares of n / (cosh(alpha (v - v0)) + lam) on the histogram, n free."""
    edges = np.linspace(0, np.quantile(s, 0.999), 61)
    dens, _ = np.histogram(s, bins=edges, density=True)
    dens *= np.mean(s <= edges[-1])
    mid = 0.5 * (edges[1:] + edges[:-1])
    med, sd = np.median(s), s.std()

    def sse(t):
        n = np.exp(t[3])
        a, lam, v0 = _ndeba_unpack(t[:3])
        h = n * np.exp(-champ_lin._logcosh_plus(a * (mid - v0), lam))
        return np.sum((h - dens) ** 2)
    starts = []
    for a0 in (1.0, 2.0, 4.0):
        for l1 in (0.2, 1.0, 4.0):
            a, lam = a0 / sd, l1 - 1
            n0 = np.exp(champ_lin.logn(a, lam))
            starts.append([np.log(a), np.log(l1), med, np.log(n0)])
    best = min((nm(sse, st) for st in sorted(starts, key=sse)[:3]), key=lambda r: r.fun)
    p = _ndeba_unpack(best.x[:3])
    n_norm = np.exp(champ_lin.logn(*p[:2]) - np.log1p(-champ_lin.G(0.0, *p)))
    LS_RATIO.append(np.exp(best.x[3]) / n_norm)
    return (lambda v, p=p: champ_lin.logpdf(v, *p)), (lambda q, p=p: champ_lin.ppf(q, *p)), 4


FAMILIES = {
    "Weibull 2p": fit_weibull2,
    "Weibull 3p": fit_weibull3,
    "Exp-Weibull 3p": fit_expweib,
    "Champ-Ndeba LS 4p": fit_ndeba_ls,
    "Champ-Ndeba MLE": fit_ndeba_mle,
}


def evaluate(s):
    """In-sample AIC, out-of-sample log-likelihood (fit on 1st half, score 2nd), quantile errors."""
    half = len(s) // 2
    train, test = s[:half], s[half:]
    emp_q = np.quantile(s, PROBS)
    out = {}
    for name, fit in FAMILIES.items():
        logpdf, ppf, npar = fit(s)
        aic = 2 * npar - 2 * logpdf(s).sum()
        lp_tr, _, _ = fit(train)
        oos = lp_tr(test).sum()
        qerr = [(float(np.ravel(ppf(p))[0]) - q) / q * 100 for p, q in zip(PROBS, emp_q)]
        out[name] = dict(aic=aic, oos=oos, qerr=qerr)
    return out


def show(title, s, res):
    base = res["Weibull 2p"]
    print(f"\n### {title}   (n = {len(s):,}, mean {s.mean():.2f} m/s, max {s.max():.1f} m/s)")
    print(f"{'family':<18}{'dAIC':>9}{'dLogLik held-out':>18}   " + "".join(f"{p*100:>8g}%" for p in PROBS))
    for name, r in res.items():
        print(f"{name:<18}{r['aic']-base['aic']:>9.0f}{r['oos']-base['oos']:>18.0f}   "
              + "".join(f"{e:>+8.1f}%" for e in r["qerr"]))


# ---------------------------------------------------------------- data
irl = pyreadr.read_r(HERE / "data" / "gstat_wind.rda")["wind"]
stations = [c for c in irl.columns if c not in ("year", "month", "day")]
lon = pyreadr.read_r(HERE / "data" / "openair_mydata.rda")["mydata"]

print("dAIC: vs Weibull 2p, negative = better.  dLogLik held-out: fit 1st half of record, "
      "score 2nd half, positive = better.\nPercent columns: fitted quantile minus observed quantile.")

# London hourly and daily
ws = lon[["date", "ws"]].dropna()
n_zero = (ws.ws <= 0).sum()
ws_h = ws.ws[ws.ws > 0].to_numpy()
print(f"\n[London hourly: dropped {n_zero} calm/zero hours of {len(ws)}; "
      f"{np.unique(ws_h).size} distinct values -> data are rounded]")
show("London, hourly, 1998-2005", ws_h, evaluate(ws_h))
daily = ws.set_index("date").ws.resample("D").mean().dropna().to_numpy()
daily = daily[daily > 0]
show("London, daily mean (temporal mixing)", daily, evaluate(daily))

# Irish regional mean (spatial mixing)
regional = irl[stations].mean(axis=1).to_numpy() * KNOT
show("Ireland, 12-station average, daily (spatial mixing)", regional, evaluate(regional))

# Irish stations individually
print("\n### Ireland, 12 stations individually, daily means 1961-1978")
summary = {f: dict(daic=[], doos=[], best_aic=0, best_oos=0, q99=[], q999=[]) for f in FAMILIES}
for st in stations:
    s = irl[st].to_numpy() * KNOT
    s = s[s > 0]
    res = evaluate(s)
    best_aic = min(res, key=lambda f: res[f]["aic"])
    best_oos = max(res, key=lambda f: res[f]["oos"])
    summary[best_aic]["best_aic"] += 1
    summary[best_oos]["best_oos"] += 1
    for f, r in res.items():
        summary[f]["daic"].append(r["aic"] - res["Weibull 2p"]["aic"])
        summary[f]["doos"].append(r["oos"] - res["Weibull 2p"]["oos"])
        summary[f]["q99"].append(r["qerr"][2])
        summary[f]["q999"].append(r["qerr"][3])
    print(f"   {st}: best AIC = {best_aic:<17} best held-out = {best_oos}")
print(f"\n{'family':<18}{'median dAIC':>12}{'med dLL out':>12}{'#best AIC':>10}{'#best out':>10}"
      f"{'99% err mean (worst)':>24}{'99.9% err mean (worst)':>25}")
for f, r in summary.items():
    print(f"{f:<18}{np.median(r['daic']):>12.0f}{np.median(r['doos']):>12.0f}{r['best_aic']:>10}{r['best_oos']:>10}"
          f"{np.mean(r['q99']):>+15.1f}% ({np.max(np.abs(r['q99'])):>4.1f})"
          f"{np.mean(r['q999']):>+16.1f}% ({np.max(np.abs(r['q999'])):>4.1f})")
print(f"\nLS fits: fitted amplitude n / normalising n = median {np.median(LS_RATIO):.3f}, range {min(LS_RATIO):.3f}..{max(LS_RATIO):.3f}")
