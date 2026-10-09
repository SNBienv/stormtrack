"""
The publishable test: does a physics-based Rice law beat standard statistical post-processing of
WRF wind at the GEP mast?

Standard baseline (Thorarinsdottir & Gneiting 2010): EMOS with a truncated normal,
    obs ~ N0(a + b U_wrf, (c + d U_wrf)^2),   fitted by minimum CRPS.
Physics-based alternatives (this project): the measured speed is the speed of a vector,
    obs ~ Rice(alpha |V_wrf|, sigma),   the exact law of |alpha V_wrf + v'| for v' ~ N(0, sigma^2 I),
  Rice-EMOS          sigma^2 = s0 + s1 U_wrf^2                       (unresolved variance vs speed)
  Rice-gradient      sigma^2 = s0 + s1 U_wrf^2 + g (D^2/24) |grad V_wrf|^2   (sub-grid spread from the
                     resolved gradients around the mast cell, needs the WRF neighbourhood)
All models are fitted on one month parity and scored on the other (out of sample), the same
two-fold split as noise_characterisation.py. Raw WRF is scored as a point forecast.

The claim worth publishing would be: with the same or fewer fitted parameters, the physically
structured law (Rice: rectification built in, Rayleigh/Weibull-2 limit at calm) is better
calibrated and sharper than the standard EMOS, in particular at low wind and in the tails, and
the fitted unresolved variance agrees with the variance measured independently at the mast
(intrinsic_floor.py). If it does not beat EMOS, the physics is not adding forecast value here.

Outputs (results/rice_vs_emos/): scores.csv, pit_<model>.csv, summary.json

    python rice_vs_emos.py                       # mast + WRF U10/V10 on the user's drives
    python rice_vs_emos.py --window-min 10       # mast 10-min mean (default) or 60
    python rice_vs_emos.py --wrf-csv table.csv   # no neighbourhood -> no Rice-gradient model
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

import config as C
from noise_characterisation import load_mast, load_wrf_csv
from stats import regularise_minutes

OUT = os.path.join(C.RESULTS, "rice_vs_emos")


# ----------------------------------------------------------------------------- data
def load_wrf_neighbourhood(directory=C.WRF_U10V10_DIR, lat=C.MAST_LAT, lon=C.MAST_LON, half=1):
    """Mast-cell U10/V10 plus |grad V|^2 from the (2 half + 1)^2 neighbourhood (centred differences)."""
    import xarray as xr
    from noise_characterisation import _nearest_cell
    files = sorted(glob.glob(os.path.join(directory, "*_u10v10.nc")))
    rows, cell = [], None
    for fp in files:
        if any(m in os.path.basename(fp) for m in C.WRF_PLACEHOLDER_MONTHS):
            continue
        ds = xr.open_dataset(fp)
        if cell is None:
            j, i, dist = _nearest_cell(ds["lat"].values, ds["lon"].values, lat, lon)
            cell = (j, i)
            dx = C.WRF_NATIVE_DX_KM * 1000.0
        j, i = cell
        U = ds["u10"].values[:, j - half:j + half + 1, i - half:i + half + 1]
        V = ds["v10"].values[:, j - half:j + half + 1, i - half:i + half + 1]
        spread = ds["u10"].std(dim=[d for d in ds["u10"].dims if d != "time"]).values
        t = pd.to_datetime(ds["time"].values)
        ds.close()
        c = half
        g2 = (((U[:, c, c + 1] - U[:, c, c - 1]) / (2 * dx)) ** 2 + ((U[:, c + 1, c] - U[:, c - 1, c]) / (2 * dx)) ** 2
              + ((V[:, c, c + 1] - V[:, c, c - 1]) / (2 * dx)) ** 2 + ((V[:, c + 1, c] - V[:, c - 1, c]) / (2 * dx)) ** 2)
        keep = np.isfinite(U[:, c, c]) & np.isfinite(V[:, c, c]) & (spread > 1e-6)
        rows.append(pd.DataFrame({"time": t[keep], "u": U[keep, c, c], "v": V[keep, c, c], "grad2": g2[keep]}))
    wrf = pd.concat(rows).sort_values("time").drop_duplicates("time")
    wrf["ws"] = np.hypot(wrf.u, wrf.v)
    return wrf.reset_index(drop=True)


def pair(minutes, wrf, window_min):
    w = int(window_min)
    obs = minutes["ws"].rolling(w, center=True, min_periods=max(1, int(0.8 * w))).mean().reindex(wrf["time"].values)
    df = wrf.copy()
    df["obs"] = obs.values
    return df.dropna(subset=["obs", "ws"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- scores
def crps_tn0(y, mu, sig):
    """Closed-form CRPS of a normal truncated at 0 (Thorarinsdottir & Gneiting 2010)."""
    sig = np.maximum(sig, 1e-6)
    p = stats.norm.cdf(mu / sig)
    z = (y - mu) / sig
    return sig / p ** 2 * (z * p * (2 * stats.norm.cdf(z) + p - 2) + 2 * stats.norm.pdf(z) * p
                           - stats.norm.cdf(np.sqrt(2) * mu / sig) / np.sqrt(np.pi))


_Z = np.random.default_rng(0).standard_normal((2, 256))


def crps_rice(y, nu, sig):
    """Sample CRPS of Rice(nu, sig) with fixed draws (common random numbers)."""
    d = np.hypot(nu[:, None] + sig[:, None] * _Z[0], sig[:, None] * _Z[1])
    t1 = np.abs(d - y[:, None]).mean(1)
    ds = np.sort(d, axis=1)
    m = ds.shape[1]
    t2 = (ds * (2 * np.arange(1, m + 1) - m - 1)).sum(1) / m ** 2          # E|X-X'|/2 via order stats
    return t1 - t2


def rice_mean(nu, sig):
    x = -nu ** 2 / (2 * sig ** 2)
    return sig * np.sqrt(np.pi / 2) * ((1 - x) * special.i0e(-x / 2) - x * special.i1e(-x / 2))


# ----------------------------------------------------------------------------- models
def fit_emos(df):
    y, U = df.obs.to_numpy(), df.ws.to_numpy()

    def obj(t):
        a, b, lc, ld = t
        return crps_tn0(y, a + b * U, np.exp(lc) + np.exp(ld) * U).mean()
    r = optimize.minimize(obj, [0.0, 1.0, np.log(0.5), np.log(0.1)], method="Nelder-Mead",
                          options=dict(maxiter=4000, xatol=1e-6, fatol=1e-8))
    a, b, lc, ld = r.x
    return dict(kind="tn", f=lambda d: (a + b * d.ws.to_numpy(), np.exp(lc) + np.exp(ld) * d.ws.to_numpy()),
                params=dict(a=a, b=b, c=np.exp(lc), d=np.exp(ld)))


def fit_rice(df, use_grad):
    y, U = df.obs.to_numpy(), df.ws.to_numpy()
    G = df.grad2.to_numpy() * (C.WRF_NATIVE_DX_KM * 1000) ** 2 / 24 if use_grad else np.zeros_like(U)

    def unpack(t):
        return np.exp(t[0]), np.exp(t[1]), np.exp(t[2]), (np.exp(t[3]) if use_grad else 0.0)

    def nll(t):
        al, s0, s1, g = unpack(t)
        sig = np.sqrt(s0 + s1 * U ** 2 + g * G)
        v = -stats.rice.logpdf(y, al * U / sig, scale=sig).sum()
        return v if np.isfinite(v) else 1e300
    x0 = [0.0, np.log(0.3), np.log(0.01)] + ([0.0] if use_grad else [])
    r = optimize.minimize(nll, x0, method="Nelder-Mead", options=dict(maxiter=6000, xatol=1e-6, fatol=1e-6))
    al, s0, s1, g = unpack(r.x)

    def f(d):
        Ud = d.ws.to_numpy()
        Gd = d.grad2.to_numpy() * (C.WRF_NATIVE_DX_KM * 1000) ** 2 / 24 if use_grad else 0.0
        return al * Ud, np.sqrt(s0 + s1 * Ud ** 2 + g * Gd)
    return dict(kind="rice", f=f, params=dict(alpha=al, s0=s0, s1=s1, g=g))


def score(df, model):
    y = df.obs.to_numpy()
    if model is None:
        m = df.ws.to_numpy()
        return dict(crps=np.abs(y - m).mean(), mean=m, pit=None)
    a, b = model["f"](df)
    if model["kind"] == "tn":
        crps = crps_tn0(y, a, b)
        cdf0 = stats.norm.cdf(-a / b)
        pit = (stats.norm.cdf((y - a) / b) - cdf0) / (1 - cdf0)
        p = stats.norm.cdf(a / b)
        mean = a + b * stats.norm.pdf(a / b) / p
    else:
        crps = crps_rice(y, a, b)
        pit = stats.rice.cdf(y, a / b, scale=b)
        mean = rice_mean(a, b)
    return dict(crps=crps.mean(), mean=mean, pit=pit)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mast", default=C.MAST_CSV)
    ap.add_argument("--wrf-dir", default=C.WRF_U10V10_DIR)
    ap.add_argument("--wrf-csv", default=None)
    ap.add_argument("--window-min", type=int, default=10)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    raw = load_mast(args.mast, args.start, args.end)
    minutes = regularise_minutes(raw, C.CharacterisationSettings.max_gap_min)
    del raw
    if args.wrf_csv:
        wrf = load_wrf_csv(args.wrf_csv)
        wrf["grad2"] = np.nan
    else:
        wrf = load_wrf_neighbourhood(args.wrf_dir)
    df = pair(minutes, wrf, args.window_min)
    df = df[df.obs >= 0].reset_index(drop=True)
    has_grad = df.grad2.notna().all()
    fold = (pd.to_datetime(df.time).dt.month % 2).to_numpy()

    names = ["raw WRF", "EMOS truncated normal", "Rice-EMOS"] + (["Rice-gradient"] if has_grad else [])
    pred = {n: dict(mean=np.full(len(df), np.nan), pit=np.full(len(df), np.nan), crps=np.full(len(df), np.nan))
            for n in names}
    params = {n: {} for n in names}
    for k in (0, 1):
        tr, te = df[fold != k], df[fold == k]
        idx = np.nonzero(fold == k)[0]
        models = {"raw WRF": None, "EMOS truncated normal": fit_emos(tr), "Rice-EMOS": fit_rice(tr, False)}
        if has_grad:
            models["Rice-gradient"] = fit_rice(tr, True)
        for n, m in models.items():
            s = score(te, m)
            pred[n]["mean"][idx] = s["mean"]
            if s["pit"] is not None:
                pred[n]["pit"][idx] = s["pit"]
            y = te.obs.to_numpy()
            if m is None:
                pred[n]["crps"][idx] = np.abs(y - te.ws.to_numpy())
            elif m["kind"] == "tn":
                a, b = m["f"](te)
                pred[n]["crps"][idx] = crps_tn0(y, a, b)
            else:
                a, b = m["f"](te)
                pred[n]["crps"][idx] = crps_rice(y, a, b)
            params[n][f"fold_{k}"] = {kk: float(v) for kk, v in (m["params"].items() if m else {})}

    y = df.obs.to_numpy()
    calm = df.ws.to_numpy() < 3
    rows = []
    for n in names:
        p = pred[n]
        r = dict(model=n, crps=float(np.nanmean(p["crps"])), crps_calm=float(np.nanmean(p["crps"][calm])),
                 bias=float(np.nanmean(p["mean"] - y)), rmse_mean=float(np.sqrt(np.nanmean((p["mean"] - y) ** 2))))
        if np.isfinite(p["pit"]).any():
            q = p["pit"][np.isfinite(p["pit"])]
            r.update(ks=float(stats.kstest(q, "uniform").statistic), cov90=float(np.mean((q > 0.05) & (q < 0.95))),
                     above_q99=float(np.mean(q > 0.99)), below_q01=float(np.mean(q < 0.01)))
            pd.Series(q).to_csv(os.path.join(args.out, f"pit_{n.replace(' ', '_')}.csv"), index=False)
        rows.append(r)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(args.out, "scores.csv"), index=False)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(n=int(len(df)), n_calm=int(calm.sum()), window_min=args.window_min, params=params,
                       scores=rows), fh, indent=2)
    print(f"{len(df):,} WRF hours vs the mast's {args.window_min}-min mean speed; out of sample by month parity;"
          f" {calm.sum():,} calm hours (WRF < 3 m/s)")
    print(tab.round(4).to_string(index=False))
    for n in names[1:]:
        print(f"{n}: " + "; ".join(f"{k}: " + ", ".join(f"{a}={b:.3g}" for a, b in v.items()) for k, v in params[n].items()))


if __name__ == "__main__":
    main()
