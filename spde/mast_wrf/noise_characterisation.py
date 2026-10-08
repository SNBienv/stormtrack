"""
Characterise the "noise" from measurements: what the deterministic model
cannot hold, measured at the GEP mast.

Two sources, combined into one noise target:

1. **The mast alone** (1-min logger).  Segment spectra, variances, integral
   times and increment statistics of the along- and cross-wind components.
   This is the full variability, resolved and unresolved.

2. **The mast - WRF residual** (hourly, native U10/V10 at the mast cell).
   The residual is split into
       r = obs - wrf = systematic(WRF state, hour of day) + eps
   where the systematic part is fitted by least squares and evaluated
   *out of sample* (two-fold by month parity), so eps is the part of the
   residual that cannot be predicted from the resolved state.  eps is the
   candidate noise: its variance, spectrum, integral time, flatness, and how
   its spread scales with the WRF wind speed (additive vs multiplicative).

The exact MSE split  bias^2 + (s_m - s_o)^2 + 2 s_m s_o (1 - r)  is written for
the raw and the de-systematised model.  Only the last two terms can come from
an unresolved zero-mean component in a linear model; a bias needs a nonlinear
mechanism (rectification) to arise from noise, and that is what the SPDE runs
test.

The spliced noise spectrum used to force the SPDE is
    S_noise(f) = S_eps(f)     for f below the hourly Nyquist frequency
               = S_mast(f)    above it (hourly WRF output holds nothing there)

Usage (prepares results/characterisation/*.csv + noise_summary.json):

    python noise_characterisation.py
    python noise_characterisation.py --start 2024-10-01 --end 2025-11-30
    python noise_characterisation.py --wrf-csv my_wrf_at_mast.csv
    python noise_characterisation.py --no-wrf          # mast-only target
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd

import config as C
from stats import (autocorrelation, integral_time, increment_moments,
                   mean_spectra, mse_decomposition, reduce_segments,
                   regularise_minutes, welch_psd, wind_components)

OUT = os.path.join(C.RESULTS, "characterisation")


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_mast(path: str = C.MAST_CSV, start=None, end=None) -> pd.DataFrame:
    """GEP 1-min logger -> DataFrame(time, ws, wd, wd_std, gust).

    Values outside the sensor ranges (cup 0-96 m/s, vane 0-360 deg) are set to
    NaN, never clipped.  ``TIMESTAMP`` is the END of the averaging minute.
    """
    t0 = time.time()
    chunks = []
    for ch in pd.read_csv(path, usecols=C.MAST_COLUMNS, chunksize=500_000,
                          low_memory=False):
        chunks.append(ch)
    df = pd.concat(chunks, ignore_index=True)
    out = pd.DataFrame({
        "time": pd.to_datetime(df["TIMESTAMP"], errors="coerce"),
        "ws": pd.to_numeric(df["WS_Avg"], errors="coerce"),
        "wd": pd.to_numeric(df["WD_Avg"], errors="coerce"),
        "wd_std": pd.to_numeric(df["WD_Std"], errors="coerce"),
        "gust": pd.to_numeric(df["WSgust_Max"], errors="coerce"),
    })
    out.loc[(out.ws < 0) | (out.ws > 96), "ws"] = np.nan
    out.loc[(out.wd < 0) | (out.wd > 360), "wd"] = np.nan
    out = out.dropna(subset=["time"])
    if start:
        out = out[out.time >= pd.Timestamp(start)]
    if end:
        out = out[out.time <= pd.Timestamp(end)]
    print(f"  mast: {len(out):,} rows {out.time.min()} -> {out.time.max()} "
          f"({time.time() - t0:.0f} s)")
    return out


def _nearest_cell(lat2d, lon2d, lat, lon):
    dy = (lat2d - lat) * 111.2
    dx = (lon2d - lon) * 111.2 * np.cos(np.radians(lat))
    d = np.hypot(dx, dy)
    j, i = np.unravel_index(np.argmin(d), d.shape)
    return int(j), int(i), float(d[j, i])


def load_wrf_u10v10(directory: str = C.WRF_U10V10_DIR,
                    lat: float = C.MAST_LAT, lon: float = C.MAST_LON) -> pd.DataFrame:
    """Native WRF U10/V10 at the mast cell, hourly.

    Guards against the ERA5+QM placeholder months twice: by file name, and by
    dropping every hour whose u10 has zero spatial spread over the grid (the
    placeholders broadcast one value per hour; a NaN check would pass them).
    """
    import xarray as xr

    files = sorted(glob.glob(os.path.join(directory, "*_u10v10.nc")))
    if not files:
        raise FileNotFoundError(f"no *_u10v10.nc in {directory} (is F: mounted?)")
    rows, cell = [], None
    for fp in files:
        name = os.path.basename(fp)
        if any(m in name for m in C.WRF_PLACEHOLDER_MONTHS):
            print(f"  skip {name}: known ERA5+QM placeholder month")
            continue
        ds = xr.open_dataset(fp)
        if cell is None:
            j, i, dist = _nearest_cell(ds["lat"].values, ds["lon"].values, lat, lon)
            cell = (j, i, dist, float(ds["lat"].values[j, i]), float(ds["lon"].values[j, i]))
            print(f"  WRF cell ({j},{i}) at {cell[3]:.4f},{cell[4]:.4f}, {dist:.2f} km from mast")
        j, i = cell[0], cell[1]
        u_all = ds["u10"]
        spread = u_all.std(dim=[d for d in u_all.dims if d != "time"]).values
        u = ds["u10"].values[:, j, i]
        v = ds["v10"].values[:, j, i]
        t = pd.to_datetime(ds["time"].values)
        ds.close()
        keep = np.isfinite(u) & np.isfinite(v) & (np.abs(u) < 100) & (spread > 1e-6)
        dropped = int((~keep).sum())
        if dropped:
            print(f"  {name}: {dropped} hours dropped (fill or zero spatial spread)")
        rows.append(pd.DataFrame({"time": t[keep], "u": u[keep], "v": v[keep]}))
    wrf = pd.concat(rows).sort_values("time").drop_duplicates("time")
    wrf["ws"] = np.hypot(wrf.u, wrf.v)
    wrf.attrs["cell"] = cell
    return wrf.reset_index(drop=True)


def load_wrf_csv(path: str) -> pd.DataFrame:
    """Any WRF-at-mast table with ``time`` and either ``u,v`` or ``ws,wd``."""
    df = pd.read_csv(path)
    tcol = [c for c in df.columns if c.lower() in ("time", "timestamp", "t")][0]
    out = pd.DataFrame({"time": pd.to_datetime(df[tcol])})
    cols = {c.lower(): c for c in df.columns}
    if "u" in cols or "u10" in cols:
        out["u"] = df[cols.get("u", cols.get("u10"))].astype(float)
        out["v"] = df[cols.get("v", cols.get("v10"))].astype(float)
    else:
        u, v = wind_components(df[cols["ws"]], df[cols["wd"]])
        out["u"], out["v"] = u, v
    out["ws"] = np.hypot(out.u, out.v)
    return out.sort_values("time").drop_duplicates("time").reset_index(drop=True)


# ---------------------------------------------------------------------------
# pairing
# ---------------------------------------------------------------------------
def pair_with_mast(minutes: pd.DataFrame, wrf: pd.DataFrame, window_min: int) -> pd.DataFrame:
    """Centred ``window_min`` mast means at each WRF time (vector and scalar).

    A window is accepted only with >= 80 % valid minutes.
    """
    w = int(window_min)
    roll = minutes[["u", "v", "ws"]].rolling(w, center=True, min_periods=int(0.8 * w)).mean()
    obs = roll.reindex(wrf["time"].values)
    out = pd.DataFrame({
        "time": wrf["time"].values,
        "obs_u": obs["u"].values, "obs_v": obs["v"].values, "obs_ws": obs["ws"].values,
        "wrf_u": wrf["u"].values, "wrf_v": wrf["v"].values, "wrf_ws": wrf["ws"].values,
    }).dropna()
    out["res_ws"] = out.obs_ws - out.wrf_ws
    # vector residual in the frame of the WRF wind (along / cross)
    th = np.arctan2(out.wrf_v, out.wrf_u)
    du, dv = out.obs_u - out.wrf_u, out.obs_v - out.wrf_v
    out["res_along"] = du * np.cos(th) + dv * np.sin(th)
    out["res_cross"] = -du * np.sin(th) + dv * np.cos(th)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# systematic part of the residual, out of sample
# ---------------------------------------------------------------------------
def _design(df: pd.DataFrame) -> np.ndarray:
    th = np.arctan2(df.wrf_v, df.wrf_u)
    h = 2 * np.pi * (df.time.dt.hour + df.time.dt.minute / 60.0) / 24.0
    return np.column_stack([
        np.ones(len(df)), df.wrf_ws,
        np.cos(th), np.sin(th),
        np.cos(h), np.sin(h), np.cos(2 * h), np.sin(2 * h),
    ])


SYSTEMATIC_TERMS = ["const", "wrf_ws", "cos_dir", "sin_dir",
                    "cos_hour", "sin_hour", "cos_2hour", "sin_2hour"]


def split_systematic(pairs: pd.DataFrame, target: str):
    """Two-fold (odd / even month) out-of-sample fit of ``target``.

    Returns ``(predicted, coefficients_per_fold)``.  Fitting and predicting on
    the same hours would let the regression absorb part of the noise and
    understate it.
    """
    X = _design(pairs)
    y = pairs[target].to_numpy()
    fold = (pairs.time.dt.month % 2).to_numpy()
    pred = np.full(len(y), np.nan)
    coefs = {}
    for k in (0, 1):
        tr, te = fold != k, fold == k
        if tr.sum() < X.shape[1] * 10 or te.sum() == 0:
            continue
        beta, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
        pred[te] = X[te] @ beta
        coefs[f"fold_{k}"] = dict(zip(SYSTEMATIC_TERMS, map(float, beta)))
    return pred, coefs


# ---------------------------------------------------------------------------
# hourly residual statistics on contiguous runs
# ---------------------------------------------------------------------------
def contiguous_runs(times: pd.Series, step=pd.Timedelta("1h"), min_len: int = 48):
    """Index arrays of runs with no missing hour, at least ``min_len`` long."""
    gaps = np.where(times.diff().to_numpy() != step.to_timedelta64())[0]
    edges = np.r_[gaps, len(times)]
    runs = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a >= min_len:
            runs.append(np.arange(a, b))
    return runs


def residual_series_stats(pairs: pd.DataFrame, col: str, nperseg: int = 48):
    """ACF-integral time, Welch PSD (averaged over runs), flatness."""
    runs = contiguous_runs(pairs.time, min_len=max(nperseg, 48))
    if not runs:
        return None
    psd, w, T, n = None, 0, [], 0
    for idx in runs:
        x = pairs[col].to_numpy()[idx]
        x = x - x.mean()
        f, p = welch_psd(x, 3600.0, nperseg)
        psd = p * len(idx) if psd is None else psd + p * len(idx)
        w += len(idx)
        T.append((integral_time(autocorrelation(x, min(len(x) // 2, 72)), 3600.0), len(idx)))
        n += len(idx)
    T_s = sum(t * l for t, l in T) / sum(l for _, l in T)
    x = pairs[col].to_numpy()
    sd1, fl1 = increment_moments(x, 1)
    return {
        "f": f, "S": psd / w, "T_s": float(T_s), "n_runs": len(runs), "n_hours": n,
        "std": float(np.nanstd(x)), "skew": float(pd.Series(x).skew()),
        "flatness": float(pd.Series(x).kurt() + 3.0),
        "d1h_std": sd1, "d1h_flat": fl1,
    }


def spread_vs_speed(pairs: pd.DataFrame, col: str, nbins: int = 10) -> pd.DataFrame:
    """Std of ``col`` per WRF-speed decile, with a weighted fit s = a + b U."""
    q = pd.qcut(pairs.wrf_ws, nbins, duplicates="drop")
    g = pairs.groupby(q, observed=True)
    tab = pd.DataFrame({
        "U_wrf_mid": g.wrf_ws.mean(), "std": g[col].std(), "n": g[col].size(),
    }).reset_index(drop=True)
    wts = np.sqrt(tab.n.to_numpy())
    A = np.column_stack([np.ones(len(tab)), tab.U_wrf_mid]) * wts[:, None]
    (a, b), *_ = np.linalg.lstsq(A, tab["std"].to_numpy() * wts, rcond=None)
    tab.attrs["a"], tab.attrs["b"] = float(a), float(b)
    return tab


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mast", default=C.MAST_CSV)
    ap.add_argument("--wrf-dir", default=C.WRF_U10V10_DIR)
    ap.add_argument("--wrf-csv", default=None, help="use a ready WRF-at-mast table")
    ap.add_argument("--no-wrf", action="store_true", help="mast-only noise target")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--segment-hours", type=float, default=None)
    ap.add_argument("--min-speed", type=float, default=None)
    args = ap.parse_args()

    S = C.CharacterisationSettings()
    if args.segment_hours:
        S.segment_hours = args.segment_hours
    if args.min_speed is not None:
        S.min_mean_speed = args.min_speed
    S.start, S.end = args.start, args.end
    os.makedirs(OUT, exist_ok=True)
    summary = {"settings": dict(vars(S)),
               "mast_lat": C.MAST_LAT, "mast_lon": C.MAST_LON}

    # ---- 1. mast alone ---------------------------------------------------
    print("[1/3] mast 1-min record")
    raw = load_mast(args.mast, S.start, S.end)
    minutes = regularise_minutes(raw, S.max_gap_min)
    del raw
    seg, spec = reduce_segments(minutes, S.segment_hours, S.min_coverage,
                                S.min_mean_speed, S.welch_segment_min, source="mast")
    if seg.empty:
        raise SystemExit("no segment passed the screening - relax the settings")
    seg.to_csv(os.path.join(OUT, "mast_segments.csv"), index=False)
    spec.to_csv(os.path.join(OUT, "mast_spectra_segments.csv"))
    ms = mean_spectra(spec)
    ms.to_csv(os.path.join(OUT, "mast_spectra_mean.csv"))
    print(f"  {len(seg)} segments of {S.segment_hours:g} h accepted")
    summary["mast"] = {
        "n_segments": int(len(seg)),
        "U_vec_mean": float(seg.U_vec.mean()),
        "ws_mean": float(seg.ws_mean.mean()),
        "sigma_along": float(np.sqrt((seg.sigma_along ** 2).mean())),
        "sigma_cross": float(np.sqrt((seg.sigma_cross ** 2).mean())),
        "T_along_s": float(seg.T_along_s.median()),
        "T_cross_s": float(seg.T_cross_s.median()),
        "dA1_flat": float(seg.dA1_flat.median()),
        "dA60_flat": float(seg.dA60_flat.median()),
        "sigma_cross_subminute": float(seg.get("sigma_cross_subminute", pd.Series([np.nan])).median()),
    }

    # ---- 2. residual against WRF ---------------------------------------------
    noise = ms[["S_along", "S_cross"]].copy()
    noise["source"] = "mast"
    if not args.no_wrf:
        print("[2/3] WRF at the mast cell")
        wrf = load_wrf_csv(args.wrf_csv) if args.wrf_csv else load_wrf_u10v10(args.wrf_dir)
        wrf_cell = wrf.attrs.get("cell")
        if S.start:
            wrf = wrf[wrf.time >= pd.Timestamp(S.start)]
        if S.end:
            wrf = wrf[wrf.time <= pd.Timestamp(S.end)]
        pairs = pair_with_mast(minutes, wrf, S.wrf_match_window_min)
        print(f"  {len(pairs):,} paired hours")
        for col in ("res_ws", "res_along", "res_cross"):
            pred, coefs = split_systematic(pairs, col)
            pairs[f"sys_{col[4:]}"] = pred
            pairs[f"eps_{col[4:]}"] = pairs[col] - pred
            summary.setdefault("systematic_coefficients", {})[col] = coefs
        pairs = pairs.dropna().reset_index(drop=True)
        pairs.to_csv(os.path.join(OUT, "wrf_mast_pairs.csv"), index=False)

        budget = []
        for var, o, m in (("ws", "obs_ws", "wrf_ws"), ("u", "obs_u", "wrf_u"),
                          ("v", "obs_v", "wrf_v")):
            d = mse_decomposition(pairs[o], pairs[m])
            d.update(variable=var, model="wrf_raw")
            budget.append(d)
        d = mse_decomposition(pairs.obs_ws, pairs.wrf_ws + pairs.sys_ws)
        d.update(variable="ws", model="wrf_plus_systematic_oos")
        budget.append(d)
        pd.DataFrame(budget).to_csv(os.path.join(OUT, "error_budget.csv"), index=False)

        spread = spread_vs_speed(pairs, "eps_ws")
        spread.to_csv(os.path.join(OUT, "residual_spread_vs_speed.csv"), index=False)
        a, b = spread.attrs["a"], spread.attrs["b"]
        Ubar = float(pairs.wrf_ws.mean())

        res = {}
        for comp in ("along", "cross", "ws"):
            st = residual_series_stats(pairs, f"eps_{comp}")
            if st is not None:
                res[comp] = st
        if "along" in res and "cross" in res:
            f_h = res["along"]["f"]
            pd.DataFrame({"S_along": res["along"]["S"], "S_cross": res["cross"]["S"]},
                         index=pd.Index(f_h, name="f_hz")).to_csv(
                os.path.join(OUT, "residual_spectra.csv"))
            f_nyq_hourly = 0.5 / 3600.0
            low = pd.DataFrame({"S_along": res["along"]["S"], "S_cross": res["cross"]["S"],
                                "source": "residual"}, index=pd.Index(f_h, name="f_hz"))
            low = low[(low.index > 0) & (low.index < f_nyq_hourly)]
            high = noise[noise.index >= f_nyq_hourly]
            noise = pd.concat([low, high]).sort_index()

        summary["wrf"] = {
            "cell_j_i_km_lat_lon": list(wrf_cell) if wrf_cell else None,
            "n_pairs": int(len(pairs)),
            "budget_raw_ws": budget[0],
            "budget_after_systematic_ws": budget[-1],
            "eps": {k: {kk: vv for kk, vv in v.items() if kk not in ("f", "S")}
                    for k, v in res.items()},
            "spread_fit": {"a": a, "b": b, "U_wrf_mean": Ubar,
                           "multiplicative_share": b * Ubar / (a + b * Ubar)
                           if (a + b * Ubar) else np.nan},
        }

    # ---- 3. spliced noise target ------------------------------------------------
    print("[3/3] noise target")
    noise.index.name = "f_hz"
    noise.to_csv(os.path.join(OUT, "noise_spectrum.csv"))
    with open(os.path.join(OUT, "noise_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"  written to {OUT}")


if __name__ == "__main__":
    main()
