"""
One reduction for both the real mast and the virtual mast.

The comparison between the SPDE, the PDE and the measurements is only fair if
every series goes through the *same* code: same gap rules, same rotation into
the mean wind, same detrending, same Welch windows, same increment lags.  So
the GEP logger file and the virtual-mast output of ``run_cases.py`` are both
turned into the same minute-level DataFrame (``time``, ``ws``, ``wd``,
``wd_std``) and handed to :func:`reduce_segments`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal

__all__ = [
    "wind_components",
    "rotate_to_mean_wind",
    "welch_psd",
    "autocorrelation",
    "integral_time",
    "increment_moments",
    "regularise_minutes",
    "reduce_segments",
    "mean_spectra",
    "mse_decomposition",
]

INCREMENT_LAGS_MIN = (1, 10, 60)
CUP_EPS = 0.1          # m/s, guards the gust-factor division


# ---------------------------------------------------------------------------
# kinematics
# ---------------------------------------------------------------------------
def wind_components(ws, wd_deg):
    """Eastward/northward components from speed and meteorological direction.

    ``wd`` is the direction the wind blows FROM (0 = north, 90 = east).
    """
    rad = np.radians(np.asarray(wd_deg, dtype=float))
    ws = np.asarray(ws, dtype=float)
    return -ws * np.sin(rad), -ws * np.cos(rad)


def rotate_to_mean_wind(u, v):
    """Return ``(along', cross', U, theta)`` for one segment.

    ``U`` is the magnitude of the vector-mean wind and ``theta`` its heading
    (math convention, radians).  The primed series have zero mean.
    """
    ub, vb = float(np.mean(u)), float(np.mean(v))
    th = np.arctan2(vb, ub)
    c, s = np.cos(th), np.sin(th)
    along = u * c + v * s
    cross = -u * s + v * c
    return along - along.mean(), cross - cross.mean(), float(np.hypot(ub, vb)), th


# ---------------------------------------------------------------------------
# second-order statistics
# ---------------------------------------------------------------------------
def welch_psd(x, dt_s: float, nperseg: int):
    """One-sided PSD [unit^2/Hz] with linear detrending; ``sum(P)*df ~ var``."""
    x = np.asarray(x, dtype=float)
    nperseg = int(min(nperseg, x.size))
    f, p = signal.welch(x, fs=1.0 / dt_s, window="hann", nperseg=nperseg,
                        noverlap=nperseg // 2, detrend="linear",
                        scaling="density", return_onesided=True)
    return f, p


def autocorrelation(x, max_lag: int):
    """Biased ACF via FFT, lags 0..max_lag."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = x.size
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    xf = np.fft.rfft(x, nfft)
    acf = np.fft.irfft(xf * np.conj(xf), nfft)[: max_lag + 1]
    return acf / acf[0] if acf[0] > 0 else np.zeros(max_lag + 1)


def integral_time(acf, dt_s: float) -> float:
    """Integral of the ACF up to its first zero crossing (trapezoid), seconds."""
    neg = np.where(acf <= 0.0)[0]
    stop = int(neg[0]) if neg.size else acf.size
    if stop < 2:
        return 0.5 * dt_s
    trap = getattr(np, "trapezoid", None) or np.trapz        # numpy >= 2 renamed trapz
    return float(trap(acf[:stop], dx=dt_s))


def increment_moments(x, lag: int):
    """Standard deviation and flatness of ``x(t+lag) - x(t)``."""
    x = np.asarray(x, dtype=float)
    if lag >= x.size:
        return np.nan, np.nan
    d = x[lag:] - x[:-lag]
    var = d.var()
    if var <= 0:
        return 0.0, np.nan
    return float(np.sqrt(var)), float(np.mean((d - d.mean()) ** 4) / var ** 2)


# ---------------------------------------------------------------------------
# minute series -> segments
# ---------------------------------------------------------------------------
def regularise_minutes(df: pd.DataFrame, max_gap_min: int) -> pd.DataFrame:
    """Put a minute series on a complete 1-min index; fill short gaps only.

    Gaps are filled on the *components*, not on speed and direction, so a
    direction wrap (359 -> 1 deg) is not interpolated through 180.
    """
    df = df.sort_values("time").drop_duplicates("time").set_index("time")
    idx = pd.date_range(df.index[0].floor("min"), df.index[-1].ceil("min"), freq="1min")
    df = df.reindex(idx)
    u, v = wind_components(df["ws"].to_numpy(), df["wd"].to_numpy())
    out = pd.DataFrame({"u": u, "v": v}, index=idx)
    if "wd_std" in df:
        out["wd_std"] = df["wd_std"].to_numpy()
    if "gust" in df:
        out["gust"] = df["gust"].to_numpy()
    out["ws"] = df["ws"].to_numpy()
    out["filled"] = out["u"].isna()
    lim = int(max_gap_min)
    for col in ("u", "v", "ws"):
        out[col] = out[col].interpolate(limit=lim, limit_area="inside")
    out["filled"] &= out["u"].notna()
    out.index.name = "time"
    return out


def reduce_segments(minutes: pd.DataFrame, segment_hours: float,
                    min_coverage: float, min_mean_speed: float,
                    welch_segment_min: int, source: str = "mast"):
    """Cut a regular 1-min series into segments and reduce each one.

    Returns ``(table, spectra)``:

    * ``table`` - one row per accepted segment: mean wind, along/cross
      standard deviations, integral times, increment std/flatness at
      ``INCREMENT_LAGS_MIN``, sub-minute direction spread, gust factor;
    * ``spectra`` - DataFrame indexed by frequency [Hz] with one along and one
      cross PSD column per segment (``S_along_<i>``, ``S_cross_<i>``).
    """
    seg_len = int(round(segment_hours * 60))
    dt = 60.0
    rows, spec_cols, freq = [], {}, None
    n_seg = len(minutes) // seg_len
    for i in range(n_seg):
        blk = minutes.iloc[i * seg_len:(i + 1) * seg_len]
        ok = blk["u"].notna().to_numpy()
        cover = float(ok.mean())
        if cover < min_coverage or not ok.all():
            # any residual NaN means a gap longer than max_gap_min
            continue
        u = blk["u"].to_numpy()
        v = blk["v"].to_numpy()
        along, cross, U, th = rotate_to_mean_wind(u, v)
        if U < min_mean_speed:
            continue
        max_lag = seg_len // 2
        acf_a = autocorrelation(along, max_lag)
        acf_c = autocorrelation(cross, max_lag)
        f, Sa = welch_psd(along, dt, welch_segment_min)
        _, Sc = welch_psd(cross, dt, welch_segment_min)
        freq = f
        spec_cols[f"S_along_{i}"] = Sa
        spec_cols[f"S_cross_{i}"] = Sc

        row = {
            "segment": i,
            "source": source,
            "start": blk.index[0],
            "U_vec": U,
            "ws_mean": float(blk["ws"].mean()),
            "heading_deg": float(np.degrees(th)),
            "sigma_along": float(along.std()),
            "sigma_cross": float(cross.std()),
            "T_along_s": integral_time(acf_a, dt),
            "T_cross_s": integral_time(acf_c, dt),
            "filled_fraction": float(blk["filled"].mean()),
            "hour_utc_start": int(blk.index[0].hour),
        }
        for lag in INCREMENT_LAGS_MIN:
            s, fl = increment_moments(along, lag)
            row[f"dA{lag}_std"], row[f"dA{lag}_flat"] = s, fl
            s, fl = increment_moments(cross, lag)
            row[f"dC{lag}_std"], row[f"dC{lag}_flat"] = s, fl
        if "wd_std" in blk and blk["wd_std"].notna().any():
            # sub-minute cross-wind variance the 1-min average has removed:
            # sigma_v,sub ~ WS * sigma_theta (small-angle)
            sub = (blk["ws"] * np.radians(blk["wd_std"])) ** 2
            row["sigma_cross_subminute"] = float(np.sqrt(np.nanmean(sub)))
        if "gust" in blk and blk["gust"].notna().any():
            row["gust_factor"] = float(np.nanmean(blk["gust"] / blk["ws"].clip(lower=CUP_EPS)))
        rows.append(row)

    table = pd.DataFrame(rows)
    spectra = pd.DataFrame(spec_cols, index=pd.Index(freq if freq is not None else [],
                                                      name="f_hz"))
    return table, spectra



def mean_spectra(spectra: pd.DataFrame) -> pd.DataFrame:
    """Ensemble mean and 5/95 % bands of the along and cross PSDs."""
    out = pd.DataFrame(index=spectra.index)
    for comp in ("along", "cross"):
        cols = [c for c in spectra.columns if c.startswith(f"S_{comp}_")]
        if not cols:
            continue
        arr = spectra[cols].to_numpy()
        out[f"S_{comp}"] = arr.mean(axis=1)
        out[f"S_{comp}_p05"] = np.percentile(arr, 5, axis=1)
        out[f"S_{comp}_p95"] = np.percentile(arr, 95, axis=1)
        out[f"n_{comp}"] = len(cols)
    return out


# ---------------------------------------------------------------------------
# error budget
# ---------------------------------------------------------------------------
def mse_decomposition(obs, mod) -> dict:
    """Exact split ``MSE = bias^2 + (s_m - s_o)^2 + 2 s_m s_o (1 - r)``.

    Population standard deviations, so the three terms add up to the MSE to
    round-off.  The first term is the part no zero-mean noise can produce in a
    linear model; the last two are where an unresolved stochastic component
    shows up (missing variance, and loss of phase/correlation).
    """
    o = np.asarray(obs, dtype=float)
    m = np.asarray(mod, dtype=float)
    ok = np.isfinite(o) & np.isfinite(m)
    o, m = o[ok], m[ok]
    bias = float(m.mean() - o.mean())
    so, sm = float(o.std()), float(m.std())
    r = float(np.corrcoef(o, m)[0, 1]) if o.size > 2 else np.nan
    mse = float(np.mean((m - o) ** 2))
    t_bias = bias ** 2
    t_sd = (sm - so) ** 2
    t_corr = 2.0 * sm * so * (1.0 - r)
    return {
        "n": int(o.size), "obs_mean": float(o.mean()), "mod_mean": float(m.mean()),
        "bias": bias, "obs_std": so, "mod_std": sm, "std_ratio": sm / so if so else np.nan,
        "r": r, "rmse": np.sqrt(mse), "mse": mse,
        "mse_bias": t_bias, "mse_std": t_sd, "mse_corr": t_corr,
        "share_bias": t_bias / mse if mse else np.nan,
        "share_std": t_sd / mse if mse else np.nan,
        "share_corr": t_corr / mse if mse else np.nan,
        "closure": (t_bias + t_sd + t_corr - mse) / mse if mse else np.nan,
    }
