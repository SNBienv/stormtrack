"""
How much of the WRF-vs-mast wind-speed bias is the unresolved part of the wind?

WRF holds a resolved wind vector V for its cell and hour. The cup measures the speed of the full
wind, V + v', where v' is everything WRF does not resolve. v' has zero mean, but speed is convex in
the components, so the mean measured speed exceeds |V| by an amount set by the unresolved
variance alone (rectification, see ../rectification.py):

    E|V + v'| = E sqrt((|V| + a)^2 + c^2) ~ |V| + sigma_c^2 / (2 |V|)

v' is measured at the mast: the 1-min along/cross deviations from the centred hourly vector mean.
By Taylor's hypothesis one hour at ~5.5 m/s covers ~20 km, about the effective resolution
(~7 dx) of the 3 km WRF product, so the within-hour deviations stand in for the sub-grid part.

Outputs (results/rectification/):
  hourly_unresolved.csv   per WRF hour: |V_wrf|, sigma_along, sigma_cross, predicted and actual gap
  budget.csv              exact MSE split (bias^2, std, corr) of
                            raw WRF speed,
                            WRF + rectification with the SAME hour's sigma (oracle: upper bound),
                            WRF + rectification with sigma predicted from WRF speed, fitted on the
                            other month parity (out of sample, usable in forecasting)
  summary.json

The rectification term is not tuned: it is the expectation of the measured speed given the
resolved vector and the measured unresolved variance. If it closes most of the bias, the bias
is the missing S-; whatever is left is systematic model error (roughness, stability, terrain).

    python rectification_budget.py
    python rectification_budget.py --wrf-csv my_wrf_at_mast.csv --start 2024-10-01
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import config as C
from noise_characterisation import load_mast, load_wrf_csv, load_wrf_u10v10
from stats import mse_decomposition, regularise_minutes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rectification import gaussian_speed_mean      # noqa: E402  (spde/rectification.py)

OUT = os.path.join(C.RESULTS, "rectification")


def hourly_unresolved(minutes: pd.DataFrame, times, window_min: int = 60, min_frac: float = 0.8):
    """Centred-window mast statistics at each WRF time: vector mean, mean speed, along/cross sd.

    Deviations are taken from the window's own vector mean and rotated into its direction.
    """
    w = int(window_min)
    mp = int(min_frac * w)
    u, v, ws = minutes["u"], minutes["v"], minutes["ws"]
    roll = lambda s: s.rolling(w, center=True, min_periods=mp)
    mu, mv = roll(u).mean(), roll(v).mean()
    # second moments -> covariance of (u, v) inside each window
    cuu = roll(u * u).mean() - mu ** 2
    cvv = roll(v * v).mean() - mv ** 2
    cuv = roll(u * v).mean() - mu * mv
    th = np.arctan2(mv, mu)
    c, s = np.cos(th), np.sin(th)
    var_a = c * c * cuu + 2 * c * s * cuv + s * s * cvv
    var_c = s * s * cuu - 2 * c * s * cuv + c * c * cvv
    out = pd.DataFrame({
        "obs_U_vec": np.hypot(mu, mv), "obs_ws": roll(ws).mean(),
        "sigma_along": np.sqrt(var_a.clip(lower=0) * w / (w - 1)),
        "sigma_cross": np.sqrt(var_c.clip(lower=0) * w / (w - 1)),
    }).reindex(pd.DatetimeIndex(times))
    out.index.name = "time"
    return out


def fit_sigma_vs_speed(U, sig, fold, deg=2):
    """Out-of-sample sigma(U): polynomial in U fitted on the other month parity."""
    pred = np.full(len(U), np.nan)
    for k in (0, 1):
        tr, te = fold != k, fold == k
        if tr.sum() > 50 and te.any():
            p = np.polyfit(U[tr], sig[tr], deg)
            pred[te] = np.clip(np.polyval(p, U[te]), 0, None)
    return pred


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mast", default=C.MAST_CSV)
    ap.add_argument("--wrf-dir", default=C.WRF_U10V10_DIR)
    ap.add_argument("--wrf-csv", default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--window-min", type=int, default=C.CharacterisationSettings.wrf_match_window_min)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    raw = load_mast(args.mast, args.start, args.end)
    minutes = regularise_minutes(raw, C.CharacterisationSettings.max_gap_min)
    del raw
    wrf = load_wrf_csv(args.wrf_csv) if args.wrf_csv else load_wrf_u10v10(args.wrf_dir)
    if args.start:
        wrf = wrf[wrf.time >= pd.Timestamp(args.start)]
    if args.end:
        wrf = wrf[wrf.time <= pd.Timestamp(args.end)]

    h = hourly_unresolved(minutes, wrf["time"].values, args.window_min)
    df = pd.concat([wrf.set_index("time")[["u", "v", "ws"]].rename(columns={"ws": "wrf_ws"}), h], axis=1)
    df = df.dropna()
    df = df[df.obs_U_vec > C.CUP_THRESHOLD_MS]
    U = df.wrf_ws.to_numpy()
    sa, sc = df.sigma_along.to_numpy(), df.sigma_cross.to_numpy()

    # the mast's own rectification gap (model-free): mean speed minus speed of the mean vector
    df["gap_obs"] = df.obs_ws - df.obs_U_vec
    df["gap_obs_pred"] = gaussian_speed_mean(df.obs_U_vec.to_numpy(), sa, sc) - df.obs_U_vec
    # applied to WRF: what a cup would read if WRF's vector were the resolved truth
    df["wrf_rect_oracle"] = gaussian_speed_mean(U, sa, sc)
    fold = (df.index.month % 2).to_numpy()
    sa_hat = fit_sigma_vs_speed(U, sa, fold)
    sc_hat = fit_sigma_vs_speed(U, sc, fold)
    df["wrf_rect_oos"] = gaussian_speed_mean(U, sa_hat, sc_hat)
    df["res_ws"] = df.obs_ws - df.wrf_ws
    df.to_csv(os.path.join(args.out, "hourly_unresolved.csv"))

    rows = []
    for name, col in (("wrf_raw", "wrf_ws"), ("wrf_plus_rectification_oracle", "wrf_rect_oracle"),
                      ("wrf_plus_rectification_oos", "wrf_rect_oos")):
        d = mse_decomposition(df.obs_ws, df[col])
        d["model"] = name
        rows.append(d)
    budget = pd.DataFrame(rows)
    budget.to_csv(os.path.join(args.out, "budget.csv"), index=False)

    bias_raw = float(df.res_ws.mean())
    rect = float((df.wrf_rect_oos - df.wrf_ws).mean())
    summary = {
        "n_hours": int(len(df)),
        "mast_gap_mean": float(df.gap_obs.mean()),
        "mast_gap_predicted_from_sigma": float(df.gap_obs_pred.mean()),
        "bias_obs_minus_wrf": bias_raw,
        "rectification_predicted_oracle": float((df.wrf_rect_oracle - df.wrf_ws).mean()),
        "rectification_predicted_oos": rect,
        "share_of_bias_explained_oos": rect / bias_raw if bias_raw else None,
        "sigma_along_mean": float(sa.mean()), "sigma_cross_mean": float(sc.mean()),
        "budget": budget.set_index("model")[["rmse", "bias", "share_bias", "share_std", "share_corr"]].to_dict("index"),
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"{len(df):,} WRF hours paired with the mast")
    print(f"mast's own gap, mean speed - |mean vector|: {summary['mast_gap_mean']:.3f} m/s "
          f"(predicted from within-hour sigma: {summary['mast_gap_predicted_from_sigma']:.3f})")
    print(f"bias obs - WRF: {bias_raw:+.3f} m/s; rectification predicts {rect:+.3f} m/s out of sample "
          f"({summary['rectification_predicted_oracle']:+.3f} with the same hour's sigma)")
    print(budget[["model", "rmse", "bias", "share_bias", "share_std", "share_corr"]].to_string(index=False))


if __name__ == "__main__":
    main()
