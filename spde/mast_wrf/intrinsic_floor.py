"""
Is the WRF-vs-mast error the unresolved part (S-), or model error? A model-free split.

A mast measures the unresolved part directly. WRF holds the wind averaged over its effective
cell (L_eff ~ 7 dx: Skamarock 2004). By Taylor's hypothesis that cell average is, to first
order, the mast's own vector mean over the centred window tau_L = L_eff / U. The mast's hourly
mean speed differs from it only by what WRF cannot resolve (dual-reporter idea, Elowitz et al.
2002; Swain, Elowitz & Siggia 2002, with the second reporter built from the same mast):

    floor = E[ (mean_T |V|  -  |mean_{tau_L} V|)^2 ]       per WRF hour, T = mast window

It contains the sub-cell variance AND the rectification gap (mean speed vs speed of the mean
vector). Both are S-. Then the exact error budget of WRF against the mast is

    MSE(WRF - mast) = floor (the most S- can explain) + excess (error in the resolved state)

computed per WRF-speed bin. If the excess dominates, the residual RMSE is not the missing term;
it is fixable error in the resolved state.

Limitation: a single mast sees only the along-wind direction of the cell. The cross-wind
averaging of a 2-D cell removes more variance, so this floor is an upper bound on the
1-D (streamwise) part and does not include cross-stream inhomogeneity. Several masts in one cell
(a true dual reporter) remove that limitation.

A second, independent check: the mast averaging window T that minimises MSE(WRF - mast_T).
For a model that resolves scales down to L_eff, T* should be about L_eff / U, so T* also
measures the model's effective resolution.

Outputs in results/intrinsic_floor/: structure_function.csv, budget_by_speed.csv,
mse_vs_window.csv, summary.json.

    python intrinsic_floor.py
    python intrinsic_floor.py --wrf-csv my_wrf_at_mast.csv --start 2024-10-01
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

import config as C
from noise_characterisation import load_mast, load_wrf_csv, load_wrf_u10v10
from stats import regularise_minutes

OUT = os.path.join(C.RESULTS, "intrinsic_floor")
LAGS_MIN = (1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240)
WINDOWS_MIN = (1, 5, 10, 20, 30, 45, 60, 90, 120, 180, 240)


def structure_functions(minutes: pd.DataFrame, lags=LAGS_MIN, smooth_min: int = 10):
    """D_t(tau) of speed and components, and the local mean speed (smoothed) at each minute."""
    ws, u, v = minutes["ws"], minutes["u"], minutes["v"]
    U = ws.rolling(smooth_min, center=True, min_periods=smooth_min // 2).mean()
    out = {"U": U}
    for lag in lags:
        out[f"Dws_{lag}"] = (ws.shift(-lag) - ws) ** 2
        out[f"Dvec_{lag}"] = (u.shift(-lag) - u) ** 2 + (v.shift(-lag) - v) ** 2
    return pd.DataFrame(out, index=minutes.index)


def half_structure_at(sf_row_means: pd.Series, U: float, L_eff_m: float, lags=LAGS_MIN, kind="ws"):
    """1/2 D(tau) at tau = L_eff / U by log-log interpolation between the measured lags."""
    tau_min = L_eff_m / max(U, 0.5) / 60.0
    lag = np.array(lags, float)
    D = np.array([sf_row_means[f"D{kind}_{l}"] for l in lags], float)
    ok = np.isfinite(D) & (D > 0)
    if ok.sum() < 2:
        return np.nan, tau_min
    t = np.clip(tau_min, lag[ok][0], lag[ok][-1])
    return 0.5 * float(np.exp(np.interp(np.log(t), np.log(lag[ok]), np.log(D[ok])))), tau_min


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mast", default=C.MAST_CSV)
    ap.add_argument("--wrf-dir", default=C.WRF_U10V10_DIR)
    ap.add_argument("--wrf-csv", default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--eff-factor", type=float, default=C.WRF_EFFECTIVE_FACTOR)
    ap.add_argument("--dx-km", type=float, default=C.WRF_NATIVE_DX_KM)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    L_eff = args.eff_factor * args.dx_km * 1000.0

    raw = load_mast(args.mast, args.start, args.end)
    minutes = regularise_minutes(raw, C.CharacterisationSettings.max_gap_min)
    del raw
    wrf = load_wrf_csv(args.wrf_csv) if args.wrf_csv else load_wrf_u10v10(args.wrf_dir)
    if args.start:
        wrf = wrf[wrf.time >= pd.Timestamp(args.start)]
    if args.end:
        wrf = wrf[wrf.time <= pd.Timestamp(args.end)]
    wrf = wrf.set_index("time")

    # ---- structure functions per mast-speed bin (state-dependent floor)
    sf = structure_functions(minutes)
    sf = sf[sf.U > C.CUP_THRESHOLD_MS]
    edges = [1, 2, 3, 4, 5, 6, 8, 10, 13, 20]
    sf["bin"] = pd.cut(sf.U, edges)
    sf_bin = sf.groupby("bin", observed=True).mean()
    sf_bin["n_min"] = sf.groupby("bin", observed=True).size()
    sf_bin.to_csv(os.path.join(args.out, "structure_function.csv"))

    # ---- WRF error vs mast averaging window
    rows = []
    for T in WINDOWS_MIN:
        w = int(T)
        roll = minutes[["u", "v", "ws"]].rolling(w, center=True, min_periods=max(1, int(0.8 * w))).mean()
        o = roll.reindex(wrf.index)
        d = pd.DataFrame({"obs": o.ws, "mod": wrf.ws}).dropna()
        d = d[d.obs > C.CUP_THRESHOLD_MS]
        rows.append(dict(window_min=T, n=len(d), mse=float(((d["mod"] - d.obs) ** 2).mean()),
                         bias=float((d["mod"] - d.obs).mean())))
    mw = pd.DataFrame(rows)
    mw.to_csv(os.path.join(args.out, "mse_vs_window.csv"), index=False)
    best = mw.loc[mw.mse.idxmin()]
    edges = [1, 2, 3, 4, 5, 6, 8, 10, 13, 20]

    # ---- budget by WRF-speed bin: intrinsic floor from the mast itself (Taylor cell average)
    w = C.CharacterisationSettings.wrf_match_window_min
    t_idx = minutes.index
    cs = {c: np.r_[0.0, np.nancumsum(minutes[c].to_numpy())] for c in ("ws", "u", "v")}
    cnt = np.r_[0, np.cumsum(minutes["u"].notna().to_numpy())]

    def centred_mean(col, pos, half):
        lo = np.clip(pos - half, 0, len(t_idx))
        hi = np.clip(pos + half + 1, 0, len(t_idx))
        n = cnt[hi] - cnt[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            return (cs[col][hi] - cs[col][lo]) / n, n / np.maximum(hi - lo, 1)

    pos = t_idx.get_indexer(wrf.index)
    ok = pos >= 0
    wt = wrf.index[ok]
    pos = pos[ok]
    obs, cov = centred_mean("ws", pos, w // 2)
    mu, cov_u = centred_mean("u", pos, w // 2)
    mv, _ = centred_mean("v", pos, w // 2)
    U_loc = np.hypot(mu, mv)
    half_L = np.clip(np.round(L_eff / np.maximum(U_loc, 0.5) / 60.0 / 2).astype(int), 0, 6 * 60)
    cu, cov_L = centred_mean("u", pos, half_L)
    cv, _ = centred_mean("v", pos, half_L)
    pairs = pd.DataFrame({"obs": obs, "cell": np.hypot(cu, cv), "mod": wrf.ws.to_numpy()[ok],
                          "tau_L_min": 2 * half_L + 1, "cov": np.minimum(cov, cov_L)}, index=wt)
    pairs = pairs[(pairs["cov"] >= 0.8) & (pairs.obs > C.CUP_THRESHOLD_MS)].dropna()
    pairs["bin"] = pd.cut(pairs["mod"], edges)
    budget = []
    for b_, g in pairs.groupby("bin", observed=True):
        if len(g) < 30:
            continue
        mse = float(((g["mod"] - g.obs) ** 2).mean())
        fl = float(((g.cell - g.obs) ** 2).mean())
        budget.append(dict(U_bin=str(b_), n=len(g), U_wrf=float(g["mod"].mean()),
                           tau_min=float(g.tau_L_min.median()), mse=mse, floor=fl,
                           floor_share=fl / mse if mse else np.nan, excess=mse - fl,
                           rect_gap=float((g.obs - g.cell).mean()), bias=float((g["mod"] - g.obs).mean())))
    bud = pd.DataFrame(budget)
    bud.to_csv(os.path.join(args.out, "budget_by_speed.csv"), index=False)
    tot_mse = float((bud.mse * bud.n).sum() / bud.n.sum())
    tot_floor = float((bud.floor * bud.n).sum() / bud.n.sum())
    U_typ = float(pairs["mod"].mean())

    summary = dict(L_eff_km=L_eff / 1000, n_pairs=int(len(pairs)), mse=tot_mse, rmse=tot_mse ** 0.5,
                   intrinsic_floor=tot_floor, floor_rms=tot_floor ** 0.5, floor_share=tot_floor / tot_mse,
                   best_window_min=float(best.window_min), mse_at_best_window=float(best.mse),
                   expected_window_min=L_eff / U_typ / 60, implied_L_eff_km=float(best.window_min) * 60 * U_typ / 1000)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"{len(pairs):,} hours; L_eff = {L_eff / 1000:.0f} km")
    print(bud[["U_bin", "n", "U_wrf", "tau_min", "mse", "floor", "floor_share", "rect_gap", "bias"]].round(3).to_string(index=False))
    print(f"\nWRF rmse {summary['rmse']:.2f} m/s; intrinsic floor rms {summary['floor_rms']:.2f} m/s "
          f"-> at most {summary['floor_share'] * 100:.0f}% of the MSE can be the unresolved part (S-);"
          f" the rest is model error.")
    print(f"MSE vs mast window: minimum at T* = {best.window_min:g} min (expected {summary['expected_window_min']:.0f} min"
          f" for L_eff = {L_eff / 1000:.0f} km); implied effective resolution {summary['implied_L_eff_km']:.0f} km")
    print(mw.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
