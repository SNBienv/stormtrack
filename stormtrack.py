"""stormtrack - a data-agnostic tropical-cyclone centre tracker.

Author : SUMAILI NDEBA Bienvenu
Contact: sumailib@gmail.com; sumaili.bienvenu@um6p.ma
Code is free for use (MIT licence, see LICENSE).

Give it any gridded field it can open (NetCDF / GRIB / Zarr; WRF incl. moving nests, GFS, ERA5,
MPAS remapped to lat-lon, ...). It finds the pressure field, coordinates and times by itself,
follows the storm centre from frame to frame, refines the centre below the grid spacing and
estimates the motion vector (forward speed + heading).

    python stormtrack.py "wrfout_d02_*" --first-guess 31.3 137.9 --out track.csv --plot track.png

Algorithm (per frame)
  1. Search disc: around the first guess (frame 1) or around the position extrapolated from the
     last motion; radius = max_dev_speed * dt + 50 km (max_speed for the 2nd frame), 100-600 km.
  2. Candidate = minimum sea-level pressure inside the disc (points over terrain higher than
     --mask-terrain are skipped when the pressure had to be reduced from surface pressure).
  3. Refinement = pressure-deficit centroid within --refine-km of the candidate:
     weights max(p_env - p, 0), p_env = 90th percentile of the disc. Removes grid-step jitter.
  4. Quality: depth = p_env - p_min; frames shallower than --min-depth are flagged "weak".
  Motion: least-squares line through the centre positions within +-window_h (local tangent plane),
  so speed/heading are not tied to the output interval.

Coordinates are re-read on every frame, so a vortex-following (moving) nest is handled.
Longitudes may be 0-360 or -180-180; the dateline is handled. Output longitudes are -180..180.

Dependencies: numpy, pandas, xarray (+ netCDF4). Optional: scipy (--smooth-km), cfgrib (GRIB),
matplotlib (+ cartopy for coastlines) for --plot. MIT licence.
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

__version__ = "0.1.0"
R_KM = 6371.0
G, RD = 9.80665, 287.05

NAMES = {
    "mslp": ["msl", "prmsl", "mslp", "slp", "pmsl", "mslet", "psl", "sea_level_pressure",
             "air_pressure_at_mean_sea_level", "air_pressure_at_sea_level"],
    "psfc": ["psfc", "sp", "ps", "surface_pressure", "pres_surface", "pressfc", "surface_air_pressure"],
    "hgt": ["hgt", "hgt_m", "orog", "terrain", "z_sfc", "surface_altitude", "hgtsfc", "orography"],
    "t2": ["t2", "t2m", "2t", "tmp2m", "air_temperature_2m"],
    "u10": ["u10", "10u", "ugrd10m", "u10m", "uas", "eastward_wind_10m"],
    "v10": ["v10", "10v", "vgrd10m", "v10m", "vas", "northward_wind_10m"],
    "lat": ["lat", "latitude", "xlat", "xlat_m", "nav_lat", "lat_0", "gridlat_0", "lats"],
    "lon": ["lon", "longitude", "xlong", "xlong_m", "nav_lon", "lon_0", "gridlon_0", "lons"],
    "time": ["time", "valid_time", "xtime", "times", "date", "t"],
}
TIME_IN_NAME = re.compile(r"(\d{4})-(\d{2})-(\d{2})[_ T](\d{2})[:\-](\d{2})[:\-](\d{2})")


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def wrap180(lon):
    return (np.asarray(lon, dtype=float) + 180.0) % 360.0 - 180.0


def to_plane(lat, lon, lat0, lon0):
    """local tangent plane (km east, km north) around (lat0, lon0); dateline safe"""
    x = R_KM * np.radians(wrap180(lon - lon0)) * np.cos(np.radians(lat0))
    y = R_KM * np.radians(np.asarray(lat) - lat0)
    return x, y


def from_plane(x, y, lat0, lon0):
    lat = lat0 + np.degrees(y / R_KM)
    lon = lon0 + np.degrees(x / (R_KM * np.cos(np.radians(lat0))))
    return float(lat), float(wrap180(lon))


def _find(ds, key, required=False):
    want = NAMES[key]
    for v in list(ds.data_vars) + list(ds.coords):
        if v.lower() in want or str(ds[v].attrs.get("standard_name", "")).lower() in want:
            return v
    if required:
        raise KeyError(f"no {key} variable found; looked for {want}. Use --var / rename.")
    return None


def _open(path, grib_filter=None):
    low = path.lower()
    if low.endswith((".grb", ".grib", ".grb2", ".grib2")) or "pgrb2" in low:
        kw = {"indexpath": ""}
        tries = [grib_filter] if grib_filter else [{"typeOfLevel": "meanSea"}, {"typeOfLevel": "surface"}]
        last = None
        for flt in tries:
            try:
                return xr.open_dataset(path, engine="cfgrib", backend_kwargs={**kw, "filter_by_keys": flt})
            except Exception as e:
                last = e
        raise RuntimeError(f"cannot read GRIB {path}: {last}")
    if low.endswith(".zarr") or low.endswith(".zarr/"):
        return xr.open_zarr(path)
    return xr.open_dataset(path)


def _time_values(ds, tdim, path):
    """one pandas Timestamp per index of tdim (NaT if unknown)"""
    n = ds.sizes.get(tdim, 1) if tdim else 1
    if "Times" in ds.variables:
        raw = ds["Times"].values
        out = []
        for r in np.atleast_1d(raw):
            s = r.tobytes().decode() if hasattr(r, "tobytes") and r.dtype.kind in "SU" and r.ndim else str(r)
            s = s.replace("b'", "").replace("'", "")
            out.append(pd.Timestamp(s.replace("_", " ")))
        if len(out) == n:
            return out
    for key in ("valid_time", "time", "xtime", "Time", "XTIME"):
        if key in ds.variables and np.issubdtype(ds[key].dtype, np.datetime64):
            v = np.atleast_1d(ds[key].values).ravel()
            if v.size == n:
                return [pd.Timestamp(x) for x in v]
            if v.size == 1:
                return [pd.Timestamp(v[0])] * n
    m = TIME_IN_NAME.search(path)
    if m and n == 1:
        return [pd.Timestamp(*map(int, m.groups()))]
    return [pd.NaT] * n


@dataclass
class Frame:
    time: pd.Timestamp
    lat: np.ndarray
    lon: np.ndarray
    p: np.ndarray
    wspd: np.ndarray | None
    source: str


def frames(paths, var=None, mask_terrain=200.0, grib_filter=None):
    """return Frame objects from any list of files, sorted by time"""
    out = []
    for path in paths:
        ds = _open(path, grib_filter)
        latv, lonv = _find(ds, "lat", True), _find(ds, "lon", True)
        pv = var or _find(ds, "mslp")
        reduce = False
        if pv is None:
            pv = _find(ds, "psfc", True)
            reduce = True
        hv, tv = _find(ds, "hgt"), _find(ds, "t2")
        uv, vv = _find(ds, "u10"), _find(ds, "v10")
        spatial = ds[pv].dims[-2:]
        extra = [d for d in ds[pv].dims if d not in spatial]
        tdim = extra[0] if extra else None
        times = _time_values(ds, tdim, path)

        def at(name, k):
            """2-D slice of any variable at frame k: time dim -> k, any other non-spatial dim -> 0"""
            a = ds[name]
            if a.ndim == 1:
                return np.asarray(a.values, float)
            sel = {d: (k if d == tdim else 0) for d in a.dims if d not in spatial}
            return np.asarray(a.isel(sel).values, float).squeeze()

        for k in range(len(times)):
            lat, lon = at(latv, k), at(lonv, k)
            p = at(pv, k)
            if lat.ndim == 1:
                lon, lat = np.meshgrid(lon, lat)
            if np.nanmedian(p) > 2000:
                p = p / 100.0
            if reduce:
                if hv is None:
                    warnings.warn("only surface pressure and no terrain height: tracking raw surface pressure")
                else:
                    z = at(hv, k)
                    t = at(tv, k) if tv else 288.15 - 0.0065 * z
                    p = p * np.exp(G * z / (RD * (t + 0.0065 * z / 2)))
                    p = np.where(z > mask_terrain, np.nan, p)
            w = np.hypot(at(uv, k), at(vv, k)) if (uv and vv) else None
            out.append(Frame(times[k], lat, wrap180(lon), p, w, path))
    out.sort(key=lambda f: (pd.Timestamp.max if pd.isna(f.time) else f.time))
    return out


def _is_global(lon):
    """True for a grid that wraps around in longitude (no east/west border to avoid)"""
    span = np.nanmax(lon) - np.nanmin(lon)
    return lon.ndim == 2 and span > 350


def _smooth(p, lat, lon, km):
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        warnings.warn("scipy missing: --smooth-km ignored")
        return p
    dy = np.nanmedian(haversine_km(lat[:-1, :], lon[:-1, :], lat[1:, :], lon[1:, :]))
    sig = km / max(dy, 1e-6)
    ok = np.isfinite(p)
    num = gaussian_filter(np.where(ok, p, 0.0), sig)
    den = gaussian_filter(ok.astype(float), sig)
    return np.where(ok, num / np.maximum(den, 1e-9), np.nan)


def track(frs, first_guess=None, box=None, refine_km=150.0, max_speed=30.0, max_dev_speed=15.0,
          min_depth=1.0, smooth_km=0.0, vmax_km=200.0, max_misses=2, edge=5):
    """follow the storm through the frames; returns one row per frame (see README for columns)"""
    rows, misses = [], 0
    for f in frs:
        p = _smooth(f.p, f.lat, f.lon, smooth_km) if smooth_km > 0 else f.p
        valid = np.isfinite(p)
        if edge > 0 and min(p.shape) > 2 * edge:
            border = np.ones(p.shape, bool)
            border[edge:-edge, edge:-edge] = False
            if not _is_global(f.lon):
                valid &= ~border
        good = [r for r in rows if r["flag"] != "lost"]
        if not good:
            if first_guess is not None:
                c0, radius = first_guess, 500.0
            else:
                c0, radius = None, np.inf
                if box is not None:
                    la0, la1, lo0, lo1 = box
                    lo = f.lon
                    inside = (f.lat >= la0) & (f.lat <= la1) & (wrap180(lo - lo0) >= 0) & (wrap180(lo1 - lo) >= 0)
                    valid &= inside
        else:
            last = good[-1]
            dt = (f.time - last["time"]).total_seconds() if pd.notna(f.time) and pd.notna(last["time"]) else 3600.0
            if len(good) >= 2 and pd.notna(good[-2]["time"]):
                x, y = to_plane(last["lat"], last["lon"], good[-2]["lat"], good[-2]["lon"])
                ddt = (last["time"] - good[-2]["time"]).total_seconds() or 1.0
                c0 = from_plane(x / ddt * dt, y / ddt * dt, last["lat"], last["lon"])
                radius = max_dev_speed * dt / 1000 + 50
            else:
                c0, radius = (last["lat"], last["lon"]), max_speed * dt / 1000 + 50
            radius = float(np.clip(radius, 100, 600))
        if c0 is not None:
            valid &= haversine_km(c0[0], c0[1], f.lat, f.lon) <= radius
        if not valid.any():
            rows.append(dict(time=f.time, lat=np.nan, lon=np.nan, grid_lat=np.nan, grid_lon=np.nan,
                             pmin_hpa=np.nan, depth_hpa=np.nan, vmax_ms=np.nan, flag="lost", source=f.source))
            misses += 1
            if misses > max_misses:
                break
            continue
        misses = 0
        j, i = np.unravel_index(np.argmin(np.where(valid, p, np.inf)), p.shape)
        la0, lo0 = f.lat[j, i], f.lon[j, i]
        disc = np.isfinite(p) & (haversine_km(la0, lo0, f.lat, f.lon) <= refine_km)
        penv = np.nanpercentile(p[disc], 90)
        w = np.where(disc, np.clip(penv - p, 0, None), 0.0)
        if w.sum() > 0:
            x, y = to_plane(f.lat, f.lon, la0, lo0)
            clat, clon = from_plane((w * x).sum() / w.sum(), (w * y).sum() / w.sum(), la0, lo0)
        else:
            clat, clon = float(la0), float(lo0)
        depth = float(penv - p[j, i])
        vmax = np.nan
        if f.wspd is not None:
            near = haversine_km(clat, clon, f.lat, f.lon) <= vmax_km
            vmax = float(np.nanmax(np.where(near, f.wspd, np.nan)))
        rows.append(dict(time=f.time, lat=clat, lon=clon, grid_lat=float(la0), grid_lon=float(lo0),
                         pmin_hpa=float(f.p[j, i]), depth_hpa=depth, vmax_ms=vmax,
                         flag="ok" if depth >= min_depth else "weak", source=f.source))
    return motion(pd.DataFrame(rows))


def motion(df, window_h=3.0):
    """forward speed (m/s), heading (deg from north), east/north components from a local line fit"""
    df = df.copy()
    for c in ("speed_ms", "speed_kt", "heading_deg", "u_ms", "v_ms"):
        df[c] = np.nan
    ok = df[(df.flag != "lost") & df.time.notna()]
    for idx, r in ok.iterrows():
        win = ok[(ok.time - r.time).abs() <= pd.Timedelta(hours=window_h)]
        if len(win) < 2:
            continue
        s = (win.time - r.time).dt.total_seconds().values
        if np.ptp(s) == 0:
            continue
        x, y = to_plane(win.lat.values, win.lon.values, r.lat, r.lon)
        u = np.polyfit(s, x, 1)[0] * 1000
        v = np.polyfit(s, y, 1)[0] * 1000
        spd = np.hypot(u, v)
        df.loc[idx, ["u_ms", "v_ms", "speed_ms", "speed_kt", "heading_deg"]] = [
            u, v, spd, spd / 0.514444, np.degrees(np.arctan2(u, v)) % 360]
    return df


def plot(df, path, every=1, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df[df.flag != "lost"].reset_index(drop=True)
    lon0 = float(d.lon.iloc[0])
    lon = pd.Series(lon0 + wrap180(d.lon - lon0), index=d.index)
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        pc = ccrs.PlateCarree(central_longitude=float(np.round(lon.mean())))
        fig, ax = plt.subplots(figsize=(9, 7.5), subplot_kw={"projection": pc})
        tr = ccrs.PlateCarree()
        ax.add_feature(cfeature.LAND, facecolor="#e6e3dc")
        ax.add_feature(cfeature.COASTLINE, lw=0.7)
        gl = ax.gridlines(draw_labels=True, lw=0.3, color="gray")
        gl.top_labels = gl.right_labels = False
        kw = {"transform": tr}
    except ImportError:
        fig, ax = plt.subplots(figsize=(9, 7.5))
        ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
        ax.set_aspect(1 / np.cos(np.radians(d.lat.mean())))
        kw = {}
    ax.plot(lon, d.lat, "-", color="k", lw=1.5, **kw)
    sc = ax.scatter(lon, d.lat, c=d.pmin_hpa, cmap="viridis_r", s=40, edgecolors="k", zorder=3, **kw)
    a = d.iloc[::max(every, 1)].dropna(subset=["u_ms"])
    q = ax.quiver(lon[a.index].values, a.lat.values, a.u_ms.values, a.v_ms.values, color="#d6604d",
                  angles="uv", scale_units="inches", scale=8, width=0.005, zorder=4, **kw)
    ax.quiverkey(q, 0.85, 0.05, 8, "motion 8 m/s", labelpos="E", coordinates="axes")
    pad = 1.5
    try:
        ax.set_extent([lon.min() - pad, lon.max() + pad, d.lat.min() - pad, d.lat.max() + pad], crs=kw["transform"])
    except (KeyError, AttributeError):
        pass
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.03)
    cb.locator = MaxNLocator(5)
    cb.formatter = FormatStrFormatter("%.1f" if np.ptp(d.pmin_hpa) < 5 else "%.0f")
    cb.update_ticks()
    cb.set_label("Minimum pressure (hPa)")
    lat_b = float(d.lat.min()) - 0.8 * pad
    width_km = max(float(haversine_km(lat_b, lon.min() - pad, lat_b, lon.max() + pad)), 1.0)
    bar = min([50, 100, 200, 300, 500, 1000, 2000], key=lambda b: abs(b - width_km / 5))
    lon_b = float(lon.min()) - 0.8 * pad
    dlon = bar / (111.32 * np.cos(np.radians(lat_b)))
    ax.plot([lon_b, lon_b + dlon], [lat_b, lat_b], color="k", lw=3, zorder=5, **kw)
    ax.text(lon_b + dlon / 2, lat_b + 0.08 * pad, f"{bar} km", ha="center", va="bottom", zorder=5, **kw)
    ax.annotate("N", xy=(0.95, 0.95), xytext=(0.95, 0.87), xycoords="axes fraction", ha="center",
                va="center", fontweight="bold", arrowprops=dict(arrowstyle="-|>", lw=1.5))
    t0, t1 = d.time.iloc[0], d.time.iloc[-1]
    ax.set_title(title or f"Track {t0:%d %b %HZ} – {t1:%d %b %HZ}", loc="left")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@dataclass
class OverlayStyle:
    """Every visual choice of overlay(). Edit the defaults here, or pass OverlayStyle(...) / keywords."""
    track_color: str = "k"
    track_lw: float = 2.0
    future_ls: str | None = "--"
    centre_marker: str = "o"
    centre_size: float = 12.0
    centre_face: str = "white"
    arrow_color: str = "#d6604d"
    arrow_every: int = 3
    arrow_ms_per_inch: object = 8.0
    arrow_width: float = 0.005
    past_arrow_frac: float = 0.5
    halo: bool = True
    label: bool = True
    units: str = "kt"
    key: bool = True
    key_xy: tuple = (0.80, 0.05)
    fontsize: float = 12.0
    zorder: float = 20.0


COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def read_track(path):
    """load a CSV written by stormtrack (times parsed)"""
    return pd.read_csv(path, parse_dates=["time"])


def track_at(df, t):
    """centre and motion interpolated to time t (None if t is outside the track)"""
    d = df[df.flag != "lost"].set_index("time").sort_index()
    t = pd.Timestamp(t)
    if d.empty or t < d.index.min() or t > d.index.max():
        return None
    x = d[["lat", "u_ms", "v_ms", "pmin_hpa"]].copy()
    x["lon"] = float(d.lon.iloc[0]) + wrap180(d.lon - float(d.lon.iloc[0]))
    r = x.reindex(x.index.union([t])).interpolate(method="time").loc[t]
    spd = float(np.hypot(r.u_ms, r.v_ms))
    return dict(time=t, lat=float(r.lat), lon=float(wrap180(r.lon)), u_ms=float(r.u_ms), v_ms=float(r.v_ms),
                speed_ms=spd, heading_deg=float(np.degrees(np.arctan2(r.u_ms, r.v_ms)) % 360),
                pmin_hpa=float(r.pmin_hpa))


def overlay(ax, df, valid_time=None, style=None, **kw):
    """Draw the tracked storm on an EXISTING map: past track, centre at valid_time, motion arrows, label.

    ax          any matplotlib Axes: a cartopy GeoAxes (e.g. a wrf-python chart, any projection) or a
                plain lon/lat Axes (0-360 or -180..180 longitudes are both handled)
    df          DataFrame from track() / read_track()
    valid_time  time of the chart being drawn; None = whole track, no current centre
    style       OverlayStyle; extra keywords override single fields: overlay(ax, df, t, arrow_color="m")
    returns     dict of the matplotlib artists, so anything can still be restyled afterwards
    """
    import matplotlib.patheffects as pe
    sty = style or OverlayStyle()
    for k, v in kw.items():
        if not hasattr(sty, k):
            raise TypeError(f"unknown style field {k!r}")
        setattr(sty, k, v)
    d = df[df.flag != "lost"].sort_values("time").reset_index(drop=True)
    geo = hasattr(ax, "projection")
    if geo:
        import cartopy.crs as ccrs
        pc = ccrs.PlateCarree()
        tkw = {"transform": pc}
        text_xy = pc._as_mpl_transform(ax)
    else:
        tkw, text_xy = {}, "data"
    lon0 = float(d.lon.iloc[0])
    lon = pd.Series(lon0 + wrap180(d.lon - lon0), index=d.index)
    use360 = (not geo) and ax.get_xlim()[1] > 180
    if use360:
        lon = lon % 360
    halo = [pe.withStroke(linewidth=sty.track_lw + 2.5, foreground="white")] if sty.halo else None
    halo_q = [pe.Stroke(linewidth=1.2, foreground="white"), pe.Normal()] if sty.halo else None
    z = sty.zorder
    art = {}
    if valid_time is None:
        past, fut = d.index, d.index[:0]
    else:
        vt = pd.Timestamp(valid_time)
        past, fut = d.index[d.time <= vt], d.index[d.time >= vt]
    art["track"] = ax.plot(lon[past], d.lat[past], "-", color=sty.track_color, lw=sty.track_lw,
                           path_effects=halo, zorder=z, **tkw)
    if sty.future_ls and len(fut) > 1:
        art["future"] = ax.plot(lon[fut], d.lat[fut], sty.future_ls, color=sty.track_color,
                                lw=sty.track_lw * 0.7, path_effects=halo, zorder=z, **tkw)
    if sty.arrow_ms_per_inch == "auto":
        w_in = ax.get_window_extent().width / ax.figure.dpi
        scale = max(float(np.nanmax(d.speed_ms)), 1.0) / (w_in / 8)
    else:
        scale = float(sty.arrow_ms_per_inch)
    qkw = dict(color=sty.arrow_color, angles="uv", scale_units="inches", scale=scale, **tkw)
    q = None
    if sty.arrow_every and len(past):
        a = d.loc[past].iloc[::sty.arrow_every].dropna(subset=["u_ms"])
        if len(a):
            pkw = dict(qkw, scale=scale / max(sty.past_arrow_frac, 1e-3))
            qp = ax.quiver(lon[a.index].values, a.lat.values, a.u_ms.values, a.v_ms.values,
                           width=sty.arrow_width * 0.8, zorder=z + 1, **pkw)
            if halo_q:
                qp.set_path_effects(halo_q)
            art["arrows"] = qp
    if valid_time is not None:
        c = track_at(d, valid_time)
        if c is not None:
            clon = c["lon"] % 360 if use360 else c["lon"]
            art["centre"] = ax.plot(clon, c["lat"], sty.centre_marker, mfc=sty.centre_face, mec=sty.track_color,
                                    mew=2.5, ms=sty.centre_size, zorder=z + 2, **tkw)
            qc = ax.quiver(np.array([clon]), np.array([c["lat"]]), np.array([c["u_ms"]]), np.array([c["v_ms"]]),
                           width=sty.arrow_width * 1.6, zorder=z + 3, **qkw)
            if halo_q:
                qc.set_path_effects(halo_q)
            art["centre_arrow"] = q = qc
            if sty.label:
                spd = c["speed_ms"] / 0.514444 if sty.units == "kt" else c["speed_ms"]
                txt = f"{COMPASS[int((c['heading_deg'] + 11.25) // 22.5) % 16]} {spd:.0f} {sty.units}"
                h = np.radians(c["heading_deg"])
                off = (22 * np.cos(h), -22 * np.sin(h))
                art["label"] = ax.annotate(
                    txt, xy=(clon, c["lat"]), xycoords=text_xy, xytext=off, textcoords="offset points",
                    ha="right" if off[0] < 0 else "left", va="top" if off[1] < 0 else "bottom",
                    fontsize=sty.fontsize, fontweight="bold", zorder=z + 4,
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.3))
    if sty.key and q is None and "arrows" in art:
        q = art["arrows"]
    if sty.key and q is not None:
        ref = 10.0 if sty.units == "kt" else 5.0
        ref_ms = ref * 0.514444 if sty.units == "kt" else ref
        art["key"] = ax.quiverkey(q, *sty.key_xy, ref_ms, f"motion {ref:.0f} {sty.units}", labelpos="E",
                                  coordinates="axes", fontproperties={"size": sty.fontsize})
    return art


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+", help="files or glob patterns (quote the patterns)")
    ap.add_argument("--var", help="pressure variable name (default: auto)")
    ap.add_argument("--first-guess", nargs=2, type=float, metavar=("LAT", "LON"))
    ap.add_argument("--box", nargs=4, type=float, metavar=("LAT0", "LAT1", "LON0", "LON1"),
                    help="restrict the first search when no first guess is given")
    ap.add_argument("--refine-km", type=float, default=150.0)
    ap.add_argument("--max-speed", type=float, default=30.0, help="m/s, search radius before motion is known")
    ap.add_argument("--max-dev-speed", type=float, default=15.0, help="m/s, allowed departure from persistence")
    ap.add_argument("--min-depth", type=float, default=1.0, help="hPa; shallower centres are flagged weak")
    ap.add_argument("--mask-terrain", type=float, default=200.0, help="m; only used if pressure is reduced")
    ap.add_argument("--smooth-km", type=float, default=0.0)
    ap.add_argument("--edge", type=int, default=5, help="grid points ignored along a limited-area border")
    ap.add_argument("--window-h", type=float, default=3.0, help="half-window of the motion fit (h)")
    ap.add_argument("--grib-filter", help="cfgrib filter, e.g. typeOfLevel=meanSea")
    ap.add_argument("--out", default="track.csv")
    ap.add_argument("--plot")
    ap.add_argument("--every", type=int, default=1, help="draw a motion arrow every N fixes")
    a = ap.parse_args(argv)
    paths = sorted({p for pat in a.files for p in (glob.glob(pat) or [pat])})
    flt = dict(kv.split("=", 1) for kv in a.grib_filter.split(",")) if a.grib_filter else None
    frs = frames(paths, a.var, a.mask_terrain, flt)
    df = track(frs, a.first_guess, a.box, a.refine_km, a.max_speed, a.max_dev_speed, a.min_depth, a.smooth_km,
               edge=a.edge)
    df = motion(df, a.window_h)
    df.to_csv(a.out, index=False, float_format="%.4f")
    print(df[["time", "lat", "lon", "pmin_hpa", "depth_hpa", "speed_ms", "heading_deg", "flag"]]
          .round(2).to_string(index=False))
    if a.plot:
        plot(df, a.plot, a.every)
    return df


if __name__ == "__main__":
    main(sys.argv[1:])
