"""Self-checks for stormtrack with synthetic vortices of KNOWN motion.  Run:  python test_stormtrack.py

Author : SUMAILI NDEBA Bienvenu
Contact: sumailib@gmail.com; sumaili.bienvenu@um6p.ma
Code is free for use (MIT licence, see LICENSE).

Each case writes small NetCDF files to a temporary folder, runs the tracker exactly as a user would
(file in -> CSV out) and compares the recovered motion with the imposed one.
  1. regular 0.25 deg grid, longitudes 0-360 (GFS/ERA5 style), Pa units, one file per time
  2. curvilinear WRF-like grid (2-D XLAT/XLONG, Time dim, PSFC + HGT only, high terrain nearby)
  3. moving nest: the grid itself shifts every frame
  4. dateline crossing (179E -> 179W) with a deeper decoy low 900 km away
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import stormtrack as st

R = 6371.0


def path_points(lat0, lon0, speed, heading, n, dt_h):
    """centre positions for constant speed (m/s) and heading (deg) on the sphere"""
    out, lat, lon = [], lat0, lon0
    for _ in range(n):
        out.append((lat, lon))
        d = speed * dt_h * 3600 / 1000.0 / R
        h = np.radians(heading)
        p1, l1 = np.radians(lat), np.radians(lon)
        p2 = np.arcsin(np.sin(p1) * np.cos(d) + np.cos(p1) * np.sin(d) * np.cos(h))
        l2 = l1 + np.arctan2(np.sin(h) * np.sin(d) * np.cos(p1), np.cos(d) - np.sin(p1) * np.sin(p2))
        lat, lon = np.degrees(p2), st.wrap180(np.degrees(l2))
    return out


def vortex(lat, lon, clat, clon, dp=35.0, rkm=120.0, penv=1010.0):
    r = st.haversine_km(clat, clon, lat, lon)
    return penv - dp * np.exp(-(r / rkm) ** 2)


def check(df, speed, heading, name, tol_speed=0.06, tol_head=3.0, truth=None):
    d = df[(df.flag == "ok")].iloc[2:-2]
    if truth is not None:
        tl = np.array([t[0] for t in truth]); to = np.array([t[1] for t in truth])
        e_ref = st.haversine_km(tl, to, df.lat.values, df.lon.values)
        e_grid = st.haversine_km(tl, to, df.grid_lat.values, df.grid_lon.values)
        print(f"      centre error: grid minimum {np.mean(e_grid):5.2f} km -> refined {np.mean(e_ref):5.2f} km")
    es = np.nanmedian(np.abs(d.speed_ms - speed)) / speed
    eh = np.nanmedian(np.abs(((d.heading_deg - heading) + 180) % 360 - 180))
    ok = es <= tol_speed and eh <= tol_head and (df.flag == "ok").all()
    print(f"{'PASS' if ok else 'FAIL'}  {name:38s} speed err {100*es:4.1f} %  heading err {eh:4.2f} deg"
          f"  fixes {len(df)}")
    return ok


def case_regular(tmp):
    speed, heading = 7.5, 40.0
    lat1, lon1 = np.arange(10, 45.01, 0.25), np.arange(120, 160.01, 0.25)
    LON, LAT = np.meshgrid(lon1, lat1)
    files = []
    for k, (cl, co) in enumerate(path_points(30.0, 137.5, speed, heading, 13, 1.0)):
        p = vortex(LAT, LON, cl, co) * 100 + np.random.default_rng(k).normal(0, 30, LAT.shape)
        t = pd.Timestamp("2026-09-20 18:00") + pd.Timedelta(hours=k)
        ds = xr.Dataset({"msl": (("time", "latitude", "longitude"), p[None])},
                        coords={"time": [t], "latitude": lat1, "longitude": lon1})
        f = tmp / f"reg_{k:02d}.nc"; ds.to_netcdf(f); files.append(str(f))
    df = st.track(st.frames(files), first_guess=(30.0, 137.5))
    return check(df, speed, heading, "regular 0-360 grid, Pa, noise",
                 truth=path_points(30.0, 137.5, speed, heading, 13, 1.0))


def lambert_grid(clat, clon, nx=120, ny=110, dx=9.0, rot=12.0):
    """2-D lat/lon of a rotated, curvilinear grid (WRF-like)"""
    i, j = np.meshgrid(np.arange(nx) - nx / 2, np.arange(ny) - ny / 2)
    a = np.radians(rot)
    x, y = (i * np.cos(a) - j * np.sin(a)) * dx, (i * np.sin(a) + j * np.cos(a)) * dx
    lat = clat + np.degrees(y / R)
    lon = clon + np.degrees(x / (R * np.cos(np.radians(lat))))
    return lat, lon


def case_wrf_like(tmp, moving=False):
    speed, heading = 6.0, 25.0
    pts = path_points(31.0, 137.6, speed, heading, 13, 1.0)
    lat, lon = lambert_grid(32.5, 138.5)
    files = []
    for k, (cl, co) in enumerate(pts):
        if moving:
            lat, lon = lambert_grid(cl + 0.3, co - 0.2)
        hgt = 1500 * np.exp(-st.haversine_km(35.5, 138.5, lat, lon) ** 2 / 80 ** 2)
        t2 = 300 - 0.0065 * hgt
        slp = vortex(lat, lon, cl, co)
        psfc = slp * 100 / np.exp(st.G * hgt / (st.RD * (t2 + 0.0065 * hgt / 2)))
        t = pd.Timestamp("2026-09-20 18:00") + pd.Timedelta(hours=k)
        ds = xr.Dataset({"PSFC": (("Time", "south_north", "west_east"), psfc[None]),
                         "HGT": (("Time", "south_north", "west_east"), hgt[None]),
                         "T2": (("Time", "south_north", "west_east"), t2[None]),
                         "XLAT": (("Time", "south_north", "west_east"), lat[None]),
                         "XLONG": (("Time", "south_north", "west_east"), lon[None]),
                         "Times": (("Time",), np.array([t.strftime("%Y-%m-%d_%H:%M:%S")], dtype="S19"))})
        f = tmp / f"wrfout_d02_{t:%Y-%m-%d_%H-%M-%S}"; ds.to_netcdf(f); files.append(str(f))
    df = st.track(st.frames(files), first_guess=pts[0])
    name = "moving nest, PSFC+HGT, rotated grid" if moving else "WRF-like, PSFC+HGT, rotated grid"
    return check(df, speed, heading, name, truth=pts)


def case_dateline(tmp):
    speed, heading = 9.0, 70.0
    pts = path_points(28.0, 178.6, speed, heading, 13, 3.0)
    lat1, lon1 = np.arange(15, 45.01, 0.25), np.arange(-180, 180, 0.25)
    LON, LAT = np.meshgrid(lon1, lat1)
    files = []
    for k, (cl, co) in enumerate(pts):
        p = vortex(LAT, LON, cl, co)
        p = p - 60 * np.exp(-(st.haversine_km(cl + 8.0, co, LAT, LON) / 250) ** 2)
        t = pd.Timestamp("2026-09-20 00:00") + pd.Timedelta(hours=3 * k)
        ds = xr.Dataset({"prmsl": (("time", "lat", "lon"), p[None])}, coords={"time": [t], "lat": lat1, "lon": lon1})
        f = tmp / f"dl_{k:02d}.nc"; ds.to_netcdf(f); files.append(str(f))
    df = st.track(st.frames(files), first_guess=pts[0])
    crossed = (df.lon.min() < -170) and (df.lon.max() > 170)
    ok = check(df, speed, heading, "dateline + deeper decoy 900 km") and crossed
    return ok


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        res = [case_regular(tmp), case_wrf_like(tmp), case_wrf_like(tmp, moving=True), case_dateline(tmp)]
    print(f"{sum(res)}/{len(res)} cases passed")
    raise SystemExit(0 if all(res) else 1)
