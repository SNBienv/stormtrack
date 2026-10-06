"""Validation example: stormtrack on three different inputs vs the (provisional) IBTrACS best track.

Author : SUMAILI NDEBA Bienvenu
Contact: sumailib@gmail.com; sumaili.bienvenu@um6p.ma
Code is free for use (MIT licence, see LICENSE).

Typhoon Dujuan, 20 Sep 18Z - 21 Sep 06Z 2026. Inputs: WRF 27 km (surface pressure only), WRF 9 km
(20-min output), raw GFS 0.25 deg GRIB. Run from this folder:  python compare_dujuan.py"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import stormtrack as st

plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "legend.fontsize": 12})
obs = pd.read_csv(HERE / "ibtracs_dujuan_provisional.csv", parse_dates=["time"])
runs = {"WRF 27 km": ("dujuan_wrf27km.csv", "#d6604d", "-"),
        "WRF 9 km": ("dujuan_wrf9km.csv", "#8c1d18", "-"),
        "GFS 0.25°": ("dujuan_gfs025.csv", "#2166ac", "--")}
t0, t1 = pd.Timestamp("2026-09-20 18:00"), pd.Timestamp("2026-09-21 06:00")
o = obs[obs.time.between(t0 - pd.Timedelta(hours=3), t1 + pd.Timedelta(hours=3))].set_index("time")

fig = plt.figure(figsize=(15, 6.2))
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    ax = fig.add_axes([0.03, 0.08, 0.42, 0.84], projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#e6e3dc")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.7)
    gl = ax.gridlines(draw_labels=True, lw=0.3, color="gray", xlocs=np.arange(130, 150, 1.0),
                      ylocs=np.arange(25, 40, 0.5)); gl.top_labels = gl.right_labels = False
    kw = {"transform": ccrs.PlateCarree()}
except ImportError:
    ax = fig.add_axes([0.06, 0.1, 0.38, 0.8]); kw = {}
ax.set_extent([136.6, 140.6, 30.6, 34.4]) if kw else None
ax.plot(o.lon, o.lat, "o-", color="k", lw=2, ms=7, label="best track (provisional)", **kw)
b = fig.add_axes([0.55, 0.56, 0.42, 0.36]); c = fig.add_axes([0.55, 0.10, 0.42, 0.36], sharex=b)
oi = o[["lat", "lon", "speed_ms"]]
summary = []
for name, (f, col, ls) in runs.items():
    d = pd.read_csv(HERE / f, parse_dates=["time"])
    d = d[d.flag != "lost"]
    ax.plot(d.lon, d.lat, ls, color=col, lw=2.2, label=name, **kw)
    ref = oi.reindex(oi.index.union(d.time)).interpolate(method="time").loc[d.time]
    err = st.haversine_km(ref.lat.values, ref.lon.values, d.lat.values, d.lon.values)
    b.plot(mdates.date2num(d.time), err, ls, color=col, lw=2.2)
    c.plot(mdates.date2num(d.time), d.speed_ms, ls, color=col, lw=2.2)
    summary.append((name, len(d), err.mean(), (d.speed_ms - ref.speed_ms.values).mean()))
c.plot(mdates.date2num(o.index), o.speed_ms, "o-", color="k", lw=2, ms=6)
lo0, la0, km = 139.25, 30.8, 100
ax.plot([lo0, lo0 + km / (111.32 * np.cos(np.radians(la0)))], [la0, la0], "k", lw=3, **kw)
ax.text(lo0 + km / (2 * 111.32 * np.cos(np.radians(la0))), la0 + 0.06, f"{km} km", ha="center", **kw)
ax.annotate("N", xy=(0.94, 0.30), xytext=(0.94, 0.21), xycoords="axes fraction", ha="center",
            fontweight="bold", arrowprops=dict(arrowstyle="-|>", lw=1.5))
ax.legend(loc="upper left", framealpha=0.95)
ax.set_title("(a) Tracks")
b.set_ylabel("Distance to best track (km)"); b.set_ylim(0, None); b.set_title("(b) Centre position error")
c.set_ylabel("Forward speed (m/s)"); c.set_ylim(0, 12); c.set_title("(c) Motion")
for a in (b, c):
    a.grid(alpha=0.3)
c.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 3, 6, 18, 21]))
c.xaxis.set_major_formatter(mdates.DateFormatter("%d/%HZ"))
plt.setp(b.get_xticklabels(), visible=False)
fig.savefig(HERE / "compare_dujuan.png", dpi=180, bbox_inches="tight")
try:
    import plotedit
    plotedit.save_code(fig, str(HERE / "compare_dujuan_code.py"))
except Exception:
    pass
for s in summary:
    print(f"{s[0]:10s} fixes {s[1]:2d}  mean position error {s[2]:5.1f} km  mean speed bias {s[3]:+.2f} m/s")
