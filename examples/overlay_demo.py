"""overlay() demo: put the tracked storm on top of an existing chart with one call.

Author : SUMAILI NDEBA Bienvenu
Contact: sumailib@gmail.com; sumaili.bienvenu@um6p.ma
Code is free for use (MIT licence, see LICENSE).

(a) a WRF-style chart (10-m wind shading, sea-level pressure, barbs) on a Lambert map, the way
    wrf-python / WRF_Python_Scripts charts are built - then  st.overlay(ax, track, valid_time)
(b) the same overlay on a plain matplotlib lon/lat axes in 0-360 longitudes (no cartopy needed)

    python overlay_demo.py  [wrfout_file]  [track.csv]
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import stormtrack as st

plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.titleweight": "bold",
                     "axes.titlelocation": "left"})
if len(sys.argv) < 2:
    sys.exit("usage: python overlay_demo.py <wrfout_file> [track.csv]")
wrf_file = Path(sys.argv[1])
trk = st.read_track(sys.argv[2] if len(sys.argv) > 2 else HERE / "dujuan_wrf27km.csv")

fr = st.frames([str(wrf_file)])[0]
ds = xr.open_dataset(wrf_file)
ca, sa = ds["COSALPHA"].values[0], ds["SINALPHA"].values[0]
u, v = ds["U10"].values[0], ds["V10"].values[0]
ue, ve = u * ca - v * sa, v * ca + u * sa
spd = np.hypot(ue, ve)
lcc = ccrs.LambertConformal(central_longitude=float(ds.STAND_LON), central_latitude=float(ds.MOAD_CEN_LAT),
                            standard_parallels=(float(ds.TRUELAT1), float(ds.TRUELAT2)))
fig = plt.figure(figsize=(17, 7.6))
ax = fig.add_axes([0.03, 0.08, 0.52, 0.84], projection=lcc)
ax.set_extent([130, 150, 26, 40], ccrs.PlateCarree())
pm = ax.pcolormesh(fr.lon, fr.lat, spd, cmap="YlGnBu", vmin=0, vmax=35, transform=ccrs.PlateCarree(),
                   shading="auto")
cs = ax.contour(fr.lon, fr.lat, fr.p, levels=np.arange(960, 1024, 4), colors="k", linewidths=0.8,
                transform=ccrs.PlateCarree())
ax.clabel(cs, fmt="%d", fontsize=10)
s = (slice(None, None, 4), slice(None, None, 4))
ax.barbs(fr.lon[s], fr.lat[s], ue[s] * 1.94384, ve[s] * 1.94384, length=5, linewidth=0.6,
         transform=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.8)
gl = ax.gridlines(draw_labels=True, lw=0.3, color="gray", x_inline=False, y_inline=False)
gl.top_labels = gl.right_labels = False
gl.xlabel_style = {"rotation": 0}
fig.colorbar(pm, ax=ax, shrink=0.8, pad=0.02).set_label("10-m wind (m/s); barbs in knots")
ax.set_title(f"(a) WRF chart + overlay, {fr.time:%d %b %H:%M} UTC")

st.overlay(ax, trk, valid_time=fr.time)

b = fig.add_axes([0.64, 0.12, 0.33, 0.78])
b.set_xlim(134, 144); b.set_ylim(29, 36)
b.set_aspect(1 / np.cos(np.radians(32.5)))
b.contour(fr.lon % 360, fr.lat, fr.p, levels=np.arange(960, 1024, 4), colors="0.6", linewidths=0.8)
b.set_xlabel("Longitude (°E, 0–360)"); b.set_ylabel("Latitude (°N)")
b.grid(alpha=0.3)
b.set_title("(b) Plain axes, no cartopy")
st.overlay(b, trk, valid_time=fr.time, arrow_every=2, units="m/s", track_color="#1f4e79",
           arrow_color="#b2182b", key_xy=(0.60, 0.05))
fig.savefig(HERE / "overlay_demo.png", dpi=170)
print("saved", HERE / "overlay_demo.png")
