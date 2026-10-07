"""
Download the two public wind datasets used by real_wind_fit*.py into spde/data/.

  gstat_wind.rda     Irish daily mean wind, 12 stations, 1961-1978, knots
                     (Haslett & Raftery 1989; R package gstat, dataset `wind`)
  openair_mydata.rda London (Marylebone Road) hourly wind, 1998-2005, m/s
                     (R package openair, dataset `mydata`)

Both are fetched from the CRAN GitHub mirrors, which stay reachable when the primary
data portals (NOAA, Meteostat) are blocked.
"""
from pathlib import Path
from urllib.request import urlopen

SOURCES = {
    "gstat_wind.rda": "https://raw.githubusercontent.com/cran/gstat/master/data/wind.rda",
    "openair_mydata.rda": "https://raw.githubusercontent.com/cran/openair/master/data/mydata.rda",
}

out = Path(__file__).resolve().parent / "data"
out.mkdir(exist_ok=True)
for name, url in SOURCES.items():
    dest = out / name
    if dest.exists():
        print(f"{name}: already present ({dest.stat().st_size:,} bytes)")
        continue
    with urlopen(url, timeout=120) as r:
        dest.write_bytes(r.read())
    print(f"{name}: {dest.stat().st_size:,} bytes from {url}")
