"""
Paths, site constants and run settings for the measured-noise SPDE study.

Every number here is either a measured site fact (with its source) or a run
setting the user is expected to change.  Nothing is a fitted value.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------
# GEP / Ben Guerir met station, 1-min logger, 2023-02-21 -> 2026-09-29.
# Columns used: TIMESTAMP (end of minute, ~UTC), WS_Avg (NRG #40C cup, m/s),
# WD_Avg (NRG #200P vane, deg FROM), WD_Std (deg, within-minute), WSgust_Max.
MAST_CSV = r"E:\Mo-GEP-29-09-2026_ALL_new.csv"
MAST_COLUMNS = ["TIMESTAMP", "WS_Avg", "WD_Avg", "WD_Std", "WSgust_Max"]

# Station position confirmed by the user on 2026-09-15 (WGS84), mast height 10 m.
# NOTE: the older project-wide point 32.2231, -7.9481 is 1.9 km west and wrong
# for this station; the 2026-08 WRF-vs-mast validations used it.
MAST_LAT, MAST_LON = 32.221478, -7.927969
MAST_HEIGHT_M = 10.0

# NRG #40C range starts at 1 m/s: below that the vane direction is not reliable
# and the cup does not resolve the speed, so segments are screened on it.
CUP_THRESHOLD_MS = 1.0

# ---------------------------------------------------------------------------
# Model output
# ---------------------------------------------------------------------------
# Native WRF 10-m diagnostic winds, Meskala-domain HPC product (covers BG),
# hourly, 111 x 111, variables lat, lon, time, u10, v10.
WRF_U10V10_DIR = r"F:\HPC WRF\u10v10"
# Dec 2024 and Dec 2025 are ERA5+QM placeholders: finite, plausible, and one
# constant per hour broadcast over the whole grid.  Excluded by name AND by a
# spatial-std guard (see noise_characterisation.load_wrf_u10v10).
WRF_PLACEHOLDER_MONTHS = ("2024_12", "2025_12")
WRF_NATIVE_DX_KM = 3.0          # nominal grid spacing of the product
WRF_EFFECTIVE_FACTOR = 7.0      # effective resolution ~ 7 dx (Skamarock 2004)


@dataclass
class CharacterisationSettings:
    """How the mast record and the mast-WRF residual are reduced."""

    segment_hours: float = 6.0          # length of each spectral segment
    max_gap_min: int = 2                # gaps up to this are linearly filled
    min_coverage: float = 0.95          # segment rejected below this
    min_mean_speed: float = 3.0         # m/s, keeps direction well defined
    welch_segment_min: int = 120        # Welch sub-window inside a segment
    wrf_match_window_min: int = 60      # mast mean centred on each WRF hour
    start: str | None = None            # optional date filter, e.g. "2024-10-01"
    end: str | None = None


@dataclass
class BoxSettings:
    """Mapping between the periodic 2-D box and physical units.

    The box represents a horizontal slab advected past the mast.  Its side is
    chosen so the longest segment period fits once (``L = U * T_segment``) and
    the grid resolves the 2-min Nyquist period of the 1-min logger.
    """

    n: int = 256
    box_km: float | None = None         # None -> U_mean * segment length
    nu_grid_reynolds: float = 2.0       # nu = u_rms * dx / Re_grid
    drag_over_tint: float = 1.0         # alpha = drag_over_tint / T_int
    tau_over_tint: float | None = None  # OU time; None -> use measured residual T
    n_probes: int = 8                   # virtual masts spread across y
    cfl: float = 0.4
    k_split_effective: bool = True      # deterministic below WRF effective k


@dataclass
class RunSettings:
    seeds: tuple = (0, 1, 2)
    t_burn_hours: float = 12.0          # physical spin-up discarded
    t_record_hours: float = 72.0        # physical record kept per run
    checkpoint_every: int = 20          # steps between partial writes
    cases: tuple = ("pde", "spde_generic", "spde_measured")


@dataclass
class Settings:
    char: CharacterisationSettings = field(default_factory=CharacterisationSettings)
    box: BoxSettings = field(default_factory=BoxSettings)
    run: RunSettings = field(default_factory=RunSettings)

    def to_dict(self):
        return asdict(self)


def result_path(*parts: str) -> str:
    path = os.path.join(RESULTS, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path
