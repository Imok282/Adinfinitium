"""Exact light geometry from a skyline. Pure numpy, no AI, no torch.

Conventions match contracts/light.py (v2): elevations are measured from the TERRAIN horizon, so
  rel_elev = absolute_elev - skyline(az);  Earth link <=> rel_elev > 0;  Sun up <=> sun_fraction > 0.
Azimuth is the grid azimuth (0 = up on the terrain map, clockwise), skyline index = azimuth degree.
"""

from __future__ import annotations

import numpy as np


def skyline_at(skyline_deg: np.ndarray, az_deg: np.ndarray) -> np.ndarray:
    """skyline [N,n_az] (index = azimuth bin), az_deg [T] -> [T,N], linear between bins (periodic)."""
    n_az = skyline_deg.shape[1]
    a = np.asarray(az_deg, dtype=np.float64) % 360.0 * (n_az / 360.0)
    i0 = np.floor(a).astype(int) % n_az
    f = a - np.floor(a)
    s = skyline_deg.astype(np.float64)
    return (1 - f)[:, None] * s[:, i0].T + f[:, None] * s[:, (i0 + 1) % n_az].T


def sun_visible_fraction(sun_abs_elev_deg: np.ndarray, skyline_at_sun_deg: np.ndarray, sun_radius_deg: float) -> np.ndarray:
    """Fraction of the finite solar disk above the skyline (straight horizon, no refraction). Shapes broadcast ([T,1] and [T,N])."""
    x = np.clip((sun_abs_elev_deg - skyline_at_sun_deg) / sun_radius_deg, -1.0, 1.0)
    return 1.0 - (np.arccos(x) - x * np.sqrt(1.0 - x * x)) / np.pi


def light_from_skyline(skyline_deg: np.ndarray, sun_az: np.ndarray, sun_abs_elev: np.ndarray, earth_az: np.ndarray,
                       earth_abs_elev: np.ndarray, sun_radius_deg: float) -> dict[str, np.ndarray]:
    """Sun fraction [T,N], Earth visibility [T,N] (bool) and the contract-style relative elevations [T,N]."""
    h_sun, h_earth = skyline_at(skyline_deg, sun_az), skyline_at(skyline_deg, earth_az)
    return {
        "sun_fraction": sun_visible_fraction(np.asarray(sun_abs_elev)[:, None], h_sun, sun_radius_deg),
        "earth_visible": (np.asarray(earth_abs_elev)[:, None] - h_earth) > 0.0,
        "sun_elev_rel_deg": np.asarray(sun_abs_elev)[:, None] - h_sun,
        "earth_elev_rel_deg": np.asarray(earth_abs_elev)[:, None] - h_earth,
    }
