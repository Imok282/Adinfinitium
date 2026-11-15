"""REAL Sun and Earth directions from NASA/JPL SPICE (DE440), replacing the analytic stand-in. Needs spiceypy and the kernels in <ADI_DATA>/kernels/.

Kernels (public, NASA NAIF, https://naif.jpl.nasa.gov/pub/naif/generic_kernels/):
  lsk/naif0012.tls · spk/planets/de440s.bsp · pck/pck00011.tpc · pck/moon_pa_de440_200625.bpc · fk/satellites/moon_de440_250416.tf

Frame: MOON_ME (mean-Earth / polar-axis), the frame LOLA coordinates use. The site is a point on the lunar reference sphere (1737.4 km).
Directions are TOPOCENTRIC (parallax included: it matters for Earth, up to ~0.26 deg) and apparent (light time + stellar aberration), no refraction.
Elevation = above the local horizontal plane; azimuth 0 = north, clockwise, east longitude positive. Grid azimuth (the tile frame of the light
contract) = true azimuth + east longitude, a convention that still needs verification against the LOLA label (KAN-21).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from contracts.paths import data_root

KERNELS = ("naif0012.tls", "de440s.bsp", "pck00011.tpc", "moon_pa_de440_200625.bpc", "moon_de440_250416.tf")
MOON_RADIUS_KM = 1737.4


def kernel_dir() -> Path:
    return data_root() / "kernels"


def load_kernels(directory: Path | None = None) -> None:
    import spiceypy as sp

    d = directory or kernel_dir()
    missing = [k for k in KERNELS if not (d / k).exists()]
    if missing:
        raise FileNotFoundError(f"missing SPICE kernels in {d}: {missing} (download from naif.jpl.nasa.gov, see this module's docstring)")
    sp.kclear()
    for k in KERNELS:
        sp.furnsh(str(d / k))


def _unit(lat_deg: float, lon_deg: float) -> np.ndarray:
    la, lo = np.radians(lat_deg), np.radians(lon_deg)
    return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def local_basis(lat_deg: float, lon_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(east, north, up) unit vectors in MOON_ME at a site."""
    la, lo = np.radians(lat_deg), np.radians(lon_deg)
    up = _unit(lat_deg, lon_deg)
    east = np.array([-np.sin(lo), np.cos(lo), 0.0])
    north = np.array([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)])
    return east, north, up


def to_az_el(v: np.ndarray, lat_deg: float, lon_deg: float) -> tuple[float, float]:
    east, north, up = local_basis(lat_deg, lon_deg)
    v = v / np.linalg.norm(v)
    return float(np.degrees(np.arctan2(v @ east, v @ north)) % 360.0), float(np.degrees(np.arcsin(np.clip(v @ up, -1, 1))))


def _cache_path(site_lat_deg: float, site_lon_east_deg: float, times_utc: list[str]) -> Path:
    import hashlib

    key = f"{site_lat_deg:.6f}_{site_lon_east_deg:.6f}_{times_utc[0]}_{times_utc[-1]}_{len(times_utc)}"
    return data_root() / "ephemeris_cache" / (hashlib.sha1(key.encode()).hexdigest()[:16] + ".npz")


def sun_earth_directions(site_lat_deg: float, site_lon_east_deg: float, times_utc: list[str]) -> dict[str, np.ndarray]:
    """True azimuth / absolute elevation of the Sun and Earth centres at a lunar site, for ISO-8601 UTC times.
    Returns arrays [T]: sun_az, sun_el, earth_az, earth_el (degrees) and et_s (TDB seconds past J2000).
    Results are cached in <ADI_DATA>/ephemeris_cache/ (same SPICE numbers; lets machines where the CSPICE library cannot load reuse them)."""
    cp = _cache_path(site_lat_deg, site_lon_east_deg, times_utc)
    if cp.exists():
        z = np.load(cp)
        return {k: z[k] for k in ("sun_az", "sun_el", "earth_az", "earth_el", "et_s")}
    out = _spice_directions(site_lat_deg, site_lon_east_deg, times_utc)
    try:
        cp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cp, **out)
    except OSError:
        pass
    return out


def _spice_directions(site_lat_deg: float, site_lon_east_deg: float, times_utc: list[str]) -> dict[str, np.ndarray]:
    import spiceypy as sp

    load_kernels()
    site = _unit(site_lat_deg, site_lon_east_deg) * MOON_RADIUS_KM
    et = np.array([sp.str2et(t.replace("Z", "")) for t in times_utc])
    out = {k: np.empty(len(et)) for k in ("sun_az", "sun_el", "earth_az", "earth_el")}
    for i, e in enumerate(et):
        for body, key in (("SUN", "sun"), ("EARTH", "earth")):
            pos, _ = sp.spkpos(body, e, "MOON_ME", "LT+S", "MOON")      # km, Moon-centre -> body, in MOON_ME
            az, el = to_az_el(np.asarray(pos) - site, site_lat_deg, site_lon_east_deg)
            out[f"{key}_az"][i], out[f"{key}_el"][i] = az, el
    out["et_s"] = et
    return out


def grid_azimuth(true_az_deg: np.ndarray, site_lon_east_deg: float) -> np.ndarray:
    """Tile (grid) azimuth = true azimuth + east longitude (assumes x = r sin(lon), y = r cos(lon); unverified)."""
    return (np.asarray(true_az_deg) + site_lon_east_deg) % 360.0
