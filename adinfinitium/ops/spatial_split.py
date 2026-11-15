"""Spatial-leakage guard: drop training / validation windows that sit on or next to a TEST tile.

Splits are by site NAME, but a grid window of a training site can lie beside (or on top of) a tile of a test site.
Two square tiles are "too close" when BOTH |dx| and |dy| between their centres are < tile size + 1280 m, i.e. the
tiles grown by the light model's ray-cast range overlap. The TRAINING (or validation) window is excluded, never the test tile.

    python -m ops.spatial_split            # pipeline stage: after tiles are built, before any training. Writes data/tiles/excluded_windows.json
    python -m ops.spatial_split --verify   # recompute from the tiles on disk; exit 1 if an overlapping window is not in the file

worldmodel.hazard.lola.regions_for_site reads the file, so every trainer honours it. ADI_NO_SPATIAL_EXCLUDE=1 turns the filter off (debugging).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from contracts.paths import data_root

MARGIN_M = 1280.0
RULE = ("a train/val tile is excluded when BOTH |dx| and |dy| to a test tile centre are < "
        "(mean of the two tile sizes) + margin_m; the test tile is never excluded")
REGION_RE = re.compile(r"^lola5m_([a-z0-9]+?)(_r\d+c\d+)?$")

_cache: dict[str, tuple[tuple[int, int], frozenset[str]]] = {}


def excluded_path() -> Path:
    return data_root() / "tiles" / "excluded_windows.json"


def site_roles() -> dict[str, str]:
    """Lower-case site name -> "train" | "val" | "test"."""
    from worldmodel.hazard.lola import HOLDOUT_SITES, TRAIN_SITES, VAL_SITES

    return {s.lower(): role for role, sites in (("train", TRAIN_SITES), ("val", VAL_SITES), ("test", HOLDOUT_SITES)) for s in sites}


def role_of(region: str, roles: dict[str, str] | None = None) -> str | None:
    """Role of a region from its site name; None for generated tiles and sites outside the split."""
    m = REGION_RE.match(region)
    return (roles or site_roles()).get(m.group(1)) if m else None


def is_centre_window(region: str) -> bool:
    m = REGION_RE.match(region)
    return bool(m) and m.group(2) is None


def latlon_to_xy(lat_deg: float, lon_deg: float, radius_m: float) -> tuple[float, float]:
    """Inverse of worldmodel.hazard.lola.xy_to_latlon (south-polar stereographic, x = r sin(lon), y = r cos(lon))."""
    r = 2.0 * radius_m * math.tan(math.radians(lat_deg + 90.0) / 2.0)
    return r * math.sin(math.radians(lon_deg)), r * math.cos(math.radians(lon_deg))


def _read_centre(path: Path, radius_m: float) -> tuple[float, float, float]:
    """(x_m, y_m, size_m) from the npz header and the height array's npy header; the arrays are not decompressed."""
    with np.load(path, allow_pickle=False) as z:
        header = json.loads(str(z["header"]))
    with zipfile.ZipFile(path) as zf, zf.open("height.npy") as f:
        version = np.lib.format.read_magic(f)
        shape = (np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0)(f)[0]
    xy = (header.get("extra") or {}).get("centre_xy_m")
    if xy is not None:
        x, y = float(xy[0]), float(xy[1])
    else:
        x, y = latlon_to_xy(float(header["center_lat_deg"]), float(header["center_lon_deg"]), radius_m)
    return x, y, max(shape) * float(header["meters_per_pixel"])


def tile_centres() -> dict[str, tuple[float, float, float]]:
    """{region: (x_m, y_m, size_m)} for every non-generated tile of a split site on disk. Unreadable tiles raise."""
    from contracts.mission import load_mission

    d = data_root() / "tiles"
    if not d.exists():
        return {}
    radius = load_mission().moon.radius_m
    roles = site_roles()
    return {p.stem: _read_centre(p, radius) for p in sorted(d.glob("lola5m_*.npz")) if role_of(p.stem, roles)}


def _too_close(a: tuple[float, float, float], b: tuple[float, float, float], margin_m: float) -> tuple[bool, float, float, float]:
    dx, dy = a[0] - b[0], a[1] - b[1]
    limit = (a[2] + b[2]) / 2.0 + margin_m
    return abs(dx) < limit and abs(dy) < limit, math.hypot(dx, dy), dx, dy


def compute(margin_m: float = MARGIN_M) -> dict:
    """The exclusion record, recomputed from the tiles on disk (nothing is written)."""
    centres = tile_centres()
    roles = site_roles()
    by_role: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for region in centres:
        by_role[role_of(region, roles)].append(region)
    excluded: dict[str, dict] = {}
    nearest: dict[str, dict] = {}
    for role in ("train", "val"):
        for region in by_role[role]:
            pairs = [(t, *_too_close(centres[region], centres[t], margin_m)) for t in by_role["test"]]
            if not pairs:
                continue
            best = min(pairs, key=lambda p: p[2])
            if not nearest.get(role) or best[2] < nearest[role]["dist_m"]:
                nearest[role] = {"region": region, "near": best[0], "dist_m": round(best[2], 1)}
            hits = [p for p in pairs if p[1]]
            if hits:
                t, _, dist, dx, dy = min(hits, key=lambda p: p[2])
                excluded[region] = {"role": role, "near": t, "dist_m": round(dist, 1), "dx_m": round(dx, 1), "dy_m": round(dy, 1),
                                    "centre_window": is_centre_window(region)}
    return {
        "rule": RULE, "margin_m": margin_m, "excluded": excluded,
        "checked": {r: len(v) for r, v in by_role.items()},
        "min_train_test_dist_m": nearest.get("train", {}).get("dist_m"),
        "min_val_test_dist_m": nearest.get("val", {}).get("dist_m"),
        "nearest": nearest,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def check(margin_m: float = MARGIN_M, write: bool = True) -> dict:
    """Compute the exclusions, print one line per exclusion plus a summary, and write excluded_windows.json."""
    rec = compute(margin_m)
    n = rec["checked"]
    for region, e in rec["excluded"].items():
        loud = "  *** THIS IS THE SITE'S CENTRE WINDOW ***" if e["centre_window"] else ""
        kind = "TRAINING leak" if e["role"] == "train" else "validation (tuning) leak"
        print(f"EXCLUDE {region} [{e['role']}] {kind}: {e['dist_m']:.0f} m from test tile {e['near']} "
              f"(dx {e['dx_m']:.0f} m, dy {e['dy_m']:.0f} m){loud}", flush=True)
    if not n["test"]:
        print("WARNING: no TEST tile on disk, so nothing could be checked (build Site23 / Site42 tiles first)", flush=True)
    for role in ("train", "val"):
        near = rec["nearest"].get(role)
        if near:
            print(f"closest {role} tile to a test tile: {near['region']} -> {near['near']} at {near['dist_m']:.0f} m", flush=True)
    by = {r: sum(e["role"] == r for e in rec["excluded"].values()) for r in ("train", "val")}
    print(f"spatial split: checked {n['train']} train + {n['val']} val tiles against {n['test']} test tiles, margin {margin_m:.0f} m: "
          f"excluded {by['train']} train, {by['val']} val", flush=True)
    if write:
        p = excluded_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec, indent=2) + "\n")
        tmp.replace(p)
        print(f"wrote {p}", flush=True)
    return rec


def excluded_regions() -> frozenset[str]:
    """Region names in excluded_windows.json (empty when the file is absent). Re-read only when the file changes."""
    p = excluded_path()
    try:
        st = p.stat()
    except OSError:
        return frozenset()
    key = (st.st_mtime_ns, st.st_size)
    hit = _cache.get(str(p))
    if hit and hit[0] == key:
        return hit[1]
    names = frozenset(json.loads(p.read_text()).get("excluded", {}))
    _cache[str(p)] = (key, names)
    return names


def active_exclusions() -> frozenset[str]:
    """What regions_for_site must drop: the file's regions, or nothing when ADI_NO_SPATIAL_EXCLUDE=1."""
    return frozenset() if os.environ.get("ADI_NO_SPATIAL_EXCLUDE") == "1" else excluded_regions()


def trained_overlaps(trained_on: list[str], margin_m: float = MARGIN_M) -> list[str]:
    """Regions of a checkpoint's trained_on list that are too close to a test tile on disk (recomputed, not read from the file).
    Test regions themselves and regions whose tile is missing are not judged here."""
    centres = tile_centres()
    roles = site_roles()
    tests = [c for r, c in centres.items() if role_of(r, roles) == "test"]
    return sorted({r for r in trained_on
                   if r in centres and role_of(r, roles) != "test" and any(_too_close(centres[r], t, margin_m)[0] for t in tests)})


def verify(margin_m: float = MARGIN_M) -> list[str]:
    """Overlapping train/val regions on disk that the excluded file does not list (empty = good)."""
    return sorted(set(compute(margin_m)["excluded"]) - excluded_regions())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verify", action="store_true", help="recompute and exit 1 if an overlapping window is missing from the excluded file")
    ap.add_argument("--margin-m", type=float, default=MARGIN_M)
    a = ap.parse_args(argv)
    if a.verify:
        missing = verify(a.margin_m)
        for r in missing:
            print(f"FAIL {r} overlaps a test tile but is not in {excluded_path().name}", flush=True)
        print(f"spatial split verify: {len(missing)} overlapping window(s) not excluded", flush=True)
        return 1 if missing else 0
    check(a.margin_m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
