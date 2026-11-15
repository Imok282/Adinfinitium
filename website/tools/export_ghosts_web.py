"""Convert ghost-trajectory JSON (python -m worldmodel.ghosts) into a compact binary for the website.

    python website/tools/export_ghosts_web.py <ghosts.json> [<ghosts.json> ...] [--site site06] [--max-pts 40]

Reads runs/ghosts/{lander,rover}_<region>[...].json (MODELLED: real simulated runs, successes AND failures; nothing is
invented or tuned here) and writes, per input file:

    website/assets/data/ghosts_<kind>_<site>.bin    little-endian uint16, 4 per point: x, y, z, t
    website/assets/data/ghosts_<kind>_<site>.json   header (counts, frame, units, provenance, source, run table)

Decoding (see header["encoding"]):  x = off[0] + q * scale[0]  (same for y, z, t);  frame = tile ENU metres,
origin = tile centre, x east, y north, z = LOLA height (m); t = simulated seconds since the start of that run.
Run k occupies points runs[k][3] .. runs[k][3] + runs[k][4] - 1 (columns: see header["run_columns"]).

Processing (only geometry reduction, no change of outcome):
  * lander tracks are cut where they first climb more than `clip_above_target_m` above the target (open-loop
    runs fly up and away), exactly as the old 3D view did; the outcome label is the simulator's.
  * every track is resampled uniformly in time to <= max_pts points (first and last point kept).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "website" / "assets" / "data"
OK_OUTCOMES = {"lander": {"ok"}, "rover": {"returned"}}
CLIP_ABOVE_TARGET_M = 3000.0


def _truthy(v) -> bool:
    return v is True or v == "True" or v == "true" or v == 1


def _resample(xyz: np.ndarray, t: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(xyz)
    if n <= k:
        return xyz, t
    idx = np.unique(np.round(np.linspace(0, n - 1, k)).astype(int))
    return xyz[idx], t[idx]


def export(src: Path, site: str | None, max_pts: int | None, out_dir: Path = OUT) -> dict:
    d = json.loads(src.read_text())
    kind = d["kind"]
    region = d["region"]
    site = site or (re.search(r"site\d+", region).group(0) if re.search(r"site\d+", region) else region)
    max_pts = max_pts or (40 if kind == "lander" else 60)
    tgt = d.get("target_xyz_m")
    policies = list(OrderedDict.fromkeys(r["policy"] for r in d["runs"]))
    outcomes = list(OrderedDict.fromkeys(r["outcome"] for r in d["runs"]))
    ok_set = OK_OUTCOMES.get(kind, {"ok"})

    tracks, rows = [], []
    best_index, best_id = None, None
    for r in d["runs"]:
        xyz = np.asarray(r["xyz_m"], dtype=np.float64)
        t = np.asarray(r.get("t_s") or np.arange(len(xyz)) * float(d.get("dt_s", 1.0)), dtype=np.float64)[: len(xyz)]
        clipped = 0
        if kind == "lander" and tgt is not None:
            above = xyz[:, 2] > tgt[2] + CLIP_ABOVE_TARGET_M
            if above.any():
                start = int(np.argmax(~above)) if above[0] else 0
                rest = np.nonzero(above[start:])[0]
                stop = start + int(rest[0]) if len(rest) else len(xyz)
                clipped = 1 if stop < len(xyz) else 0
                xyz, t = xyz[start:stop], t[start:stop]
        if len(xyz) < 2:
            continue
        xyz, t = _resample(xyz, t, max_pts)
        flags = (1 if _truthy(r.get("best")) else 0) | (2 if r["outcome"] in ok_set else 0) | (4 * clipped)
        if kind == "lander":
            flags |= (8 if _truthy(r.get("engine_failed")) else 0) | (16 if _truthy(r.get("nav_drift")) else 0) \
                | (32 if _truthy(r.get("camera_blind")) else 0)
        else:
            flags |= 8 if _truthy(r.get("reached_ice")) else 0
        miss = r.get("target_dist_m")
        miss = None if miss in (None, "None") else round(float(miss), 2)
        if flags & 1:
            best_index, best_id = len(rows), r["id"]
        rows.append([policies.index(r["policy"]), outcomes.index(r["outcome"]), flags, 0, len(xyz), miss])
        tracks.append(np.concatenate([xyz, t[:, None]], 1))

    allp = np.concatenate(tracks, 0)
    lo, hi = allp.min(0), allp.max(0)
    scale = np.where(hi > lo, (hi - lo) / 65535.0, 1.0)
    start = 0
    for row, tr in zip(rows, tracks):
        row[3] = start
        start += len(tr)
    q = np.round((allp - lo) / scale).clip(0, 65535).astype("<u2")

    counts = {p: dict(Counter(d["runs"][i]["outcome"] for i in range(len(d["runs"])) if d["runs"][i]["policy"] == p))
              for p in policies}
    tag = f"ghosts_{kind}_{site}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}.bin").write_bytes(q.tobytes())
    try:
        source = str(src.resolve().relative_to(REPO))
    except ValueError:   # generated with ADI_RUNS outside the repo: worldmodel.ghosts writes $ADI_RUNS/ghosts/<name>
        source = f"runs/ghosts_web/{src.name}"
    import datetime as _dt
    generated = _dt.datetime.fromtimestamp(src.stat().st_mtime).isoformat(timespec="minutes")
    head = {
        "meta": {
            "provenance": "modelled",
            "provenance_note": d.get("provenance", ""),
            "source": source,
            "source_generated": generated,
            "generator": f"python -m worldmodel.ghosts --kind {kind} --region {region}"
                         + (f" --date {d['date']}" if d.get("date") else ""),
            "created_by": "website/tools/export_ghosts_web.py (downsampled + quantised; outcomes unchanged)",
            "frame": d.get("frame"),
            "units": "m (x, y, z), s (t)",
        },
        "kind": kind, "region": region, "site": site, "date": d.get("date"), "dt_s": d.get("dt_s"),
        "n_runs": len(d["runs"]), "n_drawn": len(rows), "n_points": int(len(allp)),
        "policies": policies, "outcomes": outcomes, "ok_outcomes": sorted(ok_set & set(outcomes)),
        "counts": counts,
        "counts_total": dict(Counter(r["outcome"] for r in d["runs"])),
        "failure_rates": d.get("failure_rates"),
        "target_xyz_m": tgt, "lander_xyz_m": d.get("lander_xyz_m"),
        "ice_targets_xyz_m": d.get("ice_targets_xyz_m"), "ice_source": d.get("ice_source"),
        "clip_above_target_m": CLIP_ABOVE_TARGET_M if kind == "lander" else None,
        "max_pts_per_track": max_pts,
        "best_index": best_index, "best_id": best_id,
        "encoding": {"file": f"{tag}.bin", "dtype": "uint16 little-endian", "stride": 4, "fields": ["x", "y", "z", "t"],
                     "offset": [float(v) for v in lo], "scale": [float(v) for v in scale],
                     "decode": "value = offset[i] + q * scale[i]"},
        "run_columns": ["policy_idx", "outcome_idx", "flags", "first_point", "n_points", "target_dist_m"],
        "flags": {"1": "best (the one run marked best by worldmodel.ghosts)", "2": "success outcome", "4": "cut above clip",
                  "8": "engine_failed" if kind == "lander" else "reached_ice", "16": "nav_drift", "32": "camera_blind"},
        "runs": rows,
    }
    (out_dir / f"{tag}.json").write_text(json.dumps(head, separators=(",", ":")))
    print(f"{tag}: {len(rows)}/{len(d['runs'])} runs, {len(allp)} pts, {q.nbytes / 1e3:.0f} kB bin, counts {counts}, best {best_id}")
    return head


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="+", type=Path)
    ap.add_argument("--site", default=None)
    ap.add_argument("--max-pts", type=int, default=None)
    a = ap.parse_args()
    for s in a.src:
        export(s, a.site, a.max_pts)


if __name__ == "__main__":
    main()
