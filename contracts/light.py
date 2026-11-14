"""Light-field contract: data/light/<region>_<date>.npz

Same grid as the matching tile. T time samples.
  times_utc     [T]        ISO-8601 strings (UTC)
  et_s          [T]        float64 SPICE ephemeris time (TDB seconds past J2000)
  sun_fraction  [T, H, W]  0..1 visible fraction of the solar disk (finite disk)
  sun_elev_deg  [T, H, W]  Sun elevation above the LOCAL horizon (terrain + curvature, no refraction)
  earth_elev_deg[T, H, W]  Earth-centre elevation above the local horizon
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from contracts.provenance import Meta

UNITS = {"sun_fraction": "fraction 0..1", "sun_elev_deg": "deg", "earth_elev_deg": "deg",
         "et_s": "s TDB past J2000", "times_utc": "ISO-8601 UTC"}


@dataclass
class LightField:
    region: str
    date: str
    times_utc: list[str]
    et_s: np.ndarray
    sun_fraction: np.ndarray
    sun_elev_deg: np.ndarray
    earth_elev_deg: np.ndarray
    meta: Meta

    def validate(self) -> None:
        t = len(self.times_utc)
        if self.et_s.shape != (t,):
            raise ValueError("et_s must have shape [T]")
        if self.sun_fraction.ndim != 3 or self.sun_fraction.shape[0] != t:
            raise ValueError("sun_fraction must be [T, H, W]")
        for name in ("sun_elev_deg", "earth_elev_deg"):
            if getattr(self, name).shape != self.sun_fraction.shape:
                raise ValueError(f"{name} shape mismatch")
        if np.any((self.sun_fraction < 0) | (self.sun_fraction > 1)):
            raise ValueError("sun_fraction must be in [0, 1]")
        if np.any(np.diff(self.et_s) <= 0):
            raise ValueError("et_s must be strictly increasing")

    def earth_visible(self) -> np.ndarray:
        return self.earth_elev_deg > 0.0


def save_light(lf: LightField, path: str | Path) -> Path:
    lf.validate()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    meta = lf.meta.model_copy(update={"units": {**UNITS, **lf.meta.units}})
    header = {"region": lf.region, "date": lf.date, "times_utc": lf.times_utc,
              "meta": json.loads(meta.model_dump_json())}
    np.savez_compressed(
        p, header=np.array(json.dumps(header)), et_s=lf.et_s.astype(np.float64),
        sun_fraction=lf.sun_fraction.astype(np.float32),
        sun_elev_deg=lf.sun_elev_deg.astype(np.float32),
        earth_elev_deg=lf.earth_elev_deg.astype(np.float32),
    )
    return p


def load_light(path: str | Path) -> LightField:
    with np.load(Path(path), allow_pickle=False) as z:
        h = json.loads(str(z["header"]))
        lf = LightField(region=h["region"], date=h["date"], times_utc=list(h["times_utc"]),
                        et_s=z["et_s"], sun_fraction=z["sun_fraction"],
                        sun_elev_deg=z["sun_elev_deg"], earth_elev_deg=z["earth_elev_deg"],
                        meta=Meta.model_validate(h["meta"]))
    lf.validate()
    return lf
