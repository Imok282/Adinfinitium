"""Download the NASA/JPL SPICE kernels we need (public, NAIF) into <ADI_DATA>/kernels/. Pure standard library: works on Windows, Colab and the Mac.

    python -I -m worldmodel.light.fetch_kernels

Skips files that are already complete. Source: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/ (about 46 MB in total).
"""

from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

from worldmodel.light.ephemeris import KERNELS, kernel_dir

BASE = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels"
PATHS = {"naif0012.tls": "lsk", "de440s.bsp": "spk/planets", "pck00011.tpc": "pck", "moon_pa_de440_200625.bpc": "pck",
         "moon_de440_250416.tf": "fk/satellites"}
MIN_BYTES = {"naif0012.tls": 4000, "de440s.bsp": 30_000_000, "pck00011.tpc": 100_000, "moon_pa_de440_200625.bpc": 10_000_000, "moon_de440_250416.tf": 10_000}


def url_for(name: str) -> str:
    return f"{BASE}/{PATHS[name]}/{name}"


def fetch(name: str, directory: Path, retries: int = 4) -> Path:
    dest = directory / name
    if dest.exists() and dest.stat().st_size >= MIN_BYTES[name]:
        return dest
    directory.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for attempt in range(retries):
        try:
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(url_for(name), timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if tmp.stat().st_size < MIN_BYTES[name]:
                raise OSError(f"{name}: only {tmp.stat().st_size} bytes")
            tmp.replace(dest)
            return dest
        except Exception as e:  # noqa: BLE001 — network flakiness: retry
            last = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"could not download {name}: {last}")


def main() -> int:
    d = kernel_dir()
    for name in KERNELS:
        p = fetch(name, d)
        print(f"ok {name} ({p.stat().st_size / 1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
