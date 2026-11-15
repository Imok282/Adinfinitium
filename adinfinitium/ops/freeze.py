"""Freeze a finished run: record SHA-256 of every trained checkpoint, evaluation file and the code, so nothing can change silently afterwards.

    python -m ops.freeze            # writes runs/FROZEN.json
    python -m ops.freeze --check    # re-hash and compare: exit 1 if anything changed since freezing
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from contracts.paths import data_root, repo_root, runs_root

NAMES = ["hazard_unet_big", "light_skyline_big", "rover_driver_mini", "terrain_diffusion_mini"]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect() -> dict[str, str]:
    files: dict[str, Path] = {}
    for n in NAMES:
        files[f"checkpoint/{n}"] = data_root() / "models" / f"{n}.pt"
    for fam in ("hazard", "light", "rover", "terrain"):
        files[f"eval/{fam}"] = runs_root() / fam / "eval.json"
    out = {k: sha(p) for k, p in files.items() if p.exists()}
    code = hashlib.sha256()
    for sub in ("worldmodel", "contracts", "config", "ops"):
        for f in sorted((repo_root() / sub).rglob("*")):
            if f.is_file() and f.suffix in (".py", ".yaml") and "__pycache__" not in f.parts:
                code.update(str(f.relative_to(repo_root())).encode())
                code.update(f.read_bytes())
    out["code"] = code.hexdigest()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    path = runs_root() / "FROZEN.json"
    now = collect()
    if a.check:
        if not path.exists():
            print("no FROZEN.json: run `python -m ops.freeze` after the run is verified")
            return 1
        then = json.loads(path.read_text(encoding="utf-8"))["sha256"]
        bad = [k for k in then if now.get(k) != then[k]]
        print("FROZEN check:", "ALL UNCHANGED" if not bad else f"CHANGED: {bad}")
        return 1 if bad else 0
    path.write_text(json.dumps({"frozen_at": datetime.now(UTC).isoformat(), "sha256": now}, indent=2), encoding="utf-8")
    print(f"frozen {len(now)} items -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
