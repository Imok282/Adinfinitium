"""Build the hand-off package for another Claude session on another machine (no GitHub login needed).

    python -m ops.make_bundle --target macmini        # -> dist/adinfinitium_macmini_package.zip

Contains: the code (tracked + untracked, not ignored), the briefing (CLAUDE.md / START_HERE.md written for the trainer), and a seed data zip
(tiles, SPICE kernels, light files, cached skylines, existing small checkpoints) so the machine does not have to re-download what we already have.
"""

from __future__ import annotations

import argparse
import subprocess
import zipfile
from pathlib import Path

from contracts.paths import data_root, repo_root

SEED_DIRS = ["raw", "tiles", "kernels", "light", "light_train", "rover_cache", "models"]
SEED_BY_TARGET = {"macmini": SEED_DIRS, "4090": ["tiles", "kernels", "rover_cache", "models/hazard_unet.pt"]}   # 4090 brain run needs only these
DROP_FROM_CODE = ("docs/inbox/", "docs/outbox/", "CLAUDE.md", "vizard/assets/blender/", "vizard/assets/ui_models/")


def code_files(root: Path) -> list[str]:
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=root, capture_output=True, text=True, check=True).stdout.split("\n")
    return [f for f in out if f and (root / f).is_file() and not f.startswith(DROP_FROM_CODE)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default="macmini", choices=["macmini", "4090"])
    ap.add_argument("--no-data", action="store_true")
    a = ap.parse_args(argv)
    root = repo_root()
    dist = root / "dist"
    dist.mkdir(exist_ok=True)
    seed = dist / "adinfinitium_seed_data.zip"
    n_seed = 0
    if not a.no_data:
        with zipfile.ZipFile(seed, "w", zipfile.ZIP_STORED) as z:
            for d in SEED_BY_TARGET[a.target]:
                src = data_root() / d
                for f in ([src] if src.is_file() else sorted(src.rglob("*")) if src.exists() else []):
                    if f.is_file():
                        z.write(f, Path("data") / f.relative_to(data_root()))
                        n_seed += 1
    out = dist / f"adinfinitium_{a.target}_package.zip"
    brief = (root / "ops" / ("windows" if a.target == "4090" else a.target) / "START_HERE.md").read_text(encoding="utf-8")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in code_files(root):
            z.write(root / f, f"adinfinitium/{f}")
        z.writestr("adinfinitium/CLAUDE.md", brief)          # the trainer session reads this first
        z.writestr("adinfinitium/START_HERE.md", brief)
        if not a.no_data:
            z.write(seed, "adinfinitium/seed/adinfinitium_seed_data.zip")
    print(f"{out}  ({out.stat().st_size / 1e6:.0f} MB; {n_seed} seed files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
