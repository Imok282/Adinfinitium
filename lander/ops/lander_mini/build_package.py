"""Build dist/adinfinitium_lander_macmini_package.zip straight from the working tree (no staged copy on disk).

    ~/venvs/adi/bin/python ops/lander_mini/build_package.py

Top-level folder in the zip: adinfinitium_lander/ with the code (current working tree, tracked or not), the ten LOLA tiles, the
production lander pilot, data/pilot_dream/ and runs/lander/. A sha256 manifest of every packed file is added as
ops/lander_mini/package_manifest.sha256 (checked by setup.sh on the Mac mini), and a short CLAUDE.md that points at START_HERE.md.
"""

from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOP = "adinfinitium_lander"
OUT = ROOT / "dist" / "adinfinitium_lander_macmini_package.zip"

CODE_DIRS = ["contracts", "config", "worldmodel", "tests", "ops", "data_tools", "sim_jax", "scripts", "sites"]
CODE_FILES = ["pyproject.toml", "README.md", "AGENTS.md", "LICENSE", "requirements-torch.txt"]
TILES = ["dm2", "shoemaker", "site01", "site04", "site06", "site07", "site11", "site20", "site23", "site42"]
DATA_FILES = [f"data/tiles/lola5m_{t}.npz" for t in TILES] + ["data/models/lander_pilot.pt"]
DATA_DIRS = ["data/pilot_dream"]
RUNS_DIR = "runs/lander"
RUNS_SUFFIXES = {".md", ".json", ".jsonl"}
SKIP_DIRS = {"__pycache__", ".git", ".venv", ".pytest_cache", ".ruff_cache", ".pristine"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".zip", ".pt", ".ckpt", ".pth", ".onnx", ".tif", ".bsp", ".bpc", ".blend", ".glb"}
SKIP_NAMES = {".DS_Store", "package_manifest.sha256"}
MAX_CODE_BYTES = 2_000_000
EXECUTABLE = {".sh"}

PACKAGE_CLAUDE_MD = """# Adinfinitium lander package (Mac mini)

This folder is a hand-off package, not the full repository (no git history, no docs/ folder).
Read `ops/lander_mini/START_HERE.md` before doing anything: it has the hard rules, the runbook and the definition of done.
The hard rules there (Site23 is flown once at the very end; never loosen the acceptance gate; never change the simulator,
mission config, failure rates or seeds; unseen maps are for the benchmark only; report honestly) override everything else.
"""


def walk(d: str, code: bool):
    for p in sorted((ROOT / d).rglob("*")):
        if not p.is_file() or p.is_symlink() or SKIP_DIRS & set(p.relative_to(ROOT).parts) or p.name in SKIP_NAMES:
            continue
        if code and (p.suffix in SKIP_SUFFIXES or p.stat().st_size > MAX_CODE_BYTES):
            print(f"  skipped (binary or big): {p.relative_to(ROOT)}")
            continue
        yield p


def main() -> int:
    files: list[Path] = []
    for d in CODE_DIRS:
        files += list(walk(d, code=True))
    files += [ROOT / f for f in CODE_FILES]
    files += [ROOT / f for f in DATA_FILES]
    for d in DATA_DIRS:
        # the Site11 gate cache holds classic outcomes flown on THIS machine: not portable, the target recomputes it
        files += [p for p in walk(d, code=False) if not p.name.startswith(("gate_", "solutions"))]
    files += [p for p in walk(RUNS_DIR, code=False) if p.suffix in RUNS_SUFFIXES]
    missing = [str(p.relative_to(ROOT)) for p in files if not p.is_file()]
    if missing:
        print("MISSING, package not built:", missing)
        return 1
    OUT.parent.mkdir(exist_ok=True)
    tmp = OUT.with_suffix(".zip.part")
    manifest = []
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            rel = p.relative_to(ROOT).as_posix()
            data = p.read_bytes()                                  # one read: the bytes hashed are the bytes packed
            manifest.append(f"{hashlib.sha256(data).hexdigest()}  {rel}")
            zi = zipfile.ZipInfo.from_file(p, f"{TOP}/{rel}")
            zi.compress_type = zipfile.ZIP_STORED if p.suffix == ".npz" else zipfile.ZIP_DEFLATED
            zi.external_attr = (0o755 if p.suffix in EXECUTABLE else 0o644) << 16
            z.writestr(zi, data)
        for rel, text in (("CLAUDE.md", PACKAGE_CLAUDE_MD), ("ops/lander_mini/package_manifest.sha256", "\n".join(manifest) + "\n")):
            zi = zipfile.ZipInfo(f"{TOP}/{rel}", date_time=(2026, 11, 15, 12, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, text)
    tmp.replace(OUT)
    h = hashlib.sha256()
    with open(OUT, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    print(f"{OUT}\n  files: {len(files) + 2}\n  size: {OUT.stat().st_size / 1e6:.1f} MB ({OUT.stat().st_size} bytes)\n  sha256: {h.hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
