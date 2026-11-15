"""Print and save what this machine has (python, torch, CUDA/MPS, GPU, VRAM, disk, RAM, library versions). Exits 1 with --need-cuda if there is no CUDA GPU."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import sys

from contracts.paths import repo_root, runs_root


def gather() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(), "libs": {}}
    for lib in ("numpy", "scipy", "pydantic", "yaml", "rasterio", "spiceypy", "matplotlib", "torch"):
        try:
            m = importlib.import_module(lib)
            info["libs"][lib] = getattr(m, "__version__", "?")
        except Exception as e:  # noqa: BLE001
            info["libs"][lib] = f"MISSING ({type(e).__name__})"
    try:
        import torch

        info["cuda"] = bool(torch.cuda.is_available())
        info["mps"] = bool(torch.backends.mps.is_available())
        if info["cuda"]:
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = {"name": props.name, "vram_gb": round(props.total_memory / 1e9, 1), "capability": f"{props.major}.{props.minor}"}
        info["torch_cuda_build"] = torch.version.cuda
    except Exception as e:  # noqa: BLE001
        info["torch_error"] = str(e)
    root = repo_root()
    info["disk_free_gb"] = round(shutil.disk_usage(root).free / 1e9, 1)
    try:
        import os

        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            info["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except Exception:  # noqa: BLE001, S110 (RAM is optional info)
        pass
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--need-cuda", action="store_true")
    a = ap.parse_args(argv)
    info = gather()
    print(json.dumps(info, indent=2))
    out = runs_root() / "overnight" / "env_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2), encoding="utf-8")
    missing = [k for k, v in info["libs"].items() if str(v).startswith("MISSING")]
    if missing:
        print(f"MISSING libraries: {missing}", file=sys.stderr)
        return 1
    if a.need_cuda and not info.get("cuda"):
        print("NO CUDA GPU visible to PyTorch (install the CUDA build of torch and the NVIDIA driver)", file=sys.stderr)
        return 1
    if info["disk_free_gb"] < 5:
        print(f"WARNING: only {info['disk_free_gb']} GB free disk", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
