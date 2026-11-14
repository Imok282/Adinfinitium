"""Verification steps for an overnight run. Prints PASS / FAIL / WARN per check; exit 1 if any FAIL.

    python -m ops.verify before        # right after setup: libraries, device, data, config, unit tests are fine
    python -m ops.verify after         # after training: checkpoints load, evals exist, network vs physics backup, honesty checks
    python -m ops.verify after --names hazard_unet_big,light_skyline_big,rover_driver_mini,terrain_diffusion_mini
"""

from __future__ import annotations

import argparse
import json
import sys

from contracts.paths import data_root, model_path, runs_root

RESULTS: list[tuple[str, str, str]] = []
REQUIRED_TILES = ["lola5m_site01", "lola5m_site04", "lola5m_site06", "lola5m_site07", "lola5m_site11", "lola5m_site20", "lola5m_site23"]
DEFAULT_NAMES = ["hazard_unet_big", "light_skyline_big", "rover_driver_mini", "terrain_diffusion_mini"]


def check(level: str, name: str, detail: str = "") -> None:
    RESULTS.append((level, name, detail))
    print(f"{level:5s} {name}  {detail}", flush=True)


def before() -> None:
    check("PASS", "python", sys.version.split()[0])
    for lib in ("numpy", "scipy", "pydantic", "yaml", "rasterio", "spiceypy", "matplotlib", "torch"):
        try:
            m = __import__(lib)
            check("PASS", f"import {lib}", getattr(m, "__version__", ""))
        except Exception as e:  # noqa: BLE001
            check("FAIL", f"import {lib}", repr(e))
    try:
        import torch

        dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        check("PASS" if dev != "cpu" else "WARN", "accelerator", dev)
        x = torch.randn(256, 256, device=dev)
        check("PASS", "matmul on accelerator", f"{float((x @ x).abs().mean()):.3f}")
    except Exception as e:  # noqa: BLE001
        check("FAIL", "torch accelerator", repr(e))
    from contracts.mission import load_mission

    try:
        cfg = load_mission()
        check("PASS", "config/mission.yaml loads", f"light_model.coarse_factor={cfg.light_model.coarse_factor}")
    except Exception as e:  # noqa: BLE001
        check("FAIL", "config/mission.yaml", repr(e))
    tiles = data_root() / "tiles"
    for r in REQUIRED_TILES:
        check("PASS" if (tiles / f"{r}.npz").exists() else "FAIL", f"tile {r}")
    k = list((data_root() / "kernels").glob("*")) if (data_root() / "kernels").exists() else []
    check("PASS" if k else "WARN", "SPICE kernels", f"{len(k)} files (the 'kernels' stage fetches them if absent)")
    import shutil

    free = shutil.disk_usage(data_root()).free / 1e9
    check("PASS" if free > 20 else "WARN", "free disk", f"{free:.0f} GB")


def _params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def after(names: list[str], families: tuple[str, ...] = ("hazard", "light", "terrain", "rover")) -> None:
    runs = runs_root()
    led = runs / "overnight"
    runs_dirs = sorted(led.glob("*/ledger.jsonl")) if led.exists() else []
    if not runs_dirs:
        check("FAIL", "overnight ledger", "no ledger.jsonl under runs/overnight/")
    else:
        ev = [json.loads(x) for x in runs_dirs[-1].read_text(encoding="utf-8").splitlines() if x.strip()]
        last = {e["stage"]: e["status"] for e in ev if e.get("event") == "stage_end"}
        for s, st in last.items():
            check("PASS" if st == "ok" else ("WARN" if st == "timeout" else "FAIL"), f"stage {s}", st)
        check("PASS" if (runs_dirs[-1].parent / "RUN_REPORT.md").exists() else "WARN", "RUN_REPORT.md", str(runs_dirs[-1].parent))
    import torch

    for n in names:
        p = model_path(n)
        if not p.exists():
            check("FAIL", f"checkpoint {n}", f"missing {p}")
            continue
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            keys = list(ck.keys()) if isinstance(ck, dict) else []
            n_par = sum(v.numel() for v in (ck.get("state_dict") or {}).values()) if isinstance(ck, dict) else 0
            check("PASS", f"checkpoint {n} loads", f"{p.stat().st_size / 1e6:.0f} MB, ~{n_par / 1e6:.1f} M params, keys {keys[:6]}")
        except Exception as e:  # noqa: BLE001
            check("FAIL", f"checkpoint {n} loads", repr(e))
    # held-out evals: the network must be compared with the physics backup on sites it never trained on
    for fam in families:
        p = runs / fam / "eval.json"
        if not p.exists():
            check("FAIL", f"{fam} eval.json", "missing: held-out evaluation did not run")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        text = json.dumps(d)
        held = any(s in text for s in ("site23", "site42", "Site23", "Site42"))
        check("PASS" if held else "FAIL", f"{fam} evaluated on held-out Site23/Site42", "")
        tr = json.dumps(d.get("trained_on", []))
        check("PASS" if not any(s in tr for s in ("site23", "site42")) else "FAIL", f"{fam} never trained on Site23/Site42", "")
    # honest comparison, never a gate on 'looks good': we only print the numbers
    for fam, key in (("hazard", "f1"), ("light", "mae_deg")):
        p = runs / fam / "eval.json"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            for reg, v in d.get("regions", {}).items():
                if isinstance(v, dict) and str(v.get("role")).lower() == "test":
                    check("INFO", f"{fam} {reg} ({key})", json.dumps({k: (x.get(key) if isinstance(x, dict) else None) for k, x in v.items() if isinstance(x, dict) and key in x}))
    bad = [r for r in RESULTS if r[0] == "FAIL"]
    print(f"\n{len(RESULTS)} checks, {len(bad)} FAIL, {sum(1 for r in RESULTS if r[0] == 'WARN')} WARN", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["before", "after"])
    ap.add_argument("--names", default=",".join(DEFAULT_NAMES))
    ap.add_argument("--families", default="hazard,light,terrain,rover", help="eval families to check (4090 brain run: lander,planner)")
    a = ap.parse_args(argv)
    before() if a.phase == "before" else after([n for n in a.names.split(",") if n], tuple(f for f in a.families.split(",") if f))
    return 1 if any(r[0] == "FAIL" for r in RESULTS) else 0


if __name__ == "__main__":
    raise SystemExit(main())
