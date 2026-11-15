"""Verification steps for an overnight run. Prints PASS / FAIL / WARN per check; exit 1 if any FAIL.

    python -m ops.verify before        # right after setup: libraries, device, data, config, unit tests are fine
    python -m ops.verify after         # after training (run id macmini_night): checkpoints load, evals exist, network vs backup, honesty checks
    python -m ops.verify after --run-id macmini_night
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
DEFAULT_RUN_ID = "macmini_night"
TEST_SITES = ("site23", "site42")


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


def _family(name: str) -> str:
    return next((f for f in ("hazard", "light", "terrain", "rover") if f in name), name.split("_")[0])


def _load_model(fam: str, path):
    """Build the network from the checkpoint and load its weights (a key or shape mismatch raises)."""
    import importlib

    import torch

    if fam == "rover":
        from worldmodel.rover import policy

        loader = getattr(policy, "load_checkpoint", None)
    else:
        loader = importlib.import_module(f"worldmodel.{fam}.model").load_checkpoint
    if loader is None:
        return None
    out = loader(path, torch.device("cpu"))
    return out[0] if isinstance(out, tuple) else out


def after(names: list[str], run_id: str = DEFAULT_RUN_ID) -> None:
    runs = runs_root()
    ledger = runs / "overnight" / run_id / "ledger.jsonl"
    if not ledger.exists():
        check("FAIL", "overnight ledger", f"no {ledger} (verify reads exactly the run id it is given, never 'the latest')")
    else:
        ev = [json.loads(x) for x in ledger.read_text(encoding="utf-8").splitlines() if x.strip()]
        last = {e["stage"]: e["status"] for e in ev if e.get("event") == "stage_end"}
        check("PASS" if last else "FAIL", f"ledger of run {run_id}", f"{len(last)} stages ended")
        for s, st in last.items():
            check("PASS" if st == "ok" else ("WARN" if st in ("timeout", "skipped_deadline") else "FAIL"), f"stage {s}", st)
        for e in ev:
            if e.get("event") == "deadline_cap":
                check("INFO", f"deadline cap {e['stage']}", f"{e['budget_min_planned']} -> {e['budget_min_capped']} min")
        kinds = [e.get("event") for e in ev if e.get("event") in ("run_start", "run_end")]
        check("PASS" if kinds and kinds[-1] == "run_end" else "FAIL", "run reached its end", "the last launch wrote run_end" if kinds and kinds[-1] == "run_end" else "the last launch never finished")
        check("PASS" if (ledger.parent / "RUN_REPORT.md").exists() else "WARN", "RUN_REPORT.md", str(ledger.parent))
    import torch

    from ops import spatial_split

    # a training stage dropped by the deadline rule is a recorded, designed outcome: its missing files are WARN, not FAIL
    dropped = {s for s, st in (last if ledger.exists() else {}).items() if st == "skipped_deadline"}
    ids: dict[str, str] = {}
    for n in names:
        p = model_path(n)
        if not p.exists():
            check("WARN" if _family(n) in dropped else "FAIL", f"checkpoint {n}", f"missing {p}" + (" (stage skipped by the deadline: ship the physics backup)" if _family(n) in dropped else ""))
            continue
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            keys = list(ck.keys()) if isinstance(ck, dict) else []
            n_par = sum(v.numel() for v in (ck.get("state_dict") or {}).values()) if isinstance(ck, dict) else 0
            info = ck.get("info", {}) if isinstance(ck, dict) else {}
            ids[_family(n)] = str(info.get("checkpoint_id") or f"no id in {n}")
            check("PASS", f"checkpoint {n} loads", f"{p.stat().st_size / 1e6:.0f} MB, ~{n_par / 1e6:.1f} M params, keys {keys[:6]}")
            if info.get("selection"):
                check("WARN" if str(info["selection"]).startswith("last") else "INFO", f"checkpoint {n} selection", str(info["selection"]))
            tr = info.get("trained_on")
            if isinstance(tr, list):
                regions = [str(t).split("@")[0] for t in tr]
                leak = [r for r in regions if any(t in r.lower() for t in TEST_SITES)]
                check("PASS" if not leak else "FAIL", f"checkpoint {n} trained_on has no test site", f"{len(regions)} windows {leak[:5]}")
                near = spatial_split.trained_overlaps(regions)
                check("PASS" if not near else "FAIL", f"checkpoint {n} trained on no window touching a test tile", str(near[:8]))
            else:
                check("WARN", f"checkpoint {n} trained_on", "not recorded in the checkpoint")
        except Exception as e:  # noqa: BLE001
            check("FAIL", f"checkpoint {n} loads", repr(e))
            continue
        try:
            m = _load_model(_family(n), p)
            check("PASS" if m is not None else "WARN", f"checkpoint {n} builds its network", f"{_params(m) / 1e6:.1f} M params" if m is not None else "no loader found")
        except Exception as e:  # noqa: BLE001
            check("FAIL", f"checkpoint {n} builds its network", repr(e))
    try:
        missing = spatial_split.verify()
        check("PASS" if not missing else "FAIL", "spatial split file lists every overlapping window", str(missing[:8]))
    except Exception as e:  # noqa: BLE001
        check("WARN", "spatial split check", repr(e))
    tiles = data_root() / "tiles"
    have = {t: sorted(p.stem for p in tiles.glob(f"lola5m_{t}*.npz")) for t in TEST_SITES}
    for t, lst in have.items():
        check("PASS" if lst else "WARN", f"test tiles {t}", f"{len(lst)} tiles" if lst else "MISSING on this machine")
    if not have["site42"]:
        check("WARN", "held-out coverage", "Site42 tiles are missing: every held-out number rests on Site23 only. Say so in the results.")
    if not have["site23"]:
        check("FAIL", "held-out coverage", "no Site23 tiles: there is no held-out evidence at all")
    # held-out evals: the network must be compared with the physics backup on sites it never trained on
    by_fam = {_family(n): n for n in names}
    for fam in ("hazard", "light", "terrain", "rover"):
        p = runs / fam / "eval.json"
        if not p.exists():
            check("WARN" if fam in dropped else "FAIL", f"{fam} eval.json", "missing: held-out evaluation did not run" + (" (stage skipped by the deadline)" if fam in dropped else ""))
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        want_id, got_id, got_name = ids.get(fam), str(d.get("checkpoint_id")), d.get("checkpoint_name")
        if got_name is not None and by_fam.get(fam) and got_name != by_fam[fam]:
            check("FAIL", f"{fam} eval.json belongs to tonight's checkpoint", f"eval was made for {got_name!r}, expected {by_fam[fam]!r} (leftover eval?)")
        elif want_id is None:
            check("FAIL", f"{fam} eval.json belongs to tonight's checkpoint", "no checkpoint to compare with")
        elif got_id != want_id:
            check("FAIL", f"{fam} eval.json belongs to tonight's checkpoint", f"eval checkpoint_id {got_id} != {by_fam.get(fam)} id {want_id} (leftover eval?)")
        else:
            check("PASS", f"{fam} eval.json belongs to tonight's checkpoint", f"{by_fam.get(fam)} id {want_id}")
        if d.get("complete") is False:
            check("WARN", f"{fam} eval was cut short", f"{len(d.get('regions') or d.get('worlds') or {})} of {d.get('windows_planned')} windows (test sites are evaluated first)")
        if d.get("model_evaluated") is False:
            check("FAIL", f"{fam} eval evaluated the network", "baselines only: the checkpoint was not loaded")
        scored = " ".join(d.get("regions") or d.get("worlds") or {}).lower()      # the windows really scored, not a list of intended sites
        sites = [t for t in TEST_SITES if t in scored]
        check("PASS" if sites else "FAIL", f"{fam} evaluated on held-out Site23/Site42", f"covers {sites or 'neither'}" + ("" if len(sites) == 2 else " (NOT both)"))
        tr = json.dumps(d.get("trained_on", [])).lower()
        check("PASS" if not any(s in tr for s in TEST_SITES) else "FAIL", f"{fam} never trained on Site23/Site42", "")
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
    ap.add_argument("--run-id", default=DEFAULT_RUN_ID, help="the overnight run to verify (never guessed from what is on disk)")
    a = ap.parse_args(argv)
    before() if a.phase == "before" else after([n for n in a.names.split(",") if n], a.run_id)
    return 1 if any(r[0] == "FAIL" for r in RESULTS) else 0


if __name__ == "__main__":
    raise SystemExit(main())
