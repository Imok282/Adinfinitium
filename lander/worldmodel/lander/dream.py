"""Dream-RSI for the lander pilot: keeps improving the pilot OFFLINE after the main training was cut off (laptop, any GPU or CPU), safely.

One generation:
  1. FIND FAILURES  fly the current pilot through many fresh missions (nominal failure rates) on the training terrain and count how it fails
                    (engine out, navigation drift, blind camera, hard landing, hazard cell, ...);
  2. DREAM HARDER   warm-start PPO from the current pilot with the failure types made more frequent (--fail-boost) for a few minutes:
                    it practises exactly the situations it is worst at, in the physics twin, with no internet and no human data;
  3. GATE           the candidate must beat the current pilot on a FROZEN validation exam (Site11, fixed mission seeds, NOMINAL failure
                    rates, 2 x 512 landers), by at least --margin. Otherwise it is thrown away and the current pilot stays;
  4. REPEAT         the next generation starts from whichever pilot won (recursive self-improvement).
The production checkpoint is <ADI_DATA>/models/lander_pilot.pt; every replaced version is kept as lander_pilot_gen<k>.pt. History, scores and
decisions: <ADI_DATA>/pilot_dream/history.jsonl. Resumable and Ctrl-C safe. The test sites (Site23, Site42) are never used.

    python -m worldmodel.lander.dream --generations 2 --minutes 10
    python -m worldmodel.lander.dream --forever --minutes 15         # keep improving whenever the machine is on
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from contracts.mission import load_mission
from contracts.paths import data_root, model_path
from worldmodel.hazard.lola import HOLDOUT_SITES, TRAIN_SITES, VAL_SITES

PROD = "lander_pilot"
CAND = "lander_pilot_dream"
EXAM_SEEDS = (9001, 9002)                     # frozen validation exam: same missions every generation


def state_dir() -> Path:
    return data_root() / "pilot_dream"


def _log(row: dict) -> None:
    state_dir().mkdir(parents=True, exist_ok=True)
    with open(state_dir() / "history.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def _state() -> dict:
    p = state_dir() / "state.json"
    return json.loads(p.read_text()) if p.exists() else {"generation": 0, "accepted": 0, "rejected": 0}


def _save_state(st: dict) -> None:
    state_dir().mkdir(parents=True, exist_ok=True)
    (state_dir() / "state.json").write_text(json.dumps(st, indent=2))


def accept(old: float, new: float, margin: float) -> bool:
    return bool(np.isfinite(new) and new >= old + margin)


def exam(name: str, n: int, device, sites=VAL_SITES) -> dict:
    """Frozen validation exam: success rate of a checkpoint (and of classic guidance) on fixed missions, nominal failure rates."""
    from worldmodel.lander.env import make_params
    from worldmodel.lander.fly import fly, summarize
    from worldmodel.lander.guidance import guidance_params
    from worldmodel.lander.pilot import load_checkpoint
    from worldmodel.lander.scenes import load_scenes

    cfg = load_mission()
    params = make_params(cfg)
    gp = guidance_params(params, cfg.scenarios.terminal.hover_gate_altitude_m, cfg.scenarios.terminal.final_descent_speed_m_s)
    model = load_checkpoint(model_path(name), device)[0] if name != "classic" else None
    rates = []
    for sc in load_scenes(list(sites), cfg, device, "backup", 1):
        for seed in EXAM_SEEDS:
            rates.append(summarize(fly(sc, params, gp, model, n, seed, 5, device=device))["success_rate"])
    return {"success": float(np.mean(rates)) if rates else float("nan"), "n": n * len(rates)}


def failure_profile(name: str, n: int, device) -> dict:
    """How the current pilot fails on fresh training-terrain missions (what the next dream should practise)."""
    from worldmodel.lander.env import make_params
    from worldmodel.lander.fly import fly, summarize
    from worldmodel.lander.guidance import guidance_params
    from worldmodel.lander.pilot import load_checkpoint
    from worldmodel.lander.scenes import load_scenes

    cfg = load_mission()
    params = make_params(cfg)
    gp = guidance_params(params, cfg.scenarios.terminal.hover_gate_altitude_m, cfg.scenarios.terminal.final_descent_speed_m_s)
    model = load_checkpoint(model_path(name), device)[0]
    out, rng = {}, np.random.default_rng(int(time.time()))
    for sc in load_scenes([s for s in TRAIN_SITES if s not in HOLDOUT_SITES][:3], cfg, device, "backup", 1):
        s = summarize(fly(sc, params, gp, model, n, int(rng.integers(1 << 30)), 5, device=device))
        for k, v in s["failures"].items():
            out[k] = out.get(k, 0) + v
        for k in ("success_when_engine_failed", "success_when_nav_drift", "success_when_camera_blind"):
            if s.get(k) is not None:
                out.setdefault(k, []).append(s[k])
    return {k: (float(np.mean(v)) if isinstance(v, list) else v) for k, v in out.items()}


def dream_generation(gen: int, minutes: float, device_name: str, margin: float, exam_n: int, envs: int, mb: int) -> dict:
    from worldmodel.lander import train as tr

    t0 = time.time()
    device = torch.device(device_name)
    prod_p = model_path(PROD)
    if not prod_p.exists():
        raise SystemExit("no trained pilot yet (python -m worldmodel.lander.train)")
    st = torch.load(prod_p, map_location="cpu", weights_only=False)
    arch = dict(st["arch"])
    prof = failure_profile(PROD, 128, device)
    weak = [k for k in ("success_when_engine_failed", "success_when_nav_drift", "success_when_camera_blind") if prof.get(k, 1.0) < 0.95]
    boost = 3.0 if weak else 2.0
    # warm start: the candidate begins as an exact copy of the production pilot
    for f in model_path(CAND).parent.glob(CAND + "*.pt"):
        f.unlink()
    torch.save({"state_dict": st["state_dict"], "arch": st["arch"], "info": {"update": 0, "best_val_success": -1.0, "warm_start_from": PROD}},
               model_path(CAND + "_last"))
    args = ["--name", CAND, "--resume", "--max-minutes", str(minutes), "--updates", "100000", "--fail-boost", str(boost), "--eval-every", "5",
            "--val-n", str(min(128, exam_n)), "--envs", str(envs), "--horizon", "64", "--mb-size", str(mb), "--epochs", "2", "--lr", "1e-4", "--device", device_name]
    if arch.get("kind") == "former":
        args += ["--arch", "former", "--d", str(arch["d"]), "--layers", str(arch["layers"]), "--heads", str(arch["heads"])]
    else:
        args += ["--arch", "mlp", "--hidden", str(arch.get("hidden", 256))]
    tr.main(args)
    if not model_path(CAND).exists():
        row = {"generation": gen, "accepted": False, "reason": "dream produced no candidate", "seconds": round(time.time() - t0)}
        return row
    old = exam(PROD, exam_n, device)
    new = exam(CAND, exam_n, device)
    ok = accept(old["success"], new["success"], margin)
    if ok:
        shutil.copy2(prod_p, model_path(f"{PROD}_gen{gen - 1:03d}"))
        shutil.copy2(model_path(CAND), prod_p)
    row = {"generation": gen, "time_utc": datetime.now(UTC).isoformat(), "failure_profile_before": prof, "fail_boost": boost,
           "dream_minutes": minutes, "exam": f"{list(VAL_SITES)} seeds {list(EXAM_SEEDS)} x {exam_n}, nominal failures",
           "exam_success_before": round(old["success"], 4), "exam_success_candidate": round(new["success"], 4), "margin": margin,
           "accepted": ok, "seconds": round(time.time() - t0)}
    print(f"pilot dream {gen}: weak at {weak or 'nothing in particular'}; boost x{boost} for {minutes} min | exam {old['success']:.4f} -> "
          f"{new['success']:.4f} {'ACCEPTED (new production pilot)' if ok else 'rejected (kept the current pilot)'}", flush=True)
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generations", type=int, default=2)
    ap.add_argument("--forever", action="store_true")
    ap.add_argument("--minutes", type=float, default=10.0, help="training minutes per dream")
    ap.add_argument("--margin", type=float, default=0.002, help="minimum gain on the frozen exam to accept a new pilot")
    ap.add_argument("--exam-n", type=int, default=512)
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--mb-size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    st = _state()
    done = 0
    try:
        while a.forever or done < a.generations:
            gen = st["generation"] + 1
            try:
                row = dream_generation(gen, a.minutes, a.device, a.margin, a.exam_n, a.envs, a.mb_size)
            except SystemExit:
                raise
            except Exception as e:  # noqa: BLE001 - a failed dream is logged, never fatal; production pilot untouched
                row = {"generation": gen, "error": repr(e)[:300]}
                print(f"pilot dream {gen}: FAILED {e!r}", flush=True)
            _log(row)
            st["generation"] = gen
            st["accepted"] += int(bool(row.get("accepted")))
            st["rejected"] += int("accepted" in row and not row["accepted"])
            _save_state(st)
            done += 1
    except KeyboardInterrupt:
        print("stopped; the production pilot is unchanged unless a dream was accepted; the next run resumes", flush=True)
    print(f"pilot Dream-RSI: {st['generation']} generations, {st['accepted']} accepted, {st['rejected']} rejected ({state_dir()})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
