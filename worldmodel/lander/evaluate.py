"""Held-out exam for the lander pilot: Site23 + Site42 (never trained on), the SAME missions flown by classic ZEM/ZEV guidance and by the learned pilot,
with the config's failure injection (engine out, nav drift, blind camera) and sensor noise. Writes <ADI_RUNS>/lander/eval.json. Honest: reports where the pilot
is worse as well as better; per-failure-type success rates; missing test tiles are skipped with a note.

    python -m worldmodel.lander.evaluate --name lander_pilot --n 1024
"""

from __future__ import annotations

import argparse
import json

from contracts.mission import load_mission
from contracts.paths import model_path, run_dir
from worldmodel.hazard.lola import HOLDOUT_SITES, VAL_SITES
from worldmodel.lander.env import make_params
from worldmodel.lander.fly import fly, summarize
from worldmodel.lander.guidance import guidance_params
from worldmodel.lander.pilot import load_checkpoint, pick_device
from worldmodel.lander.scenes import load_scenes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="lander_pilot")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--sites", nargs="+", default=HOLDOUT_SITES + VAL_SITES)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--belief", choices=["backup", "model"], default="backup")
    ap.add_argument("--seeds", type=int, default=3, help="independent mission sets per site (variance)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)

    cfg = load_mission()
    device = pick_device(a.device)
    params = make_params(cfg)
    gp = guidance_params(params, cfg.scenarios.terminal.hover_gate_altitude_m, cfg.scenarios.terminal.final_descent_speed_m_s)
    ck = model_path(a.name)
    model, info = (load_checkpoint(ck, device) if ck.exists() else (None, {}))
    if model is None:
        print(f"note: no checkpoint {ck.name}: reporting the classic baseline only", flush=True)
    trained = set(info.get("trained_on", []))
    scenes = load_scenes(a.sites, cfg, device, a.belief, 1)
    res = {"checkpoint": ck.name if model else None, "checkpoint_id": info.get("checkpoint_id"), "parameters": info.get("parameters"),
           "trained_on": sorted(trained), "n_per_seed": a.n, "seeds": a.seeds, "injection": "config failures + sensor noise", "regions": {}}
    print(f"{'region':16s} {'role':5s} {'classic':>8s} {'pilot':>8s}   (success rate, mean over {a.seeds} seeds x {a.n} landers)", flush=True)
    for s in scenes:
        short = s.region.split("lola5m_")[1]
        role = "TEST" if short.replace("site", "Site") in HOLDOUT_SITES else "val"
        if role == "TEST":
            assert s.region not in trained, f"{s.region} is a test region but the pilot was trained on it"
        runs = {"classic": [], "pilot": []}
        for seed in range(a.seeds):
            runs["classic"].append(summarize(fly(s, params, gp, None, a.n, 1000 + seed, a.repeat, device=device)))
            if model is not None:
                runs["pilot"].append(summarize(fly(s, params, gp, model, a.n, 1000 + seed, a.repeat, device=device)))
        agg = {}
        for k, v in runs.items():
            if v:
                rates = [x["success_rate"] for x in v]
                agg[k] = {"success_rate": sum(rates) / len(rates), "success_rate_per_seed": rates, "last_seed_detail": v[-1]}
        res["regions"][s.region] = {"role": role, **agg}
        print(f"{short:16s} {role:5s} {agg['classic']['success_rate']:8.3f} {agg['pilot']['success_rate'] if 'pilot' in agg else float('nan'):8.3f}", flush=True)
    out = run_dir("lander") / "eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
