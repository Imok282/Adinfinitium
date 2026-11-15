"""Rover v2 training: copy the physics planners (DAgger), select the checkpoint on the validation site by full-episode driving. See ROVER_V2_PLAN.md.

    python -m worldmodel.rover2.teach --max-minutes 110 --arch former --d 384 --layers 6 --heads 8 --hazard-name hazard_unet_big --resume

Teacher of one rover = the planner variant (VARIANTS, all calls of the unchanged RoverEnv.planner_action) with the best simulated episode return from that
rover's start. Train sites only; candidates are scored on Site11 only; the TEST sites (Site23, Site42) are never loaded here.
Writes <ADI_DATA>/models/rover_driver_v2.pt (best on Site11), rover_driver_v2_last.pt (resume) and <ADI_RUNS>/train_log.jsonl, selection.json.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from contracts.mission import load_mission
from contracts.paths import model_path, runs_root
from worldmodel.hazard.lola import HOLDOUT_SITES, TRAIN_SITES, VAL_SITES
from worldmodel.rover.env import FIELD_CAP, MAP_CHANNELS, N_ACTIONS, VEC_DIM, RoverEnv
from worldmodel.rover.policy import build_model, pick_device
from worldmodel.rover.train import TRAIN_DATES, save_atomic, world_list
from worldmodel.rover.world import build_world, resolve_hazard_name

VARIANTS = (
    ("naive", {"turn_back": False}),
    ("cautious", {"turn_back": True}),
    ("cautious_wait_light", {"turn_back": True, "wait_for_light": True}),
    ("naive_wait_light", {"turn_back": False, "wait_for_light": True}),
    ("cautious_reserve30", {"turn_back": True, "reserve": 0.3}),
    ("cautious_wait_light25", {"turn_back": True, "wait_for_light": True, "light_need": 2.5}),
)
# For each variant, the same variant WITHOUT wait-for-light: if that one also says "stop", the planner is not waiting for light, it cannot improve.
NO_WAIT = (0, 1, 1, 0, 4, 1)
STILL_KEEP = 0.125                       # a rover that did not move on the previous step is stored this much less often (near-duplicate states)
STATE_KEYS = ("pos", "t", "steps", "battery", "phase", "target_idx", "done", "ep_return")
SELECTION = "rover v2: best on validation site Site11 by returned (full greedy episodes)"
METHOD = "imitation of the physics planners with DAgger; teacher per rover = planner variant with the best simulated episode return"
VAL_SEED = 4321


class TeachEnv(RoverEnv):
    """RoverEnv with one switch: `skip_obs` leaves out the observation while only planners drive. Dynamics and reward are the parent's."""

    skip_obs = False

    def observe(self) -> dict:
        return {} if self.skip_obs else super().observe()


def assert_not_test_sites(sites: list[str], regions: list[str]) -> None:
    """The held-out test sites must never reach training or selection: by site name and by tile name."""
    bad = sorted(set(sites) & set(HOLDOUT_SITES)) + [r for r in regions if any(f"lola5m_{h.lower()}" in r.lower() for h in HOLDOUT_SITES)]
    if bad:
        raise SystemExit(f"REFUSED: held-out test site in training/selection: {bad}")


def snapshot(env: RoverEnv) -> dict:
    return {k: getattr(env, k).clone() for k in STATE_KEYS}


def restore(env: RoverEnv, snap: dict) -> None:
    for k in STATE_KEYS:
        setattr(env, k, snap[k].clone())


def first_best(returns: torch.Tensor) -> torch.Tensor:
    """[N,K] -> [N] index of the highest value per row; ties go to the lowest index."""
    k = returns.shape[1]
    top = returns >= returns.max(dim=1, keepdim=True).values - 1e-6
    return (top.float() * torch.arange(k, 0, -1, device=returns.device)).argmax(dim=1)


def assign_teachers(env: TeachEnv) -> tuple[torch.Tensor, dict]:
    """Simulate every planner variant from the env's current start; give each rover the variant with the best episode return. The start is restored."""
    start = snapshot(env)
    returns, came_back, ice = [], [], []
    env.skip_obs = True
    try:
        for _, kw in VARIANTS:
            restore(env, start)
            back = torch.zeros(env.n, dtype=torch.bool, device=env.device)
            got = torch.zeros_like(back)
            for _ in range(env.rs.max_steps):
                _, _, done, info = env.step(env.planner_action(**kw))
                back |= info["returned"]
                got |= info["reached_ice"]
                if bool(done.all()):
                    break
            returns.append(env.ep_return.clone())
            came_back.append(back)
            ice.append(got)
    finally:
        env.skip_obs = False
        restore(env, start)
    assign = first_best(torch.stack(returns, dim=1))
    pick = assign[:, None]
    stats = {"teacher_returned": float(torch.stack(came_back, 1).gather(1, pick).float().mean()),
             "teacher_reached_ice": float(torch.stack(ice, 1).gather(1, pick).float().mean()),
             "variant_returned": {n: float(c.float().mean()) for (n, _), c in zip(VARIANTS, came_back)},
             "variant_share": {n: float((assign == i).float().mean()) for i, (n, _) in enumerate(VARIANTS)}}
    return assign, stats


def teacher_action(env: RoverEnv, assign: torch.Tensor, with_stuck: bool = False):
    """What each rover's assigned planner variant does in the rover's CURRENT state. With `with_stuck` also a mask of the rovers whose planner has NO
    ROUTE: it answers "stop" even without the wait-for-light rule AND the rover's current goal is unreachable from here (cost-to-go at the cap).
    That stop is a fallback, not a decision. A rover that turned back and now stands at the lander is NOT in the mask (its target is reachable)."""
    acts = torch.stack([env.planner_action(**kw) for _, kw in VARIANTS], dim=1)
    act = acts.gather(1, assign[:, None])[:, 0]
    if not with_stuck:
        return act
    base = torch.tensor(NO_WAIT, device=assign.device)[assign]
    return act, (acts.gather(1, base[:, None])[:, 0] == N_ACTIONS - 1) & (env._field_at(env.pos) >= 0.999 * FIELD_CAP)


class Buffer:
    """Ring buffer of (observation, teacher action) on the CPU; maps in half precision."""

    def __init__(self, cap: int, crop: int):
        self.map = torch.zeros(cap, MAP_CHANNELS, crop, crop, dtype=torch.float16)
        self.vec = torch.zeros(cap, VEC_DIM)
        self.act = torch.zeros(cap, dtype=torch.long)
        self.cap, self.size, self.at = cap, 0, 0

    def add(self, map_: torch.Tensor, vec: torch.Tensor, act: torch.Tensor) -> None:
        n = len(act)
        if n == 0:
            return
        if n > self.cap:
            map_, vec, act, n = map_[-self.cap:], vec[-self.cap:], act[-self.cap:], self.cap
        ix = (self.at + torch.arange(n)) % self.cap
        self.map[ix], self.vec[ix], self.act[ix] = map_.half().cpu(), vec.float().cpu(), act.cpu()
        self.at, self.size = int((self.at + n) % self.cap), min(self.size + n, self.cap)

    def sample(self, batch: int, gen: torch.Generator, device) -> tuple[dict, torch.Tensor]:
        ix = torch.randint(0, self.size, (batch,), generator=gen)
        return {"map": self.map[ix].to(device).float(), "vec": self.vec[ix].to(device)}, self.act[ix].to(device)


def collect(env: TeachEnv, model, buf: Buffer, gen: torch.Generator, net_frac: float, keep_frac: float, eps: float = 0.05) -> dict:
    """One full episode of every rover in `env`. A `net_frac` share of the rovers is driven by the network (greedy, `eps` random), the rest by their
    teacher; every kept state is stored with the TEACHER's action. Not stored: states where the planner has no route (its "stop" is a fallback the
    network cannot tell from its observation, and such rovers repeat it until the battery is empty). Stored STILL_KEEP as often: states of a rover that
    did not move on the previous step. Returns teacher statistics and how the network-driven rovers did."""
    d = env.device
    env.reset()
    assign, stats = assign_teachers(env)
    obs = env.observe()
    net = (torch.rand(env.n, generator=gen) < net_frac).to(d)
    back = torch.zeros(env.n, dtype=torch.bool, device=d)
    got, died = torch.zeros_like(back), torch.zeros_like(back)
    agree = seen = stored = stuck_states = 0
    moved_prev = torch.ones(env.n, dtype=torch.bool, device=d)
    for _ in range(env.rs.max_steps):
        alive = ~env.done
        if not bool(alive.any()):
            break
        act, stuck = teacher_action(env, assign, with_stuck=True)
        label = act.clone()
        ix = (alive & net).nonzero()[:, 0]
        if len(ix):
            with torch.no_grad():
                a = model({"map": obs["map"][ix], "vec": obs["vec"][ix]})[0].argmax(1)
            agree += int((a == label[ix]).sum())
            seen += len(ix)
            rnd = (torch.rand(len(ix), generator=gen) < eps).to(d)
            a = torch.where(rnd, torch.randint(0, N_ACTIONS, (len(ix),), generator=gen).to(d), a)
            act[ix] = a
        p_keep = keep_frac * torch.where(moved_prev, 1.0, STILL_KEEP)
        keep = alive & ~stuck & (torch.rand(env.n, generator=gen).to(d) < p_keep)
        buf.add(obs["map"][keep], obs["vec"][keep], label[keep])
        stored += int(keep.sum())
        stuck_states += int((alive & stuck).sum())
        before = env.pos.clone()
        obs, _, _, info = env.step(act)
        moved_prev = (env.pos != before).any(dim=1)
        back |= info["returned"]
        got |= info["reached_ice"]
        died |= info["dead"]
    k = max(int(net.sum()), 1)
    stats.update(stored=stored, stuck_states_not_stored=stuck_states, net_rovers=int(net.sum()), net_returned=float((back & net).sum()) / k, net_reached_ice=float((got & net).sum()) / k,
                 net_died=float((died & net).sum()) / k, net_agrees_with_teacher=agree / max(seen, 1))
    return stats


EVAL_PLANNERS = (("naive", {"turn_back": False}), ("cautious", {"turn_back": True}), ("cautious_wait_light", {"turn_back": True, "wait_for_light": True}))


def drive(env: RoverEnv, model, seed: int, planner: dict | None = None) -> dict:
    """Full greedy episodes of the network alone from seeded starts (the same loop as rover.train.evaluate_policy's model branch).
    With `planner` (planner_action arguments) that planner drives instead, from the same starts: the reference rows of the selection record."""
    env.gen = torch.Generator(device="cpu").manual_seed(seed)
    obs = env.reset()
    ice = ret = dead = 0
    for _ in range(env.rs.max_steps):
        alive = (~env.done).nonzero()[:, 0]
        act = torch.full((env.n,), N_ACTIONS - 1, dtype=torch.long, device=env.device)      # finished rovers: the env ignores their action
        with torch.no_grad():
            if planner is not None:
                act = env.planner_action(**planner)
            elif len(alive):
                act[alive] = model({"map": obs["map"][alive], "vec": obs["vec"][alive]})[0].argmax(1)
        obs, _, done, info = env.step(act)
        ice += int(info["reached_ice"].sum()); ret += int(info["returned"].sum()); dead += int(info["dead"].sum())
        if bool(done.all()):
            break
    return {"reached_ice": ice / env.n, "returned": ret / env.n, "died": dead / env.n, "mean_return": float(env.ep_return.mean())}


def validate(val_envs: list[RoverEnv], model, planner: dict | None = None) -> dict:
    if model is not None:
        model.eval()
    per = {f"{e.world.region}@{e.world.date}": drive(e, model, VAL_SEED, planner) for e in val_envs}
    mean = {k: float(np.mean([w[k] for w in per.values()])) for k in ("reached_ice", "returned", "died", "mean_return")}
    return {"worlds": per, **mean}


def better(a: dict, b: dict | None) -> bool:
    """The pre-registered rule: higher returned; ties -> higher reached_ice, then lower died, then the earlier candidate (b) stays."""
    return b is None or (a["returned"], a["reached_ice"], -a["died"]) > (b["returned"], b["reached_ice"], -b["died"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-minutes", type=float, default=110.0, help="wall-clock budget for training INCLUDING the Site11 selection evaluations")
    ap.add_argument("--arch", choices=["conv", "former"], default="former")
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--sites", nargs="+", default=TRAIN_SITES)
    ap.add_argument("--val-sites", nargs="+", default=VAL_SITES)
    ap.add_argument("--dates", nargs="+", default=list(TRAIN_DATES))
    ap.add_argument("--envs", type=int, default=512, help="rovers per data-collection episode")
    ap.add_argument("--val-envs", type=int, default=256, help="rovers per validation world")
    ap.add_argument("--buffer", type=int, default=100000, help="stored states (about 49 kB each in CPU memory)")
    ap.add_argument("--keep-frac", type=float, default=0.25, help="share of the visited states stored per episode")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--grad-steps", type=int, default=0, help="training steps after each data-collection episode (0 = 4 passes over the states it added, 64 to 400)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--teacher-rounds", type=int, default=8, help="first episodes are driven by the teacher only")
    ap.add_argument("--net-frac", type=float, default=0.5, help="after that, share of rovers driven by the network (DAgger)")
    ap.add_argument("--candidate-minutes", type=float, default=12.0)
    ap.add_argument("--val-estimate-s", type=float, default=150.0, help="first guess of one selection evaluation's duration (measured after the first)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--name", default="rover_driver_v2")
    ap.add_argument("--hazard-name", default=None)
    a = ap.parse_args(argv)

    t_start = time.time()
    cfg = load_mission()
    device = pick_device(a.device)
    torch.manual_seed(a.seed)
    gen = torch.Generator(device="cpu").manual_seed(a.seed)
    specs, val_specs = world_list(a.sites, a.dates), world_list(a.val_sites, a.dates)
    assert_not_test_sites(list(a.sites) + list(a.val_sites), [r for r, _ in specs + val_specs])
    missing = [s for s in list(a.sites) + list(a.val_sites) if not world_list([s], a.dates)]
    if missing:
        raise SystemExit(f"REFUSED: no tiles for {missing}: every requested site must be there (a silently smaller training set is not the plan)")
    if set(a.sites) & set(a.val_sites):
        raise SystemExit("a validation site is also a training site")
    envs = [TeachEnv(build_world(r, d, cfg, device=device, hazard_name=a.hazard_name), a.envs, device, cfg, seed=a.seed + i) for i, (r, d) in enumerate(specs)]
    if {r for r, _ in val_specs} & {r for r, _ in specs}:
        raise SystemExit("a validation tile is also a training tile")
    val_envs = [RoverEnv(build_world(r, d, cfg, device=device, hazard_name=a.hazard_name), a.val_envs, device, cfg, seed=VAL_SEED) for r, d in val_specs]
    hz = sorted({e.world.hazard_source for e in envs})
    provenance = {"trained_on": [f"{r}@{d}" for r, d in specs], "train_sites": list(a.sites), "validated_on": [f"{r}@{d}" for r, d in val_specs],
                  "hazard_name": resolve_hazard_name(a.hazard_name), "hazard_source": hz[0] if len(hz) == 1 else "mixed: " + " | ".join(hz),
                  "ice_sources": sorted({e.world.ice_source for e in envs}), "method": METHOD, "teacher_variants": [n for n, _ in VARIANTS]}
    print(f"device={device} train worlds={len(envs)} validation worlds={len(val_envs)} ({a.val_sites}); test {HOLDOUT_SITES} never loaded; "
          f"ready in {time.time() - t_start:.0f}s; hazard: {provenance['hazard_source']}", flush=True)

    kw = {"width": a.width, "hidden": a.hidden} if a.arch == "conv" else {"d": a.d, "layers": a.layers, "heads": a.heads}
    model = build_model(a.arch, **kw).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-5)
    last_p, best_p = model_path(a.name + "_last"), model_path(a.name)
    out = runs_root()
    out.mkdir(parents=True, exist_ok=True)
    log_p, sel_p = out / "train_log.jsonl", out / "selection.json"
    st = {"round": 0, "steps": 0, "train_seconds": 0.0, "candidates": [], "best": None}
    if a.resume and last_p.exists():
        ck = torch.load(last_p, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        st = ck["v2_state"]
        gen.manual_seed(a.seed + 7919 * st["round"])                       # a resumed run draws new starts, not the first run's again
        for i, e in enumerate(envs):
            e.gen.manual_seed(a.seed + i + 100003 * st["round"])
        print(f"resumed at round {st['round']}, step {st['steps']}, {st['train_seconds'] / 60:.1f} min already used, {len(st['candidates'])} candidates", flush=True)
    print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    used0 = st["train_seconds"] + (time.time() - t_start)                # world building counts against the budget
    t0 = time.time()
    budget_s = a.max_minutes * 60.0

    def used() -> float:
        return used0 + time.time() - t0

    def save_last() -> None:
        st["train_seconds"] = used()
        save_atomic(last_p, model, {"round": st["round"], "steps": st["steps"], "device": str(device), **provenance},
                    extra={"optimizer": opt.state_dict(), "v2_state": st})

    def candidate(tag: str) -> float:
        tv = time.time()
        res = validate(val_envs, model)
        row = {"candidate": len(st["candidates"]), "tag": tag, "round": st["round"], "steps": st["steps"], "minutes": used() / 60.0,
               **{k: res[k] for k in ("returned", "reached_ice", "died", "mean_return")}, "worlds": res["worlds"]}
        row["selected"] = better(row, st["best"])
        st["candidates"].append(row)
        if row["selected"]:
            st["best"] = {k: row[k] for k in ("candidate", "round", "steps", "returned", "reached_ice", "died", "mean_return")}
            save_atomic(best_p, model, {"round": st["round"], "steps": st["steps"], "device": str(device), **provenance, "selection": SELECTION,
                                        "selected_candidate": row["candidate"], "validation": st["best"]})
        sel_p.write_text(json.dumps({"rule": SELECTION + "; ties: higher reached_ice, then lower died, then the earlier candidate",
                                     "validation_worlds": provenance["validated_on"], "rovers_per_world": a.val_envs, "seed": VAL_SEED,
                                     "planners_on_the_same_starts": st.get("val_planners"), "best": st["best"], "candidates": st["candidates"]}, indent=2),
                         encoding="utf-8")
        save_last()
        dt = time.time() - tv
        print(f"candidate {row['candidate']} ({tag}) round {st['round']} step {st['steps']}: Site11 returned {row['returned']:.3f} ice {row['reached_ice']:.3f} "
              f"died {row['died']:.3f}{'  <- best so far' if row['selected'] else ''} | {dt:.0f}s", flush=True)
        return dt

    if st.get("finished"):
        print("this run already finished (final candidate recorded); nothing to train", flush=True)
        return 0
    if not st.get("val_planners"):
        st["val_planners"] = {n: {k: v for k, v in validate(val_envs, None, kw).items() if k != "worlds"} for n, kw in EVAL_PLANNERS}
        print("planners on the Site11 selection starts (reference only): " + "; ".join(
            f"{n} returned {r['returned']:.3f} ice {r['reached_ice']:.3f} died {r['died']:.3f}" for n, r in st["val_planners"].items()), flush=True)
    buf = Buffer(a.buffer, cfg.rover_sim.crop_px)
    val_s, last_cand, last_save = a.val_estimate_s, used(), time.time()
    order_rng, queue = np.random.default_rng(a.seed + 1000 * st["round"]), []
    while used() + val_s + 30.0 < budget_s:
        if not queue:
            queue = [int(i) for i in order_rng.permutation(len(envs))]       # every world once per pass, in a new random order
        env = envs[queue.pop()]
        tr, steps_before = time.time(), st["steps"]
        model.eval()
        net_frac = 0.0 if st["round"] < a.teacher_rounds else a.net_frac
        cs = collect(env, model, buf, gen, net_frac, a.keep_frac)
        t_collect = time.time() - tr
        loss_v = acc_v = float("nan")
        if buf.size >= a.batch:
            model.train()
            # --grad-steps 0: as many steps as 4 passes over the states this episode added (64 to 400), sampled from the whole buffer.
            for _ in range(a.grad_steps or max(64, min(400, 4 * cs["stored"] // a.batch))):
                obs, act = buf.sample(a.batch, gen, device)
                logits, _ = model(obs)
                loss = F.cross_entropy(logits, act)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                st["steps"] += 1
            loss_v, acc_v = float(loss.detach()), float((logits.argmax(1) == act).float().mean())
        st["round"] += 1
        row = {"round": st["round"], "steps": st["steps"], "world": f"{env.world.region}@{env.world.date}", "buffer": buf.size, "loss": loss_v, "batch_acc": acc_v,
               "collect_s": t_collect, "round_s": time.time() - tr, "steps_this_round": st["steps"] - steps_before, "minutes": used() / 60.0, **cs}
        with open(log_p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        if st["round"] % 5 == 0 or st["round"] <= 2:
            print(f"round {st['round']:4d} {row['world']:26s} loss {loss_v:.3f} acc {acc_v:.2f} | teacher home {cs['teacher_returned']:.2f} | net rovers {cs['net_rovers']:3d} "
                  f"home {cs['net_returned']:.2f} ice {cs['net_reached_ice']:.2f} agree {cs['net_agrees_with_teacher']:.2f} | collect {t_collect:.0f}s + {row['steps_this_round']} steps = {row['round_s']:.0f}s, {used() / 60:.1f} min", flush=True)
        if st["steps"] > 0 and used() - last_cand >= a.candidate_minutes * 60.0 and used() + val_s + 30.0 < budget_s:
            val_s = candidate("periodic")
            last_cand = used()
        elif time.time() - last_save > 300.0:
            save_last()
            last_save = time.time()
    if st["steps"] == 0:
        print("FAILED: the budget ended before any training step; no candidate is recorded", flush=True)
        return 4
    if not st["candidates"] or st["candidates"][-1]["steps"] != st["steps"]:
        candidate("final")
    st["finished"] = True
    save_last()
    b = st["best"]
    print(f"selected candidate {b['candidate']} (round {b['round']}): Site11 returned {b['returned']:.3f} ice {b['reached_ice']:.3f} died {b['died']:.3f}; "
          f"{len(st['candidates'])} candidates; {used() / 60:.1f} min used of {a.max_minutes:.0f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
