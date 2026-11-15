"""EXPERIMENTAL ADDENDUM: the rover keeps learning by itself after training has ended. Not part of the frozen overnight run; never mix its numbers in.

    bash ops/dream/run_dream_mini.sh 140          # on the mini, from this separate folder, after the main run has frozen; run it again to resume
    python -m worldmodel.dream.run --run-id selflearn1 --max-minutes 140 --resume --main-run-dir ~/adinfinitium-run/adinfinitium --hazard-name hazard_unet_big

Starting point: the finished, frozen driver (opened read-only, never modified). From then on the only new information is what the rover experiences
when it drives. For each situation `site:change[:date]` the loop runs round after round until the time budget ends, with no human in it:
  drive    the current driver drives in the world; its own steps are recorded
  learn    the world model is updated on the recorded steps
  dream    candidate corrections to the driver practise inside the world model (no real steps)
  test     each candidate drives real episodes from the same starts as the current driver; it is accepted only on a clear gain, else the driver stays
  repeat   round N starts from round N-1's accepted driver and model
How a round spends its tries (which recipe, how many refinements, when to stop) is decided by a Dream-RSI-style strategy layer (rsi.py): every round's
scored tree is saved as a replay world, strategy variants are replayed on all saved worlds, and the best runs the next round.
Writes ONLY under <runs>/_dream/<run-id>/ (report.json, report.md, curve_*.png, ledger.jsonl, trace_pool/, models/rover_dream_*.pt) and <runs>/_dream/_cache/.
Refuses to start while run_everything.sh / ops.overnight.run_all is alive. A used --run-id is continued only with --resume, never overwritten.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from contracts.mission import load_mission
from contracts.paths import model_path, repo_root, runs_root
from worldmodel.dream import guard, rsi
from worldmodel.dream.changes import CHANGES, apply_change
from worldmodel.dream.learn import clear_gain, dream_improve, model_error, paired_gain, train_world_model
from worldmodel.dream.model import DreamNet, load_dream, save_dream
from worldmodel.dream.replay import Replay, clone_env, reseed, rollout
from worldmodel.hazard.lola import regions_for_site
from worldmodel.rover.env import RoverEnv
from worldmodel.rover.policy import load_checkpoint, pick_device

MEASURE_SEED, GATE_SEED, ACT_SEED = 90210, 50000, 70000        # score starts are never used for driving, learning, testing candidates or strategy choice
RECIPES = (("careful", 1.0), ("medium", 0.3), ("bold", 0.05))   # how far a candidate may move from the current driver while dreaming (KL weight)
PROTOCOL_KEYS = ("frozen_name", "hazard_name", "probes", "refine", "gate_se", "n_act", "act_steps", "wm_steps", "dream_steps", "n_gate", "n_measure",
                 "batch", "horizon", "latent", "hidden", "no_rsi")      # a resumed run must keep these, or its rounds would not be comparable
STAT_KEYS = ("reached_ice", "returned", "died", "mean_return", "blocked_moves_per_episode", "mean_hazard_under_rover")
NOTE = ("EXPERIMENTAL ADDENDUM, separate from the frozen overnight results. A bounded demo of continued self-learning in simulation (kinematic, hourly "
        "steps), not open-ended self-improvement. Scores after self-learning on a site are 'online self-learning on that site', never 'held-out'. "
        "The strategy layer follows Dream-RSI (record, replay, score, keep the best) but is NOT the original: no LLM rewrites the strategy and no API "
        "key is used; variants come from a fixed grid. The planners read the true map and physics: an upper reference.")


@contextlib.contextmanager
def private_cache(cache_root: Path):
    """World building may write light/hazard caches. Point those writes at the dream folder; existing caches of the main data folder are only linked."""
    import worldmodel.rover.world as w

    orig = w.data_root
    dst = Path(cache_root) / "rover_cache"
    dst.mkdir(parents=True, exist_ok=True)
    for p in dst.iterdir():
        if p.is_symlink() and not p.exists():
            p.unlink()                       # a dangling link would let a later save write through into the main data folder
    src = orig() / "rover_cache"
    if src.exists():
        for p in src.iterdir():
            if p.is_file() and not p.is_symlink() and not (dst / p.name).exists():
                (dst / p.name).symlink_to(p.resolve())
    w.data_root = lambda: Path(cache_root)
    try:
        yield
    finally:
        w.data_root = orig


def default_provider(cfg, device, hazard_name: str | None, cache_root: Path):
    from worldmodel.rover.world import build_world

    def provide(site: str, date: str):
        regions = regions_for_site(site)[:1]
        if not regions:
            return None
        with private_cache(cache_root):
            return build_world(regions[0], date, cfg, device=device, hazard_name=hazard_name)
    return provide


def planner_fn(**kw):
    return lambda env, obs: env.planner_action(**kw)


def frozen_fn(model):
    return lambda env, obs: model(obs)[0].argmax(1)


def driver_fn(frozen, net: DreamNet, greedy: bool = True):
    """The self-learning driver: the frozen driver's logits plus the learned correction (zero before any self-learning). Also hands back the frozen
    logits so they are recorded with the step."""
    def act(env, obs):
        prior = frozen(obs)[0]
        lg = prior + net.correction(obs)
        return (lg.argmax(1) if greedy else torch.distributions.Categorical(logits=lg).sample()), prior
    return act


class StalePlanner:
    """The cautious planner that was NOT told about the change: it plans on the original map and physics while the rover drives in the changed world."""

    def __init__(self, original_env: RoverEnv):
        self.orig = original_env
        self.shadow = None

    def __call__(self, env, obs):
        if self.shadow is None or self.shadow.n != env.n:
            self.shadow = clone_env(self.orig, env.n, 0)
        for k in ("pos", "t", "steps", "battery", "phase", "target_idx", "done"):
            setattr(self.shadow, k, getattr(env, k))
        return self.shadow.planner_action(turn_back=True)


def score(env: RoverEnv, act_fn, seed: int) -> dict:
    """One episode per rover from the starts fixed by `seed` (greedy drivers: the same driver always gets the same score)."""
    reseed(env, seed)
    _, s = rollout(env, act_fn, env.rs.max_steps, reset_done=False, record=False)
    return s


def checkpoint_name(site: str, change: str, date: str, r: int) -> str:
    return f"rover_dream_{site.lower()}_{change}_{date}_r{r:02d}"


def situation_key(spec: str, default_date: str) -> tuple[str, str, str, str]:
    site, change, *rest = spec.split(":")
    date = rest[0] if rest else default_date
    return f"{site}:{change}:{date}", site, change, date


class Run:
    def __init__(self, a, provider=None):
        self.a, self.t0 = a, time.time()
        self.t_end = self.t0 + a.max_minutes * 60.0
        self.cfg = load_mission()
        self.device = pick_device(a.device)
        self.out = runs_root() / "_dream" / a.run_id
        self.models = self.out / "models"
        self.provider = provider or default_provider(self.cfg, self.device, a.hazard_name, runs_root() / "_dream" / "_cache")
        self.gen = torch.Generator(device="cpu").manual_seed(a.seed + int(self.t0) % 100000 if a.resume and self.out.exists() else a.seed)
        self.replay = Replay()
        self.frozen = None
        self.start_file = getattr(a, "start_driver_file", None)             # a driver trained after the freeze: never called frozen anywhere
        self.base = "starting driver" if self.start_file else "frozen driver"
        self.strategy = rsi.DEFAULT
        self.traces: list[dict] = []
        self.cut = []                        # training calls that the time budget cut short
        self.report = {"experimental_addendum": True, "note": NOTE, "run_id": a.run_id, "acceptance_criterion": criterion(self.base), "device": str(self.device),
                       "seeds": {"score": MEASURE_SEED, "candidate_tests": GATE_SEED, "driving": ACT_SEED}, "invocations": [], "situations": {},
                       "rsi": {"what": "key-free Dream-RSI-style loop: strategies come from a fixed grid, not from an LLM (see worldmodel/dream/rsi.py)",
                               "enabled": not a.no_rsi, "strategy_now": asdict(rsi.DEFAULT), "dream_steps": []}, "skipped": [], "verdicts": {}}

    # ---------------------------------------------------------------------------------------------- bookkeeping
    def left(self) -> float:
        return self.t_end - time.time()

    def log(self, event: str, **kw) -> None:
        line = {"time": datetime.now().astimezone().isoformat(timespec="seconds"), "event": event, **kw}
        with open(self.out / "ledger.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, default=str) + "\n")
        print(f"[{(time.time() - self.t0) / 60:6.1f} min] {event} " + " ".join(f"{k}={v}" for k, v in kw.items())[:300], flush=True)

    def write(self) -> None:
        inv = self.report["invocations"][-1]
        inv.update(minutes_used=round((time.time() - self.t0) / 60, 2), over_budget_minutes=round(max(0.0, -self.left()) / 60, 2),
                   training_cut_short_by_budget=self.cut)
        fd = driver_record(self.report)
        fd["sha256_after"] = guard.sha256(Path(fd["path"]))
        fd["unchanged"] = fd["sha256_after"] == fd["sha256_before"]
        self.report["verdicts"] = {k: verdict(s, self.base) for k, s in self.report["situations"].items()}
        tmp = self.out / "report.json.tmp"
        tmp.write_text(json.dumps(self.report, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.out / "report.json")
        (self.out / "report.md").write_text(render_markdown(self.report), encoding="utf-8")

    def fit(self, kind: str, where: str, result: dict) -> dict:
        if result["steps_done"] < result["steps_requested"]:
            self.cut.append({"what": f"{kind} ({where})", "steps_done": result["steps_done"], "steps_requested": result["steps_requested"]})
        return result

    def skip(self, what: str, why: str) -> None:
        self.report["skipped"].append({"what": what, "why": why, "invocation": len(self.report["invocations"])})
        self.log("skipped", what=what, why=why)

    def start(self) -> None:
        """Fresh run, or continue a run of the same id: its report, saved worlds, strategy and accepted drivers are loaded; recorded steps are not kept
        on disk, so the rover records again as it drives."""
        a = self.a
        if (self.out / "report.json").exists():
            self.report = json.loads((self.out / "report.json").read_text())
            self.strategy = rsi.Strategy(**self.report["rsi"]["strategy_now"])
            done = {f"{k}#r{r['round']}" for k, sit in self.report["situations"].items() for r in sit["rounds"]}
            # a round killed before it reached the report may have left a trace or a checkpoint behind: only recorded rounds count
            self.traces = [t for t in (json.loads(p.read_text()) for p in sorted((self.out / "trace_pool").glob("*.json"))) if t["name"] in done]
            first = self.report["invocations"][0]["protocol"]
            changed = [k for k in PROTOCOL_KEYS if first.get(k) != getattr(a, k)]
            if changed:
                raise SystemExit(f"REFUSED to resume: these settings differ from the run's first invocation: {', '.join(changed)}")
        self.out.mkdir(parents=True, exist_ok=True)
        p = Path(self.start_file).expanduser().resolve() if self.start_file else model_path(a.frozen_name)
        self.frozen, info = load_checkpoint(p, self.device)
        fd = {"name": p.stem if self.start_file else a.frozen_name, "path": str(p), "sha256_before": guard.sha256(p), "checkpoint_id": info.get("checkpoint_id"),
              "trained_on": info.get("trained_on"), "selection": info.get("selection"), "hazard_source_in_training": info.get("hazard_source"),
              "parameters": sum(q.numel() for q in self.frozen.parameters())}
        if self.start_file:
            fd.update(label=a.start_driver_label, description=guard.start_driver_text(a.start_driver_label, fd["sha256_before"]), part_of_main_run_freeze=False,
                      naming="keys and columns named 'gain_vs_frozen_driver' / 'reached_ice_frozen_vs_final' compare with THIS starting driver; "
                             "the names are kept so that runs stay comparable")
        old, was_file = driver_record(self.report), "starting_driver" in self.report
        if old and was_file != bool(self.start_file):
            raise SystemExit(f"REFUSED to resume: this run began from {old.get('description') or 'the frozen driver ' + str(old.get('name'))}, "
                             f"not from {fd.get('description') or 'the frozen driver ' + fd['name']}")
        if old and old["sha256_before"] != fd["sha256_before"]:
            if self.start_file:
                raise SystemExit(f"REFUSED to resume: this run began from a different starting driver file (sha256 {old['sha256_before'][:12]} -> {fd['sha256_before'][:12]})")
            raise SystemExit(f"REFUSED to resume: the frozen driver file changed since this run began ({old['sha256_before'][:12]} -> {fd['sha256_before'][:12]})")
        self.report["starting_driver" if self.start_file else "frozen_driver"] = fd
        self.report["status"] = "running"                                   # only main() sets "finished": a killed run leaves "running" behind
        self.report.setdefault("last_completed_round", None)
        if self.start_file:                                                 # FROZEN.json does not list this file: only whether the main run froze is recorded
            self.report["main_run"] = guard.main_run_state(Path(a.main_run_dir)) if a.main_run_dir else None
            if (self.report["main_run"] or {}).get("note"):
                self.report["main_run"]["note"] = guard.NOT_FROZEN_NOTE_START_FILE
        else:
            self.report["main_run"] = guard.main_run_state(Path(a.main_run_dir), a.frozen_name, p) if a.main_run_dir else None
        self.report["invocations"].append({"started": datetime.now().astimezone().isoformat(timespec="seconds"), "argv": sys.argv[1:], "max_minutes": a.max_minutes,
                                           "commit": git_commit(), "protocol": dict(vars(a)),
                                           "main_run_dir": a.main_run_dir or "NOT GIVEN (--no-main-run): the freeze, folder and simulator checks were skipped"})
        self.log("start", device=str(self.device), max_minutes=a.max_minutes, resumed=len(self.report["invocations"]) > 1,
                 **({"driver": fd["description"]} if self.start_file else {}))

    # ---------------------------------------------------------------------------------------------- one situation: rounds until its time is up
    def situation(self, spec: str, until: float) -> None:
        a = self.a
        key, site, change, date = situation_key(spec, a.adapt_date)
        w0 = self.provider(site, date)
        if w0 is None:
            return self.skip(key, "no tile for this site on this machine")
        w1, cfg1, desc = apply_change(w0, self.cfg, change, a.seed)
        env = RoverEnv(w1, a.n_measure, self.device, cfg1, seed=MEASURE_SEED)
        sit = self.report["situations"].setdefault(key, {"site": site, "site_role": guard.site_role(site), "world": f"{w0.region}@{date}", "change": desc,
                                                         "references": [], "rounds": [], "real_steps": 0, "minutes": 0.0, "stopped": None})
        seen = sit.setdefault("hazard_source", w0.hazard_source)            # what the world really used: a missing checkpoint falls back to the slope rule
        if w0.hazard_source != seen and w0.hazard_source not in seen:
            sit["hazard_source"] = f"CHANGED BETWEEN INVOCATIONS: {seen} | {w0.hazard_source}"
        t_s = time.time()
        frozen_score = score(env, frozen_fn(self.frozen), MEASURE_SEED)
        frozen_returns = frozen_score["returns"]
        if not sit["references"]:
            refs = [(f"{self.base} (never adapts)", None, {}), ("planner naive", planner_fn(turn_back=False), {"knows_the_change": True}),
                    ("planner cautious", planner_fn(turn_back=True), {"knows_the_change": True}),
                    ("planner cautious + wait for light", planner_fn(turn_back=True, wait_for_light=True), {"knows_the_change": True})]
            if change != "none":
                refs.append(("planner cautious, NOT told about the change", StalePlanner(RoverEnv(w0, a.n_measure, self.device, self.cfg, seed=MEASURE_SEED)),
                             {"knows_the_change": False}))
            for name, fn, extra in refs:
                if self.left() <= 0:
                    self.skip(f"{key} reference '{name}'", "time budget used up")
                    continue
                st = frozen_score if fn is None else score(env, fn, MEASURE_SEED)
                sit["references"].append({"driver": name, **{k: st[k] for k in STAT_KEYS}, "rovers": a.n_measure, "claim": guard.claim(site, 0), **extra})
            self.write()
        if sit["rounds"]:
            net, _ = load_dream(self.models / f"{checkpoint_name(site, change, date, len(sit['rounds']))}.pt", self.device)     # the last RECORDED round's driver
        else:
            net = DreamNet(a.latent, a.hidden).to(self.device).eval()        # correction = 0: this IS the frozen driver
        self.replay = Replay()                                               # only what the rover records itself, in this situation
        last_s, made = 0.0, 0
        while True:
            now = time.time()
            if a.max_rounds and made >= a.max_rounds:
                sit["stopped"] = f"--max-rounds {a.max_rounds} reached in this invocation (round {len(sit['rounds'])})"
                break
            if self.left() <= 0 or now + max(1.3 * last_s, 45.0) > until:
                sit["stopped"] = f"time for this situation used up after round {len(sit['rounds'])}"
                break
            if guard.main_runner_now():
                sit["stopped"] = "the main overnight runner or one of its stage processes appeared: stopped at once"
                self.skip(key, sit["stopped"])
                break
            net = self.round(key, site, change, date, sit, env, cfg1, net, frozen_returns, until)
            last_s, made = time.time() - now, made + 1
            sit["minutes"] = round(sit["minutes"] + last_s / 60, 2)
            self.write()
            try:
                draw_curve(self.out, key, sit)
            except Exception as e:                                           # a picture must never end the run (bad backend, full disk, odd value ...)
                self.log("curve_failed", situation=key, round=len(sit["rounds"]), error=f"{type(e).__name__}: {e}")
        self.log("situation_end", situation=key, rounds=len(sit["rounds"]), stopped=sit["stopped"], minutes=round((time.time() - t_s) / 60, 1))
        self.write()

    def round(self, key, site, change, date, sit, env, cfg1, accepted: DreamNet, frozen_returns, until) -> DreamNet:
        a = self.a
        r = len(sit["rounds"]) + 1
        t_r = time.time()
        # Drive: the accepted driver drives in the world (sampled actions, finished rovers restart); its own steps are recorded and counted.
        chunk, act_st = rollout(clone_env(env, a.n_act, ACT_SEED + r), driver_fn(self.frozen, accepted, greedy=False), a.act_steps, True, tag=key)
        self.replay.add(chunk)
        g_env = clone_env(env, a.n_gate, GATE_SEED + r)                      # fresh test starts every round
        reseed(g_env, GATE_SEED + r)
        ch, st_inc = rollout(g_env, driver_fn(self.frozen, accepted), env.rs.max_steps, False, tag=key)
        tests = [ch]                         # test episodes join the logbook only after this round is decided: no candidate trains on its own test starts
        r_inc = st_inc["returns"]
        steps = chunk.steps + ch.steps
        nets: dict[tuple[int, int], DreamNet] = {}
        nodes: list[dict] = []
        node_s = [0.0]

        def run_node(branch: int, attempt: int) -> float:
            nonlocal steps
            t_n = time.time()
            name, anchor = RECIPES[branch]
            parent = accepted if attempt == 0 else nets[(branch, attempt - 1)]
            net = copy.deepcopy(parent)
            where = f"{key} round {r} {name} #{attempt}"
            cap = max(until - t_n, 1.0) / max(a.probes - len(nodes), 1)
            wm = self.fit("world model", where, train_world_model(net, self.replay, a.wm_steps, self.device, self.gen, a.batch, teacher=parent, until=t_n + 0.3 * cap))
            dr = self.fit("imagination", where, dream_improve(net, self.replay, a.dream_steps, self.device, self.gen, a.batch, a.horizon, anchor_weight=anchor,
                                                              until=t_n + 0.75 * cap))
            reseed(g_env, GATE_SEED + r)
            fresh, st = rollout(g_env, driver_fn(self.frozen, net), env.rs.max_steps, False, tag=key)      # unseen by every model of this round
            tests.append(fresh)
            steps += fresh.steps
            gain, se = paired_gain(r_inc, st["returns"])
            nets[(branch, attempt)] = net
            node_s[0] = time.time() - t_n
            nodes.append({"branch": branch, "recipe": name, "attempt": attempt, "score": gain, "paired_se": se, "passes": clear_gain(gain, se, a.gate_se),
                          "test_mean_return": st["mean_return"], "real_steps": fresh.steps, "imagined_return": (dr["last"] or {}).get("dream_return"),
                          "world_model_steps": wm["steps_done"], "imagination_steps": dr["steps_done"], "seconds": round(node_s[0], 1),
                          "world_model_error_on_fresh_steps": {"accepted_model": model_error(accepted, fresh, self.device, cfg1.rover.battery_wh),
                                                               "this_candidate": model_error(net, fresh, self.device, cfg1.rover.battery_wh)}})
            self.log("candidate", situation=key, round=r, recipe=name, attempt=attempt, gain=round(gain, 2), se=round(se, 2), passes=nodes[-1]["passes"])
            return gain

        live = rsi.LiveEnv(len(RECIPES), a.refine, run_node, can_continue=lambda: self.left() > 0 and time.time() + 1.2 * node_s[0] < until)
        strategy = self.strategy
        strategy.solve(live, a.probes)
        for c in tests:
            self.replay.add(c)
        passed = [n for n in nodes if n["passes"]]
        winner = max(passed, key=lambda n: n["score"]) if passed else None
        prev = sit["rounds"][-1] if sit["rounds"] else None
        sit["real_steps"] += steps
        if winner:
            accepted = nets[(winner["branch"], winner["attempt"])]
            st = score(env, driver_fn(self.frozen, accepted), MEASURE_SEED)
            gain, se = paired_gain(frozen_returns, st["returns"])
            measured = {**{k: st[k] for k in STAT_KEYS}, "gain_vs_frozen_driver": gain, "gain_se": se}
        elif prev:
            measured = {k: prev[k] for k in (*STAT_KEYS, "gain_vs_frozen_driver", "gain_se")}        # same driver, same starts: the score is repeated, not re-run
        else:
            ref = sit["references"][0]
            measured = {**{k: ref[k] for k in STAT_KEYS}, "gain_vs_frozen_driver": 0.0, "gain_se": 0.0}   # still exactly the frozen driver
        save_dream(self.models, checkpoint_name(site, change, date, r), accepted, {"run_id": a.run_id, "situation": key, "round": r, "accepted": bool(winner)})
        trace = {"name": f"{key}#r{r}", "branches": [n for n, _ in RECIPES], "max_attempts": a.refine, "budget": a.probes, "strategy": asdict(strategy),
                 "score_is": f"mean return gain over the current driver on the same {a.n_gate} real test starts",
                 "nodes": [{k: n[k] for k in ("branch", "attempt", "score", "paired_se", "passes")} for n in nodes]}
        rsi.save_trace(self.out / "trace_pool", trace)
        self.traces.append(trace)
        sit["rounds"].append({"round": r, "accepted": bool(winner), "new_driver": winner and f"{winner['recipe']} #{winner['attempt']}", **measured,
                              "rovers": a.n_measure, "real_steps_this_round": steps, "real_steps_so_far": sit["real_steps"], "claim": guard.claim(site, sit["real_steps"]),
                              "candidates_tried": len(nodes), "candidates_allowed": a.probes,
                              "acceptance_note": f"best of {len(nodes)} candidates against one {a.gate_se:g}-standard-error threshold on {a.n_gate} test starts: optimistic; "
                                                 "the verdict uses the separate score starts", "strategy": asdict(strategy), "seconds": round(time.time() - t_r, 1),
                              "minutes_into_situation": round(sit["minutes"] + (time.time() - t_r) / 60, 2), "invocation": len(self.report["invocations"]),
                              "driving": {k: v for k, v in act_st.items() if k != "returns"}, "candidates": nodes})
        self.report["last_completed_round"] = f"{key} round {r}"
        self.log("round", situation=key, round=r, accepted=bool(winner), tried=len(nodes), ice=round(measured["reached_ice"], 3),
                 mean_return=round(measured["mean_return"], 1),
                 **{"gain_vs_start" if self.start_file else "gain_vs_frozen": round(measured["gain_vs_frozen_driver"], 2)})
        if not a.no_rsi:
            d = rsi.dream(strategy, self.traces)
            self.report["rsi"]["dream_steps"].append({"after": f"{key} round {r}", **d})
            self.strategy = rsi.Strategy(**d["selected"])
            self.report["rsi"]["strategy_now"] = d["selected"]
        return accepted

    def main(self) -> int:
        self.start()
        torch.manual_seed(self.a.seed)
        self.write()
        specs = list(self.a.adapt)
        for i, spec in enumerate(specs):
            if self.left() < 60:
                self.skip(spec, "time budget used up before this situation")
                continue
            self.situation(spec, time.time() + 0.97 * self.left() / (len(specs) - i))
        self.report["status"] = "finished"
        self.write()
        fd = driver_record(self.report)
        self.log("end", verdicts={k: v["verdict"] for k, v in self.report["verdicts"].items()}, **{f"{self.base.replace(' ', '_')}_unchanged": fd["unchanged"]})
        print(render_markdown(self.report), flush=True)
        return 0 if fd["unchanged"] else 4


CRITERION = ("After the driver is finished and frozen and the training data is cut off, the rover keeps learning by itself from its own driving: round after "
             "round, with no human, each round starting from the last accepted driver and model. MET in a situation only if at least one round was accepted "
             "AND the final driver beats the frozen driver on the score starts by more than 2 standard errors. Otherwise the report says NOT MET.")


def criterion(base: str = "frozen driver") -> str:
    """The acceptance criterion; a starting driver trained after the freeze (--start-driver-file) is never called frozen."""
    if base == "frozen driver":
        return CRITERION
    return CRITERION.replace("the driver is finished and frozen", f"the {base} is finished").replace("frozen driver", base)


def driver_record(rep: dict) -> dict | None:
    """The report's record of the driver the run started from: 'starting_driver' (--start-driver-file) or 'frozen_driver'."""
    return rep.get("starting_driver") or rep.get("frozen_driver")


def base_reference(sit: dict) -> dict | None:
    """The reference row of the driver the run started from (it never adapts)."""
    return next((r for r in sit["references"] if r["driver"].endswith("driver (never adapts)")), None)


def verdict(sit: dict, base: str = "frozen driver") -> dict:
    rounds = sit["rounds"]
    if not rounds:
        return {"verdict": "NO ROUNDS RUN", "rounds": 0}
    last, acc = rounds[-1], sum(r["accepted"] for r in rounds)
    met = acc > 0 and clear_gain(last["gain_vs_frozen_driver"], last["gain_se"])
    worse = last["gain_vs_frozen_driver"] < 0 and clear_gain(-last["gain_vs_frozen_driver"], last["gain_se"])
    return {"verdict": f"MET: self-learning improved on the {base}" if met else
            f"NOT MET: the final driver is WORSE than the {base}" if worse else f"NOT MET: no clear improvement over the {base}",
            "rounds": len(rounds), "rounds_accepted": acc, "rounds_rejected": len(rounds) - acc, "final_gain_vs_frozen_driver": last["gain_vs_frozen_driver"],
            "final_gain_se": last["gain_se"], "real_steps": sit["real_steps"],
            "reached_ice_frozen_vs_final": [(base_reference(sit) or {}).get("reached_ice"), last["reached_ice"]],
            "measured_by": "mean episode return (the reward the driver was trained on); read the ice and home columns too: return can rise from fewer "
                           "blocked moves alone"}


def git_commit(root: Path | None = None) -> str | None:
    """Commit of this code. A copy without `.git` (the mini) reads DREAM_COMMIT.txt, written by the copy step; git is not asked there, because it
    would answer with the commit of whatever repository encloses the folder."""
    root = Path(root) if root else repo_root()
    if not (root / ".git").exists():
        try:
            return f"{(root / 'DREAM_COMMIT.txt').read_text(encoding='utf-8').split()[0]} (from DREAM_COMMIT.txt)"
        except (OSError, IndexError):
            return None
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def draw_curve(out: Path, key: str, sit: dict) -> None:
    """Per-round curve (best effort: skipped if matplotlib is missing; the caller logs any other failure and goes on). Round 0 is the frozen driver."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ref = {r["driver"]: r for r in sit["references"]}
    frozen = base_reference(sit)
    if not frozen or not sit["rounds"]:
        return
    xs = [0] + [r["round"] for r in sit["rounds"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    try:
        for ax, k, title in ((axes[0], "mean_return", "mean episode return"), (axes[1], "reached_ice", "reached ice (share of rovers)")):
            ys = [frozen[k]] + [r[k] for r in sit["rounds"]]
            ax.plot(xs, ys, color="#1f5fbf", lw=2, label="self-learning driver")
            for r in sit["rounds"]:
                ax.scatter([r["round"]], [r[k]], color="#1a8a3a" if r["accepted"] else "#b0b0b0", zorder=3, s=40)
            ax.axhline(frozen[k], color="#c0392b", ls="--", lw=1.5, label=frozen["driver"])
            for name, col in (("planner cautious", "#555555"), ("planner cautious, NOT told about the change", "#999999")):
                if name in ref:
                    ax.axhline(ref[name][k], color=col, ls=":", lw=1.2, label=name)
            ax.set_xlabel("self-learning round (green dot = accepted, grey = rejected)")
            ax.set_title(title)
            ax.set_xticks(xs[:: max(1, len(xs) // 12)])
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=7, loc="best")
        fig.suptitle(f"EXPERIMENTAL: online self-learning on {sit['world']}, change: {sit['change']['name']} (simulation)", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"curve_{key.replace(':', '_')}.png", dpi=130)
    finally:
        plt.close(fig)                       # also after a failure: figures must not pile up round after round


def render_markdown(rep: dict) -> str:
    fd = driver_record(rep) or {}
    start = "starting_driver" in rep                       # --start-driver-file: a driver trained after the freeze, never called frozen
    word = "start" if start else "frozen"
    inv = rep["invocations"]
    out = [f"# The rover keeps learning after training ends: run {rep['run_id']} (EXPERIMENTAL ADDENDUM)", ""]
    if (rep.get("main_run") or {}).get("note"):
        out += [f"**WARNING: {rep['main_run']['note']}.**", ""]
    last = rep.get("last_completed_round") or "no round yet"
    if rep.get("status") == "finished":
        out += [f"Status: finished (the run ended by itself; last completed round: {last}).", ""]
    else:
        out += [f"**Status: RUNNING, or CUT SHORT if the process is no longer alive.** This file was last written after: {last}. A run killed by the hard "
                "stop (or any other kill) was cut short at that round; everything below is complete only up to it.", ""]
    proto = (inv[0].get("protocol") or {}) if inv else {}
    out += [rep["note"], "",
            f"**How to read accepted gains.** Inside a round, acceptance is best-of-N: up to {proto.get('probes', 'N')} candidates are tested on the same "
            f"{proto.get('n_gate', '')} test starts and the best one that passes is kept, so the gains that got a candidate accepted are optimistic. The honest "
            f"number is the 'gain vs {word}' column, measured on the separate score starts.", "",
           f"**Acceptance criterion.** {rep['acceptance_criterion']}", "",
           (f"This run's {fd.get('description')}. It is NOT part of the main run's freeze. File `{fd.get('path')}`" if start else
            f"Started from {'the trained, NOT frozen driver' if (rep.get('main_run') or {}).get('note') else 'frozen driver'} `{fd.get('name')}`") + f" ({fd.get('parameters')} parameters, id {fd.get('checkpoint_id')}); file unchanged by this run: "
           f"{fd.get('unchanged', 'not checked yet')}. Invocations: {len(inv)} ({', '.join(str(i.get('minutes_used', '?')) for i in inv)} min), device {rep['device']}, "
           f"dream code commit {', '.join(dict.fromkeys(str(i.get('commit') or 'NOT RECORDED') for i in inv)) or 'n/a'}."]
    for key, s in rep["situations"].items():
        v = rep["verdicts"].get(key, {})
        out += ["", f"## {s['world']}, change: {s['change']['name']}: online self-learning on that site, not a held-out score", "", s["change"]["what"], "",
                f"Hazard map shown to the rover: {s.get('hazard_source', 'not recorded')} (the driver was trained with: "
                f"{fd.get('hazard_source_in_training') or 'not recorded'}).", "",
                f"**Verdict: {v.get('verdict')}** ({v.get('rounds_accepted', 0)} rounds accepted, {v.get('rounds_rejected', 0)} rejected, {s['real_steps']} real steps of its own driving). "
                f"The verdict is by mean episode return; reached-ice went {' -> '.join(f'{x:.1%}' for x in v.get('reached_ice_frozen_vs_final', []) if x is not None)}.", "",
                f"| round | accepted | ice | home | died | blocked moves / episode | hazard under rover | mean return | gain vs {word} (se) | tried | real steps so far | minutes |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        f = base_reference(s)
        if f:
            out.append(f"| 0 ({'starting' if start else 'frozen'} driver) | n/a | {f['reached_ice']:.1%} | {f['returned']:.1%} | {f['died']:.1%} | {f['blocked_moves_per_episode']:.1f} | "
                       f"{f['mean_hazard_under_rover']:.3f} | {f['mean_return']:.1f} | 0 | 0 | 0 | 0 |")
        out += [f"| {r['round']} | {'YES: ' + r['new_driver'] if r['accepted'] else 'no (driver kept)'} | {r['reached_ice']:.1%} | {r['returned']:.1%} | {r['died']:.1%} | "
                f"{r['blocked_moves_per_episode']:.1f} | {r['mean_hazard_under_rover']:.3f} | {r['mean_return']:.1f} | {r['gain_vs_frozen_driver']:+.1f} ({r['gain_se']:.1f}) | "
                f"{r['candidates_tried']} of {r['candidates_allowed']} | {r['real_steps_so_far']} | {r['minutes_into_situation']} |" for r in s["rounds"]]
        claim = s["rounds"][-1]["claim"] if s["rounds"] else "no self-learning ran"
        out += ["", f"Claim for rounds 1+: {claim}. Round 0 and the references: {s['references'][0]['claim'] if s['references'] else 'n/a'}.", "",
                "| reference (same score starts) | ice | home | died | mean return |", "|---|---|---|---|---|"]
        out += [f"| {r['driver']} | {r['reached_ice']:.1%} | {r['returned']:.1%} | {r['died']:.1%} | {r['mean_return']:.1f} |" for r in s["references"]]
        if s["stopped"]:
            out.append(f"\nStopped: {s['stopped']}.")
    ds = rep["rsi"]["dream_steps"]
    if ds:
        sw = [d for d in ds if not d["kept_incumbent"]]
        out += ["", "## Strategy layer (key-free Dream-RSI-style; a mechanism demo: far too few worlds to prove it helps)",
                f"{len(ds)} dreaming steps over {ds[-1]['worlds_saved']} saved worlds ({ds[-1]['worlds_scored']} with a usable signal); the strategy was changed {len(sw)} times. "
                f"Strategy now: {rep['rsi']['strategy_now']}."]
        out += [f"- after {d['after']}: switched to {d['selected']} (replay score {d['incumbent_reward']:.3f} -> {d['best_reward']:.3f})" for d in sw]
    cuts = [c for i in inv for c in i.get("training_cut_short_by_budget", [])]
    if cuts:
        out += ["", f"## Training cut short by the time budget ({len(cuts)} calls)"] + [f"- {c['what']}: {c['steps_done']} of {c['steps_requested']} steps" for c in cuts[:20]]
    if rep["skipped"]:
        out += ["", "## Skipped (recorded, not hidden)"] + [f"- {s['what']}: {s['why']}" for s in rep["skipped"]]
    return "\n".join(out) + "\n"


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--max-minutes", type=float, required=True, help="wall-clock budget of this invocation; the loop keeps making rounds until it ends")
    ap.add_argument("--resume", action="store_true", help="continue the run of this id (accepted drivers, saved worlds, strategy); without it a used id is refused")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--main-run-dir", default=None, help="folder of the main overnight run: start is refused unless that run has finished and frozen")
    g.add_argument("--no-main-run", action="store_true", help="this machine has no main run folder (laptop tests): only the process check is made; recorded in the report")
    ap.add_argument("--frozen-name", default="rover_driver_mini", help="the finished, frozen driver the rover starts from (opened read-only)")
    ap.add_argument("--start-driver-file", default=None,
                    help="start from this rover checkpoint (.pt) instead of <main run>/data/models/<frozen-name>.pt: a driver trained AFTER the freeze, kept "
                         "outside the main run folder. FROZEN.json does not list it; its sha256 is recorded and it is never called frozen")
    ap.add_argument("--start-driver-label", default=guard.START_DRIVER_LABEL, help="how the report names the --start-driver-file driver")
    ap.add_argument("--hazard-name", default=None, help="hazard checkpoint the simulator shows the rover (same rule as rover.train)")
    ap.add_argument("--adapt", nargs="+", default=["Site23:none", "Site42:blocked_route"],
                    help=f"situations site:change[:date], each gets an equal share of the time; changes: {', '.join(CHANGES)}")
    ap.add_argument("--adapt-date", default="2027-06-01")
    ap.add_argument("--probes", type=int, default=6, help="candidate drivers one round may try (each costs --n-gate real episodes); the strategy may stop earlier")
    ap.add_argument("--refine", type=int, default=2, help="tries per recipe (3 recipes: careful, medium, bold)")
    ap.add_argument("--gate-se", type=float, default=2.0, help="a candidate is accepted only if its paired gain exceeds this many standard errors")
    ap.add_argument("--max-rounds", type=int, default=0, help="rounds per situation in this invocation; 0 = keep going until the time ends (tests and smoke runs set it)")
    ap.add_argument("--no-rsi", action="store_true", help="fallback: keep the default strategy for every round (no strategy dreaming)")
    ap.add_argument("--n-act", type=int, default=128, help="rovers driving at the start of each round")
    ap.add_argument("--act-steps", type=int, default=100)
    ap.add_argument("--wm-steps", type=int, default=800, help="world-model steps per candidate")
    ap.add_argument("--dream-steps", type=int, default=800, help="imagination steps per candidate")
    ap.add_argument("--n-gate", type=int, default=64, help="real test episodes per candidate")
    ap.add_argument("--n-measure", type=int, default=256, help="rovers on the fixed score starts")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--latent", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    for spec in a.adapt:
        parts = spec.split(":")
        if len(parts) not in (2, 3) or parts[1] not in CHANGES:
            ap.error(f"--adapt {spec!r}: expected site:change[:date] with change in {CHANGES}")
    if a.max_minutes <= 0:
        ap.error("--max-minutes must be positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", a.run_id):
        ap.error("--run-id may only contain letters, digits, '_', '.', '-' and must start with a letter or digit (it names a folder under runs/_dream/)")
    if "frozen" in a.start_driver_label.lower() or not a.start_driver_label.strip():
        ap.error("--start-driver-label must not be empty or call the driver frozen: FROZEN.json does not list a --start-driver-file driver")
    if a.probes < 1 or a.refine < 1:
        ap.error("--probes and --refine must be at least 1")
    return a


def main(argv: list[str] | None = None, provider=None) -> int:
    a = parse(argv)
    start = Path(a.start_driver_file).expanduser() if a.start_driver_file else None
    why = guard.refusal_reason(Path(a.main_run_dir) if a.main_run_dir else None, repo_root(), runs=runs_root(), frozen_name=a.frozen_name,
                               checkpoint=model_path(a.frozen_name), start_driver_file=start)
    if why:
        print(f"REFUSED to start: {why}", flush=True)
        return 3
    out = runs_root() / "_dream" / a.run_id
    if out.exists() and not a.resume:
        print(f"REFUSED: run id {a.run_id!r} already exists at {out}. Continue it with --resume or choose a new --run-id; a run is never overwritten.", flush=True)
        return 2
    if start is not None and not start.is_file():
        print(f"REFUSED: the starting driver file {start} does not exist; the self-learning loop starts from it.", flush=True)
        return 5
    if start is None and not model_path(a.frozen_name).exists():
        print(f"REFUSED: the frozen driver {model_path(a.frozen_name)} does not exist; the self-learning loop starts from it.", flush=True)
        return 5
    return Run(a, provider).main()


if __name__ == "__main__":
    raise SystemExit(main())
