"""Experimental dream track (worldmodel/dream): change tests, world model, imagination, keep-the-best gate, safety rails. Synthetic flat world, CPU, seconds."""

import copy
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from contracts.mission import load_mission
from worldmodel.dream import guard, rsi
from worldmodel.dream import run as dream_run
from worldmodel.dream.changes import PANEL_FACTOR, TIGHT_SLOPE_DEG, apply_change
from worldmodel.dream.learn import clear_gain, dream_improve, lambda_returns, model_error, paired_gain, train_world_model
from worldmodel.dream.model import DreamNet, dream_model_path
from worldmodel.dream.replay import Replay, clone_env, reseed, rollout
from worldmodel.rover.env import RoverEnv
from worldmodel.rover.policy import build_model, save_checkpoint
from worldmodel.rover.world import N_HOURS, RoverWorld

CFG = load_mission()
S, SC = 128, 16


def flat_world(region: str = "fixture", sun: float = 1.0) -> RoverWorld:
    return RoverWorld(region, "2027-01-01", 5.0, S, (0, 0), np.zeros((S, S), np.float32), np.zeros((S, S), np.float32), np.zeros((S, S), np.float32), "test",
                      np.zeros((S, S), bool), np.full((N_HOURS, SC, SC), sun, np.float16), np.ones((N_HOURS, SC, SC), np.uint8), ["t"] * N_HOURS, (63.5, 63.5),
                      np.array([(63.5, 110.0), (20.0, 63.5)]), "test")


def planner(env, obs):
    return env.planner_action(turn_back=True)


def test_clone_env_starts_exactly_like_a_fresh_env():
    base = RoverEnv(flat_world(), 4, seed=0)
    a, b = clone_env(base, 6, 11), RoverEnv(flat_world(), 6, seed=11)
    assert torch.equal(a.observe()["vec"], b.observe()["vec"]) and a.n == 6 and base.n == 4 and base.pos.shape[0] == 4


def test_changes_alter_the_physics_and_leave_the_original_untouched():
    w = flat_world()
    slope0 = w.slope.copy()
    _, dusty, d1 = apply_change(w, CFG, "dusty_panels")
    env = RoverEnv(w, 2, cfg=dusty, seed=0)
    env.battery[:] = 50.0
    env.step(torch.full((2,), 8))
    assert torch.allclose(env.battery, torch.full((2,), 50.0 + CFG.rover.solar_panel_w * PANEL_FACTOR - CFG.rover_sim.idle_power_w))
    _, tight, _ = apply_change(w, CFG, "tight_slope")
    assert tight.rover.max_slope_deg == TIGHT_SLOPE_DEG and CFG.rover.max_slope_deg == 20.0
    w2, same, d3 = apply_change(w, CFG, "blocked_route")
    ring = w2.slope > CFG.rover.max_slope_deg
    dist = np.hypot(*(np.argwhere(ring) - 63.5).T) * w.mpp
    assert same is CFG and ring.any() and dist.min() >= 89 and dist.max() <= 136 and np.array_equal(w.slope, slope0)
    assert d3["synthetic"] and "SYNTHETIC" in d3["what"] and d1["visible_to_rover"].startswith("no")
    with pytest.raises(ValueError):
        apply_change(w, CFG, "moonquake")


def test_planner_not_told_about_the_ring_is_blocked_far_more_than_the_told_one():
    w = flat_world()
    w2, _, _ = apply_change(w, CFG, "blocked_route")
    blocked = {}
    for name, fn in (("told", planner), ("stale", dream_run.StalePlanner(RoverEnv(w, 16, seed=3)))):
        env = RoverEnv(w2, 16, seed=3)
        n = 0
        for _ in range(60):
            n += int(env.step(fn(env, None))[3]["blocked"].sum())
        blocked[name] = n
    assert blocked["stale"] > 5 * max(blocked["told"], 1)          # the told planner still clips the ring edge now and then (hourly 36 m moves)


def test_lambda_returns_match_a_hand_calculation():
    rew, disc = torch.tensor([[1.0], [2.0]]), torch.tensor([[0.5], [0.5]])
    val = torch.tensor([[10.0], [20.0], [40.0]])
    r1 = 2.0 + 0.5 * 40.0
    assert torch.allclose(lambda_returns(rew, disc, val, 1.0), torch.tensor([[1.0 + 0.5 * r1], [r1]]))
    assert torch.allclose(lambda_returns(rew, disc, val, 0.0), torch.tensor([[1.0 + 0.5 * 20.0], [r1]]))


@pytest.mark.parametrize("cand, margin, accepted", [
    ([12.0, 13.0, 14.0, 15.0], 2.0, True),      # clearly better on every start
    ([10.0, 11.0, 12.0, 13.0], 2.0, False),     # identical: the current driver stays
    ([8.0, 10.0, 11.0, 12.0], 2.0, False),      # worse
    ([40.0, 0.0, 30.0, -20.0], 2.0, False),     # better on average but far too noisy to trust
    ([11.0, 11.0, 13.0, 14.0], 2.0, True),      # +0.75 with standard error 0.25: three standard errors
    ([11.0, 11.0, 13.0, 14.0], 4.0, False),     # ... which a stricter margin rejects
])
def test_a_candidate_is_accepted_only_on_a_clear_paired_gain(cand, margin, accepted):
    gain, se = paired_gain(torch.tensor([10.0, 11.0, 12.0, 13.0]), torch.tensor(cand))
    assert clear_gain(gain, se, margin) == accepted


def test_before_any_self_learning_the_driver_is_exactly_the_frozen_driver():
    torch.manual_seed(1)
    frozen = build_model("conv", width=8, hidden=16).eval()
    env = RoverEnv(flat_world(), 8, seed=2)
    obs = env.observe()
    act, prior = dream_run.driver_fn(frozen, DreamNet(16, 32))(env, obs)
    assert torch.equal(act, frozen(obs)[0].argmax(1)) and torch.equal(prior, frozen(obs)[0])


@pytest.mark.parametrize("command, runner", [
    ("bash ops/macmini/run_everything.sh", True), ("caffeinate -dimsu bash ops/macmini/run_everything.sh", True),
    ("nohup caffeinate -dimsu python -m ops.overnight.run_all --profile macmini", True), ("/x/.venv/bin/python -s -m ops.overnight.run_all", True),
    ("env X=1 bash run_everything.sh", True), ("bash -x ops/macmini/run_everything.sh", True), ("sh dry_run_everything.sh", False), ("tail -f runs/run_everything.log", False), ("bash -lc cat ops/macmini/run_everything.sh", False),
    ("/bin/zsh -c echo run_everything.sh ops.overnight.run_all", False), ("rg -m ops.overnight.run_all", False), ("python -m worldmodel.dream.run", False),
])
def test_guard_recognises_the_real_runner_only(command, runner):
    assert guard.is_main_runner(command) == runner


@pytest.mark.parametrize("command, stage", [
    ("/main/.venv/bin/python -s -m worldmodel.rover.train --arch former", True), ("caffeinate -dimsu python3 -m worldmodel.hazard.evaluate --name x", True),
    ("python -m worldmodel.light.make_points --dates 2027-01-01", True), ("/main/.venv/bin/python -m worldmodel.dream.run --run-id selflearn1", False),
    ("python -m pytest tests/test_rover_env.py", False), ("rg -m worldmodel.rover.train", False), ("tail -f runs/rover/worldmodel.rover.train.log", False),
    ("/bin/zsh -c python -m worldmodel.rover.train", False), ("python train.py -m worldmodel.dream.learn", False),
])
def test_guard_recognises_an_orphaned_main_stage_but_not_the_dream(command, stage):
    assert guard.is_main_stage(command) == stage


def test_guard_refuses_beside_the_main_run_and_labels_claims(tmp_path):
    ps = ("  101 bash ops/macmini/run_everything.sh\n  102 python -m ops.overnight.run_all\n  103 python -m worldmodel.dream.run\n"
          "  104 /main/.venv/bin/python -s -m worldmodel.rover.train --arch former\n")
    assert [ln.split()[0] for ln in guard.main_runner_lines(ps, own_pid=102)] == ["101", "104"]
    assert "104" in guard.refusal_reason(None, tmp_path, "  104 python -m worldmodel.rover.evaluate --name rover_driver_mini\n")
    main, repo = tmp_path / "main", tmp_path / "dream"
    assert "does not exist" in guard.refusal_reason(main, repo, "")
    for root in (main, repo):
        (root / "config").mkdir(parents=True)
        (root / "config" / "mission.yaml").write_text("same")
        (root / "sites").mkdir()
        (root / "sites" / "sites.json").write_text("{}")
        (root / "worldmodel" / "rover").mkdir(parents=True)
        (root / "worldmodel" / "rover" / "env.py").write_text("x = 1")
        (root / "worldmodel" / "nn_blocks.py").write_text("y = 1")
    (main / "runs").mkdir()
    assert "still alive" in guard.refusal_reason(main, repo, ps)
    assert "freeze step" in guard.refusal_reason(main, repo, "")
    (main / "runs" / "FROZEN.json").write_text("{}")
    assert guard.refusal_reason(main, repo, "", runs=repo / "runs") is None
    ckpt = tmp_path / "driver.pt"
    ckpt.write_bytes(b"this run's driver")
    assert "stale" in guard.refusal_reason(main, repo, "", frozen_name="drv", checkpoint=ckpt)          # FROZEN.json has no hash for this driver
    (main / "runs" / "FROZEN.json").write_text(json.dumps({"sha256": {"checkpoint/drv": guard.sha256(ckpt)}}))
    assert guard.refusal_reason(main, repo, "", frozen_name="drv", checkpoint=ckpt) is None and guard.main_run_state(main, "drv", ckpt)["frozen"]
    ckpt.write_bytes(b"a retrained driver")
    assert "stale" in guard.refusal_reason(main, repo, "", frozen_name="drv", checkpoint=ckpt)
    (main / "runs" / "VERIFY_FAILED.txt").write_text("verify after FAILED")                           # not frozen: allowed, but said out loud
    state = guard.main_run_state(main, "drv", ckpt)
    assert guard.refusal_reason(main, repo, "", frozen_name="drv", checkpoint=ckpt) is None and not state["frozen"] and "NOT frozen" in state["note"]
    assert "WARNING: main run was NOT frozen" in dream_run.render_markdown({"run_id": "x", "main_run": state, "note": "", "acceptance_criterion": "", "device": "cpu",
                                                                           "invocations": [], "situations": {}, "rsi": {"dream_steps": []}, "skipped": []})
    (main / "runs" / "VERIFY_FAILED.txt").unlink()
    (main / "runs" / "FROZEN.json").write_text("{}")
    assert "output folder sits inside" in guard.refusal_reason(main, repo, "", runs=main / "runs")
    assert "this code sits inside" in guard.refusal_reason(main, main / "sub", "")
    (repo / "worldmodel" / "rover" / "env.py").write_text("x = 2")
    assert "simulator files differ" in guard.refusal_reason(main, repo, "")
    (repo / "worldmodel" / "rover" / "env.py").unlink()                           # missing on one side is a difference too
    assert "simulator files differ" in guard.refusal_reason(main, repo, "")
    (repo / "worldmodel" / "rover" / "env.py").write_text("x = 1")
    for root in (main, repo):                                                     # the mini has no sites/sites.json, only the example: that still starts
        (root / "sites" / "sites.json").unlink()
        (root / "sites" / "sites.example.json").write_text("{}")
    assert guard.refusal_reason(main, repo, "") is None
    (main / "sites" / "sites.json").write_text("{}")                              # on one side only: a real difference
    assert "sites/sites.json" in guard.refusal_reason(main, repo, "")
    (main / "sites" / "sites.json").unlink()
    (main / "runs" / "run_everything.lock").mkdir()
    assert "not finished" in guard.refusal_reason(main, repo, "")
    assert guard.claim("Site23", 0).startswith("held-out") and "NOT a held-out" in guard.claim("Site23", 5)
    assert all("online self-learning on that site" in guard.claim(s, 1) for s in ("Site42", "Site11", "Site01"))
    assert "held-out" not in guard.claim("Site11", 0).replace("not a held-out", "") and guard.claim("Site01", 0).startswith("train site")
    with pytest.raises(ValueError):
        dream_model_path(tmp_path, "rover_driver_mini")
    assert dream_model_path(tmp_path, "rover_dream_base").name == "rover_dream_base.pt"


def test_guard_takes_a_start_driver_file_from_outside_the_freeze_and_refuses_a_missing_inside_or_frozen_one(tmp_path):
    main, repo = tmp_path / "main", tmp_path / "dream"
    for root in (main, repo):
        (root / "config").mkdir(parents=True)
        (root / "config" / "mission.yaml").write_text("same")
        (root / "worldmodel").mkdir()
        (root / "worldmodel" / "nn_blocks.py").write_text("y = 1")
    (main / "runs").mkdir()
    v2, inside = tmp_path / "rover2" / "rover_driver_v2.pt", main / "data" / "models" / "rover_driver_v2.pt"
    for f in (v2, inside):
        f.parent.mkdir(parents=True)
        f.write_bytes(b"trained after the freeze")
    assert "freeze step" in guard.refusal_reason(main, repo, "", start_driver_file=v2)                 # the main run must still have reached its freeze
    (main / "runs" / "FROZEN.json").write_text(json.dumps({"sha256": {"checkpoint/drv": "0" * 64}}))
    assert "stale" in guard.refusal_reason(main, repo, "", frozen_name="drv", checkpoint=v2)           # as a frozen driver it is refused: not in FROZEN.json
    assert guard.refusal_reason(main, repo, "", frozen_name="drv", checkpoint=v2, start_driver_file=v2) is None
    state = guard.start_driver_state(v2, main)
    assert state["refuse"] is None and state["sha256"] == guard.sha256(v2) and len(state["sha256"]) == 64
    assert "does not exist" in guard.refusal_reason(main, repo, "", start_driver_file=tmp_path / "rover2" / "missing.pt")
    assert "does not exist" in guard.refusal_reason(None, repo, "", start_driver_file=tmp_path / "rover2" / "missing.pt")
    assert "inside the main run folder" in guard.refusal_reason(main, repo, "", start_driver_file=inside)
    (main / "runs" / "FROZEN.json").write_text(json.dumps({"sha256": {"checkpoint/drv": guard.sha256(v2)}}))
    assert "copy of checkpoint/drv" in guard.refusal_reason(main, repo, "", start_driver_file=v2)      # a copy of the frozen driver is not 'after the freeze'
    (main / "runs" / "run_everything.lock").write_text("")
    assert "not finished" in guard.refusal_reason(main, repo, "", start_driver_file=v2)                # the other refusals stay


def test_a_run_from_a_start_driver_file_never_calls_it_frozen_and_refuses_to_resume_from_another(tmp_path, monkeypatch):
    monkeypatch.setenv("ADI_RUNS", str(tmp_path / "runs"))
    monkeypatch.setenv("ADI_DATA", str(tmp_path / "data"))
    monkeypatch.setattr(guard, "main_runner_now", lambda: [])
    v2, other = tmp_path / "rover2" / "rover_driver_v2.pt", tmp_path / "rover2" / "other.pt"
    argv = ["--run-id", "v2", "--max-minutes", "5", "--no-main-run", "--start-driver-file", str(v2), "--adapt", "Site23:none", "--max-rounds", "1", "--probes", "1",
            "--n-act", "4", "--act-steps", "10", "--wm-steps", "3", "--dream-steps", "3", "--n-gate", "4", "--n-measure", "4", "--batch", "16", "--horizon", "3",
            "--latent", "16", "--hidden", "32", "--device", "cpu"]
    provider = lambda site, date: flat_world(f"lola5m_{site.lower()}")
    assert dream_run.main(argv, provider) == 3                    # the guard refuses a missing file
    save_checkpoint(v2, build_model("conv", width=8, hidden=16), {"trained_on": ["fixture"]})
    save_checkpoint(other, build_model("conv", width=8, hidden=16), {"trained_on": ["fixture"]})
    assert dream_run.main(argv, provider) == 0
    out = tmp_path / "runs" / "_dream" / "v2"
    named = f"starting driver: rover v2, trained after the freeze, sha256 {guard.sha256(v2)}"
    rep, md, ledger = json.loads((out / "report.json").read_text()), (out / "report.md").read_text(), (out / "ledger.jsonl").read_text()
    assert rep["starting_driver"]["description"] == named and rep["starting_driver"]["unchanged"] and "frozen_driver" not in rep
    assert named in md and named in ledger and "frozen driver" not in md and "frozen driver" not in ledger
    assert "frozen driver" not in json.dumps({k: v for k, v in rep.items() if k != "starting_driver"}).replace("gain_vs_frozen_driver", "")
    with pytest.raises(SystemExit, match="different starting driver"):
        dream_run.main([*argv[:argv.index(str(v2))], str(other), *argv[argv.index(str(v2)) + 1:], "--resume"], provider)
    with pytest.raises(SystemExit, match="began from starting driver: rover v2"):        # nor from a frozen driver under the same run id
        save_checkpoint(tmp_path / "data" / "models" / "rover_driver_mini.pt", build_model("conv", width=8, hidden=16), {})
        dream_run.main([*argv[:argv.index("--start-driver-file")], *argv[argv.index(str(v2)) + 1:], "--resume"], provider)
    with pytest.raises(SystemExit):
        dream_run.parse([*argv, "--start-driver-label", "the frozen driver"])


def test_cache_redirect_never_writes_through_into_the_main_data_folder(tmp_path, monkeypatch):
    import worldmodel.rover.world as w
    main = tmp_path / "data"
    (main / "rover_cache" / "sub").mkdir(parents=True)
    (main / "rover_cache" / "a.npy").write_text("a")
    (main / "rover_cache" / "gone.npy").write_text("g")
    monkeypatch.setattr(w, "data_root", lambda: main)
    cache = tmp_path / "dream_cache"
    with dream_run.private_cache(cache):
        assert w.data_root() == cache and (cache / "rover_cache" / "a.npy").read_text() == "a" and not (cache / "rover_cache" / "sub").exists()
    (main / "rover_cache" / "gone.npy").unlink()
    with dream_run.private_cache(cache):
        (w.data_root() / "rover_cache" / "gone.npy").write_text("new")            # what np.save does on a cache miss
    assert w.data_root() == main and not (main / "rover_cache" / "gone.npy").exists() and (cache / "rover_cache" / "gone.npy").read_text() == "new"


TREE = {"name": "w", "branches": ["careful", "medium", "bold"], "max_attempts": 3,
        "nodes": [{"branch": b, "attempt": a, "score": s} for b, a, s in ((0, 0, -1), (1, 0, 2), (2, 0, .5), (0, 1, -2), (1, 1, 5), (2, 1, 0), (0, 2, -1), (1, 2, 4), (2, 2, 1))]}


def test_replay_reveals_only_recorded_nodes_and_scores_the_default_strategy_by_hand():
    env = rsi.TraceEnv({**TREE, "nodes": TREE["nodes"][:4]})                    # the round stopped after 4 tries: nothing else can be replayed
    rsi.DEFAULT.solve(env, 9)
    assert env.revealed == {(0, 0): -1.0, (1, 0): 2.0, (2, 0): 0.5, (0, 1): -2.0} and env.legal_actions() == [] and env.best_so_far == 2.0
    # default order careful, medium, bold, then refine in turn; budgets 2, 4, 6, 7, 9 reach 0.4, 0.4, 1, 1, 1 of the best gain (5)
    assert rsi.score_strategy(rsi.DEFAULT, [TREE])["reward"] == pytest.approx((0.4 * (9 - 2) + 0.6 * (9 - 6)) / 9)
    early = rsi.TraceEnv(TREE)
    rsi.Strategy(width=2, greedy=True, patience=1, stop_after=2).solve(early, 9)
    assert list(early.revealed) == [(0, 0), (1, 0), (1, 1), (1, 2), (0, 1)]       # refines the better recipe, stops after two tries without a new best


def test_a_live_round_that_closes_mid_choice_ends_cleanly():
    calls = {"n": 0}

    def flip():                                                   # open when the actions are listed, closed when one is taken
        calls["n"] += 1
        return calls["n"] % 2 == 1
    live = rsi.LiveEnv(3, 2, lambda b, a: 1.0, can_continue=flip)
    rsi.DEFAULT.solve(live, 6)
    assert live.budget_spent == 0


def test_dreaming_switches_only_to_a_strictly_better_strategy_and_keeps_the_current_one_without_signal():
    d = rsi.dream(rsi.DEFAULT, [TREE])
    assert not d["kept_incumbent"] and d["best_reward"] > d["incumbent_reward"] and d["variants_replayed"] == 54
    again = rsi.dream(rsi.Strategy(**d["selected"]), [TREE])
    assert again["kept_incumbent"] and again["selected"] == d["selected"]         # a tie keeps the current strategy
    flat = {**TREE, "nodes": [{**n, "score": -abs(n["score"])} for n in TREE["nodes"]]}
    none = rsi.dream(rsi.DEFAULT, [flat])
    assert none["kept_incumbent"] and none["worlds_scored"] == 0 and none["selected"]["width"] == 3


def test_world_model_learns_the_battery_and_dreaming_never_touches_it():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    env = RoverEnv(flat_world(sun=0.0), 32, seed=0)             # dark world: the battery only drains (2 W idle, 12 W driving)
    rnd = lambda e, o: torch.randint(0, 9, (e.n,), generator=gen)
    chunk, st = rollout(env, rnd, 30, reset_done=True, tag="base")
    reseed(env, 5)
    fresh, _ = rollout(env, rnd, 10, reset_done=True)
    rep = Replay()
    rep.add(chunk)
    b = rep.sample(8, 4, gen)
    assert b["map"].shape == (8, 5, 6, 16, 16) and b["act"].shape == (8, 4) and b["prior"].shape == (8, 4, 9) and st["real_steps"] == 32 * 30
    net = DreamNet(32, 64)
    before = model_error(net, fresh, torch.device("cpu"), CFG.rover.battery_wh)["battery_abs_err_wh"]
    out = train_world_model(net, rep, 150, torch.device("cpu"), gen, batch=64)
    after = model_error(net, fresh, torch.device("cpu"), CFG.rover.battery_wh)["battery_abs_err_wh"]
    assert out["last"]["loss"] < out["first"]["loss"] and after < 0.5 * before
    wm0 = [p.clone() for p in net.wm_parameters()]
    delta0, prior0 = copy.deepcopy(net.delta.state_dict()), copy.deepcopy(net.prior.state_dict())
    d = dream_improve(net, rep, 12, torch.device("cpu"), gen, batch=32, horizon=5, critic_warmup=2)
    assert d["steps_done"] == 12 and all(torch.equal(a, b) for a, b in zip(wm0, net.wm_parameters()))
    assert any(not torch.equal(v, net.delta.state_dict()[k]) for k, v in delta0.items())
    assert all(torch.equal(v, net.prior.state_dict()[k]) for k, v in prior0.items())


def test_self_learning_loop_end_to_end_writes_only_dream_names_labels_honestly_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setenv("ADI_RUNS", str(tmp_path / "runs"))
    monkeypatch.setenv("ADI_DATA", str(tmp_path / "data"))
    monkeypatch.setattr(guard, "refusal_reason", lambda *a, **k: None)
    monkeypatch.setattr(guard, "main_runner_now", lambda: [])
    argv = ["--run-id", "t1", "--max-minutes", "5", "--no-main-run", "--frozen-name", "tiny_frozen", "--adapt", "Site23:none", "Site11:blocked_route",
            "--max-rounds", "2", "--probes", "2", "--n-act", "4", "--act-steps", "10", "--wm-steps", "3", "--dream-steps", "3", "--n-gate", "4", "--n-measure", "4",
            "--batch", "16", "--horizon", "3", "--latent", "16", "--hidden", "32", "--device", "cpu"]
    provider = lambda site, date: flat_world(f"lola5m_{site.lower()}")
    assert dream_run.main(argv, provider) == 5                    # no frozen driver: nothing to start from
    frozen = tmp_path / "data" / "models" / "tiny_frozen.pt"
    save_checkpoint(frozen, build_model("conv", width=8, hidden=16), {"trained_on": ["fixture"]})
    before = frozen.read_bytes()
    assert dream_run.main(argv, provider) == 0
    out = tmp_path / "runs" / "_dream" / "t1"
    files = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file())
    assert all(f.startswith("runs/_dream/t1/") or f == "data/models/tiny_frozen.pt" for f in files) and frozen.read_bytes() == before
    assert all(p.name.startswith("rover_dream_") for p in (out / "models").iterdir()) and len(list((out / "trace_pool").glob("*.json"))) == 4
    rep = json.loads((out / "report.json").read_text())
    s23, s11 = rep["situations"]["Site23:none:2027-06-01"], rep["situations"]["Site11:blocked_route:2027-06-01"]
    assert rep["experimental_addendum"] and rep["frozen_driver"]["unchanged"] and "NOT GIVEN" in rep["invocations"][0]["main_run_dir"]
    assert [r["round"] for r in s23["rounds"]] == [1, 2] and s23["references"][0]["claim"].startswith("held-out")
    assert all("online self-learning on that site" in r["claim"] and "NOT a held-out" in r["claim"] for r in s23["rounds"] + s11["rounds"])
    assert s23["rounds"][-1]["real_steps_so_far"] == s23["real_steps"] == sum(r["real_steps_this_round"] for r in s23["rounds"]) > 0
    assert any("NOT told" in r["driver"] for r in s11["references"]) and not any("NOT told" in r["driver"] for r in s23["references"])
    assert set(rep["verdicts"]) == set(rep["situations"]) and all(v["verdict"].split(":")[0] in ("MET", "NOT MET") for v in rep["verdicts"].values())
    for r in s23["rounds"]:                                       # a rejected round keeps the previous score; round 1 rejected = the frozen driver's score
        if not r["accepted"]:
            prev = s23["rounds"][r["round"] - 2] if r["round"] > 1 else s23["references"][0]
            assert r["mean_return"] == prev["mean_return"]
    md = (out / "report.md").read_text()
    assert len(rep["rsi"]["dream_steps"]) == 4 and "MET" in md
    assert s23["hazard_source"] == "test" and "Hazard map shown to the rover: test" in md
    assert rep["status"] == "finished" and "Status: finished" in md and "best-of-N" in md and "(test site)" not in md
    seen = []                                                     # report.md as a kill after each round would leave it (the curve is drawn right after the write)

    def broken_curve(*a, **k):                                    # ... and a plotting failure of any kind is logged, never fatal
        seen.append((out / "report.md").read_text())
        raise RuntimeError("no display")
    monkeypatch.setattr(dream_run, "draw_curve", broken_curve)
    assert dream_run.main(argv, provider) == 2                    # a used run id is never overwritten ...
    assert dream_run.main([*argv, "--resume"], provider) == 0     # ... but can be continued: rounds 3 and 4 start from round 2's accepted driver
    rep2 = json.loads((out / "report.json").read_text())
    assert len(seen) == 4 and all("Status: RUNNING, or CUT SHORT" in m for m in seen) and "last written after: Site23:none:2027-06-01 round 3." in seen[0]
    assert rep2["status"] == "finished" and "Status: finished" in (out / "report.md").read_text()
    assert (out / "ledger.jsonl").read_text().count('"event": "curve_failed"') == 4
    r23 = rep2["situations"]["Site23:none:2027-06-01"]["rounds"]
    assert [r["round"] for r in r23] == [1, 2, 3, 4] and [r["invocation"] for r in r23] == [1, 1, 2, 2] and r23[:2] == s23["rounds"]
    assert len(rep2["invocations"]) == 2 and len(list((out / "trace_pool").glob("*.json"))) == 8
    with pytest.raises(SystemExit, match="n_gate"):               # a resumed run must keep its settings
        dream_run.main([*argv[:argv.index("--n-gate")], "--n-gate", "8", *argv[argv.index("--n-gate") + 2:], "--resume"], provider)
    with pytest.raises(SystemExit):
        dream_run.parse(["--run-id", "../escape", "--max-minutes", "1", "--no-main-run"])
