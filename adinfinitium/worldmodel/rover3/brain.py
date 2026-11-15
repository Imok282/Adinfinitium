"""Rover v3 brain: inputs -> PLANNER -> EXECUTOR (both scripted physics, no neural network). See ROVER_V3_PLAN.md.

PLANNER  make_plan(): for every rover, roll a fixed list of candidate plans forward to the end of the episode inside the rover's own model (a RoverEnv built
         from the BELIEF world and belief physics) and choose one. Candidates: wait w hours at the lander then go, with or without charging stops; the three
         baseline planners; safe harbour (go to the lander and stay).
EXECUTOR act(): one action per hour from the chosen plan: follow the belief cost-to-go field, never step onto ground sensed as too steep, go round an
         unexpected obstacle (wall following), stop to charge. run_episode() compares real battery / position with the plan's prediction every few steps and
         asks the planner again when they differ.
The real world may differ from the belief (changed worlds); in an unchanged world the belief model is exact.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

from contracts.mission import MissionConfig
from worldmodel.rover.env import RoverEnv, _norm
from worldmodel.rover.world import RoverWorld

STATE = ("pos", "t", "steps", "battery", "phase", "target_idx", "done", "ep_return")
KIND_STRATEGY, KIND_NAIVE, KIND_CAUTIOUS, KIND_WAIT_LIGHT, KIND_SAFE, KIND_SHELTER = 0, 1, 2, 3, 4, 5
NEVER = 100000                                 # "depart" value of a shelter-only plan
SHIFT_GRID = tuple(range(-36, 37))             # hours; forecast time shifts the self-learning layer can fit
BASELINES = {"naive": dict(turn_back=False), "cautious": dict(turn_back=True), "cautious_wait_light": dict(turn_back=True, wait_for_light=True)}
INF = 1e9


class FastEnv(RoverEnv):
    """The simulator with identical physics and starts; only the observation crops (which no scripted layer reads) are skipped."""

    def observe(self) -> dict:
        return {}


@dataclass(frozen=True)
class Theta:
    """Planner / executor parameters. The defaults are the pre-registered deployment values."""
    waits: tuple = (0, 4, 8, 12, 18, 24, 36, 48, 72, 96, 120, 144, 168, 192, 216, 240)
    charge_lo: float = 0.35
    charge_hi: float = 0.95
    margin: float = 0.0
    check_every: int = 6
    bat_tol_wh: float = 3.0
    pos_tol_m: float = 20.0
    wall_max: int = 40
    v31: bool = False                # False = frozen v3. True = v3.1: stay-alive routes + strict-first choice (ROVER_V31_PLAN.md)
    grid_r: int = 12                 # stay-alive search: nodes each side of the rover, one flat hourly move (36 m) apart


@dataclass
class Brain:
    """What the rover believes (map + physics) and how it plans. `env` is the belief model, built once per belief."""
    world: RoverWorld
    cfg: MissionConfig
    theta: Theta
    env: FastEnv | None = None
    note: str = "as deployed"

    def model(self, device) -> FastEnv:
        if self.env is None:
            self.env = FastEnv(self.world, 1, device, self.cfg, seed=0)
        return self.env

    def with_theta(self, note: str, **kw) -> "Brain":
        return Brain(self.world, self.cfg, replace(self.theta, **kw), self.env, note)


def get_state(env: RoverEnv) -> dict:
    return {k: getattr(env, k).clone() for k in STATE}


def set_state(env: RoverEnv, state: dict, rep: int = 1) -> None:
    for k in STATE:
        v = state[k]
        setattr(env, k, (v.repeat_interleave(rep, dim=0) if rep > 1 else v).clone())
    env.n = env.pos.shape[0]


def new_exec_state(n: int, device) -> dict:
    return {"charging": torch.zeros(n, dtype=torch.bool, device=device), "sense": torch.zeros(n, dtype=torch.long, device=device),
            "hit": torch.zeros(n, device=device), "wsteps": torch.zeros(n, dtype=torch.long, device=device),
            "last": torch.full((n,), 8, dtype=torch.long, device=device)}


def candidate_table(theta: Theta, device) -> dict:
    """[C] tensors. Order = tie-break order: (wait, charge off), (wait, charge on) for each wait; the three baselines; safe harbour last."""
    wait, lo, hi, kind = [], [], [], []
    for w in theta.waits:
        for on in (False, True):
            wait.append(w); lo.append(theta.charge_lo if on else -1.0); hi.append(theta.charge_hi); kind.append(KIND_STRATEGY)
    for k in (KIND_NAIVE, KIND_CAUTIOUS, KIND_WAIT_LIGHT, KIND_SAFE):
        wait.append(0); lo.append(-1.0); hi.append(1.0); kind.append(k)
    if theta.v31:                                                                    # v3.1: shelter on lit ground until hour w, then go; shelter only
        for w in theta.waits[1:]:
            for on in (False, True):
                wait.append(w); lo.append(theta.charge_lo if on else -1.0); hi.append(theta.charge_hi); kind.append(KIND_SHELTER)
        wait.append(NEVER); lo.append(-1.0); hi.append(1.0); kind.append(KIND_SHELTER)
    return {"wait": torch.tensor(wait, dtype=torch.long, device=device), "lo": torch.tensor(lo, device=device), "hi": torch.tensor(hi, device=device),
            "kind": torch.tensor(kind, dtype=torch.long, device=device)}


def field_at(env: RoverEnv, fi: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Belief cost-to-go of field index fi [M] at pos [M,2] (bilinear), same gather as RoverEnv._field_at."""
    r = pos[:, 0].clamp(0, env.S - 1.001)
    c = pos[:, 1].clamp(0, env.S - 1.001)
    r0, c0 = r.floor().long(), c.floor().long()
    fr, fc = r - r0, c - c0
    f = env.fields
    return f[fi, r0, c0] * (1 - fr) * (1 - fc) + f[fi, r0 + 1, c0] * fr * (1 - fc) + f[fi, r0, c0 + 1] * (1 - fr) * fc + f[fi, r0 + 1, c0 + 1] * fr * fc


def _first_ok(ok: torch.Tensor, dstar: torch.Tensor, sign: torch.Tensor, last: torch.Tensor):
    """First passable heading in the order dstar, dstar+sign, dstar+2*sign, ...; the reverse of the last move only if nothing else is open."""
    k = torch.arange(8, device=ok.device)
    order = (dstar[:, None] + sign[:, None] * k[None]) % 8
    ok_o = ok.gather(1, order)
    not_rev = (order != ((last + 4) % 8)[:, None]) | (last[:, None] >= 8)
    ok2 = ok_o & not_rev
    use = torch.where(ok2.any(1, keepdim=True), ok2, ok_o)
    r = (use.long() * (8 - k)[None]).argmax(1)
    return order.gather(1, r[:, None])[:, 0], r, use.any(1)


def act(env: RoverEnv, P: dict, X: dict, sense_slope: torch.Tensor, wall_max: int = 40) -> torch.Tensor:
    """EXECUTOR. env holds the rovers' current state and the BELIEF fields / physics; sense_slope [1,1,S,S] is the ground the rover senses one move ahead
    (the real slope when driving for real, the belief slope when the planner imagines). P = plan parameters per rover, X = executor memory (updated)."""
    n, dev = env.n, env.device
    cap, limit = env.rv.battery_wh, env.rv.max_slope_deg
    pos, kind = env.pos, P["kind"]
    alive = ~env.done
    shel = (kind == KIND_SHELTER) & (env.phase == 0) & (env.steps < P["depart"])     # v3.1: following the stay-alive route
    strat = (kind == KIND_STRATEGY) | ((kind == KIND_SHELTER) & ~shel)
    own = strat | (kind == KIND_SAFE) | shel
    go_out = (env.phase == 0) & strat & (env.steps >= P["depart"])
    fi = torch.where(go_out, env.target_idx + 1, torch.zeros_like(env.target_idx))
    goal = torch.where(go_out[:, None], env.targets[env.target_idx], env.lander[None].expand(n, 2))
    waiting = ~go_out & (env.phase == 0)                                             # heading to, or holding at, the lander before / instead of a trip

    dirs = env.dirs
    mid = (pos[:, None, :] + dirs[None] * (env.dist_px * 0.5)).reshape(-1, 2)
    factor = (1.0 - env._sample_full(sense_slope, mid)[:, 0].reshape(n, 8) / limit).clamp(0.25, 1.0)
    new = pos[:, None, :] + dirs[None] * (env.dist_px * factor)[:, :, None]            # where each heading would really take the rover
    flat = new.reshape(-1, 2)
    inside = (new[..., 0] > 8) & (new[..., 0] < env.S - 9) & (new[..., 1] > 8) & (new[..., 1] < env.S - 9)
    ok = inside & (env._sample_full(sense_slope, flat)[:, 0].reshape(n, 8) <= limit)
    vals = field_at(env, fi[:, None].expand(n, 8).reshape(-1), flat).reshape(n, 8)
    here = field_at(env, fi, pos)
    best_v, best = torch.where(ok, vals, torch.full_like(vals, INF)).min(1)
    improving = best_v < here - 1e-3

    # close to the goal: steer by plain distance (the simulator counts a goal as reached when the move's path passes within reach)
    d_goal = (goal - pos).norm(dim=1)
    near = d_goal <= 3 * env.reach_px
    v = new - pos[:, None, :]
    w = (goal - pos)[:, None, :]
    tt = ((w * v).sum(2) / (v * v).sum(2).clamp(min=1e-9)).clamp(0, 1)
    seg_d = (goal[:, None, :] - (pos[:, None, :] + v * tt[:, :, None])).norm(dim=2)
    end_d = (goal[:, None, :] - new).norm(dim=2)
    crit = torch.where(ok, torch.where(waiting[:, None], end_d, seg_d), torch.full_like(seg_d, INF))
    app_v, app = crit.min(1)
    hold = waiting & (d_goal <= 0.95 * env.reach_px)
    app_ok = near & ~hold & (app_v < d_goal - 1e-3)

    # charging stops (strategy plans with charge on)
    sun_here = env._sample_coarse(env._slice(env.sun), pos)
    can = env.rv.solar_panel_w * sun_here > env.rs.idle_power_w + 1.0
    start = (P["lo"] > 0) & (env.battery < P["lo"] * cap) & can
    X["charging"] = (X["charging"] | start) & can & (env.battery < P["hi"] * cap) & strat

    drive = alive & own & ~hold & ~X["charging"] & ~near & ~shel
    dstar = vals.argmin(1)
    in_wall = X["sense"] != 0
    ones = torch.ones(n, dtype=torch.long, device=dev)
    dir_p, r_p, any_p = _first_ok(ok, dstar, ones, X["last"])
    dir_m, r_m, any_m = _first_ok(ok, dstar, -ones, X["last"])
    leave = in_wall & (best_v < X["hit"] - 1e-3)
    normal = drive & ((~in_wall & improving) | leave)
    enter = drive & ~in_wall & ~improving
    plus = (r_p < r_m) | ((r_p == r_m) & (vals.gather(1, dir_p[:, None])[:, 0] <= vals.gather(1, dir_m[:, None])[:, 0]))
    follow = drive & in_wall & ~leave
    dir_f = torch.where(X["sense"] > 0, dir_p, dir_m)

    stop = torch.full((n,), 8, dtype=torch.long, device=dev)
    a = stop.clone()
    a = torch.where(alive & own & ~X["charging"] & app_ok & ~shel, app, a)
    a = torch.where(normal, best, a)
    a = torch.where(enter & any_p, torch.where(plus, dir_p, dir_m), a)
    a = torch.where(follow & any_p, dir_f, a)
    X["hit"] = torch.where(enter, here, X["hit"])
    X["sense"] = torch.where(enter, torch.where(plus, ones, -ones), torch.where(normal | (alive & own & near), torch.zeros_like(ones), X["sense"]))
    X["wsteps"] = torch.where(follow, X["wsteps"] + 1, torch.where(enter | normal, torch.zeros_like(ones), X["wsteps"]))
    give_up = X["wsteps"] > wall_max                                                 # cannot get round: fall back to safe harbour until the next plan
    P["kind"] = torch.where(give_up & strat, torch.full_like(kind, KIND_SAFE), kind)
    if bool(shel.any()):                                                             # steer to this hour's waypoint by plain distance; hold when close
        wp = P["wp"][torch.arange(n, device=dev), env.steps.clamp(max=P["wp"].shape[1] - 1)]
        e_wp = torch.where(ok, (wp[:, None, :] - new).norm(dim=2), torch.full_like(vals, INF))
        bv, bd = e_wp.min(1)
        mv = bv < (wp - pos).norm(dim=1) - 0.3 * env.dist_px
        a = torch.where(shel, torch.where(mv, bd, stop), a)
    X["sense"] = torch.where(give_up, torch.zeros_like(ones), X["sense"])
    X["wsteps"] = torch.where(give_up, torch.zeros_like(ones), X["wsteps"])
    X["last"] = torch.where(alive & own & (a < 8), a, X["last"])

    for k, name in ((KIND_NAIVE, "naive"), (KIND_CAUTIOUS, "cautious"), (KIND_WAIT_LIGHT, "cautious_wait_light")):
        m = kind == k
        if bool(m.any()):
            a = torch.where(m, env.planner_action(**BASELINES[name]), a)
    return torch.where(alive, a, stop)


@dataclass
class Plan:
    cand: torch.Tensor            # [N] chosen candidate index
    P: dict                       # per-rover plan parameters
    pred_bat: torch.Tensor        # [N,T+1] predicted battery at every step count
    pred_pos: torch.Tensor        # [N,T+1,2]
    pred_returned: torch.Tensor   # [N] bool
    pred_ice: torch.Tensor
    pred_dead: torch.Tensor
    any_return: torch.Tensor      # [N] some candidate is predicted to reach ice and return alive
    pred_empty: torch.Tensor      # [N] battery predicted to be empty at some step (strict loss)
    rank: torch.Tensor            # [N] 3 strict return, 2 never empty, 1 v3 rule (always 1 for frozen v3)


def sun_at(env: RoverEnv, tidx: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Belief sunlight fraction at time index tidx [N] and positions pos [N,M,2] -> [N,M] (same bilinear sampling as the simulator)."""
    u = env._coarse_coord(pos)
    g = torch.stack([_norm(u[..., 1], env.Sc), _norm(u[..., 0], env.Sc)], -1)[:, :, None, :]
    return F.grid_sample(env.sun[tidx], g, mode="bilinear", padding_mode="border", align_corners=True)[:, 0, :, 0]


def _shift(x: torch.Tensor, di: int, dj: int, fill) -> torch.Tensor:
    """y[:, i, j] = x[:, i - di, j - dj]; cells with no source get `fill`."""
    y = torch.roll(x, shifts=(di, dj), dims=(1, 2))
    if di == 1:
        y[:, 0, :] = fill
    elif di == -1:
        y[:, -1, :] = fill
    if dj == 1:
        y[:, :, 0] = fill
    elif dj == -1:
        y[:, :, -1] = fill
    return y


MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEG = -1e9


def stay_alive_routes(env: FastEnv, state: dict, theta: Theta, step0: int, ends: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """v3.1 stay-alive search. For every rover: the highest battery it can have at each node of a grid around it at each hour without ever being empty
    (belief forecast and physics; 4 directions; an edge takes 1-4 hours depending on the slope). For every end step in `ends` a route is read backwards:
    waypoints [N,E,T,2] (where to head / hold at each step) and ok [N,E] (a never-empty route to that step exists). The route to an end step before the
    episode end finishes where battery minus the dark-driving energy to the target is highest; the route to the episode end where battery is highest."""
    dev = env.device
    N = state["pos"].shape[0]
    R, T = theta.grid_r, env.rs.max_steps
    L = 2 * R + 1
    cap, limit, h = float(env.rv.battery_wh), env.rv.max_slope_deg, env.step_h
    panel, idle, drive = env.rv.solar_panel_w, env.rs.idle_power_w, env.rs.drive_power_w
    off = torch.arange(-R, R + 1, device=dev, dtype=torch.float32) * env.dist_px
    gi, gj = torch.meshgrid(off, off, indexing="ij")
    grid = torch.stack([gi, gj], -1)                                                  # [L,L,2]
    Pn = state["pos"][:, None, None, :] + grid[None]                                  # [N,L,L,2]
    inside = (Pn[..., 0] > 8) & (Pn[..., 0] < env.S - 9) & (Pn[..., 1] > 8) & (Pn[..., 1] < env.S - 9)
    valid = inside & (env._sample_full(env.slope, Pn.reshape(-1, 2))[:, 0].view(N, L, L) <= limit)
    K = []                                                                            # per move: hours the edge INTO each node takes (0 = no edge)
    for di, dj in MOVES:
        mid = Pn - 0.5 * env.dist_px * torch.tensor([di, dj], device=dev, dtype=torch.float32)
        f = (1.0 - env._sample_full(env.slope, mid.reshape(-1, 2))[:, 0].view(N, L, L) / limit).clamp(0.25, 1.0)
        k = torch.ceil(0.8 / f - 1e-6).clamp(1, 4).long()
        K.append(torch.where(valid & _shift(valid, di, dj, False), k, torch.zeros_like(k)))
    B = torch.full((N, L, L), NEG, device=dev)
    B[:, R, R] = state["battery"]
    hist, gains = [B], []
    BP = torch.zeros((T + 1, N, L, L), dtype=torch.int8, device=dev)
    snap = {}
    Pf = Pn.reshape(N, L * L, 2)
    for s in range(step0, T):
        tidx = (state["t"] + (s - step0)).clamp(max=env.sun.shape[0] - 1)
        g = (panel * h) * sun_at(env, tidx, Pf).view(N, L, L)
        gains = [g] + gains[:3]
        best = torch.where(hist[0] > NEG / 2, (hist[0] + g - idle * h).clamp(max=cap), torch.full_like(g, NEG))
        code = torch.zeros((N, L, L), dtype=torch.int8, device=dev)
        cum = torch.zeros_like(g)
        for k in range(1, len(hist) + 1):
            cum = cum + gains[k - 1]
            src = hist[k - 1]
            for q, (di, dj) in enumerate(MOVES):
                sb = _shift(src, di, dj, NEG)
                c = (sb + cum - drive * h * k).clamp(max=cap)
                good = (K[q] == k) & (sb > NEG / 2) & (sb + gains[k - 1] - drive * h > 1.0)
                upd = good & (c > best)
                best = torch.where(upd, c, best)
                code = torch.where(upd, torch.full_like(code, 1 + q * 4 + (k - 1)), code)
        best = torch.where((best > 1.0) & valid, best, torch.full_like(best, NEG))
        BP[s + 1] = code
        hist = [best] + hist[:3]
        if (s + 1) in ends:
            snap[s + 1] = best
    hours = field_at(env, (state["target_idx"] + 1)[:, None].expand(N, L * L).reshape(-1), Pf.reshape(-1, 2)).view(N, L * L) / (env.dist_px * env.world.mpp)
    rows = torch.arange(N, device=dev)
    wps = torch.zeros((N, len(ends), T, 2), device=dev)
    oks = torch.zeros((N, len(ends)), dtype=torch.bool, device=dev)
    dshift = torch.tensor([m[0] * L + m[1] for m in MOVES], device=dev)
    for e_i, e in enumerate(ends):
        if e not in snap:
            continue
        b = snap[e].view(N, L * L)
        okn = b > NEG / 2
        score = torch.where(okn, b - (drive * h * hours if e < T else 0.0), torch.full_like(b, NEG * 2))
        cur = score.argmax(1)
        oks[:, e_i] = okn.any(1)
        W = torch.zeros((N, T), dtype=torch.long, device=dev)
        W[:, e - 1:] = cur[:, None]
        need = torch.ones(N, dtype=torch.bool, device=dev)
        switch = torch.full((N,), -1, dtype=torch.long, device=dev)
        nxt = cur.clone()
        bp = BP.view(T + 1, N, L * L)
        for s in range(e, step0, -1):
            cd = bp[s][rows, cur].long()
            mvd = cd > 0
            q = (cd - 1).clamp(min=0) // 4
            k = torch.where(mvd, (cd - 1).clamp(min=0) % 4 + 1, torch.ones_like(cd))
            src = torch.where(mvd, cur - dshift[q], cur)
            switch = torch.where(need, s - k, switch)
            nxt = torch.where(need, src, nxt)
            W[:, s - 1] = cur
            sw = switch == s - 1
            cur = torch.where(sw, nxt, cur)
            need = sw
        wps[:, e_i] = Pf[rows[:, None], W]
    return wps, oks


def _rollout(env: FastEnv, state: dict, X: dict, theta: Theta, PP: dict, step0: int) -> dict:
    """Roll candidate plans forward for every rover inside the belief model. PP: per (rover, candidate) plan parameters [N,c] (wp [N,c,T,2] or None).
    Returns [N,c] outcomes and traces."""
    dev = env.device
    N, c = PP["kind"].shape
    set_state(env, state, rep=c)
    env.ep_return = torch.zeros(N * c, device=dev)
    T = env.rs.max_steps
    P = {k: PP[k].reshape(N * c).clone() for k in ("depart", "lo", "hi", "kind")}
    if PP.get("wp") is not None:
        P["wp"] = PP["wp"].reshape(N * c, T, 2)
    Xr = {k: v.repeat_interleave(c, dim=0).clone() for k, v in X.items()}
    bat = env.battery[:, None].repeat(1, T + 1)
    posr = env.pos[:, None, :].repeat(1, T + 1, 1)
    minbat = torch.full((N * c,), float(env.rv.battery_wh), device=dev)
    minall = minbat.clone()
    returned = torch.zeros(N * c, dtype=torch.bool, device=dev)
    dead = returned.clone()
    for s in range(step0, T):
        a = act(env, P, Xr, env.slope, theta.wall_max)
        alive0 = ~env.done
        _, _, done, info = env.step(a)
        returned |= info["returned"]
        dead |= info["dead"]
        away = (env.lander[None] - env.pos).norm(dim=1) > env.reach_px
        minbat = torch.where(away, torch.minimum(minbat, env.battery), minbat)
        minall = torch.where(alive0, torch.minimum(minall, env.battery), minall)
        bat[:, s + 1] = env.battery
        posr[:, s + 1] = env.pos
        if bool(done.all()):
            bat[:, s + 2:] = env.battery[:, None]
            posr[:, s + 2:] = env.pos[:, None, :]
            break
    return {"returned": returned.view(N, c), "dead": dead.view(N, c), "ice": (env.phase == 1).view(N, c), "minbat": minbat.view(N, c),
            "minall": minall.view(N, c), "ep_return": env.ep_return.view(N, c), "bat": bat.view(N, c, T + 1), "pos": posr.view(N, c, T + 1, 2)}


def make_plan(env: FastEnv, state: dict, X: dict, theta: Theta, cur: torch.Tensor | None = None) -> Plan:
    """PLANNER. env = the belief model (its state is overwritten). state / X = the rovers' real state and executor memory now."""
    dev = env.device
    cand = candidate_table(theta, dev)
    C, N = len(cand["kind"]), state["pos"].shape[0]
    step0 = int(state["steps"].max())
    cap = env.rv.battery_wh
    T = env.rs.max_steps
    PP = {"depart": (step0 + cand["wait"])[None].repeat(N, 1), "lo": cand["lo"][None].repeat(N, 1), "hi": cand["hi"][None].repeat(N, 1),
          "kind": cand["kind"][None].repeat(N, 1)}
    is_base = (cand["kind"] >= KIND_NAIVE) & (cand["kind"] <= KIND_WAIT_LIGHT)
    is_shel = cand["kind"] == KIND_SHELTER
    groups = [torch.nonzero(~is_base & ~is_shel)[:, 0], torch.nonzero(is_base)[:, 0]]  # the baseline planners are rolled in their own small batch (speed only)
    wp_shel = None
    if bool(is_shel.any()):                                                           # v3.1: stay-alive routes for the shelter candidates
        gs = torch.nonzero(is_shel)[:, 0]
        ends = sorted({min(step0 + int(w), T) for w in cand["wait"][gs].tolist()})
        wps, oks = stay_alive_routes(env, state, theta, step0, ends)
        e_of = torch.tensor([ends.index(min(step0 + int(w), T)) for w in cand["wait"][gs].tolist()], device=dev)
        wp_shel = wps[:, e_of]                                                        # [N,len(gs),T,2]
        ok = oks[:, e_of] & (((step0 + cand["wait"][gs]) < T) | (cand["wait"][gs] >= NEVER))[None]
        PP["kind"][:, gs] = torch.where(ok, PP["kind"][:, gs], torch.full_like(PP["kind"][:, gs], KIND_SAFE))   # no route: same as safe harbour
        groups.append(gs)
    parts = []
    for gi, g in enumerate(groups):
        sub = {k: v[:, g] for k, v in PP.items()}
        sub["wp"] = wp_shel if gi == 2 else None
        parts.append(_rollout(env, state, X, theta, sub, step0))
    order = torch.argsort(torch.cat(groups))

    def full(k):
        return torch.cat([p[k] for p in parts], dim=1)[:, order]

    ret, dd = full("returned").cpu(), full("dead").cpu()
    ice = full("ice").cpu() & ~dd
    is_safe = (cand["kind"] == KIND_SAFE).cpu()[None]
    feas = (~dd & (full("minbat").cpu() >= theta.margin * cap - 1e-4)) | is_safe
    key = ret.double() * 1e4 + ice.double() * 1e2 + full("ep_return").cpu().double() / 1e3 - torch.arange(C).double()[None] * 1e-9
    minall = full("minall").cpu()
    never_empty = ~dd & (minall > 0) & (minall >= theta.margin * cap - 1e-4)
    rank = torch.ones(N, C, dtype=torch.long)
    if theta.v31:                                                                     # strict first: strict return > never empty > v3's rule
        rank = 1 + never_empty.long() + (never_empty & ret).long()
        key = key + rank.double() * 1e6
    rows = torch.arange(N)
    if cur is not None:
        key[rows, cur.cpu()] += 1e-6
    key[~feas] = -float("inf")
    choice = key.argmax(1)
    ch = choice.to(dev)
    rd = rows.to(dev)
    Pn = {k: PP[k][rd, ch].clone() for k in ("depart", "lo", "hi", "kind")}
    if theta.v31:
        Pn["wp"] = torch.zeros((N, T, 2), device=dev)
        pos_in = torch.full((C,), -1, dtype=torch.long, device=dev)
        pos_in[groups[2]] = torch.arange(len(groups[2]), device=dev)
        m = pos_in[ch] >= 0
        if bool(m.any()):
            Pn["wp"][m] = wp_shel[rd[m], pos_in[ch][m]]
    return Plan(ch, Pn, full("bat")[rd, ch], full("pos")[rd, ch], ret[rows, choice].to(dev), ice[rows, choice].to(dev), dd[rows, choice].to(dev),
                (ret & ~dd).any(1).to(dev), (minall[rows, choice] <= 0).to(dev), rank[rows, choice].to(dev))


def strict_score(real: RoverEnv, ret: torch.Tensor, dead: torch.Tensor, empty: torch.Tensor) -> torch.Tensor:
    """Per-episode score under the strict rule: simulator reward, plus the simulator's own death penalty for a battery that was empty within reach of the
    lander (not otherwise penalised), minus the return bonus for a return that had an empty battery."""
    return real.ep_return + real.rs.reward.death * (empty & ~dead).float() - real.w_return * (ret & empty).float()


def run_episode(real: RoverEnv, brain: Brain, record: bool = False) -> dict:
    """Drive one batch of real episodes with the v3 / v3.1 brain. `real` is a freshly reset environment on the TRUE world."""
    dev = real.device
    b = brain.model(dev)
    th = brain.theta
    n = real.n
    X = new_exec_state(n, dev)
    plan = make_plan(b, get_state(real), X, th)
    first = plan
    P, cand = {k: v.clone() for k, v in plan.P.items()}, plan.cand.clone()
    pred_bat, pred_pos = plan.pred_bat.clone(), plan.pred_pos.clone()
    ice = torch.zeros(n, dtype=torch.bool, device=dev)
    ret, dead, empty = ice.clone(), ice.clone(), ice.clone()
    replans = replanned_rovers = 0
    rec = None
    if record:
        rec = {"visited": torch.zeros(real.S, real.S, device=dev), "sxy": 0.0, "sxx": 0.0, "samples": 0, "bat_max": float(real.battery.max()),
               "k_sxy": torch.zeros(len(SHIFT_GRID), dtype=torch.float64), "k_sxx": torch.zeros(len(SHIFT_GRID), dtype=torch.float64), "k_syy": 0.0, "k_n": 0}
    cap = real.rv.battery_wh
    for step in range(real.rs.max_steps):
        set_state(b, get_state(real))
        a = act(b, P, X, real.slope, th.wall_max)
        b0, p0, t0, alive0 = real.battery.clone(), real.pos.clone(), real.t.clone(), ~real.done
        _, _, done, info = real.step(a)
        ice |= info["reached_ice"]; ret |= info["returned"]; dead |= info["dead"]
        empty |= alive0 & (real.battery <= 0)
        if rec is not None:
            r = real.pos[:, 0].round().long().clamp(0, real.S - 1)
            c = real.pos[:, 1].round().long().clamp(0, real.S - 1)
            rec["visited"][r[alive0], c[alive0]] = 1.0
            sun = real._sample_coarse(real.sun[t0], real.pos)
            moved = (real.pos - p0).abs().sum(1) > 1e-6
            load = torch.where(moved, torch.full_like(b0, real.rs.drive_power_w), torch.full_like(b0, real.rs.idle_power_w))
            actual = (real.battery - b0) / real.step_h + load
            pred = b.rv.solar_panel_w * sun
            okm = alive0 & (real.battery > 0) & (real.battery < cap) & (pred > 1.0)
            rec["sxy"] += float((actual * pred)[okm].sum()); rec["sxx"] += float((pred * pred)[okm].sum()); rec["samples"] += int(okm.sum())
            rec["bat_max"] = max(rec["bat_max"], float(real.battery.max()))
            oku = alive0 & (real.battery > 0) & (real.battery < cap)                  # v3.1: telemetry for the forecast time-shift fit (belief forecast)
            if bool(oku.any()):
                act_u = actual[oku].double()
                rec["k_syy"] += float((act_u * act_u).sum()); rec["k_n"] += int(oku.sum())
                for ki, k in enumerate(SHIFT_GRID):
                    pk = (b.rv.solar_panel_w * b._sample_coarse(b.sun[(t0 - k).clamp(0, b.sun.shape[0] - 1)], real.pos))[oku].double()
                    rec["k_sxy"][ki] += (act_u * pk).sum(); rec["k_sxx"][ki] += (pk * pk).sum()
        if bool(done.all()):
            break
        s = step + 1
        if s % th.check_every == 0:
            dev_b = (real.battery - pred_bat[:, s]).abs() > th.bat_tol_wh
            dev_p = (real.pos - pred_pos[:, s]).norm(dim=1) * real.world.mpp > th.pos_tol_m
            trig = (dev_b | dev_p) & ~real.done
            if bool(trig.any()):                                                     # only the rovers that left their plan are planned again
                new = make_plan(b, {k: v[trig] for k, v in get_state(real).items()}, {k: v[trig] for k, v in X.items()}, th, cur=cand[trig])
                for k in P:
                    P[k][trig] = new.P[k]
                cand[trig] = new.cand
                pred_bat[trig] = new.pred_bat
                pred_pos[trig] = new.pred_pos
                replans += 1
                replanned_rovers += int(trig.sum())
    is_safe0 = first.P["kind"] == KIND_SAFE
    sret = ret & ~empty
    out = {"n": n, "reached_ice": int(ice.sum()), "returned": int(ret.sum()), "died": int(dead.sum()), "mean_return": float(real.ep_return.mean()),
           "ep_return": real.ep_return.detach().cpu(), "returned_mask": ret.cpu(), "dead_mask": dead.cpu(), "ice_mask": ice.cpu(),
           "kept_at_lander": int((~ice & ~dead).sum()), "chose_safe_harbour": int(is_safe0.sum()), "mission_feasible": first.any_return.cpu(), "replans": replans, "replanned_rovers": replanned_rovers,
           "prediction_matched": int(((first.pred_returned == ret) & (first.pred_dead == dead)).sum()),
           "prediction_matched_strict": int(((first.pred_returned == ret) & (first.pred_dead == dead) & (first.pred_empty == empty)).sum()),
           "first_choice": first.cand.cpu(), "first_rank": first.rank.cpu(),
           "strict_lost": int(empty.sum()), "strict_returned": int(sret.sum()), "strict_lost_mask": empty.cpu(), "strict_returned_mask": sret.cpu(),
           "strict_score": strict_score(real, ret, dead, empty).detach().cpu(), "end_battery": real.battery.detach().cpu(),
           "end_dist_m": ((real.lander[None] - real.pos).norm(dim=1) * real.world.mpp).cpu()}
    if rec is not None:
        seen = torch.nn.functional.max_pool2d(rec["visited"][None, None], kernel_size=real.crop + 1, stride=1, padding=real.crop // 2)[0, 0] > 0
        fk = rec["k_sxy"] / rec["k_sxx"].clamp(min=1e-9)
        sse = (rec["k_syy"] - rec["k_sxy"] * fk).clamp(min=0.0)
        out["record"] = {"seen": seen.cpu().numpy(), "solar_factor": (rec["sxy"] / rec["sxx"]) if rec["sxx"] > 0 else None, "solar_samples": rec["samples"],
                         "battery_seen_max": rec["bat_max"], "shift_sse": sse.tolist(), "shift_factor": fk.tolist(), "shift_samples": rec["k_n"]}
    return out


def run_baseline(real: RoverEnv, name: str, belief_env: RoverEnv | None = None) -> dict:
    """One of the three baseline planners. belief_env=None: the planner reads the real world (as in the main evaluation). With belief_env: the planner
    decides from that (stale) belief while the rover drives in the real world. Also records where and how rovers died."""
    n, dev = real.n, real.device
    ice = torch.zeros(n, dtype=torch.bool, device=dev)
    ret, dead, empty = ice.clone(), ice.clone(), ice.clone()
    d_phase = torch.full((n,), -1, dtype=torch.long, device=dev)
    d_sun = torch.zeros(n, device=dev)
    d_dist = torch.zeros(n, device=dev)
    d_step = torch.zeros(n, dtype=torch.long, device=dev)
    for _ in range(real.rs.max_steps):
        if belief_env is None:
            a = real.planner_action(**BASELINES[name])
        else:
            set_state(belief_env, get_state(real))
            a = belief_env.planner_action(**BASELINES[name])
        ph0, t0, alive0 = real.phase.clone(), real.t.clone(), ~real.done
        _, _, done, info = real.step(a)
        ice |= info["reached_ice"]; ret |= info["returned"]; dead |= info["dead"]
        empty |= alive0 & (real.battery <= 0)
        nd = info["dead"]
        if bool(nd.any()):
            d_phase = torch.where(nd, torch.maximum(ph0, real.phase), d_phase)
            d_sun = torch.where(nd, real._sample_coarse(real.sun[t0], real.pos), d_sun)
            d_dist = torch.where(nd, (real.lander[None] - real.pos).norm(dim=1) * real.world.mpp, d_dist)
            d_step = torch.where(nd, real.steps, d_step)
        if bool(done.all()):
            break
    return {"n": n, "reached_ice": int(ice.sum()), "returned": int(ret.sum()), "died": int(dead.sum()), "mean_return": float(real.ep_return.mean()),
            "ep_return": real.ep_return.detach().cpu(), "returned_mask": ret.cpu(), "dead_mask": dead.cpu(), "ice_mask": ice.cpu(),
            "death_phase": d_phase.cpu(), "death_sun": d_sun.cpu(), "death_dist_m": d_dist.cpu(), "death_step": d_step.cpu(),
            "strict_lost": int(empty.sum()), "strict_returned": int((ret & ~empty).sum()), "strict_lost_mask": empty.cpu(), "strict_returned_mask": (ret & ~empty).cpu()}
