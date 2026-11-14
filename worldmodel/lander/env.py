"""Vectorised PyTorch 3-DOF powered-descent lander (a faithful port of sim_jax/lander_env.py so the pilot can train on a CUDA / MPS GPU without JAX).

Same physics, same 13-number observation (+ optional 8-direction hazard ring), same touchdown rules, same failure injection (engine out, nav drift,
camera blind), same sensor noise. Parity with the JAX environment is tested (tests/test_lander_torch.py) when JAX is installed.
Differences, stated: targets are per-episode (the pilot is trained on many landing cells); `crop()` gives the learned pilot an egocentric camera image.
Frame: tile ENU metres, z up, uniform gravity. Action a in [-1,1]^3 = wanted thrust vector / max total thrust.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from contracts.mission import MissionConfig
from contracts.runs import FAILURE_REASONS
from worldmodel.lander.terrain import TerrainT

OUTCOME = {n: i for i, n in enumerate(FAILURE_REASONS)}
FLYING = -1
OBS_DIM = 13
RING_DIRS = ((0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (1.0, -1.0), (0.0, -1.0), (-1.0, -1.0), (-1.0, 0.0), (-1.0, 1.0))


@dataclass
class EnvParams:
    g: float
    g0: float
    isp: float
    dry_mass: float
    n_engines: int
    engine_max_n: float
    min_throttle: float
    max_tilt_rad: float
    td_max_vz: float
    td_max_vh: float
    td_max_slope: float
    dt: float
    max_time: float
    alt_range: tuple[float, float]
    off_range: tuple[float, float]
    vz_range: tuple[float, float]
    vh_range: tuple[float, float]
    start_mass: float
    start_fuel: float
    p_engine_out: float
    p_nav_drift: float
    nav_drift: float
    p_camera_blind: float
    noise_alt: float
    noise_vel: float
    noise_pos: float
    lookahead: bool = True
    lookahead_m: float = 30.0

    @property
    def obs_dim(self) -> int:
        return OBS_DIM + (len(RING_DIRS) if self.lookahead else 0)

    @property
    def max_total_thrust(self) -> float:
        return self.n_engines * self.engine_max_n

    @property
    def n_steps(self) -> int:
        return round(self.max_time / self.dt)


def make_params(cfg: MissionConfig, failures: bool = True, noise: bool = True, lookahead: bool = True, lookahead_m: float = 30.0) -> EnvParams:
    s, L, f, n = cfg.scenarios.terminal, cfg.lander, cfg.failures, cfg.sensor_noise
    k, m = (1.0 if failures else 0.0), (1.0 if noise else 0.0)
    return EnvParams(
        g=cfg.moon.gravity_m_s2, g0=cfg.moon.g0_m_s2, isp=L.isp_s, dry_mass=L.dry_mass_kg, n_engines=L.engines.count, engine_max_n=L.engines.max_thrust_n,
        min_throttle=L.engines.min_throttle, max_tilt_rad=float(np.deg2rad(L.max_tilt_deg)), td_max_vz=L.touchdown_limits.max_vertical_speed_m_s,
        td_max_vh=L.touchdown_limits.max_horizontal_speed_m_s, td_max_slope=L.touchdown_limits.max_slope_deg, dt=s.dt_s, max_time=s.max_time_s,
        alt_range=tuple(s.start_altitude_m), off_range=tuple(s.start_horizontal_offset_m), vz_range=tuple(s.start_vertical_speed_m_s),
        vh_range=tuple(s.start_horizontal_speed_m_s), start_mass=s.start_mass_kg, start_fuel=s.start_propellant_kg,
        p_engine_out=k * f.engine_out_prob, p_nav_drift=k * f.nav_drift_prob, nav_drift=f.nav_drift_m_s, p_camera_blind=k * f.camera_blind_prob,
        noise_alt=m * n.altimeter_m, noise_vel=m * n.velocimeter_m_s, noise_pos=m * n.position_m, lookahead=lookahead, lookahead_m=lookahead_m)


class LanderBatch:
    """N landers flown at once. `targets` [N,2] (m) may be set per episode; `reset(mask)` restarts only the finished ones."""

    def __init__(self, params: EnvParams, terrain: TerrainT, n: int, device: torch.device | None = None, seed: int = 0, targets: torch.Tensor | None = None):
        self.p, self.terrain, self.n = params, terrain, n
        self.device = device or terrain.device
        self.gen = torch.Generator(device=self.device if self.device.type != "mps" else "cpu")
        self.gen.manual_seed(seed)
        self.default_targets = targets if targets is not None else torch.zeros(n, 2, device=self.device)
        z = lambda *s: torch.zeros(*s, device=self.device)
        self.pos, self.vel, self.nav_bias, self.thrust, self.td = z(n, 3), z(n, 3), z(n, 3), z(n, 3), z(n, 3)
        self.mass, self.t, self.target = z(n), z(n), z(n, 2)
        self.engine_out_t = torch.full((n,), float("inf"), device=self.device)
        self.engines_ok = torch.full((n,), params.n_engines, device=self.device, dtype=torch.long)
        self.camera_ok = torch.ones(n, dtype=torch.bool, device=self.device)
        self.outcome = torch.full((n,), FLYING, device=self.device, dtype=torch.long)
        self.start_mass = z(n)
        self.had_engine_out = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.had_drift = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.reset()

    # ------------------------------------------------------------------ random
    def _u(self, *shape):
        return torch.rand(*shape, generator=self.gen, device=self.gen.device).to(self.device)

    def _normal(self, *shape):
        return torch.randn(*shape, generator=self.gen, device=self.gen.device).to(self.device)

    def _uniform(self, lo, hi, *shape):
        return lo + (hi - lo) * self._u(*shape)

    # -------------------------------------------------------------------- reset
    def reset(self, mask: torch.Tensor | None = None, targets: torch.Tensor | None = None) -> torch.Tensor:
        p, T = self.p, self.terrain
        m = torch.ones(self.n, dtype=torch.bool, device=self.device) if mask is None else mask
        k = int(m.sum())
        if k == 0:
            return self.observe()
        tgt = (targets if targets is not None else self.default_targets)[m] if (targets is None or targets.shape[0] == self.n) else targets
        off = self._uniform(*p.off_range, k, 2)
        xy = tgt + off
        alt = self._uniform(*p.alt_range, k)
        vz = self._uniform(*p.vz_range, k)
        vh = self._uniform(*p.vh_range, k, 2)
        eng_fail = self._u(k) < p.p_engine_out
        eng_t = torch.where(eng_fail, self._u(k) * 0.5 * p.max_time, torch.full((k,), float("inf"), device=self.device))
        drift = self._u(k) < p.p_nav_drift
        d_dir = self._normal(k, 3)
        d_dir[:, 2] = 0.0
        d_dir = d_dir / (d_dir.norm(dim=1, keepdim=True) + 1e-6)
        bias = torch.where(drift[:, None], p.nav_drift * d_dir, torch.zeros_like(d_dir))
        cam = ~(self._u(k) < p.p_camera_blind)
        ground = T.height_at(xy[:, 0], xy[:, 1])
        self.pos[m] = torch.stack([xy[:, 0], xy[:, 1], ground + alt], dim=1)
        self.vel[m] = torch.stack([vh[:, 0], vh[:, 1], vz], dim=1)
        self.mass[m] = p.start_mass
        self.start_mass[m] = p.start_mass
        self.t[m] = 0.0
        self.outcome[m] = FLYING
        self.engines_ok[m] = p.n_engines
        self.engine_out_t[m] = eng_t
        self.nav_bias[m] = bias
        self.camera_ok[m] = cam
        self.target[m] = tgt
        self.thrust[m] = 0.0
        self.td[m] = 0.0
        self.had_engine_out[m] = eng_fail
        self.had_drift[m] = drift
        return self.observe()

    @property
    def done(self) -> torch.Tensor:
        return self.outcome != FLYING

    # -------------------------------------------------------------- observation
    def ring(self) -> torch.Tensor:
        p, T = self.p, self.terrain
        d = torch.tensor(RING_DIRS, device=self.device)
        d = d / d.norm(dim=1, keepdim=True) * p.lookahead_m
        x, y = self.pos[:, 0:1] + d[None, :, 0], self.pos[:, 1:2] + d[None, :, 1]
        flat = (x.reshape(-1), y.reshape(-1))
        haz = T.hazard_at(*flat).reshape(self.n, -1)
        slp = T.slope_at(*flat).reshape(self.n, -1)
        unsafe = torch.maximum(haz, (slp / p.td_max_slope).clamp(0.0, 1.0))
        return unsafe * self.camera_ok.float()[:, None]

    def observe(self) -> torch.Tensor:
        p, T = self.p, self.terrain
        s_pos, s_vel = self.pos, self.vel
        est = s_pos[:, :2] + self.nav_bias[:, :2] * self.t[:, None] + p.noise_pos * self._normal(self.n, 2)
        ground = T.height_at(s_pos[:, 0], s_pos[:, 1])
        alt = s_pos[:, 2] - ground + p.noise_alt * self._normal(self.n)
        vel = s_vel + self.nav_bias + p.noise_vel * self._normal(self.n, 3)
        fuel = (self.mass - p.dry_mass).clamp(min=0.0)
        cam = self.camera_ok.float()
        haz = T.hazard_at(s_pos[:, 0], s_pos[:, 1]) * cam
        slp = T.slope_at(s_pos[:, 0], s_pos[:, 1]) * cam
        d = (est - self.target) / 1000.0
        obs = torch.stack([d[:, 0], d[:, 1], alt / 1000.0, vel[:, 0] / 50.0, vel[:, 1] / 50.0, vel[:, 2] / 50.0, fuel / max(p.start_fuel, 1e-6),
                           self.engines_ok / p.n_engines, self.mass / p.start_mass, haz, slp / 12.0, cam, self.t / p.max_time], dim=1)
        if p.lookahead:
            obs = torch.cat([obs, self.ring()], dim=1)
        return obs.float()

    def crop(self, px: int = 32, span_min: float = 60.0, span_max: float = 640.0) -> torch.Tensor:
        """Camera image for the learned pilot [N,3,px,px]; blind camera -> zeros. The ground footprint grows with altitude."""
        T = self.terrain
        alt = (self.pos[:, 2] - T.height_at(self.pos[:, 0], self.pos[:, 1])).clamp(min=0.0)
        span = (alt + span_min).clamp(span_min, span_max)
        img = T.crop(self.pos[:, 0], self.pos[:, 1], span, px, self.p.td_max_slope)
        return img * self.camera_ok.float()[:, None, None, None]

    # ------------------------------------------------------------------- thrust
    def apply_limits(self, action: torch.Tensor) -> torch.Tensor:
        p = self.p
        cmd = action.clamp(-1.0, 1.0) * p.max_total_thrust
        mag = cmd.norm(dim=1)
        dir_ = cmd / (mag[:, None] + 1e-9)
        cos_max = float(np.cos(p.max_tilt_rad))
        h = dir_[:, :2]
        h_unit = h / (h.norm(dim=1, keepdim=True) + 1e-9)
        cone = torch.cat([h_unit * float(np.sin(p.max_tilt_rad)), torch.full_like(mag[:, None], cos_max)], dim=1)
        dir_ = torch.where((dir_[:, 2] < cos_max)[:, None], cone, dir_)
        avail = self.engines_ok.float() * p.engine_max_n
        floor = p.min_throttle * p.engine_max_n
        mag = torch.where(mag < 0.5 * floor, torch.zeros_like(mag), torch.minimum(mag.clamp(min=floor), avail))
        mag = torch.where(self.mass > p.dry_mass + 1e-3, mag, torch.zeros_like(mag))
        return dir_ * mag[:, None]

    # --------------------------------------------------------------------- step
    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One dt. Returns (obs, reward, just_done). Finished landers are frozen (same as the JAX env)."""
        p, T = self.p, self.terrain
        was_done = self.done
        self.engines_ok = torch.where(self.t >= self.engine_out_t, torch.full_like(self.engines_ok, p.n_engines - 1), torch.full_like(self.engines_ok, p.n_engines))
        thrust = self.apply_limits(action)
        mag = thrust.norm(dim=1)
        dm = torch.minimum(mag / (p.isp * p.g0) * p.dt, (self.mass - p.dry_mass).clamp(min=0.0))
        acc = thrust / self.mass[:, None] + torch.tensor([0.0, 0.0, -p.g], device=self.device)
        vel = self.vel + acc * p.dt
        pos = self.pos + vel * p.dt
        mass = self.mass - dm
        t = self.t + p.dt
        ground = T.height_at(pos[:, 0], pos[:, 1])
        touched = pos[:, 2] <= ground
        vz_td, vh_td = -vel[:, 2], vel[:, :2].norm(dim=1)
        slope_td, haz_td = T.slope_at(pos[:, 0], pos[:, 1]), T.hazard_at(pos[:, 0], pos[:, 1])
        out_of_fuel = mass <= p.dry_mass + 1e-3
        land = torch.full_like(self.outcome, OUTCOME["ok"])
        for cond, name in reversed([(vz_td > p.td_max_vz, "hard_vertical"), (vh_td > p.td_max_vh, "hard_horizontal"),
                                    (slope_td > p.td_max_slope, "steep_slope"), (haz_td >= 0.5, "hazard_cell")]):
            land = torch.where(cond, torch.full_like(land, OUTCOME[name]), land)
        land = torch.where((land != OUTCOME["ok"]) & out_of_fuel, torch.full_like(land, OUTCOME["out_of_fuel"]), land)
        new = torch.full_like(self.outcome, FLYING)
        new = torch.where(t >= p.max_time, torch.full_like(new, OUTCOME["timeout"]), new)
        new = torch.where(~T.in_bounds(pos[:, 0], pos[:, 1]), torch.full_like(new, OUTCOME["out_of_bounds"]), new)
        new = torch.where(touched, land, new)
        pos = torch.cat([pos[:, :2], torch.where(touched, ground, pos[:, 2])[:, None]], dim=1)
        vel = torch.where(touched[:, None], torch.zeros_like(vel), vel)
        td = torch.where(touched[:, None], torch.stack([vz_td, vh_td, slope_td], dim=1), self.td)
        keep = was_done                                   # frozen once done
        k3 = keep[:, None]
        self.pos, self.vel = torch.where(k3, self.pos, pos), torch.where(k3, self.vel, vel)
        self.mass, self.t = torch.where(keep, self.mass, mass), torch.where(keep, self.t, t)
        self.outcome = torch.where(keep, self.outcome, new)
        self.thrust, self.td = torch.where(k3, self.thrust, thrust), torch.where(k3, self.td, td)
        just_done = (~was_done) & self.done
        return self.observe(), self.shaped_reward(just_done), just_done

    # ------------------------------------------------------------------- reward
    def shaped_reward(self, just_done: torch.Tensor) -> torch.Tensor:
        """Terminal only. Success: 1 minus small distance and fuel costs. Failure: negative, gentler the closer to legal."""
        p = self.p
        dist = (self.pos[:, :2] - self.target).norm(dim=1)
        fuel = self.start_mass - self.mass
        ok = self.outcome == OUTCOME["ok"]
        over = (self.td[:, 0] / p.td_max_vz - 1).clamp(min=0) + (self.td[:, 1] / p.td_max_vh - 1).clamp(min=0) + (self.td[:, 2] / p.td_max_slope - 1).clamp(min=0)
        touched_bad = (self.outcome >= OUTCOME["hard_vertical"]) & (self.outcome <= OUTCOME["out_of_fuel"])
        fail = torch.where(touched_bad, -(0.5 + 0.25 * over.clamp(max=3.0)), torch.full_like(over, -1.0))
        r = torch.where(ok, 1.0 - 0.0004 * dist - 0.002 * fuel, fail)
        return torch.where(just_done, r, torch.zeros_like(r))

    def metrics(self) -> dict:
        ok = self.outcome == OUTCOME["ok"]
        return {"success": ok, "outcome": self.outcome, "target_dist_m": (self.pos[:, :2] - self.target).norm(dim=1), "fuel_used_kg": self.start_mass - self.mass,
                "time_s": self.t, "td_vz": self.td[:, 0], "td_vh": self.td[:, 1], "td_slope": self.td[:, 2],
                "engine_failed": self.had_engine_out, "nav_drift": self.had_drift, "camera_blind": ~self.camera_ok}
