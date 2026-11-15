"""Vectorised rover environment (PyTorch, any device; native Windows fine). N rovers drive in parallel on one RoverWorld.

One step = one hour of mission time (`rover_sim.step_s`). Actions (9): 8 compass headings (N, NE, E, SE, S, SW, W, NW) or stop/charge.
Mission: drive to an ice (PSR-edge) target, then return to the lander, before the battery dies. All numbers come from config/mission.yaml
(`rover`, `rover_sim`, `scoring`); terrain and sunlight are real NASA data (see world.py). Kinematic: Basilisk adds the wheel physics.

Observation (dict): map [N,6,crop,crop] = relative height, slope/limit, hazard, PSR, sunlight fraction, Earth-up (north up);
                    vec [N,18] = battery, sun here, Earth here, sun forecast at +1/3/6/12/24/48 h, goal vector (dr, dc, dist), lander vector, phase, time, link.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from contracts.mission import MissionConfig, load_mission
from worldmodel.rover.planner import fields_for
from worldmodel.rover.world import N_HOURS, RoverWorld

N_ACTIONS = 9
MAP_CHANNELS = 6
VEC_DIM = 18
_D = 1 / math.sqrt(2)
DIRS = torch.tensor([(-1.0, 0.0), (-_D, _D), (0.0, 1.0), (_D, _D), (1.0, 0.0), (_D, -_D), (0.0, -1.0), (-_D, -_D)])    # (d_row, d_col), north = -row
FORECAST_H = (1, 3, 6, 12, 24, 48)
LOS_SAMPLES = 16
FIELD_CAP = 5000.0                       # cost-to-go cap where unreachable (m-equivalent)


def _norm(coord: torch.Tensor, size: int) -> torch.Tensor:
    return 2.0 * coord / (size - 1) - 1.0


class RoverEnv:
    def __init__(self, world: RoverWorld, n_envs: int, device: torch.device | None = None, cfg: MissionConfig | None = None, seed: int = 0):
        self.cfg = cfg or load_mission()
        self.rs, self.rv = self.cfg.rover_sim, self.cfg.rover
        self.world, self.n = world, n_envs
        self.device = device or torch.device("cpu")
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        d = self.device
        s = world.size
        self.S, self.Sc = s, world.sun_frac.shape[-1]
        self.stride = self.rs.light_stride_px
        self.crop = self.rs.crop_px
        h = torch.from_numpy(world.height)
        lh = float(world.height[round(world.lander_rc[0]), round(world.lander_rc[1])])
        self.static_map = torch.stack([((h - lh) / 50.0).clamp(-4, 4), (torch.from_numpy(world.slope) / self.rv.max_slope_deg).clamp(0, 2),
                                       torch.from_numpy(world.hazard), torch.from_numpy(world.psr.astype(np.float32))])[None].to(d)    # [1,4,S,S]
        self.height = h[None, None].to(d)
        self.slope = torch.from_numpy(world.slope)[None, None].to(d)
        self.hazard = torch.from_numpy(world.hazard)[None, None].to(d)
        self.sun = torch.from_numpy(world.sun_frac.astype(np.float32))[:, None].to(d)          # [T,1,Sc,Sc]
        self.earth = torch.from_numpy(world.earth_up.astype(np.float32))[:, None].to(d)
        goals = np.vstack([np.array([world.lander_rc]), world.targets_rc])
        f = fields_for(world.slope, world.hazard, self.rv.max_slope_deg, world.mpp, goals)
        self.fields = torch.from_numpy(np.nan_to_num(np.minimum(f, FIELD_CAP), posinf=FIELD_CAP)).to(d)       # [K+1,S,S], index 0 = lander
        self.lander = torch.tensor(world.lander_rc, dtype=torch.float32, device=d)
        self.targets = torch.from_numpy(world.targets_rc.astype(np.float32)).to(d)
        self.dirs = DIRS.to(d)
        mpp = world.mpp
        self.dist_px = self.rv.speed_m_s * self.rs.step_s / mpp
        self.reach_px = self.rs.reach_radius_m / mpp
        self.step_h = self.rs.step_s / 3600.0
        self.w_ice, self.w_return = self.cfg.scoring.w_ice_reached, self.cfg.scoring.w_rover_returned
        off = (torch.arange(self.crop, dtype=torch.float32) - (self.crop - 1) / 2.0)
        self.off_r, self.off_c = torch.meshgrid(off, off, indexing="ij")
        self.off_r, self.off_c = self.off_r.to(d), self.off_c.to(d)
        self.reset()

    # ------------------------------------------------------------------------------------------ sampling helpers
    def _sample_full(self, img: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """img [1,C,S,S] sampled at pos [N,2] (row, col) -> [N,C]."""
        g = torch.stack([_norm(pos[:, 1], self.S), _norm(pos[:, 0], self.S)], -1).reshape(1, -1, 1, 2)
        return F.grid_sample(img, g, mode="bilinear", padding_mode="border", align_corners=True)[0, :, :, 0].T

    def _coarse_coord(self, pos: torch.Tensor) -> torch.Tensor:
        return (pos - (self.stride - 1) / 2.0) / self.stride

    def _sample_coarse(self, img: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """img [N,1,Sc,Sc] (one time slice per env) sampled at pos [N,2] -> [N]."""
        u = self._coarse_coord(pos)
        g = torch.stack([_norm(u[:, 1], self.Sc), _norm(u[:, 0], self.Sc)], -1).reshape(-1, 1, 1, 2)
        return F.grid_sample(img, g, mode="bilinear", padding_mode="border", align_corners=True)[:, 0, 0, 0]

    def _field_at(self, pos: torch.Tensor) -> torch.Tensor:
        """Cost-to-go of each env's CURRENT goal at pos [N,2] (bilinear gather)."""
        fi = torch.where(self.phase == 0, self.target_idx + 1, torch.zeros_like(self.target_idx))
        r = pos[:, 0].clamp(0, self.S - 1.001)
        c = pos[:, 1].clamp(0, self.S - 1.001)
        r0, c0 = r.floor().long(), c.floor().long()
        fr, fc = (r - r0)[:, None].squeeze(1), (c - c0)[:, None].squeeze(1)
        f = self.fields
        v = (f[fi, r0, c0] * (1 - fr) * (1 - fc) + f[fi, r0 + 1, c0] * fr * (1 - fc) + f[fi, r0, c0 + 1] * (1 - fr) * fc + f[fi, r0 + 1, c0 + 1] * fr * fc)
        return v

    def _slice(self, arr: torch.Tensor, t_off: int = 0) -> torch.Tensor:
        return arr[(self.t + t_off).clamp(max=N_HOURS - 1)]

    # ------------------------------------------------------------------------------------------ reset / state
    def reset(self, mask: torch.Tensor | None = None) -> dict:
        d = self.device
        m = torch.ones(self.n, dtype=torch.bool, device=d) if mask is None else mask
        k = int(m.sum())
        if mask is None:
            self.pos = torch.zeros(self.n, 2, device=d)
            self.t = torch.zeros(self.n, dtype=torch.long, device=d)
            self.steps = torch.zeros(self.n, dtype=torch.long, device=d)
            self.battery = torch.zeros(self.n, device=d)
            self.phase = torch.zeros(self.n, dtype=torch.long, device=d)
            self.target_idx = torch.zeros(self.n, dtype=torch.long, device=d)
            self.done = torch.zeros(self.n, dtype=torch.bool, device=d)
            self.ep_return = torch.zeros(self.n, device=d)
        if k:
            ang = torch.rand(k, generator=self.gen).to(d) * 2 * math.pi
            rad = torch.rand(k, generator=self.gen).to(d).sqrt() * self.rs.lander_scatter_m / self.world.mpp
            self.pos[m] = self.lander + torch.stack([-torch.cos(ang), torch.sin(ang)], -1) * rad[:, None]
            hi = max(N_HOURS - 1 - self.rs.max_steps, 1)
            self.t[m] = (torch.rand(k, generator=self.gen) * hi).long().to(d)
            self.steps[m] = 0
            self.battery[m] = self.rv.battery_wh
            self.phase[m] = 0
            self.target_idx[m] = torch.randint(0, len(self.targets), (k,), generator=self.gen).to(d)
            self.done[m] = False
            self.ep_return[m] = 0.0
        return self.observe()

    def goal(self) -> torch.Tensor:
        return torch.where((self.phase == 0)[:, None], self.targets[self.target_idx], self.lander[None].expand(self.n, 2))

    # ------------------------------------------------------------------------------------------ observation
    def link(self) -> torch.Tensor:
        """Rover -> lander relay: within range, terrain line of sight, and the lander sees Earth now."""
        d_m = (self.pos - self.lander).norm(dim=1) * self.world.mpp
        ts = torch.linspace(0, 1, LOS_SAMPLES, device=self.device)
        pts = self.pos[:, None, :] * (1 - ts[None, :, None]) + self.lander[None, None, :] * ts[None, :, None]            # [N,L,2]
        g = torch.stack([_norm(pts[..., 1], self.S), _norm(pts[..., 0], self.S)], -1).reshape(1, -1, 1, 2)
        hh = F.grid_sample(self.height, g, mode="bilinear", padding_mode="border", align_corners=True).reshape(self.n, LOS_SAMPLES)
        ha, hb = hh[:, 0] + self.rs.rover_antenna_height_m, hh[:, -1] + self.rs.lander_antenna_height_m
        line = ha[:, None] * (1 - ts[None]) + hb[:, None] * ts[None]
        los = (hh[:, 1:-1] <= line[:, 1:-1]).all(dim=1)
        earth_l = self._sample_coarse(self._slice(self.earth).expand(-1, 1, -1, -1), self.lander[None].expand(self.n, 2)) > 0.5
        return ((d_m <= self.rs.comms_range_m) & los & earth_l).float()

    def observe(self) -> dict:
        n, c = self.n, self.crop
        rr, cc = self.pos[:, 0, None, None] + self.off_r[None], self.pos[:, 1, None, None] + self.off_c[None]
        g = torch.stack([_norm(cc, self.S), _norm(rr, self.S)], -1).reshape(1, n * c, c, 2)
        stat = F.grid_sample(self.static_map, g, mode="bilinear", padding_mode="border", align_corners=True)[0].reshape(4, n, c, c).permute(1, 0, 2, 3)
        u_r, u_c = self._coarse_coord(rr), self._coarse_coord(cc)
        gc = torch.stack([_norm(u_c, self.Sc), _norm(u_r, self.Sc)], -1)
        dyn = [F.grid_sample(self._slice(a), gc, mode="bilinear", padding_mode="border", align_corners=True) for a in (self.sun, self.earth)]
        map_ = torch.cat([stat, dyn[0], dyn[1]], dim=1)
        sun_here = self._sample_coarse(self._slice(self.sun), self.pos)
        earth_here = self._sample_coarse(self._slice(self.earth), self.pos)
        fc = torch.stack([self._sample_coarse(self._slice(self.sun, h), self.pos) for h in FORECAST_H], dim=1)
        mpp = self.world.mpp
        gv = (self.goal() - self.pos)
        lv = (self.lander[None] - self.pos)
        vec = torch.cat([
            (self.battery / self.rv.battery_wh)[:, None], sun_here[:, None], earth_here[:, None], fc,
            (gv * mpp / 500.0).clamp(-4, 4), (gv.norm(dim=1, keepdim=True) * mpp / 1000.0),
            (lv * mpp / 500.0).clamp(-4, 4), (lv.norm(dim=1, keepdim=True) * mpp / 1000.0),
            self.phase[:, None].float(), (self.steps.float() / self.rs.max_steps)[:, None], self.link()[:, None]], dim=1)
        return {"map": map_, "vec": vec}

    @staticmethod
    def _segment_distance(p0: torch.Tensor, p1: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Closest distance (px) from point g to the straight path p0 -> p1: a target counts as reached if the path passes within reach."""
        v = p1 - p0
        w = g - p0
        tt = ((w * v).sum(1) / (v * v).sum(1).clamp(min=1e-9)).clamp(0, 1)
        return (g - (p0 + v * tt[:, None])).norm(dim=1)

    # ------------------------------------------------------------------------------------------ dynamics
    def step(self, action: torch.Tensor) -> tuple[dict, torch.Tensor, torch.Tensor, dict]:
        """action [N] long in 0..8. Finished envs keep returning done=True until reset(mask)."""
        rw = self.rs.reward
        alive = ~self.done
        move = (action < 8) & alive
        dirs = self.dirs[action.clamp(max=7)]
        mid = self.pos + dirs * self.dist_px * 0.5
        slope_mid = self._sample_full(self.slope, mid)[:, 0]
        factor = (1.0 - slope_mid / self.rv.max_slope_deg).clamp(0.25, 1.0)
        new = self.pos + dirs * (self.dist_px * factor)[:, None]
        slope_new = self._sample_full(self.slope, new)[:, 0]
        inside = (new[:, 0] > 8) & (new[:, 0] < self.S - 9) & (new[:, 1] > 8) & (new[:, 1] < self.S - 9)
        blocked = move & (~inside | (slope_new > self.rv.max_slope_deg))
        moved = move & ~blocked
        old_cost = self._field_at(self.pos)
        old_pos = self.pos
        self.pos = torch.where(moved[:, None], new, self.pos)
        new_cost = self._field_at(self.pos)
        haz = self._sample_full(self.hazard, self.pos)[:, 0]
        sun_here = self._sample_coarse(self._slice(self.sun), self.pos)
        power = self.rv.solar_panel_w * sun_here - torch.where(moved, self.rs.drive_power_w, self.rs.idle_power_w)
        self.battery = torch.where(alive, (self.battery + power * self.step_h).clamp(0, self.rv.battery_wh), self.battery)
        self.t = torch.where(alive, (self.t + 1).clamp(max=N_HOURS - 1), self.t)
        self.steps = torch.where(alive, self.steps + 1, self.steps)

        reward = torch.zeros(self.n, device=self.device)
        reward += alive * rw.step + blocked * rw.blocked + moved * haz * rw.hazard
        reward += (old_cost - new_cost).clamp(-3 * self.dist_px * self.world.mpp, 3 * self.dist_px * self.world.mpp) * rw.progress * moved
        d_goal = self._segment_distance(old_pos, self.pos, self.goal())
        reached_ice = alive & (self.phase == 0) & (d_goal <= self.reach_px)
        reward += reached_ice * self.w_ice
        self.phase = torch.where(reached_ice, torch.ones_like(self.phase), self.phase)
        d_lander_end = (self.lander[None] - self.pos).norm(dim=1)
        d_lander = torch.minimum(d_lander_end, self._segment_distance(old_pos, self.pos, self.lander[None].expand(self.n, 2)))
        returned = alive & (self.phase == 1) & ~reached_ice & (d_lander <= self.reach_px)
        reward += returned * self.w_return
        dead = alive & (self.battery <= 0) & (d_lander_end > self.reach_px)
        reward += dead * rw.death
        timeout = alive & (self.steps >= self.rs.max_steps)
        newly_done = returned | dead | timeout
        self.done = self.done | newly_done
        self.ep_return += reward
        info = {"reached_ice": reached_ice, "returned": returned, "dead": dead, "blocked": blocked, "timeout": timeout, "newly_done": newly_done,
                "battery": self.battery.clone()}
        return self.observe(), reward, self.done.clone(), info

    # ------------------------------------------------------------------------------------------ physics backup policy
    def planner_action(self, turn_back: bool = True, reserve: float = 0.1, wait_for_light: bool = False, light_need: float = 1.5) -> torch.Tensor:
        """Physics route planner (the backup): follow the cost-to-go field of the current goal.
        turn_back=True (cautious): head home when the battery would not cover the trip home at the CURRENT net power (drive load minus the solar income here
        and now) plus a reserve. turn_back=False (naive): never turns back. wait_for_light=True (smartest physics baseline): while still at the lander before the
        outbound leg, stay put unless the sunlight fraction now + at +1 h and +3 h adds up to `light_need` (a simple forecast rule). The learned driver has to beat all three."""
        r = self.pos[:, 0].round().long().clamp(0, self.S - 1)
        c = self.pos[:, 1].round().long().clamp(0, self.S - 1)
        saved = self.phase.clone()
        if turn_back:
            hours_home = self.fields[0][r, c] / (self.rv.speed_m_s * self.rs.step_s)
            sun_here = self._sample_coarse(self._slice(self.sun), self.pos)
            net_w = (self.rs.drive_power_w - self.rv.solar_panel_w * sun_here).clamp(min=0.0)
            abort = (self.phase == 0) & (self.battery - hours_home * net_w < reserve * self.rv.battery_wh)
            self.phase = torch.where(abort, torch.ones_like(self.phase), self.phase)
        cand = self.pos[:, None, :] + self.dirs[None] * self.dist_px                                             # [N,8,2]
        vals = torch.stack([self._field_at(cand[:, k]) for k in range(8)], dim=1)
        here = self._field_at(self.pos)
        self.phase = saved
        best = vals.argmin(dim=1)
        stuck = vals.min(dim=1).values >= here - 1e-3
        if wait_for_light:
            sun_now = self._sample_coarse(self._slice(self.sun), self.pos)
            sun_soon = self._sample_coarse(self._slice(self.sun, 1), self.pos) + self._sample_coarse(self._slice(self.sun, 3), self.pos)
            at_lander = (self.lander[None] - self.pos).norm(dim=1) < 5 * self.reach_px
            stuck = stuck | ((self.phase == 0) & at_lander & (sun_now + sun_soon < light_need))
        return torch.where(stuck, torch.full((self.n,), 8, device=self.device), best)
