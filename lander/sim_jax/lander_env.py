"""3-DOF powered-descent lander environment (gymnax-style, pure JAX).

    env = LanderEnv(params)                     # params from make_params(cfg, terrain)
    obs, state = env.reset(key)
    obs, state, reward, done, info = env.step(key, state, action)

Everything is a pure function of (key, state, action) so it jit/vmaps over 10,000 landers.

Frame: tile ENU metres; z = height on the same datum as the tile. Gravity is uniform
(terminal descent). Thrust is a 3-vector — a 3-DOF stand-in for attitude: the thrust may
tilt at most `max_tilt_deg` from vertical (6-DOF is a later upgrade, see ADR-004).

Action: a in [-1, 1]^3 = desired thrust vector / max_total_thrust. The env enforces:
  * magnitude <= healthy engines x max thrust; tiny commands -> engines off;
    non-zero commands below one engine's minimum throttle are raised to that minimum
  * tilt cone, no thrust when propellant is empty
Observation: OBS_NAMES (normalised, noisy, may be biased by injected failures).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from contracts.mission import MissionConfig
from contracts.runs import FAILURE_REASONS
from sim_jax import terrain as tr

OUTCOME = {name: i for i, name in enumerate(FAILURE_REASONS)}
FLYING = -1

OBS_NAMES = (
    "dx_km", "dy_km",            # estimated position minus target (nav drift accumulates here)
    "alt_km",                    # radar altimeter: height above terrain directly below
    "vx_50", "vy_50", "vz_50",   # velocimeter / 50 m/s
    "fuel_frac", "engines_frac", "mass_frac",
    "hazard_below", "slope_below_12", "camera_ok",
    "time_frac",
)
OBS_DIM = len(OBS_NAMES)   # base observation; indices 0..12 NEVER change (classic guidance + Basilisk rely on them)
ACT_DIM = 3

# Optional hazard look-ahead (EnvParams.lookahead, default OFF): appended AFTER the base vector.
# 8 compass directions (ENU) at `lookahead_m` from the point below the lander; each value is
# max(hazard, slope/limit) in 0..1 at that ring point, zeroed when the camera is blind (same
# rule as hazard_below). Lets a pilot see WHERE the safe ground is, not only what is below it.
LOOKAHEAD_DIRS = (("n", 0.0, 1.0), ("ne", 1.0, 1.0), ("e", 1.0, 0.0), ("se", 1.0, -1.0),
                  ("s", 0.0, -1.0), ("sw", -1.0, -1.0), ("w", -1.0, 0.0), ("nw", -1.0, 1.0))
LOOKAHEAD_NAMES = tuple(f"ring_unsafe_{n}" for n, _, _ in LOOKAHEAD_DIRS)


class EnvParams(NamedTuple):
    terrain: tr.TerrainJ
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
    target_xy: tuple[float, float] = (0.0, 0.0)
    lookahead: bool = False       # append LOOKAHEAD_NAMES to the observation (default off)
    lookahead_m: float = 30.0     # ring radius for the look-ahead

    @property
    def obs_dim(self) -> int:
        return OBS_DIM + (len(LOOKAHEAD_NAMES) if self.lookahead else 0)

    @property
    def n_steps(self) -> int:
        return round(self.max_time / self.dt)

    @property
    def max_total_thrust(self) -> float:
        return self.n_engines * self.engine_max_n


def make_params(cfg: MissionConfig, terrain: tr.TerrainJ, target_xy=(0.0, 0.0),
                failures: bool = True, noise: bool = True, lookahead: bool = False,
                lookahead_m: float = 30.0) -> EnvParams:
    s = cfg.scenarios.terminal
    L = cfg.lander
    f = cfg.failures
    n = cfg.sensor_noise
    k = 1.0 if failures else 0.0
    m = 1.0 if noise else 0.0
    return EnvParams(
        terrain=terrain, g=cfg.moon.gravity_m_s2, g0=cfg.moon.g0_m_s2, isp=L.isp_s,
        dry_mass=L.dry_mass_kg, n_engines=L.engines.count, engine_max_n=L.engines.max_thrust_n,
        min_throttle=L.engines.min_throttle, max_tilt_rad=float(jnp.deg2rad(L.max_tilt_deg)),
        td_max_vz=L.touchdown_limits.max_vertical_speed_m_s,
        td_max_vh=L.touchdown_limits.max_horizontal_speed_m_s,
        td_max_slope=L.touchdown_limits.max_slope_deg,
        dt=s.dt_s, max_time=s.max_time_s,
        alt_range=tuple(s.start_altitude_m), off_range=tuple(s.start_horizontal_offset_m),
        vz_range=tuple(s.start_vertical_speed_m_s), vh_range=tuple(s.start_horizontal_speed_m_s),
        start_mass=s.start_mass_kg, start_fuel=s.start_propellant_kg,
        p_engine_out=k * f.engine_out_prob, p_nav_drift=k * f.nav_drift_prob,
        nav_drift=f.nav_drift_m_s, p_camera_blind=k * f.camera_blind_prob,
        noise_alt=m * n.altimeter_m, noise_vel=m * n.velocimeter_m_s, noise_pos=m * n.position_m,
        target_xy=tuple(float(v) for v in target_xy),
        lookahead=lookahead, lookahead_m=lookahead_m,
    )


class EnvState(NamedTuple):
    pos: jnp.ndarray          # [3] m
    vel: jnp.ndarray          # [3] m/s
    mass: jnp.ndarray         # kg
    t: jnp.ndarray            # s
    outcome: jnp.ndarray      # int32, FLYING until done
    engines_ok: jnp.ndarray   # int32
    engine_out_t: jnp.ndarray # s (inf if no failure)
    nav_bias: jnp.ndarray     # [3] m/s velocity bias in the estimate
    camera_ok: jnp.ndarray    # bool
    target: jnp.ndarray       # [2] m
    thrust: jnp.ndarray       # [3] N, last applied
    td: jnp.ndarray           # [3] touchdown vz, vh, slope (0 until landing)

    @property
    def done(self):
        return self.outcome != FLYING


class LanderEnv:
    def __init__(self, params: EnvParams):
        self.params = params

    # ------------------------------------------------------------------ reset
    def reset(self, key):
        p = self.params
        ks = jax.random.split(key, 10)
        tgt = jnp.asarray(p.target_xy, jnp.float32)
        off = jax.random.uniform(ks[0], (2,), minval=p.off_range[0], maxval=p.off_range[1])
        xy = tgt + off
        ground = tr.height_at(p.terrain, xy[0], xy[1])
        alt = jax.random.uniform(ks[1], (), minval=p.alt_range[0], maxval=p.alt_range[1])
        vz = jax.random.uniform(ks[2], (), minval=p.vz_range[0], maxval=p.vz_range[1])
        vh = jax.random.uniform(ks[3], (2,), minval=p.vh_range[0], maxval=p.vh_range[1])

        eng_fail = jax.random.uniform(ks[4]) < p.p_engine_out
        eng_t = jnp.where(eng_fail, jax.random.uniform(ks[5], (), maxval=0.5 * p.max_time), jnp.inf)
        drift = jax.random.uniform(ks[6]) < p.p_nav_drift
        d_dir = jax.random.normal(ks[7], (3,)).at[2].set(0.0)
        d_dir = d_dir / (jnp.linalg.norm(d_dir) + 1e-6)
        bias = jnp.where(drift, p.nav_drift * d_dir, jnp.zeros(3))
        cam_ok = ~(jax.random.uniform(ks[8]) < p.p_camera_blind)

        state = EnvState(
            pos=jnp.array([xy[0], xy[1], ground + alt], jnp.float32),
            vel=jnp.array([vh[0], vh[1], vz], jnp.float32),
            mass=jnp.asarray(p.start_mass, jnp.float32),
            t=jnp.asarray(0.0, jnp.float32),
            outcome=jnp.asarray(FLYING, jnp.int32),
            engines_ok=jnp.asarray(p.n_engines, jnp.int32),
            engine_out_t=jnp.asarray(eng_t, jnp.float32),
            nav_bias=bias.astype(jnp.float32),
            camera_ok=cam_ok,
            target=tgt,
            thrust=jnp.zeros(3, jnp.float32),
            td=jnp.zeros(3, jnp.float32),
        )
        return self.observe(ks[9], state), state

    # ------------------------------------------------------------ observation
    def observe(self, key, s: EnvState):
        p = self.params
        k1, k2, k3 = jax.random.split(key, 3)
        est_pos = s.pos[:2] + s.nav_bias[:2] * s.t + p.noise_pos * jax.random.normal(k1, (2,))
        ground = tr.height_at(p.terrain, s.pos[0], s.pos[1])
        alt = s.pos[2] - ground + p.noise_alt * jax.random.normal(k2, ())
        vel = s.vel + s.nav_bias + p.noise_vel * jax.random.normal(k3, (3,))
        fuel = jnp.maximum(s.mass - p.dry_mass, 0.0)
        cam = s.camera_ok.astype(jnp.float32)
        haz = tr.hazard_at(p.terrain, s.pos[0], s.pos[1]) * cam
        slp = tr.slope_at(p.terrain, s.pos[0], s.pos[1]) * cam
        d = (est_pos - s.target) / 1000.0
        obs = jnp.stack([
            d[0], d[1], alt / 1000.0,
            vel[0] / 50.0, vel[1] / 50.0, vel[2] / 50.0,
            fuel / jnp.maximum(p.start_fuel, 1e-6),
            s.engines_ok / p.n_engines,
            s.mass / p.start_mass,
            haz, slp / 12.0, cam,
            s.t / p.max_time,
        ])
        if p.lookahead:
            obs = jnp.concatenate([obs, self.lookahead_ring(s)])
        return obs.astype(jnp.float32)

    def lookahead_ring(self, s: EnvState):
        """[8] unsafe-ness (0..1) of the ground at lookahead_m in each compass direction."""
        p = self.params
        d = jnp.array([[dx, dy] for _, dx, dy in LOOKAHEAD_DIRS], jnp.float32)
        d = d / jnp.linalg.norm(d, axis=1, keepdims=True) * p.lookahead_m
        x, y = s.pos[0] + d[:, 0], s.pos[1] + d[:, 1]
        haz = jax.vmap(lambda a, b: tr.hazard_at(p.terrain, a, b))(x, y)
        slp = jax.vmap(lambda a, b: tr.slope_at(p.terrain, a, b))(x, y)
        unsafe = jnp.maximum(haz, jnp.clip(slp / p.td_max_slope, 0.0, 1.0))
        return unsafe * s.camera_ok.astype(jnp.float32)

    # --------------------------------------------------------------- thrust
    def apply_limits(self, s: EnvState, action):
        p = self.params
        cmd = jnp.clip(action, -1.0, 1.0) * p.max_total_thrust
        mag = jnp.linalg.norm(cmd)
        dir_ = cmd / (mag + 1e-9)
        # Tilt cone around +z.
        cos_max = jnp.cos(p.max_tilt_rad)
        h = dir_[:2]
        h_norm = jnp.linalg.norm(h)
        h_unit = h / (h_norm + 1e-9)
        cone = jnp.concatenate([h_unit * jnp.sin(p.max_tilt_rad), jnp.array([cos_max])])
        dir_ = jnp.where(dir_[2] < cos_max, cone, dir_)
        # Magnitude: off / min-throttle floor / healthy-engine ceiling.
        avail = s.engines_ok * p.engine_max_n
        floor = p.min_throttle * p.engine_max_n
        mag = jnp.where(mag < 0.5 * floor, 0.0, jnp.clip(mag, floor, avail))
        has_fuel = s.mass > p.dry_mass + 1e-3
        mag = jnp.where(has_fuel, mag, 0.0)
        return dir_ * mag

    # ----------------------------------------------------------------- step
    def step(self, key, s: EnvState, action):
        p = self.params
        engines_ok = jnp.where(s.t >= s.engine_out_t, p.n_engines - 1, p.n_engines).astype(jnp.int32)
        s = s._replace(engines_ok=engines_ok)
        thrust = self.apply_limits(s, action)
        mag = jnp.linalg.norm(thrust)

        dm = jnp.minimum(mag / (p.isp * p.g0) * p.dt, jnp.maximum(s.mass - p.dry_mass, 0.0))
        acc = thrust / s.mass + jnp.array([0.0, 0.0, -p.g])
        vel = s.vel + acc * p.dt                     # semi-implicit Euler
        pos = s.pos + vel * p.dt
        mass = s.mass - dm
        t = s.t + p.dt

        ground = tr.height_at(p.terrain, pos[0], pos[1])
        touched = pos[2] <= ground
        vz_td = -vel[2]
        vh_td = jnp.linalg.norm(vel[:2])
        slope_td = tr.slope_at(p.terrain, pos[0], pos[1])
        haz_td = tr.hazard_at(p.terrain, pos[0], pos[1])
        out_of_fuel = mass <= p.dry_mass + 1e-3

        land_outcome = jnp.select(
            [vz_td > p.td_max_vz, vh_td > p.td_max_vh, slope_td > p.td_max_slope, haz_td >= 0.5],
            [OUTCOME["hard_vertical"], OUTCOME["hard_horizontal"], OUTCOME["steep_slope"],
             OUTCOME["hazard_cell"]],
            default=OUTCOME["ok"],
        )
        land_outcome = jnp.where((land_outcome != OUTCOME["ok"]) & out_of_fuel,
                                 OUTCOME["out_of_fuel"], land_outcome)
        new_outcome = jnp.select(
            [touched, ~tr.in_bounds(p.terrain, pos[0], pos[1]), t >= p.max_time],
            [land_outcome, OUTCOME["out_of_bounds"], OUTCOME["timeout"]],
            default=FLYING,
        ).astype(jnp.int32)

        pos = pos.at[2].set(jnp.where(touched, ground, pos[2]))
        new = EnvState(
            pos=pos, vel=jnp.where(touched, jnp.zeros(3), vel), mass=mass, t=t,
            outcome=new_outcome, engines_ok=engines_ok, engine_out_t=s.engine_out_t,
            nav_bias=s.nav_bias, camera_ok=s.camera_ok, target=s.target, thrust=thrust,
            td=jnp.where(touched, jnp.stack([vz_td, vh_td, slope_td]), s.td),
        )
        # Once done, freeze.
        new = jax.tree_util.tree_map(lambda old, nw: jnp.where(s.done, old, nw), s, new)
        just_done = (~s.done) & new.done
        reward = jnp.where(just_done & (new.outcome == OUTCOME["ok"]), 1.0, 0.0)
        info = {"just_done": just_done}
        return self.observe(key, new), new, reward, new.done, info
