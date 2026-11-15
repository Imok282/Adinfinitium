"""Live pilot adapter: fly the learned lander pilot one control period at a time from an outside simulator (Basilisk).

    pilot = LivePilot.create(cfg, terrain, target_xy, checkpoint)      # checkpoint None or missing -> untrained network
    out = pilot.act(PilotState(t_s, x_m, y_m, alt_m, vel_enu, mass_kg, engines_ok))
    out["thrust_enu_n"]   wanted thrust vector [N] in the local east/north/up frame at the target
    out["classic_enu_n"]  what the classic ZEM/ZEV law alone would command (the pilot adds a residual to it)
    out["trace"]          what the network looked at: attention rollout over the camera patches, the strongest attention chain
                          from the output back to the input tokens, and per-layer activity (for the Mission Control "AI view")

The observation is built exactly like `LanderBatch.observe` (13 sensors + 8-direction hazard ring) but WITHOUT sensor noise: Basilisk gives the
true state. The camera image is `TerrainT.crop` at the lander position. The action is the same residual rule as training:
action = clip(classic + RESIDUAL_SCALE * clamp(mu), -1, 1), thrust = action * max total thrust.

An untrained PilotFormer has a zero last layer, so it flies exactly like the classic law (stated in pilot.py). Its attention is real but
meaningless; `info["trained"]` says which one is running. Physics backup first: callers must fall back to their own classic law when `act`
raises or returns a non-finite thrust.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from contracts.mission import MissionConfig
from worldmodel.lander.env import RING_DIRS, EnvParams, make_params
from worldmodel.lander.guidance import GuidanceParams, classic_action, guidance_params
from worldmodel.lander.pilot import PilotFormer, build_model, load_checkpoint, pilot_features, residual_action
from worldmodel.lander.terrain import TerrainT

CAMERA_SPAN_MIN_M = 60.0   # [m] same defaults as LanderBatch.crop
CAMERA_SPAN_MAX_M = 640.0  # [m]


@dataclass
class PilotState:
    """True lander state in the tile frame of the target (x east, y north, metres from the TARGET, not the tile centre)."""

    t_s: float
    x_m: float
    y_m: float
    alt_m: float
    vel_enu_m_s: tuple[float, float, float]
    mass_kg: float
    engines_ok: int


class LivePilot:
    def __init__(self, model: torch.nn.Module, info: dict, params: EnvParams, gp: GuidanceParams, terrain: TerrainT,
                 target_xy: tuple[float, float], device: torch.device):
        self.model, self.info, self.p, self.gp, self.terrain = model.eval(), info, params, gp, terrain
        self.target = torch.tensor([target_xy], dtype=torch.float32, device=device)
        self.device = device
        self.crop_px = int(getattr(model, "crop", 32))

    @classmethod
    def create(cls, cfg: MissionConfig, terrain: TerrainT, target_xy: tuple[float, float], checkpoint: str | Path | None = None,
               device: torch.device | None = None, seed: int = 0) -> LivePilot:
        dev = device or torch.device("cpu")
        if checkpoint is not None and Path(checkpoint).exists():
            model, info = load_checkpoint(checkpoint, dev)
            info = {**info, "trained": True, "checkpoint": str(checkpoint)}
        else:
            torch.manual_seed(seed)
            model = build_model("former").to(dev)
            info = {"trained": False, "checkpoint": None,
                    "note": "untrained PilotFormer (zero last layer): flies exactly like the classic law until the real checkpoint is loaded"}
        info["parameters"] = int(sum(p.numel() for p in model.parameters()))
        params = make_params(cfg, failures=False, noise=False, lookahead=True)
        return cls(model, info, params, guidance_params(params), terrain, target_xy, dev)

    # ------------------------------------------------------------------ inputs
    def _pos(self, s: PilotState) -> torch.Tensor:
        return self.target + torch.tensor([[s.x_m, s.y_m]], dtype=torch.float32, device=self.device)

    def observe(self, s: PilotState) -> torch.Tensor:
        p, T = self.p, self.terrain
        xy = self._pos(s)
        x, y = xy[:, 0], xy[:, 1]
        fuel = max(s.mass_kg - p.dry_mass, 0.0)
        vel = torch.tensor([s.vel_enu_m_s], dtype=torch.float32, device=self.device)
        d = (xy - self.target) / 1000.0
        one = lambda v: torch.tensor([float(v)], device=self.device)
        obs = torch.stack([d[:, 0], d[:, 1], one(s.alt_m / 1000.0), vel[:, 0] / 50.0, vel[:, 1] / 50.0, vel[:, 2] / 50.0,
                           one(fuel / max(p.start_fuel, 1e-6)), one(s.engines_ok / p.n_engines), one(s.mass_kg / p.start_mass),
                           T.hazard_at(x, y), T.slope_at(x, y) / 12.0, one(1.0), one(s.t_s / p.max_time)], dim=1)
        dirs = torch.tensor(RING_DIRS, device=self.device)
        dirs = dirs / dirs.norm(dim=1, keepdim=True) * p.lookahead_m
        rx, ry = x[:, None] + dirs[None, :, 0], y[:, None] + dirs[None, :, 1]
        unsafe = torch.maximum(T.hazard_at(rx.reshape(-1), ry.reshape(-1)), (T.slope_at(rx.reshape(-1), ry.reshape(-1)) / p.td_max_slope).clamp(0.0, 1.0))
        return torch.cat([obs, unsafe.reshape(1, -1)], dim=1).float()

    def camera(self, s: PilotState) -> torch.Tensor:
        xy = self._pos(s)
        span = torch.tensor([min(max(s.alt_m + CAMERA_SPAN_MIN_M, CAMERA_SPAN_MIN_M), CAMERA_SPAN_MAX_M)], device=self.device)
        return self.terrain.crop(xy[:, 0], xy[:, 1], span, self.crop_px, self.p.td_max_slope)

    # ------------------------------------------------------------------ act
    @torch.no_grad()
    def act(self, s: PilotState, trace: bool = True) -> dict:
        t0 = time.perf_counter()
        obs, img = self.observe(s), self.camera(s)
        classic = classic_action(self.gp, obs)
        feats = pilot_features(obs, classic)
        if trace and isinstance(self.model, PilotFormer):
            mu, value, tr = traced_forward(self.model, feats, img)
        else:
            (mu, value), tr = self.model(feats, img), None
        action = residual_action(classic, mu)
        scale = self.p.max_total_thrust
        return {"thrust_enu_n": (action[0] * scale).tolist(), "classic_enu_n": (classic[0].clamp(-1, 1) * scale).tolist(),
                "residual": mu[0].tolist(), "value": float(value[0]), "obs": obs[0].tolist(), "camera": img[0], "trace": tr,
                "ms": (time.perf_counter() - t0) * 1000.0}


# ---------------------------------------------------------------------- tracing
@torch.no_grad()
def traced_forward(model: PilotFormer, state: torch.Tensor, img: torch.Tensor, chain_top: int = 3, chain_next: int = 2,
                   max_per_layer: int = 6) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """PilotFormer.forward re-run layer by layer with the attention weights kept (eval mode: dropout is off, so the result is the same).

    Tokens: 0 = output token (CLS), 1 = state token (sensors + classic command), 2..65 = the 8 x 8 camera patches (row-major, north up).
    """
    tok = model.stem(img).flatten(2).transpose(1, 2) + model.pe
    x = torch.cat([model.cls.expand(tok.shape[0], -1, -1), model.state(state)[:, None], tok], dim=1)
    attn, act = [], []
    for layer in model.encoder.enc.layers:
        xn = layer.norm1(x)
        a, w = layer.self_attn(xn, xn, xn, need_weights=True, average_attn_weights=False)
        x = x + layer.dropout1(a)
        hid = layer.activation(layer.linear1(layer.norm2(x)))
        x = x + layer.dropout2(layer.linear2(layer.dropout(hid)))
        attn.append(w[0])                                   # [heads, N, N]
        act.append(hid[0])                                  # [N, 4d]
    if model.encoder.enc.norm is not None:
        x = model.encoder.enc.norm(x)
    h = model.encoder.norm(x)[:, 0]
    mu, value = torch.tanh(model.mu(h)), model.v(h)[:, 0]
    return mu, value, summarize_trace(attn, act, chain_top, chain_next, max_per_layer)


def summarize_trace(attn: list[torch.Tensor], act: list[torch.Tensor], chain_top: int, chain_next: int, max_per_layer: int) -> dict:
    n_layers, n_tok = len(attn), attn[0].shape[-1]
    mean = [a.mean(0) for a in attn]                        # [N, N] per layer, averaged over heads
    # attention rollout (Abnar & Zuidema 2020): how much each input token reaches the output token through all layers
    eye = torch.eye(n_tok, device=mean[0].device)
    roll = eye
    for m in mean:
        a = 0.5 * m + 0.5 * eye
        roll = (a / a.sum(-1, keepdim=True)) @ roll
    rollout = roll[0]
    # strongest chain: from the output token at the last layer back to the first layer
    edges, frontier = [], [0]
    for li in range(n_layers - 1, -1, -1):
        nxt: dict[int, float] = {}
        k = chain_top if li == n_layers - 1 else chain_next
        for q in frontier:
            w, idx = mean[li][q].topk(k)
            for wi, ki in zip(w.tolist(), idx.tolist()):
                head = int(attn[li][:, q, ki].argmax())
                edges.append([li, q, ki, round(wi, 4), head])
                nxt[ki] = max(nxt.get(ki, 0.0), wi)
        frontier = [t for t, _ in sorted(nxt.items(), key=lambda kv: -kv[1])[:max_per_layer]]
    activity = [a.abs().mean(-1) for a in act]              # [N] per layer
    top_neurons = []
    for a in act:
        v, i = a[0].abs().topk(32)                          # output token's strongest MLP neurons
        top_neurons.append([[int(j), round(float(a[0, j]), 4)] for j in i.tolist()])
    return {"tokens": n_tok, "layers": n_layers, "heads": attn[0].shape[0], "mlp_width": act[0].shape[-1],
            "rollout": [round(float(v), 5) for v in rollout], "chain": edges,
            "activity": [[round(float(v), 4) for v in a] for a in activity], "top_neurons": top_neurons}


def camera_png_array(img: torch.Tensor) -> np.ndarray:
    """[3,px,px] camera tensor -> uint8 [px,px,3] for display: R = slope/limit, G = hazard belief, B = relative height (+-4 -> 0..255)."""
    a = img.detach().cpu().numpy()
    rgb = np.stack([a[0].clip(0, 1), a[1].clip(0, 1), (a[2].clip(-4, 4) + 4) / 8], axis=-1)
    return (rgb * 255).astype(np.uint8)
