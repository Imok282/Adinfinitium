"""Live mission bridge: Basilisk flies the descent in real time with the learned pilot, and every control period is streamed to
Mission Control and the 3D view (``/ws/ingest`` of ``lunar_ops.server``) and, optionally, to a running Vizard (live stream).

    python -m lunar_ops.server --port 8600                       # terminal 1: Mission Control + 3D view
    python -m scripts.live_mission --rtf 1                       # terminal 2: Basilisk + pilot, real time
    python -m scripts.live_mission --rtf 1 --live-viz            # same, also streaming to Vizard (start Vizard in live mode first)
    python -m scripts.live_mission --no-ingest --out runs/live/x.jsonl --rtf 0   # offline: write the stream to a file

This is glue between folders (basilisk_mods, worldmodel, lunar_ops) and talks to them only through their public entry points.
Physics backup first: the pilot is called through ``DescentGuidance.command_fn``; any pilot error falls back to classic ZEM/ZEV.

Frame mapping: Basilisk flies in a local east/north/up frame at the target; tile_xy = target_tile_xy + (x_m, y_m).
Ground: the terrain tile is used for both the guidance ground height and the pilot camera, relative to the target cell height.
Provenance: MODELLED (Basilisk simulation; terrain heights OBSERVED by LRO LOLA when the tile is a lola5m_* tile).
"""

from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path

import numpy as np
import torch
from Basilisk.utilities import RigidBodyKinematics as rbk

from basilisk_mods import scenario_descent
from basilisk_mods.bsk_common import local_frame
from contracts.mission import load_mission
from contracts.paths import model_path, tile_path
from contracts.tiles import load_tile
from worldmodel.lander.live import LivePilot, PilotState, camera_png_array
from worldmodel.lander.terrain import safe_target_cells, terrain_from_tile

N_ENGINES_NOTE = "engine index 0 = centre, 1..4 = outer (+x, +y, -x, -y body); lander_B nodes engine_centre, engine_outer0..3"


def pick_target(tile) -> tuple[float, float]:
    """Safe cell closest to the tile centre (backup hazard map), in tile metres."""
    cells = safe_target_cells(tile)
    h, w = tile.height.shape
    d = (cells[:, 0] - (h - 1) / 2) ** 2 + (cells[:, 1] - (w - 1) / 2) ** 2
    r, c = cells[int(np.argmin(d))]
    mpp = tile.meters_per_pixel
    return float((c - (w - 1) / 2) * mpp), float(((h - 1) / 2 - r) * mpp)


def body_to_enu_quat(sigma_bn, E, N, U) -> list[float]:
    """Quaternion [w, x, y, z] rotating body vectors into the local east/north/up frame."""
    BN = np.array(rbk.MRP2C(np.asarray(sigma_bn)))          # N -> B
    EN = np.stack([E, N, U])                                 # N -> ENU
    R = EN @ BN.T                                            # B -> ENU
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w > 1e-6:
        return [w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)]
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(max(0.0, 1.0 + R[i, i] - R[j, j] - R[k, k])) * 2.0
    q = [0.0, 0.0, 0.0, 0.0]
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = s / 4.0
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


class Sink:
    """Where frames go: the Mission Control ingest socket and/or a JSONL file."""

    def __init__(self, url: str | None, out: Path | None):
        self.ws = None
        if url:
            from websockets.sync.client import connect

            self.ws = connect(url, max_size=None)
        self.f = open(out, "w", encoding="utf-8") if out else None

    def send(self, msg: dict) -> None:
        text = json.dumps(msg, separators=(",", ":"))
        if self.ws is not None:
            self.ws.send(text)
        if self.f is not None:
            self.f.write(text + "\n")

    def close(self) -> None:
        if self.ws is not None:
            self.ws.close()
        if self.f is not None:
            self.f.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--region", default="lola5m_site23", help="terrain tile; falls back to lola5m_site01 if missing")
    ap.add_argument("--target", type=float, nargs=2, default=None, metavar=("X_M", "Y_M"), help="target in tile metres")
    ap.add_argument("--checkpoint", default=None, help="pilot checkpoint (default data/models/lander_pilot.pt)")
    ap.add_argument("--classic", action="store_true", help="fly with the classic law only (no pilot)")
    ap.add_argument("--rtf", type=float, default=1.0, help="real-time factor (1 = wall clock, 0 = as fast as possible)")
    ap.add_argument("--ingest", default="ws://127.0.0.1:8600/ws/ingest")
    ap.add_argument("--no-ingest", action="store_true")
    ap.add_argument("--out", default=None, help="also write the stream to this JSONL file")
    ap.add_argument("--live-viz", action="store_true", help="stream to a running Vizard")
    ap.add_argument("--alt", type=float, default=2000.0)
    ap.add_argument("--offset-east", type=float, default=300.0)
    ap.add_argument("--offset-north", type=float, default=-200.0)
    ap.add_argument("--v-down", type=float, default=45.0)
    ap.add_argument("--engine-out-at", type=float, default=None)
    ap.add_argument("--id", default="live")
    a = ap.parse_args()

    cfg = load_mission()
    region = a.region if tile_path(a.region).exists() else "lola5m_site01"
    if region != a.region:
        print(f"[live] tile {a.region} not found, using {region} as a stand-in", flush=True)
    tile = load_tile(tile_path(region))
    terrain = terrain_from_tile(tile)
    target = tuple(a.target) if a.target else pick_target(tile)
    tx, ty = torch.tensor([target[0]]), torch.tensor([target[1]])
    h_target = float(terrain.height_at(tx, ty)[0])
    ckpt = Path(a.checkpoint) if a.checkpoint else model_path("lander_pilot")
    pilot = None if a.classic else LivePilot.create(cfg, terrain, target, ckpt)
    print(f"[live] region {region}, target {target[0]:.0f} {target[1]:.0f} m, pilot: "
          f"{'classic only' if pilot is None else ('trained ' + str(ckpt)) if pilot.info['trained'] else 'UNTRAINED network (flies like classic)'}",
          flush=True)

    sink = Sink(None if a.no_ingest else a.ingest, Path(a.out) if a.out else None)
    tgt_N = scenario_descent.target_point_N()
    E0, N0, U0 = local_frame(tgt_N)
    frame_ctx: dict = {"E": (E0, N0, U0, tgt_N), "last": None, "n_dec": 0, "prev_ok": 5}

    def enu(r, tgt, E, N, U):
        rel = r - tgt
        return float(rel @ E), float(rel @ N), float(rel @ U)

    def ground_height(r_N) -> float:
        E, N, U, tgt = frame_ctx["E"]
        x, y, _ = enu(np.asarray(r_N), tgt, E, N, U)
        return float(terrain.height_at(torch.tensor([target[0] + x]), torch.tensor([target[1] + y]))[0]) - h_target

    def command(t_s, r, v, mass, engines_ok, classic_N):
        E, N, U, tgt = frame_ctx["E"]
        x, y, z = enu(r, tgt, E, N, U)
        alt = z - ground_height(r)
        out = pilot.act(PilotState(t_s, x, y, alt, (float(v @ E), float(v @ N), float(v @ U)), mass, engines_ok))
        frame_ctx["last"] = (t_s, out)
        te = out["thrust_enu_n"]
        return te[0] * E + te[1] * N + te[2] * U

    def on_step(t, h, guid, st, tgt, frame):
        E, N, U = frame
        r, v = np.array(st.r_BN_N), np.array(st.v_BN_N)
        x, y, z = enu(r, tgt, E, N, U)
        alt = z - ground_height(r)
        last = guid.last
        ok = int(last.get("engines_ok", 5))
        on = last.get("on_time_s", [0.0] * 5)
        period = 0.2  # [s] control period of scenario_descent
        events = [{"t_s": round(td, 2), "kind": "decision", "msg": d} for td, d in guid.decisions[frame_ctx["n_dec"]:]]
        frame_ctx["n_dec"] = len(guid.decisions)
        f = {"type": "frame", "t_s": round(t, 2), "mission_phase": "DESCENT", "control": "autonomy",
             "phase": str(last.get("phase", "")).replace("_", " ").upper(), "reason": "",
             "lander": {"x_m": target[0] + x, "y_m": target[1] + y, "alt_m": alt, "vx_m_s": float(v @ E), "vy_m_s": float(v @ N),
                        "vz_m_s": float(v @ U), "mass_kg": h["mass_fn"](), "fuel_kg": h["fuel_fn"](), "thrust_n": float(last.get("thrust_n", 0.0)),
                        "engines_ok": ok, "q_enu": body_to_enu_quat(st.sigma_BN, E, N, U),
                        "engine_throttle": [min(1.0, o / period) for o in on[:5]], "source": last.get("source", "classic")},
             "events": events}
        if pilot is not None and frame_ctx["last"] is not None:
            tp, out = frame_ctx["last"]
            f["pilot"] = {"trained": pilot.info["trained"], "parameters": pilot.info["parameters"], "t_s": round(tp, 2),
                          "thrust_enu_n": out["thrust_enu_n"], "classic_enu_n": out["classic_enu_n"], "residual": out["residual"],
                          "value": out["value"], "ms": round(out["ms"], 1), "obs": [round(o, 4) for o in out["obs"]]}
            if out["trace"] is not None:
                cam = camera_png_array(out["camera"])
                sink.send({"type": "ai", "t_s": round(tp, 2), "trace": out["trace"], "camera_rgb_b64": base64.b64encode(cam.tobytes()).decode(),
                           "camera_px": int(cam.shape[0]), "camera_channels": "R slope/limit, G hazard belief, B relative height"})
        if ok != frame_ctx["prev_ok"]:
            f["events"].append({"t_s": round(t, 2), "kind": "event", "msg": f"{ok} of 5 engines available"})
            frame_ctx["prev_ok"] = ok
        sink.send(f)

    try:
        report = scenario_descent.run(a.id, a.alt, a.offset_east, a.offset_north, a.v_down, engine_out_at=a.engine_out_at,
                                      viz=False, live_viz=a.live_viz, command_fn=None if pilot is None else command,
                                      ground_height_fn=ground_height,
                                      on_step=on_step, rtf=a.rtf or None)
        msg = f"Touchdown {report['outcome']}" if "vertical_speed_m_s" in report else f"Run ended: {report['outcome']}"
        if "vertical_speed_m_s" in report:
            msg += f": vertical {report['vertical_speed_m_s']:.2f} m/s, horizontal {report['horizontal_speed_m_s']:.2f} m/s, " \
                   f"{report['fuel_left_kg']:.0f} kg fuel left, {math.hypot(report['x_m'], report['y_m']):.1f} m from target"
        sink.send({"type": "frame", "t_s": round(report["t_s"], 2), "mission_phase": "LANDED" if report["outcome"] == "ok" else "DESCENT",
                   "control": "autonomy", "phase": "TOUCHDOWN", "reason": report["outcome"], "events": [{"t_s": round(report["t_s"], 2), "kind": "sample", "msg": msg}]})
        print("[live]", msg, flush=True)
    finally:
        sink.close()


if __name__ == "__main__":
    main()
