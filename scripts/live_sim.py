"""Live simulator: the full Basilisk mission (basilisk_mods.mission) run in real time with time warp, driven from the operator
console, streamed to Mission Control, the 3D view and (optionally) Vizard.

    python -m lunar_ops.server --port 8600          # terminal 1: Mission Control, 3D view, AI view
    python -m scripts.live_sim                      # terminal 2: the simulator (connects to /ws/ingest)
    python -m scripts.live_sim --live-viz           # also stream to Vizard (start Vizard with -directComm tcp://localhost:5556)

Console commands (JSON {"type": "cmd", "cmd": ...} sent by the UI, forwarded by the server):
  play | pause | warp {value} (also "speed")        time control (1x = wall clock; the sim runs ~100x at most on a laptop)
  ai {on}                                            learned pilot on/off in the terminal descent (classic backup otherwise)
  fail {kind: "engine_out", index}                   inject a failure now
  restart {setup: {region, target_xy, epoch_utc, attitude, engine_out_at_s, ...}}  new run

Glue between folders: talks to basilisk_mods and worldmodel through their public entry points only. Provenance: MODELLED.
"""

from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path

import numpy as np
import torch

from basilisk_mods.mission import CTRL_DT_S, MissionSetup, MissionSim
from contracts.paths import model_path
from worldmodel.lander.live import LivePilot, PilotState, camera_png_array
from worldmodel.lander.terrain import terrain_from_tile

REPO = Path(__file__).resolve().parents[1]
VIZ_MODELS = {"lander": REPO / "lunar_ops" / "vizard" / "assets" / "lander_B.obj",
              "propulsion_module": REPO / "lunar_ops" / "vizard" / "assets" / "propulsion_module_B.obj"}
FRAME_EVERY_S = 0.1  # [s] wall clock between frames
MAX_STEP_S = 5.0  # [s] most simulated time per loop turn (keeps the console responsive at high warp)


class Link:
    """Mission Control ingest socket (reconnecting) and/or a JSONL recording."""

    def __init__(self, url: str | None, out: Path | None):
        self.url, self.ws = url, None
        self.f = open(out, "w", encoding="utf-8") if out else None
        self._connect()

    def _connect(self):
        if not self.url:
            return
        from websockets.sync.client import connect

        import contextlib

        try:
            self._stack = contextlib.ExitStack()
            self.ws = self._stack.enter_context(connect(self.url, max_size=None, open_timeout=2))  # websockets 17: must be entered
        except OSError:
            self.ws = None

    def send(self, msg: dict) -> None:
        text = json.dumps(msg, separators=(",", ":"))
        if self.f:
            self.f.write(text + "\n")
        if self.url and self.ws is None:
            self._connect()
        if self.ws is not None:
            try:
                self.ws.send(text)
            except Exception:  # noqa: BLE001 - server restarted: reconnect next time
                self.ws = None

    def commands(self) -> list[dict]:
        out = []
        while self.ws is not None:
            try:
                out.append(json.loads(self.ws.recv(timeout=0)))
            except TimeoutError:
                break
            except Exception:  # noqa: BLE001
                self.ws = None
                break
        return out


class LiveSim:
    def __init__(self, setup: MissionSetup, checkpoint: Path, ai: bool):
        self.checkpoint, self.ai = checkpoint, ai
        self.warp, self.playing = 1.0, True
        self.start(setup)

    # ------------------------------------------------------------------ run control
    def start(self, setup: MissionSetup):
        self.setup = setup
        self.ms = MissionSim(setup, command_fn=None)
        tile = self.ms.terrain.tile
        self.pilot = LivePilot.create(self.ms.cfg, terrain_from_tile(tile), self.ms.site.xy, self.checkpoint)
        self.ms.guid.command_fn = self.command if self.ai else None
        self.last_pilot = None
        self.sent_ai_t = -1.0
        self.n_dec = 0
        self.debt = 0.0
        self.started = time.monotonic()

    def command(self, t_s, r, v, mass, engines_ok, classic_N):
        """Learned pilot in the tile frame of the site (DescentGuidance.command_fn)."""
        ms = self.ms
        ex, ey, up = ms.tile_basis_N()
        x, y, alt = ms.site.tile_xy_alt(ms.PN() @ r)
        out = self.pilot.act(PilotState(t_s, x - ms.site.xy[0], y - ms.site.xy[1], alt - 1.45, (float(v @ ex), float(v @ ey), float(v @ up)),
                                        mass, engines_ok))
        self.last_pilot = (t_s, out)
        te = out["thrust_enu_n"]
        return te[0] * ex + te[1] * ey + te[2] * up

    def handle(self, cmds: list[dict]) -> list[dict]:
        events = []
        for m in cmds:
            c = m.get("cmd")
            print(f"[sim] command {m}", flush=True)
            if c == "play":
                self.playing = True
            elif c == "pause":
                self.playing = False
            elif c in ("warp", "speed"):
                self.warp = float(min(max(float(m.get("value", 1.0)), 0.1), 10000.0))
            elif c == "ai":
                self.ai = bool(m.get("on"))
                self.ms.guid.command_fn = self.command if self.ai else None
                events.append(f"Operator: AI pilot {'ON' if self.ai else 'OFF (classic guidance)'}")
            elif c == "fail" and m.get("kind") == "engine_out":
                g = self.ms.guid
                g.engine_out_index = int(m.get("index", 1))
                g.engine_out_at_s = self.ms.t
                events.append(f"Operator: inject engine {g.engine_out_index} failure")
            elif c == "restart":
                base = self.setup.__dict__.copy()
                base.update({k: v for k, v in (m.get("setup") or {}).items() if k in base})
                self.start(MissionSetup(**base))
                events.append(f"Operator: restart ({self.setup.region})")
            else:
                continue
        return events

    def step(self, wall_dt: float) -> None:
        if not self.playing or self.ms.done:
            return
        self.debt = min(self.debt + wall_dt * self.warp, MAX_STEP_S)
        n = int(self.debt / CTRL_DT_S)
        if n:
            self.ms.advance(n * CTRL_DT_S)
            self.debt -= n * CTRL_DT_S

    # ------------------------------------------------------------------ output
    def messages(self, extra_events: list[str]) -> list[dict]:
        ms = self.ms
        f = ms.frame()
        g = ms.guid
        f["events"] = [{"t_s": round(t, 2), "kind": "decision", "msg": d} for t, d in g.decisions[self.n_dec:]]
        f["events"] += [{"t_s": round(ms.t, 2), "kind": "event", "msg": e} for e in extra_events]
        self.n_dec = len(g.decisions)
        f["sim"] = {"warp": self.warp, "playing": self.playing, "ai": self.ai, "pilot_trained": bool(self.pilot.info["trained"]),
                    "pilot_parameters": self.pilot.info["parameters"], "attitude": self.setup.attitude, "epoch_utc": self.setup.epoch_utc,
                    "done": ms.done, "report": ms.report}
        out = [f]
        if self.last_pilot is not None and "lander" in f:
            tp, p = self.last_pilot
            f["pilot"] = {"trained": bool(self.pilot.info["trained"]), "parameters": self.pilot.info["parameters"], "t_s": round(tp, 2),
                          "thrust_enu_n": p["thrust_enu_n"], "classic_enu_n": p["classic_enu_n"], "residual": p["residual"],
                          "value": p["value"], "ms": round(p["ms"], 1), "obs": [round(o, 4) for o in p["obs"]]}
            if p["trace"] is not None and tp > self.sent_ai_t + 0.19:
                cam = camera_png_array(p["camera"])
                out.append({"type": "ai", "t_s": round(tp, 2), "trace": p["trace"], "camera_rgb_b64": base64.b64encode(cam.tobytes()).decode(),
                            "camera_px": int(cam.shape[0]), "camera_channels": "R slope/limit, G hazard belief, B relative height"})
                self.sent_ai_t = tp
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--region", default="lola5m_site23")
    ap.add_argument("--attitude", default="rcs", choices=["rcs", "ideal"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--no-ai", action="store_true")
    ap.add_argument("--warp", type=float, default=1.0)
    ap.add_argument("--engine-out-at", type=float, default=None)
    ap.add_argument("--ingest", default="ws://127.0.0.1:8600/ws/ingest")
    ap.add_argument("--no-ingest", action="store_true")
    ap.add_argument("--out", default=None, help="record the stream to this JSONL file")
    ap.add_argument("--live-viz", action="store_true")
    ap.add_argument("--max-wall-s", type=float, default=None, help="stop after this many wall seconds (tests)")
    a = ap.parse_args()

    from contracts.paths import tile_path

    region = a.region if tile_path(a.region).exists() else "lola5m_site01"
    if region != a.region:
        print(f"[sim] tile {a.region} not found, using {region} as a stand-in", flush=True)
    setup = MissionSetup(region=region, attitude=a.attitude, engine_out_at_s=a.engine_out_at, live_viz=a.live_viz,
                         viz_models={k: str(v) for k, v in VIZ_MODELS.items()} if a.live_viz else {})
    sim = LiveSim(setup, Path(a.checkpoint) if a.checkpoint else model_path("lander_pilot"), ai=not a.no_ai)
    sim.warp = a.warp
    link = Link(None if a.no_ingest else a.ingest, Path(a.out) if a.out else None)
    print(f"[sim] {region}, target {sim.ms.site.xy}, pilot {'trained' if sim.pilot.info['trained'] else 'UNTRAINED'}, "
          f"attitude {a.attitude}, warp {sim.warp}x", flush=True)
    t_prev = t_frame = time.monotonic()
    t0 = t_prev
    while True:
        now = time.monotonic()
        events = sim.handle(link.commands())
        sim.step(now - t_prev)
        t_prev = now
        if now - t_frame >= FRAME_EVERY_S or events:
            for m in sim.messages(events):
                link.send(m)
            t_frame = now
        if a.max_wall_s and now - t0 > a.max_wall_s:
            break
        if a.no_ingest and sim.ms.done:
            for m in sim.messages([]):
                link.send(m)
            print("[sim]", sim.ms.report, flush=True)
            break
        time.sleep(0.005 if sim.playing and not sim.ms.done else 0.05)


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    torch.set_num_threads(2)
    main()
