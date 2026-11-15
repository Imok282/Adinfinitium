# Copyright (c) 2026, Adinfinitium team
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
"""Start everything on one localhost: the Mission Control server (every screen) and the live Basilisk simulator.

    python -m scripts.mission_control                  # server on http://127.0.0.1:8600 + simulator at 1x
    python -m scripts.mission_control --warp 25        # simulator time warp
    python -m scripts.mission_control --no-sim         # server only (replays the last recording)
    python -m scripts.mission_control --live-viz       # also stream to Vizard (start it with -directComm tcp://localhost:5556)

All screens are served by the one server; open http://127.0.0.1:8600/hub.html to reach them, or use the Screens menu in the
banner of any screen. If a server is already listening on the port it is reused. Ctrl+C stops what this script started.

Glue between folders: starts lunar_ops.server and scripts.live_sim as separate processes and talks to neither directly.
"""

from __future__ import annotations

import argparse
import signal
import socket
import subprocess
import sys
import time
import urllib.request

SCREENS = (("All screens (hub)", "/hub.html"), ("Mission control", "/"), ("Mission wall", "/showcase/wall.html"),
           ("3D + Vizard", "/showcase/duo.html"), ("Pilot brain", "/ai.html"), ("3D view", "/showcase/?live=1"))
SERVER_WAIT_S = 20.0  # [s] longest wait for the server to accept connections


def server_answers(base: str) -> bool:
    """Return True when a Mission Control server at ``base`` serves the hub page (not just a socket closing down).

    :param base: server root, e.g. ``http://127.0.0.1:8600``
    :return: whether ``/hub.html`` answered with HTTP 200
    """
    try:
        with urllib.request.urlopen(f"{base}/hub.html", timeout=2) as r:
            return r.status == 200
    except OSError:
        return False


def port_open(host: str, port: int) -> bool:
    """Return True when something accepts TCP connections on ``host:port``.

    :param host: host name or address
    :param port: TCP port
    :return: whether a connection succeeded
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) == 0


def _interrupt(*_: object) -> None:
    """Turn SIGTERM into KeyboardInterrupt so the clean-up below runs."""
    raise KeyboardInterrupt


def main() -> None:
    """Start the server (unless one is already running) and the simulator, print the screen links, then wait."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--no-sim", action="store_true", help="start only the server")
    ap.add_argument("--warp", type=float, default=1.0, help="simulator time warp")
    ap.add_argument("--region", default=None, help="terrain tile for the simulator (its default otherwise)")
    ap.add_argument("--no-ai", action="store_true", help="classic guidance only")
    ap.add_argument("--live-viz", action="store_true", help="also stream to Vizard")
    ap.add_argument("--viz-terrain", action="store_true", help="with --live-viz: Vizard's Moon gets the real landing-site terrain (experimental)")
    a = ap.parse_args()

    signal.signal(signal.SIGTERM, _interrupt)  # a plain kill also stops the children
    procs: list[subprocess.Popen] = []
    base = f"http://{a.host}:{a.port}"
    try:
        if server_answers(base):
            print(f"[mission control] a server is already running on {base}; using it")
        elif port_open(a.host, a.port):
            sys.exit(f"[mission control] port {a.port} is taken by something that is not Mission Control; use --port")
        else:
            procs.append(subprocess.Popen([sys.executable, "-m", "lunar_ops.server", "--host", a.host, "--port", str(a.port)]))
            t0 = time.monotonic()
            while not port_open(a.host, a.port):
                if procs[0].poll() is not None or time.monotonic() - t0 > SERVER_WAIT_S:
                    sys.exit("[mission control] the server did not start; see the messages above")
                time.sleep(0.2)
        if not a.no_sim:
            cmd = [sys.executable, "-m", "scripts.live_sim", "--warp", str(a.warp), "--ingest", f"ws://{a.host}:{a.port}/ws/ingest"]
            cmd += ["--region", a.region] if a.region else []
            cmd += ["--no-ai"] if a.no_ai else []
            cmd += ["--live-viz"] if a.live_viz else []
            cmd += ["--viz-terrain"] if a.viz_terrain else []
            procs.append(subprocess.Popen(cmd))
        print("\n[mission control] every screen is on one server:")
        for name, path in SCREENS:
            print(f"  {name:<20} {base}{path}")
        print("  Ctrl+C stops what this script started.\n", flush=True)
        server = procs[0] if procs and procs[0].args[2] == "lunar_ops.server" else None
        sim = procs[-1] if procs and procs[-1].args[2] == "scripts.live_sim" else None
        if server is None and sim is None:
            return  # nothing started here: the running server already has every screen
        while True:  # the server keeps serving (and replaying) if the simulator stops; stop when the server stops
            time.sleep(0.5)
            if sim is not None and sim.poll() is not None:
                print(f"[mission control] the simulator stopped (exit code {sim.returncode}); the screens stay up", flush=True)
                sim = None
            if (server is not None and server.poll() is not None) or (server is None and sim is None):
                break
    except KeyboardInterrupt:
        pass
    finally:
        for p in reversed(procs):
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()
