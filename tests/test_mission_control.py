# Copyright (c) 2026, Adinfinitium team
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
"""Fast checks for the one-command launcher (scripts.mission_control)."""

import socket

from scripts.mission_control import SCREENS, port_open, server_answers


def _free_port() -> int:
    """Return a TCP port nothing is listening on."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_free_port_is_neither_open_nor_a_server() -> None:
    """A port with no listener is reported closed and is never mistaken for a running Mission Control server."""
    port = _free_port()
    assert not port_open("127.0.0.1", port)
    assert not server_answers(f"http://127.0.0.1:{port}")


def test_every_screen_is_on_the_one_server() -> None:
    """The launcher lists the hub and every screen as paths on the same server, each once."""
    paths = [p for _, p in SCREENS]
    assert "/hub.html" in paths and "/" in paths and "/showcase/wall.html" in paths and "/ai.html" in paths
    assert all(p.startswith("/") for p in paths) and len(set(paths)) == len(paths)
