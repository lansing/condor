#!/usr/bin/env python3
"""Replay recorded metrics to a local Unix socket for TUI development.

Loads dev/metrics_sample.jsonl and serves it on the stats socket so condor-tui
can be run without a live Condor/GPU instance.  Frames are replayed at 1-second
intervals, looping indefinitely.

Usage:
    python scripts/metrics_replay.py          # socket at /tmp/condor-metrics.sock
    python scripts/metrics_replay.py PATH     # custom socket path

Then in another terminal:
    uv run condor-tui
    # or: make tui
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
from pathlib import Path

SOCKET_PATH = os.environ.get("CONDOR_STATS_SOCKET", "/tmp/condor-metrics.sock")
DATA_FILE = Path(__file__).parent.parent / "dev" / "metrics_sample.jsonl"


def _load_frames(path: Path) -> list[bytes]:
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    if not lines:
        sys.exit(f"error: no frames found in {path}")
    return [line.encode() + b"\n" for line in lines]


def _client(conn: socket.socket, frames: list[bytes], stop: threading.Event) -> None:
    idx = 0
    try:
        while not stop.is_set():
            conn.sendall(frames[idx % len(frames)])
            idx += 1
            stop.wait(1.0)
            # Drain any config messages sent by the TUI (window_s / sparkline_len).
            conn.setblocking(False)
            try:
                conn.recv(4096)
            except (BlockingIOError, OSError):
                pass
            finally:
                conn.setblocking(True)
    except (BrokenPipeError, OSError):
        pass
    finally:
        try:
            conn.close()
        except OSError:
            pass


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else SOCKET_PATH

    frames = _load_frames(DATA_FILE)
    print(f"Loaded {len(frames)} frames from {DATA_FILE}")

    if os.path.exists(path):
        os.unlink(path)

    stop = threading.Event()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(8)
    srv.settimeout(1.0)
    print(f"Replaying on {path}  (Ctrl-C to stop)")

    try:
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            t = threading.Thread(target=_client, args=(conn, frames, stop), daemon=True)
            t.start()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stop.set()
        srv.close()
        if os.path.exists(path):
            os.unlink(path)


if __name__ == "__main__":
    main()
