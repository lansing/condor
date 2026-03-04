"""In-process stats accumulator and Unix socket server for the TUI.

StatsCollector maintains rolling windows of metric measurements and per-worker
counters.  It is updated by the ``_Tel`` singleton on every metric call.

StatsServer runs a background thread that accepts Unix socket connections and
pushes a JSON snapshot to each connected client once per second.

Protocol: persistent connection; server pushes one JSON line per second.
Socket path: /tmp/condor-metrics.sock
"""

from __future__ import annotations

import collections
import json
import logging
import os
import socket
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# Override with CONDOR_STATS_SOCKET env var, e.g. when running in Docker
# and the socket directory is bind-mounted to the host.
SOCKET_PATH = os.environ.get("CONDOR_STATS_SOCKET", "/tmp/condor-metrics.sock")
_WINDOW_S = 5.0       # rolling window length for per-worker latency stats
_SPARKLINE_LEN = 60   # sparkline history depth (one point per second)


# ---------------------------------------------------------------------------
# Rolling window
# ---------------------------------------------------------------------------

class _RollingWindow:
    """Thread-safe deque of (monotonic_timestamp, value) pairs."""

    def __init__(self, window_s: float = _WINDOW_S) -> None:
        self._window_s = window_s
        self._data: collections.deque[tuple[float, float]] = collections.deque()
        self._lock = threading.Lock()

    def add(self, value: float) -> None:
        now = time.monotonic()
        with self._lock:
            self._data.append((now, value))
            self._evict(now)

    def _evict(self, now: float) -> None:
        cutoff = now - self._window_s
        while self._data and self._data[0][0] < cutoff:
            self._data.popleft()

    def stats(self) -> dict[str, float] | None:
        """Return {avg, min, max} or None if no data in the window."""
        now = time.monotonic()
        with self._lock:
            self._evict(now)
            if not self._data:
                return None
            vals = [v for _, v in self._data]
        n = len(vals)
        return {
            "avg": round(sum(vals) / n, 2),
            "min": round(min(vals), 2),
            "max": round(max(vals), 2),
        }

    def rate(self) -> float:
        """Events per second in the rolling window."""
        now = time.monotonic()
        with self._lock:
            self._evict(now)
            return len(self._data) / self._window_s

    def count_in_window(self, window_s: float) -> int:
        """Count events in the most recent *window_s* seconds."""
        now = time.monotonic()
        cutoff = now - window_s
        with self._lock:
            return sum(1 for t, _ in self._data if t >= cutoff)

    def stats_for_window(self, window_s: float) -> dict[str, float] | None:
        """Return {avg, min, max} for events in the most recent *window_s* seconds."""
        now = time.monotonic()
        cutoff = now - window_s
        with self._lock:
            vals = [v for t, v in self._data if t >= cutoff]
        if not vals:
            return None
        n = len(vals)
        return {
            "avg": round(sum(vals) / n, 2),
            "min": round(min(vals), 2),
            "max": round(max(vals), 2),
        }

    def cur_min_max(self, tick_s: float) -> dict[str, float]:
        """Return {cur, min, max} — always a dict, zeros when no data.

        cur: avg of values in the most recent *tick_s* seconds (instantaneous).
        min/max: extremes over the full rolling window.
        """
        now = time.monotonic()
        cutoff_cur = now - tick_s
        with self._lock:
            self._evict(now)
            all_vals = [v for _, v in self._data]
            cur_vals = [v for t, v in self._data if t >= cutoff_cur]
        cur = round(sum(cur_vals) / len(cur_vals), 2) if cur_vals else 0.0
        mn = round(min(all_vals), 2) if all_vals else 0.0
        mx = round(max(all_vals), 2) if all_vals else 0.0
        return {"cur": cur, "min": mn, "max": mx}

    def set_window(self, window_s: float) -> None:
        """Change the rolling window duration. Old data outside the new window
        will be evicted on the next stats() / rate() call."""
        with self._lock:
            self._window_s = window_s


# ---------------------------------------------------------------------------
# Per-worker stats bucket
# ---------------------------------------------------------------------------

class _WorkerStats:
    def __init__(self) -> None:
        self.requests_total = 0
        self.inference_total = 0
        self.e2e = _RollingWindow()
        self.infer = _RollingWindow()
        self.postprocess = _RollingWindow()


# ---------------------------------------------------------------------------
# Stats collector
# ---------------------------------------------------------------------------

class StatsCollector:
    """Thread-safe accumulator for all Condor runtime metrics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start = time.monotonic()

        # Global gauges
        self._workers_active = 0
        self._inference_concurrent = 0
        self._active_model = ""

        # Config metadata (set once from main)
        self._provider = ""
        self._num_workers = 1
        self._base_port = 5555

        # Per-worker buckets (created on first access)
        self._workers: dict[int, _WorkerStats] = {}

        # Global timing windows (backend-level; no worker_id available)
        self._sem_wait = _RollingWindow()
        self._trt_host_copy = _RollingWindow()
        self._trt_h2d = _RollingWindow()
        self._trt_execute = _RollingWindow()
        self._trt_d2h = _RollingWindow()

        # Sparkline history — appended once per tick by _maybe_update_sparklines
        self._sparkline_latency: collections.deque[float] = collections.deque(
            maxlen=_SPARKLINE_LEN
        )
        self._sparkline_throughput: collections.deque[float] = collections.deque(
            maxlen=_SPARKLINE_LEN
        )
        self._last_sparkline = 0.0
        self._sparkline_tick_s = 2.0   # seconds between sparkline points; updated by set_window_config
        self._sparkline_lock = threading.Lock()

    # --- helpers -----------------------------------------------------------

    def _get_worker(self, wid: int) -> _WorkerStats:
        with self._lock:
            if wid not in self._workers:
                self._workers[wid] = _WorkerStats()
            return self._workers[wid]

    # --- configuration -----------------------------------------------------

    def configure(self, provider: str, num_workers: int, base_port: int) -> None:
        with self._lock:
            self._provider = provider
            self._num_workers = num_workers
            self._base_port = base_port

    def set_active_model(self, model: str) -> None:
        with self._lock:
            self._active_model = model

    # --- gauge updates -----------------------------------------------------

    def inc_workers_active(self) -> None:
        with self._lock:
            self._workers_active += 1

    def dec_workers_active(self) -> None:
        with self._lock:
            self._workers_active = max(0, self._workers_active - 1)

    def inc_inference_concurrent(self) -> None:
        with self._lock:
            self._inference_concurrent += 1

    def dec_inference_concurrent(self) -> None:
        with self._lock:
            self._inference_concurrent = max(0, self._inference_concurrent - 1)

    # --- per-worker counter updates ----------------------------------------

    def count_request(self, worker_id: int) -> None:
        w = self._get_worker(worker_id)
        with self._lock:
            w.requests_total += 1

    def count_inference(self, worker_id: int) -> None:
        w = self._get_worker(worker_id)
        with self._lock:
            w.inference_total += 1

    # --- per-worker latency updates ----------------------------------------

    def record_e2e(self, worker_id: int, ms: float) -> None:
        self._get_worker(worker_id).e2e.add(ms)

    def record_infer(self, worker_id: int, ms: float) -> None:
        self._get_worker(worker_id).infer.add(ms)

    def record_postprocess(self, worker_id: int, ms: float) -> None:
        self._get_worker(worker_id).postprocess.add(ms)

    # --- global latency updates --------------------------------------------

    def record_sem_wait(self, ms: float) -> None:
        self._sem_wait.add(ms)

    def record_trt_host_copy(self, ms: float) -> None:
        self._trt_host_copy.add(ms)

    def record_trt_h2d(self, ms: float) -> None:
        self._trt_h2d.add(ms)

    def record_trt_execute(self, ms: float) -> None:
        self._trt_execute.add(ms)

    def record_trt_d2h(self, ms: float) -> None:
        self._trt_d2h.add(ms)

    # --- time config -------------------------------------------------------

    def set_window_config(self, window_s: float, sparkline_len: int) -> None:
        """Update rolling window duration and sparkline depth for all metrics.

        Called when the TUI changes the tick settings.  Old data outside the
        new window expires naturally on the next stats() call.
        """
        window_s = max(1.0, window_s)
        sparkline_len = max(10, sparkline_len)

        with self._lock:
            for w in self._workers.values():
                w.e2e.set_window(window_s)
                w.infer.set_window(window_s)
                w.postprocess.set_window(window_s)
            for rw in (self._sem_wait, self._trt_host_copy, self._trt_h2d,
                       self._trt_execute, self._trt_d2h):
                rw.set_window(window_s)

        with self._sparkline_lock:
            old_lat = list(self._sparkline_latency)
            old_tput = list(self._sparkline_throughput)
            self._sparkline_latency = collections.deque(
                old_lat[-sparkline_len:], maxlen=sparkline_len
            )
            self._sparkline_throughput = collections.deque(
                old_tput[-sparkline_len:], maxlen=sparkline_len
            )
            self._sparkline_tick_s = window_s / sparkline_len

    # --- sparkline ---------------------------------------------------------

    def _maybe_update_sparklines(self) -> None:
        now = time.monotonic()
        with self._sparkline_lock:
            tick_s = self._sparkline_tick_s
            elapsed = now - self._last_sparkline
            if elapsed < tick_s:
                return
            self._last_sparkline = now

        # Gather worker data outside the main lock (rolling windows are self-locking)
        with self._lock:
            worker_refs = list(self._workers.values())

        # Instantaneous measurements: count events that arrived in the last `elapsed` seconds.
        all_e2e: list[float] = []
        total_count = 0
        for w in worker_refs:
            s = w.e2e.stats_for_window(elapsed)
            if s:
                all_e2e.append(s["avg"])
            total_count += w.e2e.count_in_window(elapsed)

        # Instantaneous throughput = events in elapsed window / elapsed seconds
        instant_rps = round(total_count / elapsed, 2) if elapsed > 0 else 0.0

        # Always append so sparklines scroll smoothly each tick
        if all_e2e:
            self._sparkline_latency.append(round(sum(all_e2e) / len(all_e2e), 1))
        else:
            self._sparkline_latency.append(
                self._sparkline_latency[-1] if self._sparkline_latency else 0.0
            )
        self._sparkline_throughput.append(instant_rps)

    # --- snapshot ----------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot for the TUI."""
        self._maybe_update_sparklines()

        now = time.monotonic()
        tick_s = self._sparkline_tick_s

        with self._lock:
            workers_snap: dict[str, Any] = {}
            for wid, w in self._workers.items():
                workers_snap[str(wid)] = {
                    "requests_total": w.requests_total,
                    "inference_total": w.inference_total,
                    "req_per_sec": round(w.e2e.rate(), 2),
                    "e2e_ms": w.e2e.cur_min_max(tick_s),
                    "infer_ms": w.infer.cur_min_max(tick_s),
                    "postprocess_ms": w.postprocess.cur_min_max(tick_s),
                }
            cfg = {
                "provider": self._provider,
                "num_workers": self._num_workers,
                "base_port": self._base_port,
            }
            active_model = self._active_model
            workers_active = self._workers_active
            inference_concurrent = self._inference_concurrent
            uptime = now - self._start

        def _agg(stats_list: list[dict]) -> dict[str, float]:
            """Aggregate cur_min_max dicts across workers; ignore workers with no data."""
            active = [s for s in stats_list if s["max"] > 0]
            if not active:
                return {"cur": 0.0, "min": 0.0, "max": 0.0}
            return {
                "cur": round(sum(s["cur"] for s in active) / len(active), 2),
                "min": round(min(s["min"] for s in active), 2),
                "max": round(max(s["max"] for s in active), 2),
            }

        global_e2e = _agg([workers_snap[w]["e2e_ms"] for w in workers_snap])
        global_infer = _agg([workers_snap[w]["infer_ms"] for w in workers_snap])
        global_pp = _agg([workers_snap[w]["postprocess_ms"] for w in workers_snap])
        global_rps = round(
            sum(workers_snap[w]["req_per_sec"] for w in workers_snap), 2
        )

        return {
            "config": cfg,
            "uptime_s": round(uptime, 1),
            "active_workers": workers_active,
            "inference_concurrent": inference_concurrent,
            "active_model": active_model,
            "workers": workers_snap,
            "global_e2e_ms": global_e2e,
            "global_throughput_rps": global_rps,
            "global_sem_wait_ms": self._sem_wait.cur_min_max(tick_s),
            "global_trt_host_copy_ms": self._trt_host_copy.cur_min_max(tick_s),
            "global_trt_h2d_ms": self._trt_h2d.cur_min_max(tick_s),
            "global_trt_execute_ms": self._trt_execute.cur_min_max(tick_s),
            "global_trt_d2h_ms": self._trt_d2h.cur_min_max(tick_s),
            "global_infer_ms": global_infer,
            "global_postprocess_ms": global_pp,
            "sparkline_latency": list(self._sparkline_latency),
            "sparkline_throughput": list(self._sparkline_throughput),
        }


# ---------------------------------------------------------------------------
# Socket server
# ---------------------------------------------------------------------------

class StatsServer:
    """Background thread that pushes JSON snapshots over a Unix domain socket.

    Spawns one per-client thread for each connected TUI instance.
    """

    def __init__(
        self,
        collector: StatsCollector,
        path: str = SOCKET_PATH,
    ) -> None:
        self._collector = collector
        self._path = path
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if os.path.exists(self._path):
            try:
                os.unlink(self._path)
            except OSError:
                pass
        self._thread = threading.Thread(
            target=self._accept_loop,
            name="condor-stats-server",
            daemon=True,
        )
        self._thread.start()
        logger.info("Stats socket server: %s", self._path)

    def stop(self) -> None:
        self._stop.set()

    def _accept_loop(self) -> None:
        try:
            srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            srv.bind(self._path)
            srv.listen(8)
            srv.settimeout(1.0)
            while not self._stop.is_set():
                try:
                    conn, _ = srv.accept()
                except socket.timeout:
                    continue
                t = threading.Thread(
                    target=self._client_loop,
                    args=(conn,),
                    name="condor-stats-client",
                    daemon=True,
                )
                t.start()
        except Exception:
            logger.exception("Stats socket server error")
        finally:
            try:
                srv.close()
            except Exception:
                pass
            if os.path.exists(self._path):
                try:
                    os.unlink(self._path)
                except OSError:
                    pass

    def _client_loop(self, conn: socket.socket) -> None:
        try:
            while not self._stop.is_set():
                snap = json.dumps(self._collector.snapshot()) + "\n"
                conn.sendall(snap.encode())
                self._stop.wait(1.0)
                # Non-blocking check for config messages sent back by the TUI.
                conn.setblocking(False)
                try:
                    data = conn.recv(4096)
                    if data:
                        self._apply_client_config(data.decode(errors="replace"))
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

    def _apply_client_config(self, raw: str) -> None:
        """Parse and apply JSON config messages received from the TUI."""
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                ws = msg.get("window_s")
                sl = msg.get("sparkline_len")
                if ws is not None and sl is not None:
                    self._collector.set_window_config(float(ws), int(sl))
            except (json.JSONDecodeError, ValueError, KeyError):
                pass
