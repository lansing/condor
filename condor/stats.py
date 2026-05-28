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
_WINDOW_S = 5.0  # rolling window length for per-worker latency stats
_SPARKLINE_LEN = 200  # sparkline history depth at 2 s/tick ≈ 6.7 min
_SPARKLINE_TICK_S = 2.0  # fixed tick interval — one sparkline point every 2 seconds


def _pad(dq: collections.deque) -> list[float]:
    """Zero-pad to maxlen so the TUI always receives a full-length array."""
    deficit = (dq.maxlen or 0) - len(dq)
    return [0.0] * deficit + list(dq)


class _RollingWindow:
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
        now = time.monotonic()
        with self._lock:
            self._evict(now)
            return len(self._data) / self._window_s

    def count_in_window(self, window_s: float) -> int:
        now = time.monotonic()
        cutoff = now - window_s
        with self._lock:
            return sum(1 for t, _ in self._data if t >= cutoff)

    def stats_for_window(self, window_s: float) -> dict[str, float] | None:
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

    def avg_p99(self) -> dict[str, float]:
        now = time.monotonic()
        with self._lock:
            self._evict(now)
            vals = [v for _, v in self._data]
        if not vals:
            return {"avg": 0.0, "p99": 0.0}
        n = len(vals)
        avg = round(sum(vals) / n, 2)
        sorted_vals = sorted(vals)
        p99_idx = min(int(0.99 * n), n - 1)
        p99 = round(sorted_vals[p99_idx], 2)
        return {"avg": avg, "p99": p99}

    def set_window(self, window_s: float) -> None:
        with self._lock:
            self._window_s = window_s


class GpuPoller:

    _POLL_INTERVAL = 0.2  # seconds between NVML queries

    def __init__(self, device_index: int = 0) -> None:
        self._device_index = device_index
        self._lock = threading.Lock()
        self._available = False
        self._name = ""
        self._util_pct = 0.0
        self._power_w = 0.0
        self._power_limit_w = 0.0
        self._mem_used_mb = 0.0
        self._mem_total_mb = 0.0
        self._temp_c = 0
        self._util_samples: list[float] = []  # accumulated since last consume
        self._stop = threading.Event()

    def start(self) -> bool:
        try:
            import pynvml  # type: ignore[import-untyped]

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            with self._lock:
                self._available = True
                self._name = name
                self._power_limit_w = round(limit_mw / 1000.0, 1)
                self._mem_total_mb = round(mem.total / (1024 * 1024))
        except Exception:
            return False
        threading.Thread(
            target=self._poll_loop, name="condor-gpu-poller", daemon=True
        ).start()
        return True

    def _poll_loop(self) -> None:
        try:
            import pynvml  # type: ignore[import-untyped]

            handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
            while not self._stop.is_set():
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    temp = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                    with self._lock:
                        self._util_pct = float(util.gpu)
                        self._util_samples.append(self._util_pct)
                        self._power_w = round(power_mw / 1000.0, 1)
                        self._mem_used_mb = round(mem.used / (1024 * 1024))
                        self._temp_c = temp
                except Exception:
                    pass
                self._stop.wait(self._POLL_INTERVAL)
        except Exception:
            pass

    def stop(self) -> None:
        self._stop.set()

    def consume_avg_util(self) -> float:
        with self._lock:
            samples = self._util_samples
            self._util_samples = []
        return sum(samples) / len(samples) if samples else 0.0

    def latest(self) -> dict | None:
        with self._lock:
            if not self._available:
                return None
            return {
                "name": self._name,
                "index": self._device_index,
                "util_pct": self._util_pct,
                "power_w": self._power_w,
                "power_limit_w": self._power_limit_w,
                "mem_used_mb": self._mem_used_mb,
                "mem_total_mb": self._mem_total_mb,
                "temp_c": self._temp_c,
            }


class _WorkerStats:
    def __init__(self, window_s: float = _WINDOW_S) -> None:
        self.requests_total = 0
        self.inference_total = 0
        self.e2e = _RollingWindow(window_s)
        self.infer = _RollingWindow(window_s)
        self.postprocess = _RollingWindow(window_s)


class StatsCollector:

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start = time.monotonic()

        self._workers_active = 0
        self._inference_concurrent = 0
        self._active_model = ""
        self._active_postprocessor = ""

        self._provider = ""
        self._num_workers = 1
        self._base_port = 5555

        self._workers: dict[int, _WorkerStats] = {}
        self._current_window_s: float = _WINDOW_S

        self._gpu_poller: GpuPoller | None = None
        self._sparkline_gpu_util: collections.deque[float] = collections.deque(
            maxlen=_SPARKLINE_LEN
        )

        self._sem_wait = _RollingWindow()
        self._trt_host_copy = _RollingWindow()
        self._trt_h2d = _RollingWindow()
        self._trt_execute = _RollingWindow()
        self._trt_d2h = _RollingWindow()

        self._sparkline_latency: collections.deque[float] = collections.deque(
            maxlen=_SPARKLINE_LEN
        )
        self._sparkline_throughput: collections.deque[float] = collections.deque(
            maxlen=_SPARKLINE_LEN
        )
        self._sparkline_mcpy: collections.deque[float] = collections.deque(
            maxlen=_SPARKLINE_LEN
        )
        self._sparkline_h2d: collections.deque[float] = collections.deque(
            maxlen=_SPARKLINE_LEN
        )
        self._sparkline_swait: collections.deque[float] = collections.deque(
            maxlen=_SPARKLINE_LEN
        )
        self._sparkline_exec: collections.deque[float] = collections.deque(
            maxlen=_SPARKLINE_LEN
        )
        self._sparkline_d2h: collections.deque[float] = collections.deque(
            maxlen=_SPARKLINE_LEN
        )
        self._sparkline_pp: collections.deque[float] = collections.deque(
            maxlen=_SPARKLINE_LEN
        )
        self._last_sparkline = 0.0
        self._sparkline_lock = threading.Lock()

    def _get_worker(self, wid: int) -> _WorkerStats:
        with self._lock:
            if wid not in self._workers:
                self._workers[wid] = _WorkerStats(self._current_window_s)
            return self._workers[wid]

    def configure(self, provider: str, num_workers: int, base_port: int) -> None:
        with self._lock:
            self._provider = provider
            self._num_workers = num_workers
            self._base_port = base_port

    def configure_gpu(self, device_index: int = 0) -> None:
        poller = GpuPoller(device_index)
        if poller.start():
            with self._lock:
                self._gpu_poller = poller
            gpu = poller.latest()
            name = gpu["name"] if gpu else "?"
            logger.info("GPU poller started: device %d (%s)", device_index, name)
        else:
            logger.info(
                "GPU metrics unavailable (pynvml not installed or no GPU found)"
            )

    def set_active_model(self, model: str) -> None:
        with self._lock:
            self._active_model = model

    def set_active_postprocessor(self, postprocessor: str) -> None:
        with self._lock:
            self._active_postprocessor = postprocessor

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

    def count_request(self, worker_id: int) -> None:
        w = self._get_worker(worker_id)
        with self._lock:
            w.requests_total += 1

    def count_inference(self, worker_id: int) -> None:
        w = self._get_worker(worker_id)
        with self._lock:
            w.inference_total += 1

    def record_e2e(self, worker_id: int, ms: float) -> None:
        self._get_worker(worker_id).e2e.add(ms)

    def record_infer(self, worker_id: int, ms: float) -> None:
        self._get_worker(worker_id).infer.add(ms)

    def record_postprocess(self, worker_id: int, ms: float) -> None:
        self._get_worker(worker_id).postprocess.add(ms)

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


    def _maybe_update_sparklines(self) -> None:
        now = time.monotonic()
        with self._sparkline_lock:
            if now - self._last_sparkline < 2.0:
                return
            self._last_sparkline = now

        with self._lock:
            worker_refs = list(self._workers.values())

        all_e2e: list[float] = []
        total_count = 0
        for w in worker_refs:
            s = w.e2e.stats_for_window(elapsed)
            if s:
                all_e2e.append(s["avg"])
            total_count += w.e2e.count_in_window(elapsed)

        instant_rps = round(total_count / elapsed, 2) if elapsed > 0 else 0.0

        def _stage_avg(rw: _RollingWindow) -> float:
            s = rw.stats_for_window(elapsed)
            return round(s["avg"], 1) if s else 0.0

        pp_stats = [w.postprocess.stats_for_window(elapsed) for w in worker_refs]
        active_pp = [s["avg"] for s in pp_stats if s]
        pp_val = round(sum(active_pp) / len(active_pp), 1) if active_pp else 0.0

        if all_e2e:
            self._sparkline_latency.append(round(sum(all_e2e) / len(all_e2e), 1))
        else:
            self._sparkline_latency.append(
                self._sparkline_latency[-1] if self._sparkline_latency else 0.0
            )
        self._sparkline_throughput.append(instant_rps)
        self._sparkline_mcpy.append(_stage_avg(self._trt_host_copy))
        self._sparkline_h2d.append(_stage_avg(self._trt_h2d))
        self._sparkline_swait.append(_stage_avg(self._sem_wait))
        self._sparkline_exec.append(_stage_avg(self._trt_execute))
        self._sparkline_d2h.append(_stage_avg(self._trt_d2h))
        self._sparkline_pp.append(pp_val)
        gpu_util = (
            self._gpu_poller.consume_avg_util() if self._gpu_poller is not None else 0.0
        )
        self._sparkline_gpu_util.append(gpu_util)

    def snapshot(self) -> dict[str, Any]:
        self._maybe_update_sparklines()

        now = time.monotonic()

        with self._lock:
            workers_snap: dict[str, Any] = {}
            for wid, w in self._workers.items():
                workers_snap[str(wid)] = {
                    "requests_total": w.requests_total,
                    "inference_total": w.inference_total,
                    "req_per_sec": round(w.e2e.rate(), 2),
                    "e2e_ms": w.e2e.avg_p99(),
                    "infer_ms": w.infer.avg_p99(),
                    "postprocess_ms": w.postprocess.avg_p99(),
                }
            cfg = {
                "provider": self._provider,
                "num_workers": self._num_workers,
                "base_port": self._base_port,
            }
            active_model = self._active_model
            active_postprocessor = self._active_postprocessor
            workers_active = self._workers_active
            inference_concurrent = self._inference_concurrent
            uptime = now - self._start

        def _agg(stats_list: list[dict]) -> dict[str, float]:
            active = [s for s in stats_list if s["p99"] > 0]
            if not active:
                return {"avg": 0.0, "p99": 0.0}
            return {
                "avg": round(sum(s["avg"] for s in active) / len(active), 2),
                "p99": round(max(s["p99"] for s in active), 2),
            }

        global_e2e = _agg([workers_snap[w]["e2e_ms"] for w in workers_snap])
        global_infer = _agg([workers_snap[w]["infer_ms"] for w in workers_snap])
        global_pp = _agg([workers_snap[w]["postprocess_ms"] for w in workers_snap])
        global_rps = round(sum(workers_snap[w]["req_per_sec"] for w in workers_snap), 2)

        return {
            "config": cfg,
            "uptime_s": round(uptime, 1),
            "active_workers": workers_active,
            "inference_concurrent": inference_concurrent,
            "active_model": active_model,
            "active_postprocessor": active_postprocessor,
            "workers": workers_snap,
            "global_e2e_ms": global_e2e,
            "global_throughput_rps": global_rps,
            "global_sem_wait_ms": self._sem_wait.avg_p99(),
            "global_trt_host_copy_ms": self._trt_host_copy.avg_p99(),
            "global_trt_h2d_ms": self._trt_h2d.avg_p99(),
            "global_trt_execute_ms": self._trt_execute.avg_p99(),
            "global_trt_d2h_ms": self._trt_d2h.avg_p99(),
            "global_infer_ms": global_infer,
            "global_postprocess_ms": global_pp,
            "sparkline_latency": _pad(self._sparkline_latency),
            "sparkline_throughput": _pad(self._sparkline_throughput),
            "sparkline_stages": {
                "mcpy": _pad(self._sparkline_mcpy),
                "h2d": _pad(self._sparkline_h2d),
                "swait": _pad(self._sparkline_swait),
                "exec": _pad(self._sparkline_exec),
                "d2h": _pad(self._sparkline_d2h),
                "pp": _pad(self._sparkline_pp),
            },
            "gpu": (
                {
                    **self._gpu_poller.latest(),
                    "sparkline": _pad(self._sparkline_gpu_util),
                }
                if self._gpu_poller is not None
                and self._gpu_poller.latest() is not None
                else None
            ),
        }


class StatsServer:

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
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass


# Module-level singleton shared by all instrumented modules.
tel = StatsCollector()
