"""Entry point for the Frigate remote detector server."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import threading

from ..config.settings import AppConfig, load_config
from .zmq_handler import AsyncZMQHandler

logger = logging.getLogger(__name__)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        stream=sys.stdout,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frigate ZMQ Remote Object Detector Server"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        metavar="PATH",
        help="Path to YAML config file (default: config/config.yaml).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Multi-worker coordination
# ---------------------------------------------------------------------------

class _WorkerCoordinator:
    """Thread-safe registry that propagates shutdown to all worker event loops."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop_requested = False
        self._workers: list[tuple[asyncio.AbstractEventLoop, asyncio.Task]] = []

    def register(
        self,
        loop: asyncio.AbstractEventLoop,
        task: asyncio.Task,
    ) -> None:
        """Called from inside each worker thread once its asyncio task is running."""
        with self._lock:
            self._workers.append((loop, task))
            # If shutdown was already requested before this worker registered,
            # cancel it immediately.
            if self._stop_requested:
                loop.call_soon_threadsafe(task.cancel)

    def shutdown_all(self) -> None:
        """Cancel every registered worker task (thread-safe, callable from signal handler)."""
        with self._lock:
            self._stop_requested = True
            for loop, task in self._workers:
                loop.call_soon_threadsafe(task.cancel)


def _run_worker(
    config: AppConfig,
    endpoint: str,
    coordinator: _WorkerCoordinator,
    worker_idx: int,
) -> None:
    """Target function for each worker thread.

    Each thread runs its own asyncio event loop so workers are fully
    concurrent.  The ZMQ REP socket inside each worker still maintains the
    strict send→recv ordering Frigate requires, but workers on different ports
    proceed independently.
    """

    async def _main() -> None:
        handler = AsyncZMQHandler(config, endpoint=endpoint)
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        assert task is not None
        coordinator.register(loop, task)
        try:
            await handler.run()
        except asyncio.CancelledError:
            pass

    logger.info("Worker %d starting on %s", worker_idx, endpoint)
    asyncio.run(_main())
    logger.info("Worker %d stopped.", worker_idx)


# ---------------------------------------------------------------------------
# Single-worker (original) path
# ---------------------------------------------------------------------------

async def _run_single(config: AppConfig) -> None:
    handler = AsyncZMQHandler(config)

    loop = asyncio.get_running_loop()
    current_task = asyncio.current_task()

    def _signal_handler() -> None:
        logger.info("Shutdown signal received.")
        if current_task and not current_task.done():
            current_task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    try:
        await handler.run()
    except asyncio.CancelledError:
        pass  # Normal shutdown path


# ---------------------------------------------------------------------------
# Multi-worker path
# ---------------------------------------------------------------------------

def _run_multi(config: AppConfig) -> None:
    num_workers = config.server.num_workers
    base_port = config.server.base_port

    coordinator = _WorkerCoordinator()
    threads: list[threading.Thread] = []

    for i in range(num_workers):
        endpoint = f"tcp://*:{base_port + i}"
        t = threading.Thread(
            target=_run_worker,
            args=(config, endpoint, coordinator, i),
            name=f"condor-worker-{i}",
            daemon=True,
        )
        t.start()
        threads.append(t)

    logger.info(
        "Started %d workers on ports %d–%d.",
        num_workers,
        base_port,
        base_port + num_workers - 1,
    )

    def _signal_handler(signum: int, frame: object) -> None:
        logger.info("Shutdown signal received (signal %d).", signum)
        coordinator.shutdown_all()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        # Fallback in case KeyboardInterrupt slips through despite the handler.
        coordinator.shutdown_all()
        for t in threads:
            t.join()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    _setup_logging(config.logging.level)

    if config.server.num_workers > 1:
        _run_multi(config)
    else:
        try:
            asyncio.run(_run_single(config))
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
