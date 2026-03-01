"""Entry point for the Frigate remote detector server."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys

from ..config.settings import load_config
from .zmq_handler import AsyncZMQHandler


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


async def _run(config_path: str) -> None:
    config = load_config(config_path)
    _setup_logging(config.logging.level)

    handler = AsyncZMQHandler(config)

    loop = asyncio.get_running_loop()

    # Graceful shutdown on SIGINT / SIGTERM
    def _signal_handler() -> None:
        logging.getLogger(__name__).info("Shutdown signal received.")
        asyncio.create_task(handler.shutdown())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    await handler.run()


def main() -> None:
    args = _parse_args()
    try:
        asyncio.run(_run(args.config))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
