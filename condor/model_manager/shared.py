"""Process-level registry for shared backend resources.

When multiple workers load the same model, expensive one-time initialisation
(TRT engine deserialisation, OpenVINO graph compilation) should happen only
once.  SharedStateRegistry caches the result of each backend's
``load_shared_sync()`` call and returns it to every subsequent worker.

Threading model
---------------
Workers run in separate OS threads, each with its own asyncio event loop.
The registry uses a plain ``threading.Lock`` (not an asyncio lock) so it is
usable from any thread.  ``get_or_load`` is called via ``asyncio.to_thread``
from within the model manager, so the calling event loop is never blocked.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

from ..backends.base import SharedBackendState

logger = logging.getLogger(__name__)


class SharedStateRegistry:
    """Thread-safe cache of shared backend state keyed by ``"provider:model_path"``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: dict[str, SharedBackendState] = {}

    def get_or_load(
        self,
        key: str,
        loader: Callable[[], SharedBackendState],
    ) -> SharedBackendState:
        """Return the cached state for *key*, or call *loader()* to create it.

        *loader* is called at most once per key; all concurrent callers for
        the same key block on the lock until the first caller's ``loader()``
        returns, then receive the cached result.

        This method is synchronous and is intended to run inside
        ``asyncio.to_thread``.
        """
        with self._lock:
            if key not in self._cache:
                logger.debug("SharedStateRegistry: loading shared state for %r", key)
                self._cache[key] = loader()
                logger.debug("SharedStateRegistry: cached shared state for %r", key)
            else:
                logger.debug("SharedStateRegistry: reusing shared state for %r", key)
            return self._cache[key]

    def contains(self, key: str) -> bool:
        """Return True if *key* is already in the cache (best-effort, for metrics)."""
        with self._lock:
            return key in self._cache

    def invalidate(self, key: str) -> None:
        """Remove *key* from the cache so the next caller triggers a fresh load."""
        with self._lock:
            if self._cache.pop(key, None) is not None:
                logger.debug("SharedStateRegistry: invalidated %r", key)
