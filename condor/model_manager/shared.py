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
        with self._lock:
            if key not in self._cache:
                logger.debug("SharedStateRegistry: loading shared state for %r", key)
                self._cache[key] = loader()
                logger.debug("SharedStateRegistry: cached shared state for %r", key)
            else:
                logger.debug("SharedStateRegistry: reusing shared state for %r", key)
            return self._cache[key]

    def contains(self, key: str) -> bool:
        with self._lock:
            return key in self._cache

    def cached_keys(self) -> list[str]:
        with self._lock:
            return list(self._cache.keys())

    def invalidate(self, key: str) -> None:
        with self._lock:
            if self._cache.pop(key, None) is not None:
                logger.debug("SharedStateRegistry: invalidated %r", key)
