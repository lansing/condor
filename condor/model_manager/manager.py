"""Async model lifecycle manager.

Responsibilities:
  - Check whether a model file exists in the models directory.
  - Save model bytes received from Frigate (async, via aiofiles).
  - Load a model into the inference backend (async, with cleanup of the
    previously loaded model to free VRAM / memory before loading the new one).
  - Expose the active backend for inference calls.

An asyncio.Lock guards all load/unload operations so that concurrent model
management messages cannot corrupt state.

Shared-resource protocol
------------------------
When a SharedStateRegistry is supplied (multi-worker mode), load_model()
calls backend.load_shared_sync() via the registry — ensuring that expensive
one-time work (engine deserialisation, model compilation) happens at most once
regardless of how many workers race to load simultaneously.  The resulting
SharedBackendState is passed to backend.load() so each worker can set up its
own per-worker resources (execution context, infer request, I/O buffers).

In single-worker mode (no registry), backend.load() receives shared=None and
is responsible for doing everything itself.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path

import aiofiles

from ..backends.base import BaseBackend, ModelInfo
from ..backends.onnx_backend import OnnxRuntimeBackend
from ..backends.openvino_backend import OpenVINOBackend
from ..backends.tensorrt_backend import TensorRTBackend
from ..telemetry import tel, tracer
from opentelemetry.trace import StatusCode
from .shared import SharedStateRegistry

logger = logging.getLogger(__name__)


class AsyncModelManager:
    def __init__(
        self,
        models_dir: str,
        inference_config: dict,
        shared_registry: SharedStateRegistry | None = None,
        infer_sem: threading.BoundedSemaphore | None = None,
    ) -> None:
        self.models_dir = Path(models_dir)
        self.inference_config = inference_config
        self._shared_registry = shared_registry
        self._infer_sem = infer_sem

        self._backend: BaseBackend | None = None
        self._active_model: str | None = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def backend(self) -> BaseBackend | None:
        return self._backend

    @property
    def active_model(self) -> str | None:
        return self._active_model

    @property
    def model_info(self) -> ModelInfo | None:
        return self._backend.model_info if self._backend is not None else None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_backend(self) -> BaseBackend:
        """Instantiate the correct backend based on the configured provider."""
        provider = self.inference_config.get("provider", "onnx").lower()
        if provider == "tensorrt":
            return TensorRTBackend()
        if provider == "openvino":
            return OpenVINOBackend()
        if provider == "cpu":
            # Legacy alias — treated as ONNX Runtime with default EP selection.
            logger.warning(
                "provider: 'cpu' is deprecated; use provider: 'onnx' instead."
            )
            return OnnxRuntimeBackend()
        return OnnxRuntimeBackend()

    def _shared_key(self, model_path: str) -> str:
        provider = self.inference_config.get("provider", "onnx").lower()
        return f"{provider}:{model_path}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def model_exists(self, model_name: str) -> bool:
        """Return True if *model_name* exists in the models directory."""
        return (self.models_dir / model_name).exists()

    async def save_model(self, model_name: str, data: bytes) -> bool:
        """Write *data* to ``<models_dir>/<model_name>`` asynchronously."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.models_dir / model_name
        try:
            async with aiofiles.open(model_path, "wb") as f:
                await f.write(data)
            logger.info("Model saved: %s (%d bytes)", model_name, len(data))
            return True
        except Exception:
            logger.exception("Failed to save model %s.", model_name)
            return False

    async def lazy_load_from_registry(self) -> bool:
        """Load per-worker resources from the shared registry cache without waiting
        for Frigate to explicitly send a model-request to this worker.

        In multi-worker mode Frigate addresses only one worker with the initial
        model-request message.  All other workers must self-initialise on first
        inference so they can serve requests immediately.  The expensive shared
        work (engine deserialisation, etc.) is already cached; only the cheap
        per-worker setup (execution context, I/O buffers) is repeated here.

        Returns True if a model is now loaded (already was, or just finished).
        """
        if self._backend is not None:
            return True
        if self._shared_registry is None:
            return False

        provider = self.inference_config.get("provider", "onnx").lower()
        prefix = f"{provider}:"
        for key in self._shared_registry.cached_keys():
            if key.startswith(prefix):
                # key = "tensorrt:/abs/path/to/model.engine"
                model_path = key[len(prefix):]
                model_name = Path(model_path).name
                logger.info(
                    "Worker has no model; lazy-loading %s from registry cache.",
                    model_name,
                )
                return await self.load_model(model_name)

        return False

    async def load_model(self, model_name: str) -> bool:
        """Load *model_name* into the backend, replacing any currently loaded model.

        The lock ensures sequential load/unload even if multiple coroutines
        race to load at the same time.

        When a SharedStateRegistry is present, expensive one-time resources
        (engine deserialisation, model compilation) are loaded via the registry
        and reused across all workers.
        """
        provider = self.inference_config.get("provider", "onnx")

        with tracer.start_as_current_span("condor.model.load") as load_span:
            load_span.set_attribute("model_name", model_name)
            load_span.set_attribute("provider", provider)

            t_lock = time.perf_counter()
            async with self._lock:
                lock_wait_ms = (time.perf_counter() - t_lock) * 1000
                tel.record_model_lock_wait(lock_wait_ms)
                load_span.set_attribute("lock_wait_ms", round(lock_wait_ms, 2))

                model_path = self.models_dir / model_name
                if not model_path.exists():
                    logger.error("Model file not found: %s", model_path)
                    load_span.set_status(StatusCode.ERROR, "model file not found")
                    tel.count_model_load(model_name=model_name, provider=provider, status="error")
                    return False

                # Cleanup the existing backend before loading the new model so that
                # GPU/NPU memory is released first (requirement §3.3).
                if self._backend is not None:
                    logger.info(
                        "Unloading current model (%s) before loading %s.",
                        self._active_model,
                        model_name,
                    )
                    with tracer.start_as_current_span("condor.model.cleanup") as cleanup_span:
                        cleanup_span.set_attribute("previous_model", self._active_model or "")
                        await self._backend.cleanup()
                    self._backend = None
                    self._active_model = None

                try:
                    backend = self._make_backend()
                    shared = None

                    if self._shared_registry is not None:
                        # Load shared resources via the registry (at most once per
                        # model across all workers; subsequent workers get cached state).
                        key = self._shared_key(str(model_path))

                        # Probe whether a cache hit is likely (rough check; the
                        # real serialised check is inside get_or_load under its lock).
                        cache_hit_before = self._shared_registry.contains(key)

                        with tracer.start_as_current_span("condor.shared_state.get_or_load") as ss_span:
                            ss_span.set_attribute("model_key", key)
                            shared = await asyncio.to_thread(
                                self._shared_registry.get_or_load,
                                key,
                                lambda: backend.load_shared_sync(
                                    str(model_path), self.inference_config
                                ),
                            )

                        if cache_hit_before:
                            ss_span.set_attribute("cache_hit", True)
                            tel.count_cache_hit(provider=provider, model_name=model_name)
                        else:
                            ss_span.set_attribute("cache_hit", False)
                            tel.count_cache_miss(provider=provider, model_name=model_name)

                    with tracer.start_as_current_span("condor.backend.load"):
                        await backend.load(
                            str(model_path),
                            self.inference_config,
                            shared=shared,
                            infer_sem=self._infer_sem,
                        )
                    self._backend = backend
                    self._active_model = model_name
                    logger.info("Model loaded: %s  info=%s", model_name, backend.model_info)
                    tel.count_model_load(model_name=model_name, provider=provider, status="ok")
                    tel.set_active_model(model_name)
                    return True
                except Exception:
                    logger.exception("Failed to load model %s.", model_name)
                    load_span.set_status(StatusCode.ERROR, "backend load failed")
                    tel.count_model_load(model_name=model_name, provider=provider, status="error")
                    self._backend = None
                    self._active_model = None
                    return False
