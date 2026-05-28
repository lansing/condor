from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path

import aiofiles

from ..backends.base import BaseBackend, ModelInfo
from ..backends.tensorrt_backend import TensorRTBackend
from ..stats import tel
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

    @property
    def backend(self) -> BaseBackend | None:
        return self._backend

    @property
    def active_model(self) -> str | None:
        return self._active_model

    @property
    def model_info(self) -> ModelInfo | None:
        return self._backend.model_info if self._backend is not None else None

    def _make_backend(self) -> BaseBackend:
        return TensorRTBackend()

    def _shared_key(self, model_path: str) -> str:
        return f"tensorrt:{model_path}"

    def model_exists(self, model_name: str) -> bool:
        return (self.models_dir / model_name).exists()

    async def save_model(self, model_name: str, data: bytes) -> bool:
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
        if self._backend is not None:
            return True
        if self._shared_registry is None:
            return False

        prefix = "tensorrt:"
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

    async def auto_load_from_disk(self) -> bool:
        if self._backend is not None:
            return True

        _MODEL_SUFFIXES = {".engine", ".trt"}
        try:
            candidates = sorted(
                p.name
                for p in self.models_dir.iterdir()
                if p.is_file() and p.suffix.lower() in _MODEL_SUFFIXES
            )
        except OSError:
            return False

        if not candidates:
            logger.warning(
                "auto_load_from_disk: no model files in %s.", self.models_dir
            )
            return False

        model_name = candidates[0]
        logger.info(
            "Auto-loading '%s' from disk (Frigate skipped model-request handshake "
            "after Condor restart).",
            model_name,
        )
        return await self.load_model(model_name)

    async def load_model(self, model_name: str) -> bool:
        async with self._lock:
            model_path = self.models_dir / model_name
            if not model_path.exists():
                logger.error("Model file not found: %s", model_path)
                return False

            if self._backend is not None:
                logger.info(
                    "Unloading current model (%s) before loading %s.",
                    self._active_model,
                    model_name,
                )
                await self._backend.cleanup()
                self._backend = None
                self._active_model = None

            try:
                backend = self._make_backend()
                shared = None

                if self._shared_registry is not None:
                    key = self._shared_key(str(model_path))
                    shared = await asyncio.to_thread(
                        self._shared_registry.get_or_load,
                        key,
                        lambda: backend.load_shared_sync(
                            str(model_path), self.inference_config
                        ),
                    )

                await backend.load(
                    str(model_path),
                    self.inference_config,
                    shared=shared,
                    infer_sem=self._infer_sem,
                )
                self._backend = backend
                self._active_model = model_name
                logger.info("Model loaded: %s  info=%s", model_name, backend.model_info)
                tel.set_active_model(model_name)
                return True
            except Exception:
                logger.exception("Failed to load model %s.", model_name)
                self._backend = None
                self._active_model = None
                return False
