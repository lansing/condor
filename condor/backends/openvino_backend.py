"""Native OpenVINO inference backend.

Uses the ``openvino`` Python package directly — NOT the onnxruntime-openvino
execution provider.  Safe to import without OpenVINO installed; OV_SUPPORT will
be False and load() will raise RuntimeError with a clear message.

Config example::

    inference:
      provider: "openvino"
      provider_options:
        device: "CPU"   # or GPU, AUTO, NPU, etc.  Default: CPU.

Shared-resource model
---------------------
``ov.CompiledModel`` is documented as thread-safe: multiple ``InferRequest``
objects can be created from one ``CompiledModel`` and used concurrently.  The
expensive compile step (device JIT, graph optimisation, NPU/GPU upload) runs
only once per (provider, model_path) key, shared across all workers.

``OVSharedState`` (created in ``load_shared_sync()``) holds the compiled model
and extracted ``ModelInfo``.  Each worker's ``load()`` calls
``compiled.create_infer_request()`` to obtain its own ``InferRequest``.
``cleanup()`` releases the per-worker request only.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from .base import BaseBackend, ModelInfo, SharedBackendState
from ..telemetry import tel, tracer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional import — guarded so the server starts without OpenVINO installed.
# ---------------------------------------------------------------------------
try:
    import openvino as ov

    OV_SUPPORT = True

    # Map OpenVINO element types → numpy dtype strings.
    _OV_DTYPE_MAP: dict = {
        ov.Type.f32:     "float32",
        ov.Type.f16:     "float16",
        ov.Type.f64:     "float64",
        ov.Type.u8:      "uint8",
        ov.Type.u16:     "uint16",
        ov.Type.u32:     "uint32",
        ov.Type.u64:     "uint64",
        ov.Type.i8:      "int8",
        ov.Type.i16:     "int16",
        ov.Type.i32:     "int32",
        ov.Type.i64:     "int64",
        ov.Type.boolean: "bool",
    }
except ImportError:
    ov = None          # type: ignore[assignment]
    OV_SUPPORT = False
    _OV_DTYPE_MAP = {}


# ---------------------------------------------------------------------------
# Shape helper
# ---------------------------------------------------------------------------

def _shape_to_list(partial_shape) -> list[int]:
    """Convert an OV PartialShape to a plain int list (dynamic dims → -1)."""
    return [
        -1 if dim.is_dynamic else int(dim)
        for dim in partial_shape
    ]


# ---------------------------------------------------------------------------
# Shared state dataclass
# ---------------------------------------------------------------------------

@dataclass
class OVSharedState(SharedBackendState):
    """Resources shared across all OpenVINO worker instances for one model."""
    compiled: object    # ov.CompiledModel (thread-safe for create_infer_request)
    model_info: ModelInfo


# ---------------------------------------------------------------------------
# OpenVINOBackend
# ---------------------------------------------------------------------------

class OpenVINOBackend(BaseBackend):
    """Async-wrapped native OpenVINO inference backend.

    Blocking OpenVINO calls are offloaded via ``asyncio.to_thread`` so the
    event loop is never stalled.

    Lifecycle::

        load_shared_sync()  [once]
          → ov.Core() → core.read_model() → core.compile_model(device)
          → extract ModelInfo
          → return OVSharedState(compiled, model_info)

        load()  [per worker]
          → compiled.create_infer_request()  (per-worker InferRequest)

        infer()  [per request]
          → [acquire infer_sem] request.infer() [release]
          → collect output tensors

        cleanup()  [per worker]
          → release InferRequest reference only
    """

    def __init__(self) -> None:
        self._compiled: ov.CompiledModel | None = None
        self._request: ov.InferRequest | None = None
        self._model_info: ModelInfo | None = None
        self._infer_sem: threading.BoundedSemaphore | None = None

    # ------------------------------------------------------------------
    # Shared-resource interface
    # ------------------------------------------------------------------

    def load_shared_sync(self, model_path: str, config: dict) -> OVSharedState:
        """Compile the model once; result shared across all workers."""
        if not OV_SUPPORT:
            raise RuntimeError(
                "openvino is not installed. "
                "Install it with: uv pip install openvino"
            )

        provider_options = config.get("provider_options", {})
        device = str(provider_options.get("device", "CPU")).upper()
        logger.info(
            "Compiling shared OpenVINO model %s on device %s.", model_path, device
        )

        core = ov.Core()
        model = core.read_model(model_path)
        compiled = core.compile_model(model, device)
        model_info = self._extract_model_info(compiled)
        logger.info("OpenVINO shared state ready: %s", model_info)
        return OVSharedState(compiled=compiled, model_info=model_info)

    # ------------------------------------------------------------------
    # BaseBackend interface
    # ------------------------------------------------------------------

    async def load(
        self,
        model_path: str,
        config: dict,
        shared: SharedBackendState | None = None,
        infer_sem: threading.BoundedSemaphore | None = None,
    ) -> None:
        """Create a per-worker InferRequest from the shared CompiledModel."""
        await asyncio.to_thread(self._load_sync, model_path, config, shared, infer_sem)

    async def infer(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        """Run OpenVINO inference in a thread and return raw output tensors."""
        if self._request is None:
            raise RuntimeError("Backend has no loaded model; call load() first.")
        return await asyncio.to_thread(self._infer_sync, input_tensor)

    async def cleanup(self) -> None:
        """Release the per-worker InferRequest."""
        self._request = None
        self._compiled = None
        self._model_info = None
        logger.info("OpenVINOBackend cleaned up.")

    @property
    def model_info(self) -> ModelInfo | None:
        return self._model_info

    # ------------------------------------------------------------------
    # Synchronous helpers — all run inside asyncio.to_thread
    # ------------------------------------------------------------------

    def _load_sync(
        self,
        model_path: str,
        config: dict,
        shared: OVSharedState | None,
        infer_sem: threading.BoundedSemaphore | None,
    ) -> None:
        if shared is None:
            # Single-worker mode: compile inline (no registry).
            shared = self.load_shared_sync(model_path, config)

        # Borrow compiled model reference.
        self._compiled = shared.compiled
        self._model_info = shared.model_info
        self._infer_sem = infer_sem

        # Each worker gets its own InferRequest (not thread-safe to share).
        self._request = self._compiled.create_infer_request()
        logger.info("OpenVINO worker ready (InferRequest created).")

    def _infer_sync(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        assert self._request is not None
        assert self._model_info is not None

        if self._infer_sem is not None:
            t_sem = time.perf_counter()
            with tracer.start_as_current_span("condor.infer_sem.wait"):
                self._infer_sem.acquire()
            tel.record_sem_wait((time.perf_counter() - t_sem) * 1000)
        tel.inc_inference_concurrent()
        try:
            with tracer.start_as_current_span("condor.ov.infer"):
                self._request.infer({self._model_info.input_name: input_tensor})
        finally:
            tel.dec_inference_concurrent()
            if self._infer_sem is not None:
                self._infer_sem.release()

        return [
            self._request.get_output_tensor(i).data.copy()
            for i in range(len(self._model_info.output_names))
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_model_info(compiled) -> ModelInfo:
        """Build ModelInfo from a compiled model's tensor metadata."""
        inp = compiled.input(0)
        input_name = inp.any_name
        input_shape = _shape_to_list(inp.partial_shape)
        input_dtype = _OV_DTYPE_MAP.get(inp.element_type, "float32")

        output_names: list[str] = []
        output_shapes: list[list[int]] = []
        output_dtypes: list[str] = []
        for i in range(len(compiled.outputs)):
            out = compiled.output(i)
            output_names.append(out.any_name)
            output_shapes.append(_shape_to_list(out.partial_shape))
            output_dtypes.append(_OV_DTYPE_MAP.get(out.element_type, "float32"))

        return ModelInfo(
            input_name=input_name,
            input_shape=input_shape,
            input_dtype=input_dtype,
            output_names=output_names,
            output_shapes=output_shapes,
            output_dtypes=output_dtypes,
        )
