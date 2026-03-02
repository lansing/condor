"""Native OpenVINO inference backend.

Uses the ``openvino`` Python package directly — NOT the onnxruntime-openvino
execution provider.  Safe to import without OpenVINO installed; OV_SUPPORT will
be False and load() will raise RuntimeError with a clear message.

Config example::

    inference:
      provider: "openvino"
      provider_options:
        device: "CPU"   # or GPU, AUTO, NPU, etc.  Default: CPU.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from .base import BaseBackend, ModelInfo

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
# OpenVINOBackend
# ---------------------------------------------------------------------------

class OpenVINOBackend(BaseBackend):
    """Async-wrapped native OpenVINO inference backend.

    Blocking OpenVINO calls are offloaded via ``asyncio.to_thread`` so the
    event loop is never stalled.

    Lifecycle::

        load()    → ov.Core() → core.read_model() → core.compile_model(device)
                    → compiled.create_infer_request() → extract ModelInfo
        infer()   → request.infer({input_name: tensor})
                    → collect output tensors
        cleanup() → release references (GC handles OV resources)
    """

    def __init__(self) -> None:
        self._compiled: ov.CompiledModel | None = None
        self._request: ov.InferRequest | None = None
        self._model_info: ModelInfo | None = None

    # ------------------------------------------------------------------
    # BaseBackend interface
    # ------------------------------------------------------------------

    async def load(self, model_path: str, config: dict) -> None:
        """Compile the model for the target device in a thread."""
        await asyncio.to_thread(self._load_sync, model_path, config)

    async def infer(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        """Run OpenVINO inference in a thread and return raw output tensors."""
        if self._request is None:
            raise RuntimeError("Backend has no loaded model; call load() first.")
        return await asyncio.to_thread(self._infer_sync, input_tensor)

    async def cleanup(self) -> None:
        """Release all OpenVINO resources."""
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

    def _load_sync(self, model_path: str, config: dict) -> None:
        if not OV_SUPPORT:
            raise RuntimeError(
                "openvino is not installed. "
                "Install it with: uv pip install openvino"
            )

        provider_options = config.get("provider_options", {})
        device = str(provider_options.get("device", "CPU")).upper()
        logger.info(
            "Loading model %s on OpenVINO device %s.", model_path, device
        )

        core = ov.Core()
        model = core.read_model(model_path)
        self._compiled = core.compile_model(model, device)
        self._request = self._compiled.create_infer_request()
        self._model_info = self._extract_model_info()
        logger.info("OpenVINO model loaded: %s", self._model_info)

    def _infer_sync(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        assert self._request is not None
        assert self._model_info is not None

        self._request.infer({self._model_info.input_name: input_tensor})

        return [
            self._request.get_output_tensor(i).data.copy()
            for i in range(len(self._model_info.output_names))
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_model_info(self) -> ModelInfo:
        """Build ModelInfo from the compiled model's tensor metadata."""
        assert self._compiled is not None

        inp = self._compiled.input(0)
        input_name = inp.any_name
        input_shape = _shape_to_list(inp.partial_shape)
        input_dtype = _OV_DTYPE_MAP.get(inp.element_type, "float32")

        output_names: list[str] = []
        output_shapes: list[list[int]] = []
        output_dtypes: list[str] = []
        for i in range(len(self._compiled.outputs)):
            out = self._compiled.output(i)
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
