"""ONNX Runtime backend.

The provider is always ``"onnx"`` in config; the specific execution provider
(EP) is selected via ``provider_options``:

  provider: "onnx"
  provider_options:
    # Optional — if omitted, ORT uses its default priority order.
    execution_provider: "OpenVINOExecutionProvider"
    # Optional — device hint forwarded to the EP (EP-specific meaning).
    device: "GPU"

If ``execution_provider`` is specified it MUST be available; a RuntimeError is
raised immediately if it is not, rather than silently falling back.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from .base import ONNX_TYPE_TO_NUMPY, BaseBackend, ModelInfo

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    ONNXRUNTIME_SUPPORT = True
except ImportError:
    ort = None  # type: ignore[assignment]
    ONNXRUNTIME_SUPPORT = False


class OnnxRuntimeBackend(BaseBackend):
    """Async-wrapped ONNX Runtime inference backend.

    Blocking ONNX Runtime calls are offloaded to a thread-pool executor via
    ``asyncio.to_thread`` so the event loop is never blocked.
    """

    def __init__(self) -> None:
        self._session: ort.InferenceSession | None = None
        self._model_info: ModelInfo | None = None

    # ------------------------------------------------------------------
    # BaseBackend interface
    # ------------------------------------------------------------------

    async def load(self, model_path: str, config: dict) -> None:
        """Load model from *model_path* in a thread to avoid blocking the loop."""
        await asyncio.to_thread(self._load_sync, model_path, config)

    async def infer(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        """Run ONNX inference in a thread and return output tensor list."""
        if self._session is None:
            raise RuntimeError("Backend has no loaded model; call load() first.")
        return await asyncio.to_thread(self._infer_sync, input_tensor)

    async def cleanup(self) -> None:
        """Release the ONNX session and model metadata."""
        self._session = None
        self._model_info = None
        logger.info("OnnxRuntimeBackend cleaned up.")

    @property
    def model_info(self) -> ModelInfo | None:
        return self._model_info

    # ------------------------------------------------------------------
    # Synchronous helpers (run inside thread-pool executor)
    # ------------------------------------------------------------------

    def _load_sync(self, model_path: str, config: dict) -> None:
        if not ONNXRUNTIME_SUPPORT:
            raise RuntimeError(
                "onnxruntime is not installed. "
                "Install it with: uv pip install onnxruntime"
            )

        provider_options = config.get("provider_options", {})
        requested_ep = provider_options.get("execution_provider")
        device = provider_options.get("device")

        providers = self._resolve_providers(requested_ep, device)
        logger.info(
            "Loading ONNX model %s with providers %s",
            model_path,
            providers if providers is not None else "(ORT defaults)",
        )

        self._session = ort.InferenceSession(model_path, providers=providers)
        self._model_info = self._extract_model_info()
        logger.info("Model loaded successfully: %s", self._model_info)
        # Log the providers that ORT actually activated (may differ from the
        # requested list if some ops fell back to a lower-priority provider).
        logger.info("Active session providers: %s", self._session.get_providers())

    def _infer_sync(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        assert self._session is not None
        assert self._model_info is not None
        return self._session.run(
            None, {self._model_info.input_name: input_tensor}
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_providers(
        self,
        execution_provider: str | None,
        device: str | None,
    ) -> list[str | tuple[str, dict]] | None:
        """Build the ORT providers list from config.

        Returns ``None`` to let ORT use its own default priority order.
        Raises ``RuntimeError`` if a specified EP is not available.
        """
        if execution_provider is None:
            if device is not None:
                logger.warning(
                    "provider_options.device=%r ignored: "
                    "no execution_provider was specified.",
                    device,
                )
            return None  # let ORT pick

        available = set(ort.get_available_providers())
        if execution_provider not in available:
            raise RuntimeError(
                f"Requested execution_provider {execution_provider!r} is not "
                f"available in this environment. "
                f"Available providers: {sorted(available)}"
            )

        opts = self._ep_device_options(execution_provider, device)
        providers: list[str | tuple[str, dict]] = (
            [(execution_provider, opts)] if opts else [execution_provider]
        )
        # Always append CPU as a fallback so ORT can handle any ops not
        # supported by the primary EP.
        if execution_provider != "CPUExecutionProvider":
            providers.append("CPUExecutionProvider")
        return providers

    def _ep_device_options(self, ep: str, device: str | None) -> dict:
        """Return the EP-specific dict for the given device name."""
        if device is None:
            return {}
        if ep == "OpenVINOExecutionProvider":
            return {"device_type": device}
        if ep == "CUDAExecutionProvider":
            return {"device_id": str(device)}
        logger.warning(
            "No device-key mapping known for EP %r; "
            "passing device as a generic 'device' key.",
            ep,
        )
        return {"device": device}

    def _extract_model_info(self) -> ModelInfo:
        assert self._session is not None
        inp = self._session.get_inputs()[0]
        outputs = self._session.get_outputs()

        input_dtype = ONNX_TYPE_TO_NUMPY.get(inp.type, "float32")

        return ModelInfo(
            input_name=inp.name,
            input_shape=list(inp.shape),
            input_dtype=input_dtype,
            output_names=[o.name for o in outputs],
            output_shapes=[list(o.shape) for o in outputs],
            output_dtypes=[
                ONNX_TYPE_TO_NUMPY.get(o.type, "float32") for o in outputs
            ],
        )
