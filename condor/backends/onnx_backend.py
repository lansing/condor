"""ONNX Runtime backend supporting CPU and Intel OpenVINO execution providers."""

from __future__ import annotations

import asyncio
import logging

import numpy as np
import onnxruntime as ort

from .base import ONNX_TYPE_TO_NUMPY, BaseBackend, ModelInfo

logger = logging.getLogger(__name__)


class OnnxRuntimeBackend(BaseBackend):
    """Async-wrapped ONNX Runtime inference backend.

    Blocking ONNX Runtime calls are offloaded to a thread-pool executor via
    ``asyncio.to_thread`` so the event loop is never blocked.

    Supported providers (configured via *config["provider"]*):
      - ``"cpu"``      → CPUExecutionProvider
      - ``"openvino"`` → OpenVINOExecutionProvider (falls back to CPU if unavailable)
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
        provider_name = config.get("provider", "cpu").lower()
        provider_options = config.get("provider_options", {})

        providers = self._resolve_providers(provider_name, provider_options)
        logger.info(
            "Loading ONNX model %s with providers %s",
            model_path,
            [p if isinstance(p, str) else p[0] for p in providers],
        )

        self._session = ort.InferenceSession(model_path, providers=providers)
        self._model_info = self._extract_model_info()
        logger.info("Model loaded successfully: %s", self._model_info)

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
        self, provider: str, options: dict
    ) -> list[str | tuple[str, dict]]:
        """Build the ONNX Runtime provider list from config.

        If the requested provider is unavailable, warn and fall back to CPU.
        """
        available = set(ort.get_available_providers())

        if provider == "openvino":
            if "OpenVINOExecutionProvider" not in available:
                logger.warning(
                    "OpenVINOExecutionProvider is not available "
                    "(install onnxruntime-openvino). Falling back to CPU."
                )
                return ["CPUExecutionProvider"]

            ov_opts: dict = {
                "device_type": options.get("device_type", "CPU"),
            }
            logger.info("OpenVINO EP options: %s", ov_opts)
            return [
                ("OpenVINOExecutionProvider", ov_opts),
                "CPUExecutionProvider",
            ]

        if provider != "cpu":
            logger.warning(
                "Unknown provider %r; falling back to CPU.", provider
            )

        return ["CPUExecutionProvider"]

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
