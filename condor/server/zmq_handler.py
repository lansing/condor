"""Async ZMQ REP server and Frigate protocol dispatcher.

Architecture
------------
* A single ``asyncio`` event loop runs ``AsyncZMQHandler.run()``.
* The loop calls ``await socket.recv_multipart()`` (non-blocking thanks to
  ``zmq.asyncio``) and dispatches each request.
* Heavy operations (ONNX inference, file I/O, post-processing) are already
  wrapped in ``asyncio.to_thread`` at lower layers, so the event loop is
  never blocked.

Protocol dispatch
-----------------
Frame 0 (JSON header) determines request type:
  ``model_request`` → model availability check
  ``model_data``    → model file transfer
  anything else     → inference request

State machine
-------------
``_active_model`` tracks the model negotiated during the last successful
Model Management handshake.  All subsequent inference requests target it.
"""

from __future__ import annotations

import asyncio
import json
import logging

import numpy as np
import zmq
import zmq.asyncio

from ..config.settings import AppConfig
from ..model_manager.manager import AsyncModelManager
from ..post_process.yolov10 import YoloV10PostProcessor

logger = logging.getLogger(__name__)

_ZEROS_HEADER = json.dumps({"shape": [20, 6], "dtype": "float32"}).encode()
_ZEROS_BODY = np.zeros((20, 6), dtype=np.float32).tobytes(order="C")


def _zeros_response() -> list[bytes]:
    return [_ZEROS_HEADER, _ZEROS_BODY]


class AsyncZMQHandler:
    """Async ZMQ REP socket + Frigate protocol dispatcher."""

    def __init__(self, config: AppConfig, *, endpoint: str | None = None) -> None:
        self.config = config
        self._endpoint = endpoint or config.server.endpoint
        self.manager = AsyncModelManager(
            models_dir=config.server.models_dir,
            inference_config=config.inference.model_dump(),
        )
        self.post_processor = YoloV10PostProcessor(
            confidence_threshold=config.post_process.confidence_threshold,
            max_detections=config.post_process.max_detections,
        )

        self._ctx = zmq.asyncio.Context()
        self._socket: zmq.asyncio.Socket = self._ctx.socket(zmq.REP)  # type: ignore[attr-defined]
        self._running = False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        endpoint = self._endpoint
        self._socket.bind(endpoint)
        self._running = True
        logger.info("ZMQ server bound to %s. Ready.", endpoint)

        try:
            while self._running:
                try:
                    frames = await self._socket.recv_multipart()
                except asyncio.CancelledError:
                    break
                except zmq.ZMQError as exc:
                    logger.error("ZMQ receive error: %s", exc)
                    continue

                try:
                    response = await self._dispatch(frames)
                except Exception:
                    logger.exception("Unhandled error during dispatch.")
                    response = _zeros_response()

                try:
                    await self._socket.send_multipart(response)
                except zmq.ZMQError as exc:
                    logger.error("ZMQ send error: %s", exc)
        finally:
            # linger=0: discard unsent messages immediately so ctx.term() doesn't block.
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.close()
            self._ctx.term()
            logger.info("ZMQ server shut down.")

    async def shutdown(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    async def _dispatch(self, frames: list[bytes]) -> list[bytes]:
        try:
            header: dict = json.loads(frames[0])
        except Exception:
            logger.exception("Failed to parse request header.")
            return _zeros_response()

        if "model_request" in header:
            return await self._handle_model_request(header)

        if "model_data" in header:
            data = frames[1] if len(frames) > 1 else b""
            return await self._handle_model_data(header, data)

        # Inference request
        tensor_bytes = frames[1] if len(frames) > 1 else None
        return await self._handle_inference(header, tensor_bytes)

    # ------------------------------------------------------------------
    # Model management handlers
    # ------------------------------------------------------------------

    async def _handle_model_request(self, header: dict) -> list[bytes]:
        """Stage 1A – model availability check."""
        model_name: str = header.get("model_name", "")

        model_available = self.manager.model_exists(model_name)
        model_loaded = False

        if model_available:
            # Try to load; if already the active model this is a no-op lock acquire
            if self.manager.active_model != model_name:
                model_loaded = await self.manager.load_model(model_name)
            else:
                model_loaded = True
        else:
            logger.info(
                "Model '%s' not found locally; Frigate will transfer it.", model_name
            )

        resp = {"model_available": model_available, "model_loaded": model_loaded}
        logger.info("Model request '%s' → %s", model_name, resp)
        return [json.dumps(resp).encode()]

    async def _handle_model_data(self, header: dict, data: bytes) -> list[bytes]:
        """Stage 1B – receive model file from Frigate and load it."""
        model_name: str = header.get("model_name", "")

        saved = await self.manager.save_model(model_name, data)
        model_loaded = False
        if saved:
            model_loaded = await self.manager.load_model(model_name)

        resp = {"model_saved": saved, "model_loaded": model_loaded}
        logger.info("Model data '%s' → %s", model_name, resp)
        return [json.dumps(resp).encode()]

    # ------------------------------------------------------------------
    # Inference handler
    # ------------------------------------------------------------------

    async def _handle_inference(
        self, header: dict, tensor_bytes: bytes | None
    ) -> list[bytes]:
        """Stage 2 – run inference on the active model."""
        if tensor_bytes is None:
            logger.error("Inference request missing tensor frame.")
            return _zeros_response()

        backend = self.manager.backend
        if backend is None:
            logger.warning("Inference request received but no model is loaded.")
            return _zeros_response()

        model_info = self.manager.model_info
        if model_info is None:
            logger.warning("Backend loaded but model_info is unavailable.")
            return _zeros_response()

        # --- dtype validation ---
        request_dtype: str = header.get("dtype", "uint8")
        expected_dtype: str = model_info.input_dtype
        if request_dtype != expected_dtype:
            logger.error(
                "Input dtype mismatch: Frigate sent '%s', model expects '%s'. "
                "Returning zero detections.",
                request_dtype,
                expected_dtype,
            )
            return _zeros_response()

        # --- reconstruct tensor ---
        try:
            shape = tuple(int(d) for d in header["shape"])
            tensor = np.frombuffer(tensor_bytes, dtype=request_dtype).reshape(shape)
        except Exception:
            logger.exception(
                "Failed to reconstruct input tensor from header %s.", header
            )
            return _zeros_response()

        # --- inference ---
        try:
            outputs = await backend.infer(tensor)
        except Exception:
            logger.exception("Inference failed.")
            return _zeros_response()

        # --- post-process ---
        # Extract spatial dims respecting the model's declared input layout.
        try:
            if model_info.input_layout == "nhwc":
                # [N, H, W, C] → H = shape[1], W = shape[2]
                input_h = int(shape[1])
                input_w = int(shape[2])
            else:
                # [N, C, H, W] → H = shape[2], W = shape[3]
                input_h = int(shape[2])
                input_w = int(shape[3])
            result = await self.post_processor.process(outputs, (input_h, input_w))
        except Exception:
            logger.exception("Post-processing failed.")
            return _zeros_response()

        resp_header = {"shape": list(result.shape), "dtype": "float32"}
        return [json.dumps(resp_header).encode(), result.tobytes(order="C")]
