"""YOLOv10 post-processor.

YOLOv10 models exported from Ultralytics output a single tensor of shape
``(1, N, 6)`` where each detection row is:
    ``[x1, y1, x2, y2, confidence, class_id]``

Coordinates are in pixel space scaled to the model input size (e.g. 320×320).
The model already performs NMS internally, so no external NMS step is needed.

Post-processing steps:
    1. Squeeze the batch dimension → shape ``(N, 6)``.
    2. Filter rows where ``confidence >= confidence_threshold``.
    3. Normalise box coordinates by the input tensor dimensions (÷ width/height).
    4. Clip coordinates to ``[0.0, 1.0]``.
    5. Pack into a ``(max_detections, 6)`` float32 output as
       ``[class_id, score, ymin, xmin, ymax, xmax]``.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from .base import BasePostProcessor

logger = logging.getLogger(__name__)


class YoloV10PostProcessor(BasePostProcessor):
    """Post-processor for YOLOv10 ONNX models with ``(1, N, 6)`` output."""

    def __init__(
        self,
        confidence_threshold: float = 0.4,
        max_detections: int = 20,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.max_detections = max_detections

    async def process(
        self,
        inference_output: list[np.ndarray],
        input_shape: tuple[int, int],
    ) -> np.ndarray:
        """Async entry point — delegates CPU work to a thread-pool executor."""
        return await asyncio.to_thread(
            self._process_sync, inference_output, input_shape
        )

    # ------------------------------------------------------------------
    # Synchronous implementation (runs inside thread-pool executor)
    # ------------------------------------------------------------------

    def _process_sync(
        self,
        inference_output: list[np.ndarray],
        input_shape: tuple[int, int],
    ) -> np.ndarray:
        result = np.zeros((self.max_detections, 6), dtype=np.float32)

        if not inference_output:
            logger.warning("YoloV10PostProcessor: empty inference output.")
            return result

        raw = inference_output[0]  # shape: (1, N, 6) or (N, 6)

        # Normalise to float32 regardless of model output dtype (float16, etc.)
        raw = raw.astype(np.float32)

        # Squeeze batch dimension if present
        if raw.ndim == 3:
            raw = raw[0]  # (N, 6)

        if raw.ndim != 2 or raw.shape[1] != 6:
            logger.error(
                "YoloV10PostProcessor: unexpected output shape %s; expected (N, 6).",
                raw.shape,
            )
            return result

        input_h, input_w = input_shape

        # Filter by confidence (column index 4)
        confidences = raw[:, 4]
        mask = confidences >= self.confidence_threshold
        filtered = raw[mask]

        if filtered.shape[0] == 0:
            return result

        # Sort by confidence descending and take top max_detections
        order = np.argsort(filtered[:, 4])[::-1][: self.max_detections]
        top = filtered[order]

        for i, det in enumerate(top):
            x1, y1, x2, y2, confidence, class_id = det

            ymin = float(y1) / input_h
            xmin = float(x1) / input_w
            ymax = float(y2) / input_h
            xmax = float(x2) / input_w

            result[i] = [
                float(class_id),
                float(confidence),
                max(0.0, min(1.0, ymin)),
                max(0.0, min(1.0, xmin)),
                max(0.0, min(1.0, ymax)),
                max(0.0, min(1.0, xmax)),
            ]

        n_dets = len(top)
        if n_dets:
            logger.debug(
                "YoloV10PostProcessor: %d detection(s) above threshold %.2f.",
                n_dets,
                self.confidence_threshold,
            )

        return result
