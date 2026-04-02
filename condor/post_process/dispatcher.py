"""Auto-detecting post-processor that dispatches to the appropriate model-specific handler."""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from .base import BasePostProcessor
from .yolov10 import YoloV10PostProcessor
from .yolov9 import YoloV9PostProcessor

logger = logging.getLogger(__name__)


class DispatchedPostProcessor(BasePostProcessor):
    """Auto-detecting post-processor that routes to YoloV9 or YoloV10 based on output shape.

    Init args match YoloV10PostProcessor for drop-in replacement compatibility.
    """

    short_name = "Auto"

    def __init__(
        self,
        confidence_threshold: float = 0.4,
        max_detections: int = 20,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.max_detections = max_detections
        self._last_processor_type: str | None = None
        self._active_short_name: str = "?"

    @property
    def active_short_name(self) -> str:
        """Return the short name of the currently active post-processor."""
        return self._active_short_name

    async def process(
        self,
        inference_output: list[np.ndarray],
        input_shape: tuple[int, int],
    ) -> np.ndarray:
        """Auto-detect model type and delegate to appropriate post-processor."""
        return await asyncio.to_thread(
            self._process_sync, inference_output, input_shape
        )

    def _process_sync(
        self,
        inference_output: list[np.ndarray],
        input_shape: tuple[int, int],
    ) -> np.ndarray:
        if not inference_output:
            logger.warning("DispatchedPostProcessor: empty inference output.")
            return np.zeros((self.max_detections, 6), dtype=np.float32)

        model_type = self.detect_output_type(inference_output)

        if model_type == "yolov10":
            processor = YoloV10PostProcessor(
                confidence_threshold=self.confidence_threshold,
                max_detections=self.max_detections,
            )
            processor_name = "YoloV10PostProcessor"
            self._active_short_name = processor.short_name
        elif model_type == "yolov9":
            processor = YoloV9PostProcessor(
                confidence_threshold=self.confidence_threshold,
                nms_threshold=0.4,
                max_detections=self.max_detections,
            )
            processor_name = "YoloV9PostProcessor"
            self._active_short_name = processor.short_name
        else:
            logger.error(
                "DispatchedPostProcessor: unknown model type '%s', returning zeros.",
                model_type,
            )
            return np.zeros((self.max_detections, 6), dtype=np.float32)

        if self._last_processor_type is None:
            logger.info(
                "DispatchedPostProcessor: detected %s output, using %s",
                model_type,
                processor_name,
            )
        elif self._last_processor_type != model_type:
            logger.warning(
                "DispatchedPostProcessor: model output format changed from %s to %s, switching to %s",
                self._last_processor_type,
                model_type,
                processor_name,
            )

        self._last_processor_type = model_type

        return processor._process_sync(inference_output, input_shape)

    @staticmethod
    def detect_output_type(inference_output: list[np.ndarray]) -> str:
        """Detect model type from output tensor shapes.

        Detection rules:
        - YoloV10: Shape (1, N, 6) or (N, 6) - last dimension is 6
        - YoloV9:  Shape (1, num_attrs, num_preds) where num_attrs >= 5 and != 6
        - Unknown: Cannot determine from shape

        Args:
            inference_output: List of raw output tensors from inference

        Returns:
            'yolov10', 'yolov9', or 'unknown'
        """
        if not inference_output:
            return "unknown"

        raw = inference_output[0]

        if raw.ndim == 3:
            if raw.shape[0] == 1:
                if raw.shape[2] == 6:
                    return "yolov10"
                num_attrs = raw.shape[1]
                if num_attrs >= 5:
                    return "yolov9"

        elif raw.ndim == 2:
            if raw.shape[1] == 6:
                return "yolov10"
            num_attrs = raw.shape[0]
            if num_attrs >= 5:
                return "yolov9"

        return "unknown"
