"""Abstract post-processor interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BasePostProcessor(ABC):
    """Async plugin interface for model post-processors.

    Implementations should offload heavy CPU work to a thread-pool executor
    (e.g. ``asyncio.to_thread``) so the event loop stays responsive.
    """

    @abstractmethod
    async def process(
        self,
        inference_output: list[np.ndarray],
        input_shape: tuple[int, int],
    ) -> np.ndarray:
        """Convert raw model output to a ``[max_detections, 6]`` float32 array.

        Args:
            inference_output: Raw output tensors from the inference backend.
            input_shape: ``(height, width)`` of the model input tensor (used to
                         normalise bounding-box coordinates to ``[0, 1]``).

        Returns:
            ``np.ndarray`` of shape ``(max_detections, 6)`` with dtype float32.
            Each row: ``[class_id, score, ymin, xmin, ymax, xmax]`` where
            coordinates are normalised to ``[0.0, 1.0]``.
        """
