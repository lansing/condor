from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BasePostProcessor(ABC):
    short_name: str = "?"

    @abstractmethod
    async def process(
        self,
        inference_output: list[np.ndarray],
        input_shape: tuple[int, int],
    ) -> np.ndarray:
        """Convert raw model output to a ``[max_detections, 6]`` float32 array."""
