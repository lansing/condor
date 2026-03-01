"""Abstract backend interface and shared type definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


# Map ONNX Runtime tensor type strings to numpy dtype names.
ONNX_TYPE_TO_NUMPY: dict[str, str] = {
    "tensor(float)": "float32",
    "tensor(float16)": "float16",
    "tensor(double)": "float64",
    "tensor(uint8)": "uint8",
    "tensor(int8)": "int8",
    "tensor(uint16)": "uint16",
    "tensor(int16)": "int16",
    "tensor(int32)": "int32",
    "tensor(int64)": "int64",
    "tensor(bool)": "bool",
}


@dataclass
class ModelInfo:
    """Describes a loaded model's input/output tensor specifications."""

    input_name: str
    input_shape: list[int | str]  # may contain symbolic dims (e.g. "batch_size")
    input_dtype: str              # numpy dtype string, e.g. "float32"

    output_names: list[str] = field(default_factory=list)
    output_shapes: list[list[int | str]] = field(default_factory=list)
    output_dtypes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"ModelInfo(input={self.input_name} {self.input_shape} {self.input_dtype}, "
            f"outputs={list(zip(self.output_names, self.output_shapes, self.output_dtypes))})"
        )


class BaseBackend(ABC):
    """Async plugin interface for inference backends.

    All methods that touch hardware / file I/O must be awaitable so that the
    asyncio event loop is never blocked.
    """

    @abstractmethod
    async def load(self, model_path: str, config: dict) -> None:
        """Load a model from *model_path* using provider settings in *config*."""

    @abstractmethod
    async def infer(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        """Run inference and return the raw output tensor list."""

    @abstractmethod
    async def cleanup(self) -> None:
        """Release all hardware / memory resources held by this backend."""

    @property
    @abstractmethod
    def model_info(self) -> ModelInfo | None:
        """Return :class:`ModelInfo` for the currently loaded model, or None."""
