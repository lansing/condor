from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


def _detect_layout(shape: list) -> str:
    """Infer NCHW vs NHWC from tensor shape.

    Heuristic: for a shape ``[N, A, B, C]``, if ``A > C`` the channel axis is
    last (NHWC) because spatial dimensions are always > 3 while the channel
    count is typically 1 or 3.  If ``A <= C`` the channel axis is second
    (NCHW).
    """
    if len(shape) != 4:
        raise Exception(f"Tried to detect layout of tensor with {len(shape)} dims, it needs to be 4 dim (either nchw or nhwc)")
    dim1 = int(shape[1])
    dim3 = int(shape[3])
    return "nhwc" if dim1 > dim3 else "nchw"


@dataclass
class ModelInfo:
    input_name: str
    input_shape: list[int | str]
    input_dtype: str

    output_names: list[str] = field(default_factory=list)
    output_shapes: list[list[int | str]] = field(default_factory=list)
    output_dtypes: list[str] = field(default_factory=list)

    input_layout: str = field(default="nchw", init=False)

    def __post_init__(self) -> None:
        self.input_layout = _detect_layout(self.input_shape)

    def __str__(self) -> str:
        return (
            f"ModelInfo(input={self.input_name} {self.input_shape} "
            f"{self.input_dtype} {self.input_layout}, "
            f"outputs={list(zip(self.output_names, self.output_shapes, self.output_dtypes))})"
        )


@dataclass
class SharedBackendState:
    pass

class BaseBackend(ABC):


    def load_shared_sync(
        self, model_path: str, config: dict
    ) -> SharedBackendState:
        """Load and return resources shared across all worker instances."""
        return SharedBackendState()

    @abstractmethod
    async def load(
        self,
        model_path: str,
        config: dict,
        shared: SharedBackendState | None = None,
        infer_sem: threading.BoundedSemaphore | None = None,
    ) -> None:
        """Load per-worker resources."""

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
