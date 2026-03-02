"""Abstract backend interface and shared type definitions."""

from __future__ import annotations

import threading
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


def _detect_layout(shape: list) -> str:
    """Infer NCHW vs NHWC from a 4-D tensor shape.

    Heuristic: for a shape ``[N, A, B, C]``, if ``A > C`` the channel axis is
    last (NHWC) because spatial dimensions are always > 3 while the channel
    count is typically 1 or 3.  If ``A <= C`` the channel axis is second
    (NCHW).

    Returns ``"nhwc"`` or ``"nchw"``.  Defaults to ``"nchw"`` for shapes that
    are not 4-D or contain non-integer (symbolic) dimensions.
    """
    if len(shape) != 4:
        return "nchw"
    try:
        dim1 = int(shape[1])
        dim3 = int(shape[3])
    except (ValueError, TypeError):
        return "nchw"  # symbolic dim — cannot determine
    return "nhwc" if dim1 > dim3 else "nchw"


@dataclass
class ModelInfo:
    """Describes a loaded model's input/output tensor specifications."""

    input_name: str
    input_shape: list[int | str]  # may contain symbolic dims (e.g. "batch_size")
    input_dtype: str              # numpy dtype string, e.g. "float32"

    output_names: list[str] = field(default_factory=list)
    output_shapes: list[list[int | str]] = field(default_factory=list)
    output_dtypes: list[str] = field(default_factory=list)

    # Derived from input_shape; not part of the constructor.
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
    """Opaque container for resources shared across worker instances of the same model.

    Backend subclasses define their own typed dataclass that inherits from this.
    The SharedStateRegistry holds one instance per (provider, model_path) key;
    each worker's backend instance receives a reference at load() time.
    """


class BaseBackend(ABC):
    """Async plugin interface for inference backends.

    All methods that touch hardware / file I/O must be awaitable so that the
    asyncio event loop is never blocked.

    Shared-resource protocol
    ------------------------
    When multiple workers load the same model, expensive one-time work
    (engine deserialisation, model compilation) should happen only once.
    Override ``load_shared_sync()`` to return a ``SharedBackendState`` containing
    those resources.  ``load()`` receives the cached state via the ``shared``
    parameter and uses it to initialise per-worker state only.

    ``load_shared_sync()`` is called synchronously inside ``asyncio.to_thread``
    under a ``threading.Lock`` in ``SharedStateRegistry``, so it runs at most
    once regardless of how many workers race to load simultaneously.
    """

    def load_shared_sync(
        self, model_path: str, config: dict
    ) -> SharedBackendState:
        """Load and return resources shared across all worker instances.

        Default implementation returns an empty ``SharedBackendState``
        (no shared resources).  Override for backends with expensive
        one-time initialisation (engine deserialisation, graph compilation).

        Called synchronously; must be thread-safe.  The caller holds a
        ``threading.Lock`` so this is never called concurrently for the
        same (provider, model_path) key.
        """
        return SharedBackendState()

    @abstractmethod
    async def load(
        self,
        model_path: str,
        config: dict,
        shared: SharedBackendState | None = None,
        infer_sem: threading.BoundedSemaphore | None = None,
    ) -> None:
        """Load per-worker resources.

        *shared* contains the pre-loaded shared state from
        ``load_shared_sync()``, or ``None`` in single-worker mode (in which
        case the backend is responsible for loading everything itself).

        *infer_sem* is an optional ``threading.BoundedSemaphore`` that guards
        the hardware inference call.  ``None`` means unlimited concurrency.
        """

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
