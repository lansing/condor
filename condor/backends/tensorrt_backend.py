"""TensorRT inference backend.

All blocking GPU operations are offloaded via asyncio.to_thread so the event
loop is never stalled.  Uses the CUDA driver API (cuda-python) for explicit
memory management and TensorRT Python bindings for engine execution.

This module is safe to import in environments without TensorRT installed;
TRT_SUPPORT will be False and load() will raise RuntimeError with a clear
message.
"""

from __future__ import annotations

import asyncio
import ctypes
import logging

import numpy as np

from .base import BaseBackend, ModelInfo

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — guarded so the server starts without TRT installed.
# ---------------------------------------------------------------------------
try:
    import tensorrt as trt
    from cuda.bindings import driver as cu

    TRT_SUPPORT = True
    _ILogger_base: type = trt.ILogger
except ImportError:
    TRT_SUPPORT = False
    trt = None  # type: ignore[assignment]
    cu = None   # type: ignore[assignment]
    _ILogger_base = object


# ---------------------------------------------------------------------------
# CUDA helpers
# ---------------------------------------------------------------------------

def _check(retval, op: str) -> None:
    """Raise RuntimeError on a non-zero CUDA result code.

    Accepts either a bare CUresult or a tuple whose first element is CUresult
    (the shape returned by most cuda-python driver API functions).
    """
    code = retval[0] if isinstance(retval, tuple) else retval
    if int(code) != 0:
        raise RuntimeError(f"CUDA error in {op}: error code {int(code)}")


def _unwrap(retval, op: str):
    """Check for error and return the second element of a 2-tuple result."""
    _check(retval[0], op)
    return retval[1]


# ---------------------------------------------------------------------------
# TrtLogger
# ---------------------------------------------------------------------------

class TrtLogger(_ILogger_base):
    """Bridges TensorRT log messages into Python's logging module."""

    def log(self, severity, msg: str) -> None:
        if not TRT_SUPPORT:
            return
        if severity == trt.ILogger.Severity.VERBOSE:
            logger.debug("[TRT] %s", msg)
        elif severity == trt.ILogger.Severity.INFO:
            logger.info("[TRT] %s", msg)
        elif severity == trt.ILogger.Severity.WARNING:
            logger.warning("[TRT] %s", msg)
        elif severity == trt.ILogger.Severity.ERROR:
            logger.error("[TRT] %s", msg)
        else:  # INTERNAL_ERROR
            logger.critical("[TRT] %s", msg)


# ---------------------------------------------------------------------------
# HostDeviceMem
# ---------------------------------------------------------------------------

class HostDeviceMem:
    """Paired page-locked host buffer and CUDA device buffer.

    The host buffer is allocated with cuMemHostAlloc (CU_MEMHOSTALLOC_PORTABLE)
    so it is pinned for fast DMA.  A numpy array view over the host buffer
    allows zero-copy fill of input data.
    """

    # CU_MEMHOSTALLOC_PORTABLE = 0x01 — pinned, any-context accessible.
    _ALLOC_FLAGS = 0x01

    def __init__(self, size: int, dtype: np.dtype) -> None:
        nbytes = int(size * dtype.itemsize)
        self._nbytes = nbytes

        # Allocate pinned host memory.
        err, self._host_ptr = cu.cuMemHostAlloc(nbytes, self._ALLOC_FLAGS)
        _check(err, "cuMemHostAlloc")

        # Allocate device memory.
        err, self._dev_ptr = cu.cuMemAlloc(nbytes)
        _check(err, "cuMemAlloc")

        # Build a numpy view over the pinned host memory.
        raw = (ctypes.c_byte * nbytes).from_address(int(self._host_ptr))
        self.host: np.ndarray = np.frombuffer(raw, dtype=dtype)

    @property
    def device(self) -> int:
        """Device pointer as a plain integer for use in the bindings list."""
        return int(self._dev_ptr)

    def __del__(self) -> None:
        if cu is None:
            return
        try:
            cu.cuMemFreeHost(self._host_ptr)
        except Exception:
            pass
        try:
            cu.cuMemFree(self._dev_ptr)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# TensorRTBackend
# ---------------------------------------------------------------------------

class TensorRTBackend(BaseBackend):
    """Async-wrapped TensorRT inference backend.

    Follows the CUDA session lifecycle defined in docs/requirements/TENSORRT.md:
      load()    → cuInit → cuCtxCreate → cuStreamCreate → engine deserialise
                  → allocate buffers → extract ModelInfo
      infer()   → cuCtxPush → H→D copy → execute_v2 → D→H copy → sync → pop
      cleanup() → free buffers → destroy stream/context
    """

    def __init__(self) -> None:
        self._cu_ctx = None
        self._cu_stream = None
        self._engine = None
        self._context = None
        self._inputs: list[HostDeviceMem] = []
        self._outputs: list[HostDeviceMem] = []
        self._bindings: list[int] = []
        self._model_info: ModelInfo | None = None

    # ------------------------------------------------------------------
    # BaseBackend interface
    # ------------------------------------------------------------------

    async def load(self, model_path: str, config: dict) -> None:
        """Initialise CUDA, deserialise the TRT engine, allocate GPU buffers."""
        await asyncio.to_thread(self._load_sync, model_path, config)

    async def infer(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        """Run TRT inference in a thread and return raw output tensors."""
        if self._context is None:
            raise RuntimeError("Backend has no loaded model; call load() first.")
        return await asyncio.to_thread(self._infer_sync, input_tensor)

    async def cleanup(self) -> None:
        """Release all CUDA resources (buffers, stream, context)."""
        await asyncio.to_thread(self._cleanup_sync)
        logger.info("TensorRTBackend cleaned up.")

    @property
    def model_info(self) -> ModelInfo | None:
        return self._model_info

    # ------------------------------------------------------------------
    # Synchronous helpers — all run inside asyncio.to_thread
    # ------------------------------------------------------------------

    def _load_sync(self, model_path: str, config: dict) -> None:
        if not TRT_SUPPORT:
            raise RuntimeError(
                "TensorRT / cuda-python libraries are not available. "
                "Run inside the condor:tensorrt Docker container "
                "(docker run --runtime nvidia ...)."
            )

        device_idx = int(config.get("provider_options", {}).get("device", 0))
        logger.info("Initialising TensorRT backend on CUDA device %d.", device_idx)

        # 1. Init CUDA driver.
        _check(cu.cuInit(0), "cuInit")

        # 2. Validate device index.
        err, count = cu.cuDeviceGetCount()
        _check(err, "cuDeviceGetCount")
        if device_idx >= count:
            raise RuntimeError(
                f"CUDA device {device_idx} requested but only {count} "
                "device(s) are available."
            )
        cu_device = _unwrap(cu.cuDeviceGet(device_idx), "cuDeviceGet")

        # 3. Create CUDA context.
        # cuda.bindings 12.x added ctxCreateParams as the first arg (None = defaults).
        self._cu_ctx = _unwrap(cu.cuCtxCreate(None, 0, cu_device), "cuCtxCreate")

        # 4. Create CUDA stream.
        self._cu_stream = _unwrap(cu.cuStreamCreate(0), "cuStreamCreate")

        # 5. TRT logger.
        trt_logger = TrtLogger()

        # 6. Register built-in plugins (non-fatal for YOLOv10 which uses no plugins).
        try:
            trt.init_libnvinfer_plugins(trt_logger, "")
        except OSError as exc:
            logger.warning("init_libnvinfer_plugins failed (non-fatal): %s", exc)

        # 7. Deserialise TRT engine.
        logger.info("Deserialising engine: %s", model_path)
        runtime = trt.Runtime(trt_logger)
        with open(model_path, "rb") as fh:
            engine_data = fh.read()
        self._engine = runtime.deserialize_cuda_engine(engine_data)
        if self._engine is None:
            raise RuntimeError(
                f"Failed to deserialise TRT engine '{model_path}'. "
                "The engine may have been compiled for a different GPU architecture."
            )

        # 8. Execution context.
        self._context = self._engine.create_execution_context()

        # 9. Allocate host + device buffers and build bindings list.
        self._inputs, self._outputs, self._bindings = self._allocate_buffers()

        # 10. Extract ModelInfo from engine metadata.
        self._model_info = self._extract_model_info()
        logger.info("TRT engine loaded: %s", self._model_info)

    def _infer_sync(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        # 1. Push CUDA context onto this thread's context stack.
        _check(cu.cuCtxPushCurrent(self._cu_ctx), "cuCtxPushCurrent")
        try:
            # 2. Copy input data into the pinned host buffer and transfer H→D.
            np.copyto(self._inputs[0].host, input_tensor.ravel())
            _check(
                cu.cuMemcpyHtoDAsync(
                    self._inputs[0].device,
                    self._inputs[0]._host_ptr,
                    self._inputs[0]._nbytes,
                    self._cu_stream,
                ),
                "cuMemcpyHtoDAsync",
            )

            # 3. Execute engine.
            ok = self._context.execute_v2(self._bindings)
            if not ok:
                logger.warning("TRT execute_v2 returned False — output may be invalid.")

            # 4. Transfer outputs D→H.
            for out in self._outputs:
                _check(
                    cu.cuMemcpyDtoHAsync(
                        out._host_ptr,
                        out.device,
                        out._nbytes,
                        self._cu_stream,
                    ),
                    "cuMemcpyDtoHAsync",
                )

            # 5. Synchronise — blocks this thread (not the event loop) until GPU
            #    work is complete.
            _check(cu.cuStreamSynchronize(self._cu_stream), "cuStreamSynchronize")

        finally:
            # 6. Always pop the context, even on error.
            cu.cuCtxPopCurrent()

        # 7. Return a copy of each output buffer reshaped to its declared shape.
        return [
            out.host.copy().reshape(shape)
            for out, shape in zip(self._outputs, self._model_info.output_shapes)
        ]

    def _cleanup_sync(self) -> None:
        # Free output buffers first, then inputs (matches alloc order reversed).
        for buf in self._outputs:
            try:
                del buf
            except Exception:
                pass
        self._outputs = []

        for buf in self._inputs:
            try:
                del buf
            except Exception:
                pass
        self._inputs = []

        if self._cu_stream is not None:
            try:
                cu.cuStreamDestroy(self._cu_stream)
            except Exception:
                pass
            self._cu_stream = None

        # engine and context don't need explicit CUDA calls — Python GC handles it.
        self._context = None
        self._engine = None

        if self._cu_ctx is not None:
            try:
                cu.cuCtxDestroy(self._cu_ctx)
            except Exception:
                pass
            self._cu_ctx = None

        self._model_info = None
        self._bindings = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _allocate_buffers(
        self,
    ) -> tuple[list[HostDeviceMem], list[HostDeviceMem], list[int]]:
        """Allocate HostDeviceMem for every engine I/O tensor.

        The bindings list preserves engine tensor index order, which is what
        execute_v2 requires.
        """
        inputs: list[HostDeviceMem] = []
        outputs: list[HostDeviceMem] = []
        bindings: list[int] = []

        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            shape = tuple(self._engine.get_tensor_shape(name))
            dtype = np.dtype(trt.nptype(self._engine.get_tensor_dtype(name)))
            size = int(np.prod(shape))

            buf = HostDeviceMem(size, dtype)
            bindings.append(buf.device)

            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append(buf)
            else:
                outputs.append(buf)

        return inputs, outputs, bindings

    def _extract_model_info(self) -> ModelInfo:
        """Build ModelInfo from the deserialised TRT engine's tensor metadata."""
        input_names: list[str] = []
        input_shapes: list[list[int]] = []
        input_dtypes: list[str] = []
        output_names: list[str] = []
        output_shapes: list[list[int]] = []
        output_dtypes: list[str] = []

        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            shape = list(self._engine.get_tensor_shape(name))
            dtype_str = np.dtype(trt.nptype(self._engine.get_tensor_dtype(name))).name

            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                input_names.append(name)
                input_shapes.append(shape)
                input_dtypes.append(dtype_str)
            else:
                output_names.append(name)
                output_shapes.append(shape)
                output_dtypes.append(dtype_str)

        return ModelInfo(
            input_name=input_names[0],
            input_shape=input_shapes[0],
            input_dtype=input_dtypes[0],
            output_names=output_names,
            output_shapes=output_shapes,
            output_dtypes=output_dtypes,
        )
