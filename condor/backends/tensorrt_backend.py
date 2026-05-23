from __future__ import annotations

import asyncio
import ctypes
import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from ..telemetry import tel, tracer
from .base import BaseBackend, ModelInfo, SharedBackendState

logger = logging.getLogger(__name__)


try:
    import tensorrt as trt
    from cuda.bindings import driver as cu

    TRT_SUPPORT = True
    _ILogger_base: type = trt.ILogger
except ImportError:
    TRT_SUPPORT = False
    trt = None
    cu = None
    _ILogger_base = object


# CUDA helpers


def _check(retval, op: str) -> None:
    """Raise RuntimeError on a non-zero CUDA result code."""
    code = retval[0] if isinstance(retval, tuple) else retval
    if int(code) != 0:
        raise RuntimeError(f"CUDA error in {op}: error code {int(code)}")


def _unwrap(retval, op: str):
    """Check for error and return the second element of a 2-tuple result."""
    _check(retval[0], op)
    return retval[1]


# TrtLogger

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


# HostDeviceMem


class HostDeviceMem:
    # CU_MEMHOSTALLOC_PORTABLE = 0x01 — pinned, any-context accessible.
    _ALLOC_FLAGS = 0x01

    def __init__(self, size: int, dtype: np.dtype) -> None:
        nbytes = int(size * dtype.itemsize)
        self._nbytes = nbytes
        self._freed = False

        err, self._host_ptr = cu.cuMemHostAlloc(nbytes, self._ALLOC_FLAGS)
        _check(err, "cuMemHostAlloc")

        err, self._dev_ptr = cu.cuMemAlloc(nbytes)
        _check(err, "cuMemAlloc")

        raw = (ctypes.c_byte * nbytes).from_address(int(self._host_ptr))
        self.host: np.ndarray = np.frombuffer(raw, dtype=dtype)

    @property
    def device(self) -> int:
        return int(self._dev_ptr)

    def free(self) -> None:
        if cu is None or self._freed:
            return
        self._freed = True
        try:
            cu.cuMemFreeHost(self._host_ptr)
        except Exception:
            pass
        try:
            cu.cuMemFree(self._dev_ptr)
        except Exception:
            pass

    def __del__(self) -> None:
        self.free()


# Shared state dataclass


@dataclass
class TrtSharedState(SharedBackendState):
    """Resources shared across all TensorRT worker instances for one model."""

    cu_device: object  # CUdevice handle
    cu_ctx: object  # Primary CUDA context (cuDevicePrimaryCtxRetain)
    engine: object  # trt.ICudaEngine (immutable, thread-safe)
    model_info: ModelInfo


# Module-level helper


def _extract_model_info(engine) -> ModelInfo:
    """Build ModelInfo from a deserialised TRT engine's tensor metadata."""
    input_names: list[str] = []
    input_shapes: list[list[int]] = []
    input_dtypes: list[str] = []
    output_names: list[str] = []
    output_shapes: list[list[int]] = []
    output_dtypes: list[str] = []

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = list(engine.get_tensor_shape(name))
        dtype_str = np.dtype(trt.nptype(engine.get_tensor_dtype(name))).name

        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
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


# TensorRTBackend


class TensorRTBackend(BaseBackend):

    def __init__(self) -> None:
        self._cu_ctx = None  # primary CUDA context
        self._engine = None  # trt.ICudaEngine

        self._context = None  # trt.IExecutionContext
        self._stream = None
        self._inputs: list[HostDeviceMem] = []
        self._outputs: list[HostDeviceMem] = []
        self._bindings: list[int] = []
        self._model_info: ModelInfo | None = None
        self._infer_sem: threading.BoundedSemaphore | None = None

        self._ev_start = None
        self._ev_h2d_done = None
        self._ev_exec_start = None
        self._ev_exec_done = None
        self._ev_d2h_done = None

    # Shared-resource interface

    def load_shared_sync(self, model_path: str, config: dict) -> TrtSharedState:
        if not TRT_SUPPORT:
            raise RuntimeError(
                "TensorRT / cuda-python libraries are not available. "
            )

        device_idx = int(config.get("provider_options", {}).get("device", 0))
        logger.info("Loading TRT shared resources on CUDA device %d.", device_idx)

        _check(cu.cuInit(0), "cuInit")

        err, count = cu.cuDeviceGetCount()
        _check(err, "cuDeviceGetCount")
        if device_idx >= count:
            raise RuntimeError(
                f"CUDA device {device_idx} requested but only {count} "
                "device(s) are available."
            )
        cu_device = _unwrap(cu.cuDeviceGet(device_idx), "cuDeviceGet")

        cu_ctx = _unwrap(
            cu.cuDevicePrimaryCtxRetain(cu_device), "cuDevicePrimaryCtxRetain"
        )

        _check(cu.cuCtxPushCurrent(cu_ctx), "cuCtxPushCurrent")
        try:
            trt_logger = TrtLogger()
            try:
                trt.init_libnvinfer_plugins(trt_logger, "")
            except OSError as exc:
                logger.warning("init_libnvinfer_plugins failed (non-fatal): %s", exc)

            logger.info("Deserialising TRT engine: %s", model_path)
            runtime = trt.Runtime(trt_logger)
            with open(model_path, "rb") as fh:
                engine_data = fh.read()
            engine = runtime.deserialize_cuda_engine(engine_data)
            if engine is None:
                raise RuntimeError(
                    f"Failed to deserialise TRT engine '{model_path}'. "
                    "The engine may have been compiled for a different GPU "
                    "architecture."
                )

            model_info = _extract_model_info(engine)
            logger.info("TRT shared state ready: %s", model_info)

        finally:
            cu.cuCtxPopCurrent()

        return TrtSharedState(
            cu_device=cu_device,
            cu_ctx=cu_ctx,
            engine=engine,
            model_info=model_info,
        )

    # BaseBackend interface

    async def load(
        self,
        model_path: str,
        config: dict,
        shared: SharedBackendState | None = None,
        infer_sem: threading.BoundedSemaphore | None = None,
    ) -> None:
        await asyncio.to_thread(self._load_sync, model_path, config, shared, infer_sem)

    async def infer(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        if self._context is None:
            raise RuntimeError("Backend has no loaded model; call load() first.")
        return await asyncio.to_thread(self._infer_sync, input_tensor)

    async def cleanup(self) -> None:
        await asyncio.to_thread(self._cleanup_sync)
        logger.info("TensorRTBackend worker cleaned up.")

    @property
    def model_info(self) -> ModelInfo | None:
        return self._model_info

    # Synchronous helpers

    def _load_sync(
        self,
        model_path: str,
        config: dict,
        shared: TrtSharedState | None,
        infer_sem: threading.BoundedSemaphore | None,
    ) -> None:
        if not TRT_SUPPORT:
            raise RuntimeError(
                "TensorRT / cuda-python libraries are not available. "
            )

        if shared is None:
            shared = self.load_shared_sync(model_path, config)

        self._cu_ctx = shared.cu_ctx
        self._engine = shared.engine
        self._model_info = shared.model_info
        self._infer_sem = infer_sem

        _check(cu.cuCtxPushCurrent(self._cu_ctx), "cuCtxPushCurrent")
        try:
            self._context = self._engine.create_execution_context()
            if self._context is None:
                raise RuntimeError(
                    "TRT engine.create_execution_context() returned None."
                )
            err, self._stream = cu.cuStreamCreate(0)
            _check(err, "cuStreamCreate")
            self._inputs, self._outputs, self._bindings = self._allocate_buffers()

            for i in range(self._engine.num_io_tensors):
                name = self._engine.get_tensor_name(i)
                ptr = self._bindings[i]
                self._context.set_tensor_address(name, ptr)

            # Create timing events (CU_EVENT_DEFAULT = 0 enables timing).
            self._ev_start = _unwrap(cu.cuEventCreate(0), "cuEventCreate")
            self._ev_h2d_done = _unwrap(cu.cuEventCreate(0), "cuEventCreate")
            self._ev_exec_start = _unwrap(cu.cuEventCreate(0), "cuEventCreate")
            self._ev_exec_done = _unwrap(cu.cuEventCreate(0), "cuEventCreate")
            self._ev_d2h_done = _unwrap(cu.cuEventCreate(0), "cuEventCreate")
        finally:
            cu.cuCtxPopCurrent()

        logger.info("TRT worker ready (exec ctx + I/O buffers allocated).")

    def _infer_sync(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        _check(cu.cuCtxPushCurrent(self._cu_ctx), "cuCtxPushCurrent")
        try:
            t_input_copy = time.perf_counter()
            with tracer.start_as_current_span("condor.trt.host_copy"):
                np.copyto(self._inputs[0].host, input_tensor.ravel())
            input_copy_ms = (time.perf_counter() - t_input_copy) * 1000

            with tracer.start_as_current_span("condor.trt.h2d_copy") as h2d_span:
                h2d_span.set_attribute("bytes_transferred", self._inputs[0]._nbytes)
                _check(cu.cuEventRecord(self._ev_start, self._stream), "cuEventRecord:start")
                _check(
                    cu.cuMemcpyHtoDAsync(
                        self._inputs[0].device,
                        self._inputs[0]._host_ptr,
                        self._inputs[0]._nbytes,
                        self._stream,
                    ),
                    "cuMemcpyHtoDAsync",
                )
                _check(cu.cuEventRecord(self._ev_h2d_done, self._stream), "cuEventRecord:h2d_done")

            if self._infer_sem is not None:
                t_sem = time.perf_counter()
                with tracer.start_as_current_span("condor.infer_sem.wait"):
                    self._infer_sem.acquire()
                tel.record_sem_wait((time.perf_counter() - t_sem) * 1000)

            tel.inc_inference_concurrent()
            try:
                _check(cu.cuEventRecord(self._ev_exec_start, self._stream), "cuEventRecord:exec_start")
                with tracer.start_as_current_span("condor.trt.execute"):
                    ok = self._context.execute_async_v3(int(self._stream))
                _check(cu.cuEventRecord(self._ev_exec_done, self._stream), "cuEventRecord:exec_done")
                if not ok:
                    logger.warning("TRT execute_async_v3 returned False — output may be invalid.")
                _check(cu.cuStreamSynchronize(self._stream), "cuStreamSynchronize:exec")
            finally:
                tel.dec_inference_concurrent()
                if self._infer_sem is not None:
                    self._infer_sem.release()

            with tracer.start_as_current_span("condor.trt.d2h_copy"):
                for out in self._outputs:
                    _check(
                        cu.cuMemcpyDtoHAsync(
                            out._host_ptr, out.device, out._nbytes, self._stream
                        ),
                        "cuMemcpyDtoHAsync",
                    )
                _check(cu.cuEventRecord(self._ev_d2h_done, self._stream), "cuEventRecord:d2h_done")

            _check(cu.cuStreamSynchronize(self._stream), "cuStreamSynchronize:d2h")

            t_output_copy = time.perf_counter()
            with tracer.start_as_current_span("condor.trt.output_copy"):
                outputs = [
                    out.host.copy().reshape(shape)
                    for out, shape in zip(self._outputs, self._model_info.output_shapes)
                ]
            output_copy_ms = (time.perf_counter() - t_output_copy) * 1000

            tel.record_trt_host_copy(input_copy_ms + output_copy_ms)

            h2d_ms = _unwrap(
                cu.cuEventElapsedTime(self._ev_start, self._ev_h2d_done),
                "cuEventElapsedTime:h2d",
            )
            exec_ms = _unwrap(
                cu.cuEventElapsedTime(self._ev_exec_start, self._ev_exec_done),
                "cuEventElapsedTime:exec",
            )
            d2h_ms = _unwrap(
                cu.cuEventElapsedTime(self._ev_exec_done, self._ev_d2h_done),
                "cuEventElapsedTime:d2h",
            )
            tel.record_trt_h2d(h2d_ms)
            tel.record_trt_execute(exec_ms)
            tel.record_trt_d2h(d2h_ms)

        finally:
            cu.cuCtxPopCurrent()

        return outputs

    def _cleanup_sync(self) -> None:
        if self._cu_ctx is not None:
            try:
                _check(cu.cuCtxPushCurrent(self._cu_ctx), "cuCtxPushCurrent")
                try:
                    # Destroy execution context first (frees TRT activation memory).
                    self._context = None

                    if self._stream is not None:
                        cu.cuStreamDestroy(self._stream)
                        self._stream = None

                    # Destroy timing events.
                    for ev in (self._ev_start, self._ev_h2d_done, self._ev_exec_start,
                               self._ev_exec_done, self._ev_d2h_done):
                        if ev is not None:
                            cu.cuEventDestroy(ev)
                    self._ev_start = self._ev_h2d_done = self._ev_exec_start = None
                    self._ev_exec_done = self._ev_d2h_done = None

                    # Explicitly free I/O buffers while context is active.
                    for buf in (*self._outputs, *self._inputs):
                        buf.free()
                finally:
                    cu.cuCtxPopCurrent()
            except Exception:
                logger.exception("Error during TRT worker cleanup.")

        self._outputs = []
        self._inputs = []
        self._bindings = []

        self._engine = None
        self._cu_ctx = None
        self._model_info = None

    # Private helpers

    def _allocate_buffers(
        self,
    ) -> tuple[list[HostDeviceMem], list[HostDeviceMem], list[int]]:
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
