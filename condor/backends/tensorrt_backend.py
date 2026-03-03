"""TensorRT inference backend — multi-worker Strategy A.

Shared resources (loaded once, cached in SharedStateRegistry)
--------------------------------------------------------------
* CUDA device handle
* Device primary context  (cuDevicePrimaryCtxRetain — one per device per process)
* trt.ICudaEngine          (immutable after deserialisation; thread-safe factory)
* ModelInfo                (derived from engine; stateless)

Per-worker resources (created fresh for each worker instance)
--------------------------------------------------------------
* trt.IExecutionContext    (holds mutable per-inference activation memory)
* HostDeviceMem I/O buffers (pinned host + device memory for each tensor)

All blocking GPU operations are offloaded via asyncio.to_thread so the event
loop is never stalled.  This module is safe to import without TensorRT;
TRT_SUPPORT will be False and load() will raise RuntimeError with a clear
message.
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from .base import BaseBackend, ModelInfo, SharedBackendState
from ..telemetry import tel, tracer

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
    """Raise RuntimeError on a non-zero CUDA result code."""
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

    Call ``free()`` explicitly while the appropriate CUDA context is pushed to
    release device memory deterministically.  ``__del__`` will call ``free()``
    as a safety net, but GC timing is unpredictable — always prefer explicit
    ``free()`` in cleanup code.
    """

    # CU_MEMHOSTALLOC_PORTABLE = 0x01 — pinned, any-context accessible.
    _ALLOC_FLAGS = 0x01

    def __init__(self, size: int, dtype: np.dtype) -> None:
        nbytes = int(size * dtype.itemsize)
        self._nbytes = nbytes
        self._freed = False

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

    def free(self) -> None:
        """Explicitly release CUDA resources.  Safe to call multiple times.

        Must be called while the appropriate CUDA context is pushed.
        """
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
        # Safety net: free any resources not already released by explicit free().
        self.free()


# ---------------------------------------------------------------------------
# Shared state dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrtSharedState(SharedBackendState):
    """Resources shared across all TensorRT worker instances for one model."""
    cu_device: object   # CUdevice handle
    cu_ctx: object      # Primary CUDA context (cuDevicePrimaryCtxRetain)
    engine: object      # trt.ICudaEngine (immutable, thread-safe)
    model_info: ModelInfo


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# TensorRTBackend
# ---------------------------------------------------------------------------

class TensorRTBackend(BaseBackend):
    """Async-wrapped TensorRT inference backend.

    Multi-worker lifecycle (Strategy A — shared primary CUDA context)
    -----------------------------------------------------------------
    load_shared_sync()  [called once, under SharedStateRegistry lock]
      cuInit → cuDevicePrimaryCtxRetain → push ctx
      → init_libnvinfer_plugins → deserialize engine → extract ModelInfo
      → pop ctx
      → return TrtSharedState(cu_device, cu_ctx, engine, model_info)

    load()  [called per worker]
      → store refs to shared cu_ctx, engine, model_info
      → push primary ctx
      → create_execution_context() → allocate I/O buffers
      → pop ctx

    infer()  [called per request, inside asyncio.to_thread]
      → push primary ctx
      → H→D copy (sync, DMA engine — not semaphore-guarded)
      → [acquire infer_sem] execute_v2 [release infer_sem]   ← compute engine
      → D→H copy (sync — not semaphore-guarded)
      → pop ctx

    cleanup()  [called per worker]
      → push primary ctx
      → destroy IExecutionContext (frees activation memory)
      → explicitly free all I/O buffers
      → pop ctx
      → drop references to shared resources (do NOT release primary ctx)

    The primary context is retained once at load_shared_sync() time and lives
    for the process lifetime.  Workers push/pop it as a thread-local operation;
    CUDA correctly serialises any concurrent operations that would conflict.
    """

    def __init__(self) -> None:
        # Shared (references only — do not own/destroy):
        self._cu_ctx = None       # primary CUDA context
        self._engine = None       # trt.ICudaEngine

        # Per-worker (owned):
        self._context = None      # trt.IExecutionContext
        self._inputs: list[HostDeviceMem] = []
        self._outputs: list[HostDeviceMem] = []
        self._bindings: list[int] = []
        self._model_info: ModelInfo | None = None
        self._infer_sem: threading.BoundedSemaphore | None = None

    # ------------------------------------------------------------------
    # Shared-resource interface
    # ------------------------------------------------------------------

    def load_shared_sync(self, model_path: str, config: dict) -> TrtSharedState:
        """Deserialise the TRT engine once and retain the device primary context.

        Called at most once per (provider, model_path) key, under the
        SharedStateRegistry threading.Lock.
        """
        if not TRT_SUPPORT:
            raise RuntimeError(
                "TensorRT / cuda-python libraries are not available. "
                "Run inside the condor:tensorrt Docker container "
                "(docker run --runtime nvidia ...)."
            )

        device_idx = int(config.get("provider_options", {}).get("device", 0))
        logger.info(
            "Loading TRT shared resources on CUDA device %d.", device_idx
        )

        # 1. Init CUDA driver (idempotent).
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

        # 3. Retain the device's primary context.
        #    cuDevicePrimaryCtxRetain is reference-counted; one Retain here,
        #    one Release on process exit (handled by CUDA runtime cleanup).
        cu_ctx = _unwrap(
            cu.cuDevicePrimaryCtxRetain(cu_device), "cuDevicePrimaryCtxRetain"
        )

        # 4. Push primary context so TRT initialisation uses it.
        _check(cu.cuCtxPushCurrent(cu_ctx), "cuCtxPushCurrent")
        try:
            # 5. TRT logger + plugins.
            trt_logger = TrtLogger()
            try:
                trt.init_libnvinfer_plugins(trt_logger, "")
            except OSError as exc:
                logger.warning(
                    "init_libnvinfer_plugins failed (non-fatal): %s", exc
                )

            # 6. Deserialise engine.  Weight tensors are allocated in the
            #    primary context; all workers' execution contexts will share
            #    this device memory via the primary context.
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

            # 7. Extract ModelInfo (shared; stateless).
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

    # ------------------------------------------------------------------
    # BaseBackend interface
    # ------------------------------------------------------------------

    async def load(
        self,
        model_path: str,
        config: dict,
        shared: SharedBackendState | None = None,
        infer_sem: threading.BoundedSemaphore | None = None,
    ) -> None:
        """Set up per-worker execution context and I/O buffers."""
        await asyncio.to_thread(self._load_sync, model_path, config, shared, infer_sem)

    async def infer(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        """Run TRT inference in a thread and return raw output tensors."""
        if self._context is None:
            raise RuntimeError("Backend has no loaded model; call load() first.")
        return await asyncio.to_thread(self._infer_sync, input_tensor)

    async def cleanup(self) -> None:
        """Release per-worker CUDA resources."""
        await asyncio.to_thread(self._cleanup_sync)
        logger.info("TensorRTBackend worker cleaned up.")

    @property
    def model_info(self) -> ModelInfo | None:
        return self._model_info

    # ------------------------------------------------------------------
    # Synchronous helpers — all run inside asyncio.to_thread
    # ------------------------------------------------------------------

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
                "Run inside the condor:tensorrt Docker container "
                "(docker run --runtime nvidia ...)."
            )

        if shared is None:
            # Single-worker mode: load shared resources inline (no registry).
            shared = self.load_shared_sync(model_path, config)

        # Borrow references from shared state (do not own/destroy these).
        self._cu_ctx = shared.cu_ctx
        self._engine = shared.engine
        self._model_info = shared.model_info
        self._infer_sem = infer_sem

        # Per-worker: create execution context and allocate I/O buffers.
        # Both must happen while the primary context is pushed.
        _check(cu.cuCtxPushCurrent(self._cu_ctx), "cuCtxPushCurrent")
        try:
            self._context = self._engine.create_execution_context()
            if self._context is None:
                raise RuntimeError(
                    "TRT engine.create_execution_context() returned None."
                )
            self._inputs, self._outputs, self._bindings = self._allocate_buffers()
        finally:
            cu.cuCtxPopCurrent()

        logger.info("TRT worker ready (exec ctx + I/O buffers allocated).")

    def _infer_sync(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        # 1. Push primary context onto this thread's context stack.
        _check(cu.cuCtxPushCurrent(self._cu_ctx), "cuCtxPushCurrent")
        try:
            # 2. H→D copy (synchronous, DMA engine).
            #
            # IMPORTANT: cuMemcpyHtoD (not HtoDAsync) is required here.
            # execute_v2 uses TRT's own internal CUDA stream; an async copy on
            # a different stream has no ordering guarantee w.r.t. that stream
            # and would cause one-behind data leakage.  The synchronous copy
            # blocks this CPU thread until DMA is complete.
            #
            # The DMA engine and the GPU compute engine are independent
            # hardware blocks.  While this worker is copying H→D, another
            # worker that already holds the infer_sem can be executing
            # execute_v2 simultaneously on the compute engine — pipelining.
            t_h2d = time.perf_counter()
            with tracer.start_as_current_span("condor.trt.h2d_copy") as h2d_span:
                h2d_span.set_attribute("bytes_transferred", self._inputs[0]._nbytes)
                np.copyto(self._inputs[0].host, input_tensor.ravel())
                _check(
                    cu.cuMemcpyHtoD(
                        self._inputs[0].device,
                        self._inputs[0]._host_ptr,
                        self._inputs[0]._nbytes,
                    ),
                    "cuMemcpyHtoD",
                )
            tel.record_trt_h2d((time.perf_counter() - t_h2d) * 1000)

            # 3. Execute engine — guarded by the inference semaphore.
            #    The semaphore serialises (or limits) compute-engine usage
            #    across all workers.  H→D and D→H are intentionally outside
            #    the guarded region so DMA can overlap with another worker's
            #    compute step.
            if self._infer_sem is not None:
                t_sem = time.perf_counter()
                with tracer.start_as_current_span("condor.infer_sem.wait"):
                    self._infer_sem.acquire()
                tel.record_sem_wait((time.perf_counter() - t_sem) * 1000)
            # inc_inference_concurrent brackets only the compute step so the
            # TUI "concurrent" gauge accurately reflects GPU utilisation
            # (≤ max_inference_concurrency), not the wider H→D/D→H pipeline.
            tel.inc_inference_concurrent()
            try:
                t_exec = time.perf_counter()
                with tracer.start_as_current_span("condor.trt.execute"):
                    ok = self._context.execute_v2(self._bindings)
                tel.record_trt_execute((time.perf_counter() - t_exec) * 1000)
                if not ok:
                    logger.warning(
                        "TRT execute_v2 returned False — output may be invalid."
                    )
            finally:
                tel.dec_inference_concurrent()
                if self._infer_sem is not None:
                    self._infer_sem.release()

            # 4. D→H copy (synchronous).
            #    execute_v2 has returned; device output buffers are fully written.
            t_d2h = time.perf_counter()
            with tracer.start_as_current_span("condor.trt.d2h_copy"):
                for out in self._outputs:
                    _check(
                        cu.cuMemcpyDtoH(
                            out._host_ptr,
                            out.device,
                            out._nbytes,
                        ),
                        "cuMemcpyDtoH",
                    )
            tel.record_trt_d2h((time.perf_counter() - t_d2h) * 1000)

        finally:
            # 5. Always pop the context, even on error.
            cu.cuCtxPopCurrent()

        # 6. Return copies of output buffers reshaped to their declared shapes.
        return [
            out.host.copy().reshape(shape)
            for out, shape in zip(self._outputs, self._model_info.output_shapes)
        ]

    def _cleanup_sync(self) -> None:
        if self._cu_ctx is not None:
            try:
                _check(cu.cuCtxPushCurrent(self._cu_ctx), "cuCtxPushCurrent")
                try:
                    # Destroy execution context first (frees TRT activation memory).
                    # Setting to None triggers Python GC → TRT destructor while
                    # the primary context is pushed, as required.
                    self._context = None

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

        # Drop references to shared resources — do NOT destroy them.
        # The SharedStateRegistry owns the TrtSharedState; engine and primary
        # context live until the registry entry is invalidated or process exits.
        self._engine = None
        self._cu_ctx = None
        self._model_info = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _allocate_buffers(
        self,
    ) -> tuple[list[HostDeviceMem], list[HostDeviceMem], list[int]]:
        """Allocate HostDeviceMem for every engine I/O tensor.

        Must be called while the primary CUDA context is pushed.
        The bindings list preserves engine tensor index order for execute_v2.
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
