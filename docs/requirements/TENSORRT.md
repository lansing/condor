# TensorRT Backend — Engineering Requirements

This document specifies the design and implementation requirements for adding a
TensorRT inference backend (`TensorRTBackend`) to the Condor detector server.
It follows the same async-plugin architecture established in Phase 1 (ONNX
Runtime) and draws on the reference implementation in
`frigate/detectors/plugins/tensorrt.py`.

---

> **HOST SYSTEM RULE — NON-NEGOTIABLE**
>
> **NEVER** install, upgrade, downgrade, or otherwise modify NVIDIA drivers,
> CUDA toolkit, cuDNN, TensorRT, cuda-python, or any other component of the
> GPU/NVIDIA software stack on the **host machine**.
>
> The host requires **only** the NVIDIA kernel driver and
> `nvidia-container-toolkit` (to enable `--runtime nvidia`). Every other
> piece of the NVIDIA/CUDA/TensorRT stack lives exclusively inside the
> Docker container described in §2. Violating this rule risks destabilising
> the host GPU driver and breaking unrelated workloads.

---

## 1. Overview

TensorRT (TRT) enables high-throughput, low-latency inference on NVIDIA GPUs
using engine files compiled ahead-of-time for a specific GPU architecture.
The backend wraps the TRT Python bindings and the CUDA Python bindings
(`cuda-python`) in the existing `BaseBackend` interface, keeping the asyncio
event loop non-blocking throughout.

**Test model:** `models/MDV6-yolov10-e_demo_export.engine`
- Architecture: YOLOv10-e
- Input: `float32`, NCHW, shape `[1, 3, 640, 640]`
- Output: `(1, 300, 6)` — same post-NMS format as the ONNX YOLOv10 model
- Classes: animal, person, vehicle (same `md.classes.txt`)
- Post-processor: **reuse `YoloV10PostProcessor` unchanged**

---

## 2. Development Lifecycle & Container Environment

### 2.1 Container-first mandate

All development, testing, and execution of the TensorRT backend takes place
**inside** `docker/tensorrt/Dockerfile`. No TensorRT, CUDA, or NVIDIA Python
package is ever installed on the host. The only host-level prerequisites are:

- NVIDIA kernel driver (installed by the GPU vendor toolchain, not by us).
- `nvidia-container-toolkit` — enables Docker's `--runtime nvidia` flag.

### 2.2 Base image

```
nvcr.io/nvidia/tensorrt:26.01-py3   (NVIDIA NGC Container Registry)
```

This image provides TensorRT Python bindings, `cuda-python`, CUDA runtime and
driver API, cuDNN, libnvinfer plugins, and Python 3.12. **Do not change the
base image without updating this document and the Dockerfile together.**

### 2.3 Build the image

```bash
docker build -f docker/tensorrt/Dockerfile -t condor:tensorrt .
```

Rebuild whenever `docker/tensorrt/Dockerfile` or `pyproject.toml` changes.

### 2.4 Development workflow

Source code and dependencies are **baked into the image** at build time,
identical to the ONNX Runtime image.  The default entrypoint launches the
Condor server:

```bash
# Run the server (models and config can be overridden with bind mounts)
docker run --rm -it --runtime nvidia \
  -v /path/to/models:/app/models \
  -v /path/to/config:/app/config \
  -p 5555:5555 \
  condor:tensorrt

# Open an interactive shell (override entrypoint)
make docker-shell-tensorrt

# Run the test suite
make docker-test-tensorrt
```

> `--runtime nvidia` is **required** for every container that needs GPU
> access. Without it the CUDA driver is not exposed and TensorRT will fail
> to initialise.

After any code change, rebuild the image before testing:

```bash
make docker-build-tensorrt
```

### 2.5 Running the test suite

```bash
make docker-test-tensorrt
# Equivalent: docker run --rm --runtime nvidia --entrypoint python \
#               condor:tensorrt -m pytest tests/ -v
```

### 2.6 Running the test client

```bash
# Start server in one terminal:
docker run --rm --runtime nvidia \
  -v /path/to/models:/app/models \
  -p 5555:5555 \
  condor:tensorrt

# Run the client from the host or another container:
python scripts/test_client.py \
  --model MDV6-yolov10-e_demo_export.engine \
  --input-size 640
```

### 2.7 Running the Condor server

```bash
docker run --rm -it --runtime nvidia \
  -v /path/to/models:/app/models \
  -v /path/to/config:/app/config \
  -p 5555:5555 \
  condor:tensorrt
```

Config and models directories can be overridden with bind mounts; everything
else is baked into the image.

### 2.8 Quick-reference: what lives where

| Layer | What is here | Installed by |
|---|---|---|
| Host kernel | NVIDIA GPU driver only | OS / GPU vendor |
| Host userspace | `nvidia-container-toolkit` only | distro package manager |
| Container base image | TensorRT, CUDA, cuDNN, Python 3.12 | NGC image |
| Container venv | pyzmq, aiofiles, pydantic, opencv, condor | Dockerfile |

---

## 3. Dependencies

| Package | Source | Notes |
|---|---|---|
| `tensorrt` | NVIDIA PyPI index or TRT wheel | Python bindings for TRT engine loading and execution |
| `cuda-python` | NVIDIA PyPI index | `cuda.cuda` module for CUDA driver API (context, stream, memory) |

These packages are **not** installable from the standard PyPI index; they
require the NVIDIA package index or a local wheel bundled with the TRT
installation. They are optional dependencies — the backend must guard against
`ImportError` and raise a clear error if TRT support is absent (same pattern
as the reference implementation's `TRT_SUPPORT` flag).

The `pyproject.toml` optional extra `tensorrt` should list these packages.
The `docker/tensorrt/Dockerfile` provides the canonical environment.

---

## 4. CUDA Session Lifecycle

Follow the reference implementation's proven initialization sequence exactly:

### 4.1 Initialization (`load()`)

1. **CUDA init:** `cuda.cuInit(0)` — initialise the CUDA driver.
2. **Device validation:** `cuda.cuDeviceGetCount()` — assert the configured
   device index is valid.
3. **Context creation:** `cuda.cuCtxCreate(CU_CTX_MAP_HOST, device)` — create
   a CUDA context bound to the target GPU. `CU_CTX_MAP_HOST` enables
   host-memory-mapped allocations used by `HostDeviceMem`.
4. **Stream creation:** `cuda.cuStreamCreate(0)` — create a CUDA stream for
   async memory transfers and kernel launches.
5. **TRT logger:** Instantiate a custom `trt.ILogger` subclass that bridges
   TRT severity levels to Python `logging` (see §5.1).
6. **Plugin init:** `trt.init_libnvinfer_plugins(trt_logger, "")` — registers
   built-in TRT plugins. Wrap in `try/except OSError` and log a warning on
   failure (non-fatal for YOLOv10).
7. **Engine deserialization:** Open the engine file in binary mode and call
   `runtime.deserialize_cuda_engine(data)` within a `trt.Runtime` context
   manager.
8. **Execution context:** `engine.create_execution_context()`.
9. **Buffer allocation:** Allocate `HostDeviceMem` objects for every
   input/output tensor (see §5.2).
10. **ModelInfo extraction:** Derive input name, shape, and dtype from the
    engine's tensor metadata (see §5.3).

All of the above runs inside `asyncio.to_thread` so the event loop is not
blocked during potentially-slow initialization (especially engine
deserialization).

### 4.2 Inference (`infer()`)

The entire inference sequence runs inside a single `asyncio.to_thread` call:

1. **Push CUDA context:** `cuda.cuCtxPushCurrent(cu_ctx)`.
2. **H→D transfer (async):** `cuda.cuMemcpyHtoDAsync(device_ptr, host_ptr,
   nbytes, stream)` for each input buffer.
3. **Execute:** `context.execute_v2(bindings)` — enqueues GPU kernel(s) on the
   stream. Log a warning if it returns `False`.
4. **D→H transfer (async):** `cuda.cuMemcpyDtoHAsync(host_ptr, device_ptr,
   nbytes, stream)` for each output buffer.
5. **Synchronize:** `cuda.cuStreamSynchronize(stream)` — blocks the **thread**
   (not the event loop) until all GPU work is complete.
6. **Pop CUDA context:** `cuda.cuCtxPopCurrent()`.
7. Return reshaped output arrays.

The key async insight: steps 2–4 are non-blocking GPU-side operations. The
thread blocks only at step 5. Because the entire function runs in
`asyncio.to_thread`, the event loop can freely process other coroutines (e.g.
model management heartbeats from Frigate) while the GPU executes.

### 4.3 Cleanup (`cleanup()`)

Explicit `async cleanup()` must free all CUDA resources in order:

1. Delete output `HostDeviceMem` objects (triggers `cuda.cuMemFreeHost` +
   `cuda.cuMemFree` in `__del__`).
2. Delete input `HostDeviceMem` objects.
3. `cuda.cuStreamDestroy(stream)`.
4. Delete execution context and engine.
5. `cuda.cuCtxDestroy(cu_ctx)`.

Wrap each step in a try/except to tolerate partially-initialised state (e.g.
if `load()` failed mid-way).

---

## 5. Component Specifications

### 5.1 TrtLogger

```python
class TrtLogger(trt.ILogger):
    def log(self, severity: trt.ILogger.Severity, msg: str) -> None:
        # Map TRT severity → Python logging level
```

Severity mapping (matches reference implementation):
- `VERBOSE` → `DEBUG`
- `INFO` → `INFO`
- `WARNING` → `WARNING`
- `ERROR` → `ERROR`
- `INTERNAL_ERROR` → `CRITICAL`

### 5.2 HostDeviceMem

Identical to the reference implementation. Allocates page-locked (pinned)
host memory for zero-copy DMA and a corresponding device buffer:

```python
class HostDeviceMem:
    def __init__(self, size: int, dtype: np.dtype) -> None:
        # cuda.cuMemHostAlloc with CU_MEMHOSTALLOC_DEVICEMAP
        # cuda.cuMemAlloc for device buffer
        # np.frombuffer view over the pinned host memory
    def __del__(self) -> None:
        # cuda.cuMemFreeHost + cuda.cuMemFree
```

The `host` attribute is a NumPy array view over the pinned memory, enabling
`np.copyto` to fill input data without an extra allocation.

### 5.3 ModelInfo Extraction

Derive `ModelInfo` from the TRT engine after deserialization:

- Iterate `engine.num_io_tensors` using `engine.get_tensor_name(i)`.
- Distinguish input vs output using `engine.get_tensor_mode(name)` compared to
  `trt.TensorIOMode.INPUT`.
- Shapes: `engine.get_tensor_shape(name)` → `list[int]`.
- Dtypes: `trt.nptype(engine.get_tensor_dtype(name))` → numpy dtype string.

The first `INPUT` tensor provides `input_name`, `input_shape`, `input_dtype`.

### 5.4 TensorRTBackend class

```python
class TensorRTBackend(BaseBackend):
    async def load(self, model_path: str, config: dict) -> None: ...
    async def infer(self, input_tensor: np.ndarray) -> list[np.ndarray]: ...
    async def cleanup(self) -> None: ...
    @property
    def model_info(self) -> ModelInfo | None: ...
```

Config keys read from `config` dict:
- `device` (int, default `0`) — CUDA device index.

---

## 6. Configuration

Add TensorRT as a new provider option in `config/config.yaml`:

```yaml
inference:
  provider: "tensorrt"
  provider_options:
    device: 0           # CUDA device index (0 = first GPU)
```

The `OnnxRuntimeBackend._resolve_providers` pattern does not apply here.
The provider routing lives in `AsyncModelManager`, which must be updated to
instantiate `TensorRTBackend` when `provider == "tensorrt"`.

---

## 7. Model Manager Integration

`AsyncModelManager.load_model()` currently hardcodes `OnnxRuntimeBackend()`.
This must be refactored into a factory:

```python
def _make_backend(self, provider: str) -> BaseBackend:
    if provider == "tensorrt":
        return TensorRTBackend()
    return OnnxRuntimeBackend()
```

The provider string is already available in `self.inference_config["provider"]`.

---

## 8. Post-Processing

`MDV6-yolov10-e_demo_export.engine` outputs the same `(1, 300, 6)` post-NMS
tensor format as the ONNX YOLOv10 model. `YoloV10PostProcessor` is **reused
without modification**; only the input spatial dimensions passed at inference
time change (640×640 instead of 320×320).

---

## 9. Test / Benchmark Client

Update `scripts/test_client.py` to:
- Accept a `--model` flag that can point to the `.engine` file.
- Accept an `--input-size` flag (default `320`, override to `640` for TRT
  test).
- Preprocess the test image to the correct size before sending.

All testing runs **inside the container** (see §2). The TRT test invocation:

```bash
docker run --rm --runtime nvidia \
  -v $(pwd):/workspace -w /workspace \
  condor:tensorrt \
  python scripts/test_client.py \
    --model MDV6-yolov10-e_demo_export.engine \
    --input-size 640
```

---

## 10. Docker Image

`docker/tensorrt/Dockerfile` is the canonical environment for all TensorRT
work. Key design points:

- **Base image:** `nvcr.io/nvidia/tensorrt:26.01-py3` (NVIDIA NGC registry).
  Provides TensorRT, `cuda-python`, CUDA runtime, cuDNN, and Python 3.12.
  **Never change this without updating the document.**
- A virtualenv is created with `--system-site-packages` so that `tensorrt` and
  `cuda-python` from the base image are importable without reinstalling them.
- `onnxruntime` is **not installed** — this image is TensorRT-only.
- Project source is **baked into the image** (`COPY condor/ + uv pip install --no-deps .`), matching the ONNX Runtime image's approach.
- Default entrypoint: `condor --config /app/config/config.yaml`.
- `EXPOSE 5555`.

See §2 for all build and run commands.

---

## 11. Guards and Error Handling

- Wrap `import tensorrt` and `import cuda` in `try/except ImportError`.
  If absent, importing `TensorRTBackend` must not crash the server; raise
  `RuntimeError("TensorRT/CUDA Python libraries not available")` only when
  `load()` is actually called.
- If `cuInit` fails, raise with the CUDA error code in the message.
- If `execute_v2` returns `False`, log a warning but continue (same as
  reference).
- Engine files are compiled for a specific GPU architecture (compute
  capability). Loading an incompatible engine raises a TRT error during
  deserialization; surface this as a clear `RuntimeError`.

---

## 12. Explicitly Out of Scope

- Dynamic shapes / variable batch sizes — the engine is built for batch=1.
- `execute_async_v3` (TRT 10 fully-async path) — deferred to a future
  iteration; `execute_v2` + stream synchronize in a thread is sufficient.
- INT8 calibration or engine building from scratch — the engine is pre-built.
- `libyolo_layer.so` — that plugin handles legacy pre-NMS YOLO formats.
  YOLOv10 outputs post-NMS tensors; the plugin is not needed.
