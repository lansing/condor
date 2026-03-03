# Multi-Worker Architecture Proposal
## Phase 3 — Shared Resources + Inference Concurrency Control

---

## 1. Current state and gaps

Two problems exist in the current multi-worker implementation.

**Gap 1: No shared resources.**
Each worker calls `AsyncModelManager.load_model()`, which calls `_make_backend()` and `backend.load()` independently.  For TensorRT this means `N` full engine deserialisations: ~1–2 s each and ~34 MB of GPU weight memory multiplied by `N`.  For OpenVINO, the device graph compilation (which can be several seconds on NPU/GPU) also runs `N` times.  These are exactly the resources the design doc said should be shared.

**Gap 2: No inference concurrency control.**
All workers call `execute_v2` / `session.run()` / `request.infer()` concurrently and unconditionally.  On a constrained GPU (e.g. a Jetson Orin, Arc A380, or any iGPU), queuing multiple GPU kernels simultaneously has zero throughput benefit and incurs context-switch overhead.  There is currently no knob to limit this.

---

## 2. Per-backend thread-safety audit

### 2.1 TensorRT

| Resource | Thread-safe? | Shareable? |
|---|---|---|
| `trt.ICudaEngine` | ✅ Yes — TRT docs: "safe to use from multiple threads as a factory" | ✅ Share one instance |
| `trt.IExecutionContext` | ❌ No — holds per-inference mutable activation buffers | One per worker |
| Pinned host buffers (`HostDeviceMem`) | ❌ No — written by input, read by output | One set per worker |
| CUDA device buffers (`HostDeviceMem.device`) | ❌ No — bound to execution context | One set per worker |
| `cuInit` | ✅ Idempotent | Call once at process start |
| `trt.init_libnvinfer_plugins` | ✅ Idempotent | Call once |

**CUDA context strategy for engine sharing.**
An engine is deserialized in whichever CUDA context is current at that moment.  Its weight tensors live in the device memory of that context.  `IExecutionContext` allocates its own activation buffers in the context that is current when it is *created*.

Two viable strategies:

| | Strategy A — primary context (recommended) | Strategy B — per-worker context (current) |
|---|---|---|
| Context | All workers share the device primary context (`cuDevicePrimaryCtxRetain`) | Each worker creates its own context with `cuCtxCreate` |
| Engine | Deserialised once in the primary context | Deserialised per worker |
| Execution ctx | One per worker, created in primary context | One per worker |
| I/O buffers | One set per worker, allocated in primary context | One set per worker |
| GPU weight memory | **~34 MB × 1** (shared) | ~34 MB × N |
| Serialisation | Push/pop primary ctx around every GPU op (already the pattern) | Push/pop own ctx (current) |
| Engine sharing complexity | Low — engine lives in the one primary ctx; all workers' exec ctxs are in the same ctx | N/A — no sharing |
| CUDA stream overlap | ✅ Distinct exec ctxs on the same device use TRT's internal streams; concurrent `execute_v2` calls on different exec ctxs overlap on the GPU's compute queue | Same |

**Strategy A is recommended.** It eliminates weight memory duplication and the deserialization cost scales as O(1).

Key change in `_load_sync`: replace `cuCtxCreate` with `cuDevicePrimaryCtxRetain`.  The shared state contains `(cu_device, cu_primary_ctx, trt_engine)`.  Each worker's `load()` pushes the primary ctx, calls `engine.create_execution_context()`, allocates its own I/O buffers, and pops.

### 2.2 ONNX Runtime

| Resource | Thread-safe? | Shareable? |
|---|---|---|
| `ort.InferenceSession` | ✅ Yes — ORT docs explicitly guarantee `run()` is thread-safe; session state is read-only after creation | ✅ Share one instance |
| Per-run output buffers | N/A — `run()` returns freshly allocated numpy arrays | N/A |

ORT already manages its own inter-op and intra-op thread pools.  A single shared session with `N` concurrent `run()` calls is correct and saves the model's weight memory N-fold.  Each worker holds a reference to the shared session; `_infer_sync` just calls `session.run()`.

**No per-worker ORT state is needed.**  The shared state is the session itself.

### 2.3 OpenVINO

| Resource | Thread-safe? | Shareable? |
|---|---|---|
| `ov.Core` | ✅ Yes | Can be shared or per-worker (cheap to create) |
| `ov.CompiledModel` | ✅ Yes — OV docs: "CompiledModel can be used in multiple threads simultaneously" | ✅ Share one instance |
| `ov.InferRequest` | ❌ No — holds mutable activation memory | One per worker |

The compile step (`core.compile_model`) is expensive (JIT, graph optimisation, device upload).  The shared state is the `CompiledModel`.  Each worker calls `compiled.create_infer_request()` to get its own `InferRequest`.

---

## 3. Shared resource design

### 3.1 `SharedBackendState` and `SharedStateRegistry`

Introduce two new constructs:

```python
# condor/backends/base.py

@dataclass
class SharedBackendState:
    """Opaque container for resources shared across worker instances.
    Backend subclasses replace this with their own typed dataclass."""
    pass
```

```python
# condor/model_manager/shared.py  (new file)

class SharedStateRegistry:
    """Process-level cache of shared backend state, keyed by (provider, model_path).

    Uses a threading.Lock so workers running in different asyncio event loops
    can safely race to load the shared state; the first thread to arrive does
    the work, all subsequent threads receive the cached result.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: dict[str, SharedBackendState] = {}

    def get_or_load(
        self,
        key: str,
        loader: Callable[[], SharedBackendState],
    ) -> SharedBackendState:
        """Synchronous; runs inside asyncio.to_thread in the manager."""
        with self._lock:
            if key not in self._cache:
                self._cache[key] = loader()
        return self._cache[key]

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._cache.pop(key, None)
```

### 3.2 Changes to `BaseBackend`

Add one new method; `load()` gains an optional `shared` parameter:

```python
class BaseBackend(ABC):

    def load_shared_sync(
        self, model_path: str, config: dict
    ) -> SharedBackendState:
        """Load and return resources shared across all worker instances.

        Called synchronously inside asyncio.to_thread, under the
        SharedStateRegistry lock.  Override in backends that have
        expensive one-time initialisation (engine deserialisation,
        model compilation).  Default: no shared resources.
        """
        return SharedBackendState()

    @abstractmethod
    async def load(
        self,
        model_path: str,
        config: dict,
        shared: SharedBackendState | None = None,
    ) -> None:
        """Load per-worker resources.  shared contains pre-loaded shared state."""
```

### 3.3 Changes to `AsyncModelManager`

The manager gains an optional `SharedStateRegistry` reference.  During `load_model()`:

```python
# Inside load_model(), replacing the bare backend.load() call:

key = f"{provider}:{model_path}"
shared = await asyncio.to_thread(
    self._shared_registry.get_or_load,
    key,
    lambda: backend.load_shared_sync(str(model_path), self.inference_config),
)
await backend.load(str(model_path), self.inference_config, shared=shared)
```

If `_shared_registry` is `None` (single-worker mode), `load_shared_sync` is not called and `shared=None` is passed to `load()` — existing behavior.

### 3.4 Changes to `server/main.py`

In `_run_multi`, create one `SharedStateRegistry` and pass it to each worker's `AsyncModelManager` (via `AsyncZMQHandler`).  `AsyncZMQHandler` gains an optional `shared_registry` constructor argument.

---

## 4. Inference concurrency control

### 4.1 The pipelining opportunity

With `max_inference_concurrency: 1` and 3 workers, the timeline looks like this (TRT, all ops synchronous on the calling thread):

```
        [CPU work ←→ DMA engine ←→ Compute engine]
Worker A: recv──prep──H2D────────execute_v2──────D2H──postproc──send
Worker B:      recv──prep──H2D─────────────[wait]─execute_v2──D2H──...
Worker C:               recv──prep──H2D─────────────────────[wait]─...
```

Worker B's H2D (DMA engine) overlaps with Worker A's `execute_v2` (compute engine) because the two GPU hardware engines are independent.  The same for Worker C.  Even with strict serialisation of compute, the DMA pipeline keeps the GPU's memory controller busy and delivers close to the throughput of pipelining the entire inference stack.

This is the key insight: multi-worker with `max_inference_concurrency: 1` still outperforms single-worker because CPU-side work and DMA overlap with compute.

### 4.2 Config

```yaml
inference:
  max_inference_concurrency: 1   # 0 = unlimited (default, backward-compatible)
```

`0` means unlimited (current behavior).  `1` means at most one hardware inference call at a time.  `N` allows up to `N` concurrent calls (useful if the GPU can overlap multiple execution contexts efficiently, e.g., MIG or multi-instance TRT).

### 4.3 Implementation: `threading.BoundedSemaphore`

Because workers run in separate asyncio event loops, an `asyncio.Semaphore` cannot be shared across them.  A `threading.BoundedSemaphore` is the correct tool: it works across OS threads and is acquired/released inside `asyncio.to_thread` worker threads without blocking any event loop.

The semaphore is created once in `main.py` (or `InferenceConfig` on startup) and passed down to each backend instance.  It is stored as `self._infer_sem: threading.BoundedSemaphore | None`.

### 4.4 Placement per backend

The semaphore guards only the actual hardware call, allowing H2D/D2H and post-processing to proceed concurrently:

**TensorRT** — wrap only `execute_v2`:
```python
def _infer_sync(self, input_tensor):
    _check(cu.cuCtxPushCurrent(self._cu_ctx), "cuCtxPushCurrent")
    try:
        np.copyto(self._inputs[0].host, input_tensor.ravel())
        _check(cu.cuMemcpyHtoD(...), "cuMemcpyHtoD")      # ← not guarded

        if self._infer_sem:
            self._infer_sem.acquire()
        try:
            ok = self._context.execute_v2(self._bindings)  # ← guarded
        finally:
            if self._infer_sem:
                self._infer_sem.release()

        for out in self._outputs:
            _check(cu.cuMemcpyDtoH(...), "cuMemcpyDtoH")  # ← not guarded
    finally:
        cu.cuCtxPopCurrent()
```

**ONNX Runtime** — wrap the entire `session.run()` (no H2D/D2H separation for CPU/EP):
```python
def _infer_sync(self, input_tensor):
    if self._infer_sem:
        with self._infer_sem:
            return self._session.run(None, {self._model_info.input_name: input_tensor})
    return self._session.run(None, {self._model_info.input_name: input_tensor})
```

**OpenVINO** — wrap `request.infer()`:
```python
def _infer_sync(self, input_tensor):
    if self._infer_sem:
        with self._infer_sem:
            self._request.infer({self._model_info.input_name: input_tensor})
    else:
        self._request.infer({self._model_info.input_name: input_tensor})
    return [self._request.get_output_tensor(i).data.copy() for i in ...]
```

### 4.5 Passing the semaphore

`threading.BoundedSemaphore` is constructed in `main.py` if `max_inference_concurrency > 0`, then passed through:

```
main.py
  → _WorkerCoordinator (holds semaphore reference)
  → _run_worker(... infer_sem=...)
  → AsyncZMQHandler(... infer_sem=...)
  → AsyncModelManager(... infer_sem=...)
  → backend._infer_sem = infer_sem  (set during load())
```

The semaphore is set on the backend instance at `load()` time, not at `infer()` time, so the hot path (`_infer_sync`) never does a dict lookup.

---

## 5. Files changed

| File | Change |
|---|---|
| `condor/config/settings.py` | Add `max_inference_concurrency: int = 0` to `InferenceConfig` |
| `condor/backends/base.py` | Add `SharedBackendState`; add `load_shared_sync()` to `BaseBackend`; update `load()` signature |
| `condor/backends/tensorrt_backend.py` | `TrtSharedState` dataclass; `load_shared_sync()` — cuInit, primary ctx retain, plugin init, engine deserialise; `load()` — accept shared, push primary ctx, create exec ctx, allocate buffers, pop; `_infer_sync()` — acquire/release semaphore around `execute_v2` |
| `condor/backends/onnx_backend.py` | `OnnxSharedState` (holds `ort.InferenceSession`); `load_shared_sync()` — build session; `load()` — store session ref from shared; `_infer_sync()` — semaphore around `session.run()` |
| `condor/backends/openvino_backend.py` | `OVSharedState` (holds `ov.CompiledModel`); `load_shared_sync()` — core + compile; `load()` — `compiled.create_infer_request()`; `_infer_sync()` — semaphore around `request.infer()` |
| `condor/model_manager/manager.py` | Accept optional `SharedStateRegistry`; call `get_or_load` in `load_model()` |
| `condor/model_manager/shared.py` | **New file** — `SharedStateRegistry` |
| `condor/server/zmq_handler.py` | Accept optional `infer_sem`; pass to manager |
| `condor/server/main.py` | Construct `SharedStateRegistry` and `threading.BoundedSemaphore` in `_run_multi`; pass both to each worker |
| `config/config.yaml` | Document `max_inference_concurrency` |

No changes to Dockerfiles, Makefile, or the ZMQ protocol.

---

## 6. Backward compatibility

- `num_workers: 1` — `SharedStateRegistry` not created, `shared=None` passed to `load()`, all backends behave identically to today.
- `max_inference_concurrency: 0` (default) — `infer_sem` is `None`, backends skip all semaphore code.  No performance overhead on the hot path.
- Existing tests require no changes; the new parameters are all optional with the same defaults as today.

---

## 7. Open questions

1. **TRT primary context vs. per-worker context.**  Strategy A (primary context, engine shared) is recommended.  If a future use case needs two *different* engines on the same device in the same process (e.g., a detection model and a classification model), the primary context still works — each engine is deserialised into it.  The only downside is that all workers' CUDA operations interleave in a single context; the primary context has no memory isolation between workers.  This is acceptable since Condor is a single-purpose server.

2. **ORT session sharing and EP thread pools.**  Sharing one `InferenceSession` is correct per the ORT spec, but some EPs (e.g., TensorRT EP, CUDA EP) internally serialise inference through their own mutex.  If the target EP already serialises, the `max_inference_concurrency` semaphore is redundant but harmless.  Worth profiling before choosing `max_inference_concurrency` for ORT+CUDA EP users.

3. **Model hot-reload.**  When Frigate sends a new model while workers are running, `SharedStateRegistry.invalidate()` must be called before the new `load_shared_sync()`.  The current manager already serialises load/unload under `asyncio.Lock` per worker; with shared state, we need to also invalidate the registry entry.  Since all workers share the same `AsyncModelManager._lock` (currently they don't — each worker has its own manager), the invalidation ordering needs thought.  **Proposed:** keep one manager per worker (existing), but all share the same `SharedStateRegistry`; invalidate the registry entry only after all workers have cleanly unloaded their backends (coordinated via `_WorkerCoordinator`).  Hot-reload is not in Frigate's normal operation path (models are static at deploy time), so this can be a follow-up.
