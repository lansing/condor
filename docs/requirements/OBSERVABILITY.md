# Condor Observability Proposal

## Overview

This document proposes adding OpenTelemetry (OTel) instrumentation to Condor covering:

1. **Distributed traces** — per-request span trees showing time spent at every stage
2. **Metrics** — counters, histograms, and gauges for throughput, latency, and concurrency
3. **Two deployment modes** — a zero-extra-software local mode and a full export mode

OTel is the right foundation because:
- The API is a lightweight abstraction (~no overhead when a no-op provider is configured)
- The SDK is swappable — the same instrumentation code works with console, Prometheus, and OTLP exporters
- HyperDX, Grafana, Jaeger, and Datadog all speak OTLP natively

---

## Instrumentation Points

### 1. Request handling (`zmq_handler.py`)

Every incoming ZMQ message creates a **root span** that wraps the full round-trip:

| Span name | Created in | Key attributes |
|---|---|---|
| `condor.request` | `_dispatch` entry | `worker_id`, `request_type` (`inference`/`model_request`/`model_data`) |
| `condor.header.parse` | Inside `_dispatch` | — |
| `condor.inference` | `_handle_inference` entry | `model_name`, `provider`, `input_dtype`, `input_shape` |
| `condor.dtype_validation` | dtype check block | `expected_dtype`, `received_dtype`, `mismatch` (bool) |
| `condor.tensor.reconstruct` | `np.frombuffer` call | `input_bytes` (int) |
| `condor.infer_sem.wait` | Before `infer_sem.acquire()` | `worker_id`, `sem_limit` |
| `condor.backend.infer` | Inside backend `infer()` | `provider`, `model_name` |
| `condor.post_process` | `post_processor.process()` | `post_processor_class`, `detections_raw`, `detections_final` |

The `condor.infer_sem.wait` span is critical — its duration is the **pure queue wait time** a request experienced due to concurrency limiting. This is the primary signal for semaphore pressure.

### 2. TensorRT backend (`tensorrt_backend.py`)

TensorRT's inference has three distinguishable phases that currently cannot be separated in logs:

| Span name | Wraps | Key attributes |
|---|---|---|
| `condor.trt.h2d_copy` | `cuMemcpyHtoD` calls | `bytes_transferred` |
| `condor.trt.execute` | `context.execute_v2()` | `device_id` |
| `condor.trt.d2h_copy` | `cuMemcpyDtoH` calls | `bytes_transferred` |

These are children of `condor.backend.infer`. With `max_inference_concurrency` configured, only `condor.trt.execute` is guarded by the semaphore — the H2D/D2H spans deliberately fall outside it to allow DMA/compute overlap. This lets you see in a trace exactly how much compute overlap is happening between workers.

### 3. ONNX and OpenVINO backends

Both are simpler; a single child span suffices:

| Span name | Wraps | Key attributes |
|---|---|---|
| `condor.onnx.run` | `session.run()` | `execution_provider` |
| `condor.ov.infer` | `request.infer()` | `device` |

These are children of `condor.backend.infer`.

### 4. Model loading (`manager.py`)

Model load is infrequent but expensive — detailed spans pay off for understanding cold-start latency:

| Span name | Wraps | Key attributes |
|---|---|---|
| `condor.model.load` | All of `load_model()` | `model_name`, `provider` |
| `condor.model.lock_wait` | `asyncio.Lock.__aenter__` | — |
| `condor.model.cleanup` | `backend.cleanup()` on previous model | `previous_model_name` |
| `condor.shared_state.get_or_load` | `registry.get_or_load()` | `cache_hit` (bool), `model_key` |
| `condor.backend.load` | `backend.load()` | `provider`, `model_name` |

### 5. Worker startup (`main.py`)

One span per worker covering initialization:

| Span name | Wraps | Key attributes |
|---|---|---|
| `condor.worker.start` | Worker startup sequence | `worker_id`, `endpoint`, `provider` |

---

## Span Hierarchy

### Inference request (happy path)

```
condor.request  [~15ms total]
│ worker_id=1, request_type=inference, model_name=MDV6-yolov10-c
│
├── condor.header.parse  [~0.1ms]
│
└── condor.inference  [~14ms]
      input_dtype=float32, input_shape=[1,3,320,320], num_detections=3
      │
      ├── condor.dtype_validation  [~0.01ms]
      │     mismatch=false
      │
      ├── condor.tensor.reconstruct  [~0.1ms]
      │
      ├── condor.infer_sem.wait  [~0ms if no contention, ~5ms under load]
      │     worker_id=1, sem_limit=2
      │
      ├── condor.backend.infer  [~12ms]
      │     provider=tensorrt
      │     │
      │     ├── condor.trt.h2d_copy  [~0.3ms]
      │     ├── condor.trt.execute   [~11ms]
      │     └── condor.trt.d2h_copy  [~0.3ms]
      │
      └── condor.post_process  [~0.5ms]
            post_processor=YoloV10PostProcessor
            detections_raw=47, detections_final=3
```

### Model load (first worker, cold cache)

```
condor.model.load  [~500ms]
│ model_name=MDV6-yolov10-c, provider=tensorrt
│
├── condor.model.lock_wait  [~0ms if uncontested]
│
├── condor.model.cleanup  [~5ms if previous model existed]
│     previous_model_name=old_model
│
├── condor.shared_state.get_or_load  [~450ms, cache_hit=false]
│     Deserializes TRT engine, retains CUDA context
│
└── condor.backend.load  [~50ms]
      Allocates per-worker execution context and I/O buffers
```

### Model load (second+ worker, warm cache)

```
condor.model.load  [~55ms]
│ model_name=MDV6-yolov10-c, provider=tensorrt
│
├── condor.model.lock_wait  [~0ms]
│
├── condor.shared_state.get_or_load  [~0.1ms, cache_hit=true]
│
└── condor.backend.load  [~50ms]
      Allocates per-worker execution context and I/O buffers
```

---

## Metrics Inventory

### Counters (monotonically increasing)

| Metric name | Unit | Labels | Meaning |
|---|---|---|---|
| `condor.requests.total` | count | `worker_id`, `request_type`, `status` (`ok`/`error`) | Total requests received |
| `condor.inference.total` | count | `worker_id`, `model_name`, `provider`, `status` | Total inference calls |
| `condor.dtype_mismatch.total` | count | `worker_id`, `expected`, `received` | Dtype rejection events |
| `condor.model.loads.total` | count | `model_name`, `provider`, `status` | Model load events |
| `condor.shared_state.cache_hits.total` | count | `provider`, `model_name` | SharedStateRegistry hits |
| `condor.shared_state.cache_misses.total` | count | `provider`, `model_name` | SharedStateRegistry misses (expensive loads) |

### Histograms (latency distributions)

| Metric name | Unit | Labels | Meaning |
|---|---|---|---|
| `condor.request.duration` | ms | `worker_id`, `request_type` | Full request round-trip |
| `condor.inference.duration` | ms | `worker_id`, `model_name`, `provider` | Backend infer() only |
| `condor.trt.execute.duration` | ms | `device_id` | TRT kernel execution |
| `condor.trt.h2d.duration` | ms | `device_id` | Host→device copy |
| `condor.trt.d2h.duration` | ms | `device_id` | Device→host copy |
| `condor.postprocess.duration` | ms | `worker_id`, `post_processor` | Post-processing |
| `condor.sem_wait.duration` | ms | `worker_id` | Time waiting for `infer_sem` |
| `condor.model_lock_wait.duration` | ms | `worker_id` | Time waiting for asyncio.Lock in model manager |

Recommended histogram boundaries (ms): `[0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]`

### Gauges / UpDownCounters

| Metric name | Unit | Labels | Meaning |
|---|---|---|---|
| `condor.workers.active` | count | — | Number of active worker threads |
| `condor.inference.concurrent` | count | `worker_id` | Inflight `backend.infer()` calls right now |
| `condor.model.loaded` | 0/1 | `worker_id`, `model_name`, `provider` | Whether a model is currently loaded per worker |

### Throughput (derived)

Throughput (requests/second) is computed from `condor.requests.total` — it is **not** a separate metric. Any metrics backend can compute `rate(condor.requests.total[1m])`. For local display, the periodic exporter approach below computes this directly.

---

## OTel Architecture

### Context propagation across threads

The server uses `asyncio.to_thread()` for blocking inference calls. Python 3.7+ propagates `contextvars.Context` to threads started from async code, and the OTel Python SDK stores its active span in a `ContextVar`. This means span context flows correctly into `asyncio.to_thread()` calls automatically — no manual context copying is needed.

For multi-worker mode, each worker runs in its own OS thread with its own event loop. Spans from different workers are independent traces; they are connected only by shared attributes (`worker_id`, `model_name`) which is exactly what you want for per-worker latency comparison.

### Tracer and meter setup

```python
# condor/telemetry.py  (new file)
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider

tracer = trace.get_tracer("condor")
meter = metrics.get_meter("condor")
```

All modules import `tracer` and `meter` from `condor.telemetry`. The actual provider (console vs OTLP) is configured once at startup based on `config.observability`.

### No-op mode

When `observability.enabled: false`, configure the SDK with a `NoOpTracerProvider` and `NoOpMeterProvider`. All `tracer.start_as_current_span()` calls become no-ops with zero overhead. The instrumentation code does not need any conditionals.

---

## Configuration

Add a new top-level `observability` section to `AppConfig`:

```yaml
observability:
  enabled: true

  # "console" | "prometheus" | "otlp"
  # console: spans and metrics printed to stdout
  # prometheus: metrics at http://localhost:{metrics_port}/metrics (no trace storage)
  # otlp: export traces + metrics via OTLP to a backend
  mode: "console"

  service_name: "condor"
  service_version: "0.1.0"   # can match package version

  # Mode: console
  console:
    # Print a metrics summary every N seconds (0 = disable periodic summary)
    metrics_interval_seconds: 30
    # Print individual spans (very verbose, useful during development)
    export_traces: false

  # Mode: prometheus
  prometheus:
    host: "0.0.0.0"
    port: 9090
    # Also emit periodic metric summary to log at this interval (seconds, 0 = disable)
    log_summary_interval_seconds: 60

  # Mode: otlp
  otlp:
    endpoint: "http://localhost:4318"   # OTLP/HTTP; use 4317 for gRPC
    protocol: "http/protobuf"           # or "grpc"
    headers: {}                         # e.g. {"x-hyperdx-api-key": "YOUR_KEY"}
    # Export both traces and metrics via OTLP
    export_traces: true
    export_metrics: true
    metrics_interval_seconds: 30
```

---

## Mode 1: Lightweight Local (Zero Extra Software)

### Sub-mode A: Console exporter (development)

**How it works:**
- `ConsoleSpanExporter` prints each finished span as a JSON blob to stdout
- `ConsoleMetricExporter` prints a metric snapshot every `metrics_interval_seconds`
- No processes, no databases, no config outside `config.yaml`

**What you see (span output):**
```json
{
  "name": "condor.backend.infer",
  "context": {"trace_id": "0x4bf9...", "span_id": "0x1a2b..."},
  "start_time": "2026-03-02T14:23:01.234Z",
  "duration_ns": 11850000,
  "attributes": {
    "provider": "tensorrt",
    "model_name": "MDV6-yolov10-c",
    "worker_id": 0
  }
}
```

**What you see (metrics snapshot, every 30s):**
```
--- Condor Metrics Snapshot [2026-03-02 14:23:30] ---
condor.requests.total             = 1482  (+47/30s = 1.57 req/s)
condor.inference.total            = 1482
condor.inference.duration (p50)   = 12.3ms
condor.inference.duration (p95)   = 15.1ms
condor.inference.duration (p99)   = 18.2ms
condor.sem_wait.duration (p50)    = 0.1ms
condor.sem_wait.duration (p95)    = 4.8ms   ← semaphore pressure visible here
condor.request.duration (p99)     = 19.5ms
condor.workers.active             = 2
condor.model.loaded               = 1 (worker_id=0, model=MDV6-yolov10-c)
condor.model.loaded               = 1 (worker_id=1, model=MDV6-yolov10-c)
```

Traces are not stored, but they are visible in the log stream. Redirect stdout to a file and `grep` for span names or trace IDs to reconstruct a request path.

**Best for:** Local development, debugging latency issues, CI environments.

### Sub-mode B: Prometheus endpoint (operations)

**How it works:**
- `PrometheusMetricReader` starts a lightweight HTTP server (single Python thread, `prometheus_client`'s `make_wsgi_app`)
- No trace storage — only aggregated metrics
- View metrics with: `curl http://localhost:9090/metrics`

**Sample output:**
```
# HELP condor_inference_duration_milliseconds Inference latency
# TYPE condor_inference_duration_milliseconds histogram
condor_inference_duration_milliseconds_bucket{le="1.0",provider="tensorrt"} 0
condor_inference_duration_milliseconds_bucket{le="10.0",provider="tensorrt"} 612
condor_inference_duration_milliseconds_bucket{le="20.0",provider="tensorrt"} 1482
...
condor_inference_duration_milliseconds_count{provider="tensorrt"} 1482
condor_inference_duration_milliseconds_sum{provider="tensorrt"} 18234.5

# HELP condor_sem_wait_duration_milliseconds Semaphore wait latency
condor_sem_wait_duration_milliseconds_bucket{le="0.1",worker_id="0"} 1200
condor_sem_wait_duration_milliseconds_bucket{le="5.0",worker_id="0"} 1475
...

# HELP condor_requests_total Total requests
# TYPE condor_requests_total counter
condor_requests_total{request_type="inference",status="ok",worker_id="0"} 741
condor_requests_total{request_type="inference",status="ok",worker_id="1"} 741
```

The Prometheus text format is human-readable — no Prometheus server required to read it. You can also feed it to `prom2json` or any scraper on demand.

**Throughput from the command line (no server needed):**
```bash
# Compute req/s manually: scrape twice 10 seconds apart, diff the counter
A=$(curl -s localhost:9090/metrics | grep 'condor_requests_total{.*status="ok"' | awk '{sum+=$2} END{print sum}')
sleep 10
B=$(curl -s localhost:9090/metrics | grep 'condor_requests_total{.*status="ok"' | awk '{sum+=$2} END{print sum}')
echo "scale=2; ($B - $A) / 10" | bc
```

Or use `watch -n5 'curl -s localhost:9090/metrics | grep condor_requests_total'` for a live view.

**Best for:** Persistent deployments, home lab, server-side monitoring without a full stack.

---

## Mode 2: Full Export (HyperDX and friends)

### OTLP export

Both traces and metrics are exported via OTLP (OpenTelemetry Protocol). All major observability backends support OTLP:

| Backend | Traces | Metrics | Self-hosted | Notes |
|---|---|---|---|---|
| **HyperDX** | ✓ | ✓ | ✓ (Docker) | Excellent trace UX, built-in correlation |
| Grafana (Tempo + Mimir) | ✓ | ✓ | ✓ | More setup, very flexible |
| Jaeger | ✓ | — | ✓ | Traces only, very lightweight |
| Zipkin | ✓ | — | ✓ | Traces only |
| Datadog | ✓ | ✓ | ✗ | Managed, paid |
| Honeycomb | ✓ | ✓ | ✗ | Managed, generous free tier |

### HyperDX (recommended for "full" local setup)

HyperDX is a single Docker container with a full trace/metric/log UI:

```bash
docker run -p 8080:8080 -p 4318:4318 hyperdx/hyperdx-local
```

Config:
```yaml
observability:
  mode: "otlp"
  otlp:
    endpoint: "http://localhost:4318"
    protocol: "http/protobuf"
    export_traces: true
    export_metrics: true
    metrics_interval_seconds: 30
```

What HyperDX gives you:
- **Flame graph per request** — visualize the full span tree for any request
- **P50/P95/P99 latency charts** — across any label combination
- **Trace search** — find all requests where `condor.sem_wait.duration > 10ms`
- **Throughput charts** — `rate(condor.requests.total, 1m)` as a time series
- **Correlation** — trace IDs embedded in log lines → jump from a log line to its trace

### Managed cloud (HyperDX cloud, Honeycomb, etc.)

For managed backends, just set the OTLP endpoint and add authentication headers:

```yaml
observability:
  mode: "otlp"
  otlp:
    endpoint: "https://in-otel.hyperdx.io"
    protocol: "http/protobuf"
    headers:
      authorization: "Bearer YOUR_API_KEY"
```

No other changes — the instrumentation code is identical.

---

## Dependencies

### Always added (tiny, no-op when disabled)

```toml
[project.dependencies]
opentelemetry-api = ">=1.27"
opentelemetry-sdk = ">=1.27"
```

`opentelemetry-api` alone is ~200KB and is nearly zero-cost when no SDK is configured. `opentelemetry-sdk` is needed to configure exporters.

### Optional extras

```toml
[project.optional-dependencies]
# Mode: prometheus
observability-local = [
  "opentelemetry-exporter-prometheus>=0.48",
  "prometheus-client>=0.21",
]

# Mode: otlp (traces + metrics to HyperDX, Grafana, etc.)
observability-otlp = [
  "opentelemetry-exporter-otlp-proto-http>=1.27",
  # OR for gRPC:
  # "opentelemetry-exporter-otlp-proto-grpc>=1.27",
]
```

Install commands:
```bash
make install-observability-local  # uv sync --extra observability-local
make install-observability-otlp   # uv sync --extra observability-otlp
```

---

## Code Patterns

### Span creation (async context)

```python
from condor.telemetry import tracer

async def _handle_inference(self, header, tensor_bytes):
    with tracer.start_as_current_span("condor.inference") as span:
        span.set_attribute("model_name", self._manager.active_model)
        span.set_attribute("provider", self._config.inference.provider)
        span.set_attribute("input_dtype", header["dtype"])
        span.set_attribute("input_shape", str(header["shape"]))

        # semaphore wait as a child span
        with tracer.start_as_current_span("condor.infer_sem.wait"):
            self._infer_sem.acquire()
        try:
            result = await self._manager.backend.infer(tensor)
        finally:
            self._infer_sem.release()

        with tracer.start_as_current_span("condor.post_process") as pp_span:
            detections = await self._post_processor.process(result, input_shape)
            pp_span.set_attribute("detections_final", int((detections[:, 1] > 0).sum()))
```

### Span creation (sync thread, e.g. inside asyncio.to_thread)

Context is inherited automatically — no special handling:

```python
# Inside TensorRTBackend._infer_sync() called via asyncio.to_thread()
def _infer_sync(self, tensor):
    with tracer.start_as_current_span("condor.trt.h2d_copy") as span:
        span.set_attribute("bytes_transferred", tensor.nbytes)
        for name, hd in self._io_buffers.items():
            cu.cuMemcpyHtoD(hd.device, hd.host.ctypes.data, hd.nbytes)

    with tracer.start_as_current_span("condor.trt.execute"):
        self._context.execute_v2(self._bindings)

    with tracer.start_as_current_span("condor.trt.d2h_copy"):
        for name, hd in self._io_buffers.items():
            cu.cuMemcpyDtoH(hd.host.ctypes.data, hd.device, hd.nbytes)
```

### Counter and histogram recording

```python
from condor.telemetry import meter

# At module level (created once):
request_counter = meter.create_counter(
    "condor.requests.total",
    description="Total requests received",
)
inference_histogram = meter.create_histogram(
    "condor.inference.duration",
    unit="ms",
    description="Backend inference latency",
)

# Usage:
request_counter.add(1, {"worker_id": worker_id, "request_type": "inference", "status": "ok"})
inference_histogram.record(duration_ms, {"provider": provider, "model_name": model_name})
```

### Error recording on spans

```python
with tracer.start_as_current_span("condor.inference") as span:
    try:
        ...
    except Exception as e:
        span.record_exception(e)
        span.set_status(StatusCode.ERROR, str(e))
        raise
```

---

## Implementation Plan

### Phase A — Core instrumentation (no exporters yet)

1. Add `opentelemetry-api` and `opentelemetry-sdk` to `pyproject.toml`
2. Create `condor/telemetry.py` — module-level `tracer` and `meter`, plus `setup_telemetry(config)` function that configures the provider based on `config.observability.mode`
3. Add `ObservabilityConfig` Pydantic model to `condor/config/settings.py`
4. Instrument `zmq_handler.py` — `condor.request`, `condor.inference`, `condor.infer_sem.wait`, `condor.post_process`
5. Instrument `manager.py` — `condor.model.load`, `condor.model.lock_wait`, `condor.shared_state.get_or_load`, `condor.backend.load`
6. Instrument `onnx_backend.py` — `condor.onnx.run`
7. Instrument `tensorrt_backend.py` — `condor.trt.h2d_copy`, `condor.trt.execute`, `condor.trt.d2h_copy`
8. Instrument `openvino_backend.py` — `condor.ov.infer`
9. Call `setup_telemetry(config)` in `main.py` before workers start
10. Add counters and histograms for all metrics in the inventory

### Phase B — Local exporters

11. Add `observability-local` optional dependency group
12. Implement console exporter path in `setup_telemetry()` — `BatchSpanProcessor(ConsoleSpanExporter())` + `ConsoleMetricExporter` with periodic reader
13. Implement prometheus exporter path — `PrometheusMetricReader` (starts HTTP server), no trace storage

### Phase C — OTLP export

14. Add `observability-otlp` optional dependency group
15. Implement OTLP path in `setup_telemetry()` — `OTLPSpanExporter` + `OTLPMetricExporter` with configurable endpoint/headers
16. Document Docker setup for self-hosted HyperDX
17. Add example `config.yaml` variants for each mode under `config/examples/`

### Phase D — Tests and Makefile

18. Unit tests for `telemetry.py` setup function (no-op provider, console provider)
19. Add Makefile targets: `install-observability-local`, `install-observability-otlp`
20. Update `docker-compose.yaml` with optional HyperDX service for the full stack

---

## Answering the Specific Questions

### How much time is spent at each stage?

Every stage in the request path is a named span with a start and end timestamp. The span hierarchy gives you exact durations. With any trace viewer you get a flame graph. With console mode you can diff `start_time` and `duration_ns` from the JSON blobs.

### Wait time for semaphore?

`condor.infer_sem.wait` span duration = pure queue wait. If this is consistently >1ms under your target load, `max_inference_concurrency` is too restrictive (or you need more workers). Both the histogram and individual spans surface this.

### Throughput?

`condor.requests.total` is a counter. Rate = delta over time window. In Prometheus mode: `rate(condor_requests_total[1m])`. In console mode: the periodic summary prints `(+N/30s = X.X req/s)` by comparing the last two snapshots. In HyperDX/Grafana: a time-series chart of `rate()`.

### Local without heavy software?

**Console mode** — zero dependencies beyond the OTel SDK (which is already in the dep tree). Metrics summary in the log, individual spans as JSON blobs. No processes, no ports.

**Prometheus mode** — adds `prometheus_client` (pure Python, ~100KB). Starts one HTTP listener thread. `curl localhost:9090/metrics` gives you everything. Still no database, no UI.

### Full fancy mode?

**HyperDX** — single `docker run` command, full OTLP receiver, trace flame graphs, metric dashboards, log correlation, and alerting. No Kubernetes, no Grafana stack.
