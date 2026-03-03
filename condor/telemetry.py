"""OpenTelemetry tracing and metrics for Condor.

Usage
-----
Import the module-level singletons in any module::

    from condor.telemetry import tracer, tel

Tracing (spans)::

    from opentelemetry.trace import StatusCode

    with tracer.start_as_current_span("condor.foo") as span:
        span.set_attribute("key", "value")
        ...
        span.set_status(StatusCode.ERROR, "reason")

Metrics (null-safe — all no-ops until setup_telemetry() runs)::

    tel.count_request(worker_id=0, request_type="inference", status="ok")
    tel.record_inference_duration(12.5, provider="tensorrt", model_name="foo")

Call ``setup_telemetry(config)`` once in ``main()`` before workers start.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator

from opentelemetry import metrics as otel_metrics
from opentelemetry import trace
from opentelemetry.trace import StatusCode  # re-exported for convenience

from .stats import StatsCollector, StatsServer

if TYPE_CHECKING:
    from .config.settings import ObservabilityConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level tracer
# ---------------------------------------------------------------------------
# trace.get_tracer() returns a ProxyTracer that automatically delegates to
# whichever TracerProvider is registered at the time of each span operation.
# It is safe to obtain this at import time — it will use the real provider
# after setup_telemetry() calls trace.set_tracer_provider().
tracer = trace.get_tracer(
    "condor", schema_url="https://opentelemetry.io/schemas/1.27.0"
)

# Re-export for convenience so callers only import from condor.telemetry
__all__ = ["tracer", "tel", "StatusCode", "setup_telemetry"]


# ---------------------------------------------------------------------------
# Metric instruments container
# ---------------------------------------------------------------------------


class _Tel:
    """Null-safe container for all OTel metric instruments.

    All ``count_*`` and ``record_*`` methods are no-ops until
    ``setup_telemetry()`` has been called.  This ensures instrumented code
    never needs to guard against None or check whether observability is
    enabled.
    """

    # ------------------------------------------------------------------
    # Instrument references (None = not initialized)
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        # Always-available stats collector (feeds the TUI socket)
        self.stats = StatsCollector()

        # Counters
        self._req_ctr = None
        self._infer_ctr = None
        self._dtype_mismatch_ctr = None
        self._model_load_ctr = None
        self._shared_cache_hit_ctr = None
        self._shared_cache_miss_ctr = None

        # Histograms
        self._req_hist = None
        self._infer_hist = None
        self._trt_h2d_hist = None
        self._trt_execute_hist = None
        self._trt_d2h_hist = None
        self._postprocess_hist = None
        self._sem_wait_hist = None
        self._model_lock_wait_hist = None

        # UpDownCounters (gauges)
        self._workers_active = None
        self._inference_concurrent = None

    def _init(self, meter: otel_metrics.Meter) -> None:
        """Create all metric instruments from *meter*.

        Called by ``setup_telemetry()`` after the MeterProvider is configured.
        Instruments created from a real meter are exported; those created from
        the default no-op meter are silently discarded.
        """
        HIST_BOUNDS = [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

        self._req_ctr = meter.create_counter(
            "condor.requests.total",
            description="Total ZMQ requests received",
        )
        self._infer_ctr = meter.create_counter(
            "condor.inference.total",
            description="Total backend inference calls",
        )
        self._dtype_mismatch_ctr = meter.create_counter(
            "condor.dtype_mismatch.total",
            description="Input dtype rejection events",
        )
        self._model_load_ctr = meter.create_counter(
            "condor.model.loads.total",
            description="Model load events",
        )
        self._shared_cache_hit_ctr = meter.create_counter(
            "condor.shared_state.cache_hits.total",
            description="SharedStateRegistry cache hits (shared resources already loaded)",
        )
        self._shared_cache_miss_ctr = meter.create_counter(
            "condor.shared_state.cache_misses.total",
            description="SharedStateRegistry cache misses (expensive first-time load)",
        )

        self._req_hist = meter.create_histogram(
            "condor.request.duration",
            unit="ms",
            description="Full ZMQ request round-trip latency",
        )
        self._infer_hist = meter.create_histogram(
            "condor.inference.duration",
            unit="ms",
            description="Backend inference latency (excludes post-processing)",
        )
        self._trt_h2d_hist = meter.create_histogram(
            "condor.trt.h2d.duration",
            unit="ms",
            description="TensorRT host-to-device DMA copy latency",
        )
        self._trt_execute_hist = meter.create_histogram(
            "condor.trt.execute.duration",
            unit="ms",
            description="TensorRT kernel execution latency (compute engine only)",
        )
        self._trt_d2h_hist = meter.create_histogram(
            "condor.trt.d2h.duration",
            unit="ms",
            description="TensorRT device-to-host DMA copy latency",
        )
        self._postprocess_hist = meter.create_histogram(
            "condor.postprocess.duration",
            unit="ms",
            description="Post-processing latency (NMS filtering, coordinate normalization)",
        )
        self._sem_wait_hist = meter.create_histogram(
            "condor.sem_wait.duration",
            unit="ms",
            description="Time spent waiting to acquire the inference semaphore",
        )
        self._model_lock_wait_hist = meter.create_histogram(
            "condor.model_lock_wait.duration",
            unit="ms",
            description="Time spent waiting for model manager asyncio.Lock",
        )

        self._workers_active = meter.create_up_down_counter(
            "condor.workers.active",
            description="Number of active worker threads",
        )
        self._inference_concurrent = meter.create_up_down_counter(
            "condor.inference.concurrent",
            description="Number of GPU/compute calls currently executing (≤ max_inference_concurrency)",
        )

    # ------------------------------------------------------------------
    # Counter helpers
    # ------------------------------------------------------------------

    def set_active_model(self, model_name: str) -> None:
        self.stats.set_active_model(model_name)

    def count_request(self, *, worker_id: int, request_type: str, status: str) -> None:
        self.stats.count_request(worker_id)
        if self._req_ctr is not None:
            self._req_ctr.add(
                1,
                {
                    "worker_id": str(worker_id),
                    "request_type": request_type,
                    "status": status,
                },
            )

    def count_inference(
        self, *, worker_id: int, model_name: str, provider: str, status: str
    ) -> None:
        if status == "ok":
            self.stats.count_inference(worker_id)
        if self._infer_ctr is not None:
            self._infer_ctr.add(
                1,
                {
                    "worker_id": str(worker_id),
                    "model_name": model_name,
                    "provider": provider,
                    "status": status,
                },
            )

    def count_dtype_mismatch(self, *, expected: str, received: str) -> None:
        if self._dtype_mismatch_ctr is not None:
            self._dtype_mismatch_ctr.add(
                1, {"expected": expected, "received": received}
            )

    def count_model_load(self, *, model_name: str, provider: str, status: str) -> None:
        if self._model_load_ctr is not None:
            self._model_load_ctr.add(
                1,
                {
                    "model_name": model_name,
                    "provider": provider,
                    "status": status,
                },
            )

    def count_cache_hit(self, *, provider: str, model_name: str) -> None:
        if self._shared_cache_hit_ctr is not None:
            self._shared_cache_hit_ctr.add(
                1,
                {
                    "provider": provider,
                    "model_name": model_name,
                },
            )

    def count_cache_miss(self, *, provider: str, model_name: str) -> None:
        if self._shared_cache_miss_ctr is not None:
            self._shared_cache_miss_ctr.add(
                1,
                {
                    "provider": provider,
                    "model_name": model_name,
                },
            )

    # ------------------------------------------------------------------
    # Histogram helpers
    # ------------------------------------------------------------------

    def record_request_duration(
        self, ms: float, *, worker_id: int, request_type: str
    ) -> None:
        if request_type == "inference":
            self.stats.record_e2e(worker_id, ms)
        if self._req_hist is not None:
            self._req_hist.record(
                ms,
                {
                    "worker_id": str(worker_id),
                    "request_type": request_type,
                },
            )

    def record_inference_duration(
        self, ms: float, *, provider: str, model_name: str, worker_id: int | None = None
    ) -> None:
        if worker_id is not None:
            self.stats.record_infer(worker_id, ms)
        if self._infer_hist is not None:
            self._infer_hist.record(
                ms,
                {
                    "provider": provider,
                    "model_name": model_name,
                },
            )

    def record_trt_h2d(self, ms: float) -> None:
        self.stats.record_trt_h2d(ms)
        if self._trt_h2d_hist is not None:
            self._trt_h2d_hist.record(ms)

    def record_trt_execute(self, ms: float) -> None:
        self.stats.record_trt_execute(ms)
        if self._trt_execute_hist is not None:
            self._trt_execute_hist.record(ms)

    def record_trt_d2h(self, ms: float) -> None:
        self.stats.record_trt_d2h(ms)
        if self._trt_d2h_hist is not None:
            self._trt_d2h_hist.record(ms)

    def record_postprocess_duration(
        self, ms: float, *, post_processor: str, worker_id: int | None = None
    ) -> None:
        if worker_id is not None:
            self.stats.record_postprocess(worker_id, ms)
        if self._postprocess_hist is not None:
            self._postprocess_hist.record(ms, {"post_processor": post_processor})

    def record_sem_wait(self, ms: float) -> None:
        self.stats.record_sem_wait(ms)
        if self._sem_wait_hist is not None:
            self._sem_wait_hist.record(ms)

    def record_model_lock_wait(self, ms: float) -> None:
        if self._model_lock_wait_hist is not None:
            self._model_lock_wait_hist.record(ms)

    def record_sync(self, ms: float) -> None:
        # TODO implement me: record stat and also hist
        # report those
        pass

    # ------------------------------------------------------------------
    # UpDownCounter helpers
    # ------------------------------------------------------------------

    def inc_workers_active(self) -> None:
        self.stats.inc_workers_active()
        if self._workers_active is not None:
            self._workers_active.add(1)

    def dec_workers_active(self) -> None:
        self.stats.dec_workers_active()
        if self._workers_active is not None:
            self._workers_active.add(-1)

    def inc_inference_concurrent(self) -> None:
        self.stats.inc_inference_concurrent()
        if self._inference_concurrent is not None:
            self._inference_concurrent.add(1)

    def dec_inference_concurrent(self) -> None:
        self.stats.dec_inference_concurrent()
        if self._inference_concurrent is not None:
            self._inference_concurrent.add(-1)


# Module-level singleton imported by all other modules.
tel = _Tel()


# ---------------------------------------------------------------------------
# Context manager helpers
# ---------------------------------------------------------------------------


@contextmanager
def timed_span(name: str, **attrs: object) -> Generator[trace.Span, None, None]:
    """Start a span, set attributes, and yield it.

    Records StatusCode.ERROR on uncaught exceptions and re-raises.
    """
    with tracer.start_as_current_span(name) as span:
        for k, v in attrs.items():
            span.set_attribute(k, v)
        try:
            yield span
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise


# ---------------------------------------------------------------------------
# Provider setup
# ---------------------------------------------------------------------------


def setup_telemetry(config: ObservabilityConfig) -> None:
    """Configure OTel providers based on *config*.

    Must be called in the main process before workers start.  Configures the
    global TracerProvider and MeterProvider, then initialises metric instruments
    on the ``tel`` singleton so all subsequent metric calls are real.
    """
    if not config.enabled:
        logger.info("Observability disabled (observability.enabled: false).")
        return

    from opentelemetry.sdk.resources import Resource

    resource = Resource.create(
        {
            "service.name": config.service_name,
            "service.version": config.service_version,
        }
    )

    mode = config.mode
    if mode == "tui":
        # TUI-only mode: no OTel providers configured; all metrics flow exclusively
        # through tel.stats → the Unix socket → condor-tui.  OTel instruments
        # remain no-ops (tel._init is not called).
        logger.info(
            "Observability mode: tui — metrics streamed to stats socket only "
            "(no console / Prometheus / OTLP output)."
        )
        return
    elif mode == "console":
        _setup_console(config, resource)
    elif mode == "prometheus":
        _setup_prometheus(config, resource)
    elif mode == "otlp":
        _setup_otlp(config, resource)
    else:
        logger.warning(
            "Unknown observability.mode %r; observability disabled. "
            "Valid values: tui, console, prometheus, otlp.",
            mode,
        )
        return

    # Initialise OTel metric instruments AFTER the MeterProvider is configured
    # so instruments are real exporters, not no-ops.
    # (Skipped for mode=tui — stats socket handles everything in that case.)
    meter = otel_metrics.get_meter(
        "condor",
        schema_url="https://opentelemetry.io/schemas/1.27.0",
    )
    tel._init(meter)
    logger.info(
        "Observability configured: mode=%s service=%s version=%s",
        mode,
        config.service_name,
        config.service_version,
    )


def _setup_console(config: ObservabilityConfig, resource: object) -> None:
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import (
        ConsoleMetricExporter,
        PeriodicExportingMetricReader,
    )
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    # Traces — optional (verbose; off by default)
    tp = TracerProvider(resource=resource)  # type: ignore[arg-type]
    if config.console.export_traces:
        tp.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        logger.info("Console span export enabled (export_traces: true).")
    trace.set_tracer_provider(tp)

    # Metrics — always enabled; periodic JSON snapshot to stdout
    interval_ms = max(config.console.metrics_interval_seconds * 1000, 5_000)
    reader = PeriodicExportingMetricReader(
        ConsoleMetricExporter(),
        export_interval_millis=int(interval_ms),
    )
    mp = MeterProvider(resource=resource, metric_readers=[reader])  # type: ignore[arg-type]
    otel_metrics.set_meter_provider(mp)
    logger.info(
        "Console metrics: snapshot every %ds (set observability.console.metrics_interval_seconds to change).",
        config.console.metrics_interval_seconds,
    )


def _setup_prometheus(config: ObservabilityConfig, resource: object) -> None:
    try:
        from opentelemetry.exporter.prometheus import PrometheusMetricReader
    except ImportError:
        logger.error(
            "opentelemetry-exporter-prometheus is not installed. "
            "Install it with: make install-observability-local  "
            "Falling back to no metrics export."
        )
        return

    from opentelemetry.sdk.metrics import MeterProvider

    reader = PrometheusMetricReader()
    mp = MeterProvider(resource=resource, metric_readers=[reader])  # type: ignore[arg-type]
    otel_metrics.set_meter_provider(mp)

    # Start the HTTP server for Prometheus scraping.
    try:
        from prometheus_client import start_http_server

        start_http_server(port=config.prometheus.port, addr=config.prometheus.host)
        logger.info(
            "Prometheus metrics available at http://%s:%d/metrics",
            config.prometheus.host,
            config.prometheus.port,
        )
    except Exception:
        logger.exception("Failed to start Prometheus HTTP server.")

    # No trace provider in prometheus mode — metrics only.
    # Spans are still created (for hierarchy / timing), but discarded.


def _setup_otlp(config: ObservabilityConfig, resource: object) -> None:
    try:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
    except ImportError:
        logger.error(
            "opentelemetry-exporter-otlp-proto-http is not installed. "
            "Install it with: make install-observability-otlp"
        )
        return

    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    endpoint = config.otlp.endpoint
    headers = config.otlp.headers

    if config.otlp.export_traces:
        span_exporter = OTLPSpanExporter(
            endpoint=f"{endpoint}/v1/traces",
            headers=headers,
        )
        tp = TracerProvider(resource=resource)  # type: ignore[arg-type]
        tp.add_span_processor(BatchSpanProcessor(span_exporter))
        trace.set_tracer_provider(tp)

    if config.otlp.export_metrics:
        interval_ms = max(config.otlp.metrics_interval_seconds * 1000, 5_000)
        metric_exporter = OTLPMetricExporter(
            endpoint=f"{endpoint}/v1/metrics",
            headers=headers,
        )
        reader = PeriodicExportingMetricReader(
            metric_exporter,
            export_interval_millis=int(interval_ms),
        )
        mp = MeterProvider(resource=resource, metric_readers=[reader])  # type: ignore[arg-type]
        otel_metrics.set_meter_provider(mp)

    logger.info(
        "OTLP export configured → %s (traces=%s metrics=%s)",
        endpoint,
        config.otlp.export_traces,
        config.otlp.export_metrics,
    )
