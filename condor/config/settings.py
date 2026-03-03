"""Pydantic-validated configuration with YAML loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class ServerConfig(BaseModel):
    endpoint: str = "tcp://*:5555"
    models_dir: str = "./models"
    num_workers: int = 1
    base_port: int = 5555


class InferenceConfig(BaseModel):
    provider: str = "cpu"
    provider_options: dict[str, Any] = Field(default_factory=dict)
    max_inference_concurrency: int = 0  # 0 = unlimited


class PostProcessConfig(BaseModel):
    confidence_threshold: float = 0.4
    max_detections: int = 20


class LoggingConfig(BaseModel):
    level: str = "INFO"


class ConsoleObservabilityConfig(BaseModel):
    # Print a metric snapshot to stdout every N seconds (0 = disable).
    metrics_interval_seconds: int = 30
    # Also emit individual finished spans as JSON to stdout (very verbose).
    export_traces: bool = False


class PrometheusObservabilityConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 9090


class OtlpObservabilityConfig(BaseModel):
    # OTLP/HTTP base URL — traces go to {endpoint}/v1/traces, metrics to /v1/metrics.
    endpoint: str = "http://localhost:4318"
    # Extra HTTP headers, e.g. {"authorization": "Bearer YOUR_KEY"} for HyperDX cloud.
    headers: dict[str, str] = Field(default_factory=dict)
    export_traces: bool = True
    export_metrics: bool = True
    metrics_interval_seconds: int = 30


class ObservabilityConfig(BaseModel):
    enabled: bool = False
    # "console"    — spans + metrics summary printed to stdout (zero extra software).
    # "prometheus" — Prometheus scrape endpoint at http://host:port/metrics.
    # "otlp"       — export traces + metrics via OTLP to HyperDX, Grafana, etc.
    mode: str = "console"
    service_name: str = "condor"
    service_version: str = "0.1.0"
    console: ConsoleObservabilityConfig = Field(default_factory=ConsoleObservabilityConfig)
    prometheus: PrometheusObservabilityConfig = Field(default_factory=PrometheusObservabilityConfig)
    otlp: OtlpObservabilityConfig = Field(default_factory=OtlpObservabilityConfig)


class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    post_process: PostProcessConfig = Field(default_factory=PostProcessConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    @model_validator(mode="before")
    @classmethod
    def _coerce_none_sections(cls, data: Any) -> Any:
        """Replace null/missing config sections with empty dicts so Pydantic
        applies field defaults rather than raising a validation error."""
        if isinstance(data, dict):
            for key in ("server", "inference", "post_process", "logging", "observability"):
                if data.get(key) is None:
                    data[key] = {}
        return data


def load_config(path: str = "config/config.yaml") -> AppConfig:
    """Load config from *path* (YAML).  Returns all-defaults AppConfig if the
    file does not exist."""
    config_path = Path(path)
    if config_path.exists():
        with config_path.open() as f:
            data = yaml.safe_load(f)
        return AppConfig.model_validate(data or {})
    return AppConfig()
