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


class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    post_process: PostProcessConfig = Field(default_factory=PostProcessConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @model_validator(mode="before")
    @classmethod
    def _coerce_none_sections(cls, data: Any) -> Any:
        """Replace null/missing config sections with empty dicts so Pydantic
        applies field defaults rather than raising a validation error."""
        if isinstance(data, dict):
            for key in ("server", "inference", "post_process", "logging"):
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
