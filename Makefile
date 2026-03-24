.PHONY: install run test test-client lint \
        docker-build-tensorrt docker-rebuild-tensorrt \
        docker-run-tensorrt docker-shell-tensorrt docker-test-tensorrt \
        docker-build-tensorrt-build docker-run-tensorrt-build docker-shell-tensorrt-build \
        install-observability-local install-observability-otlp \
        install-tui run-tui tui-host tui-docker

CONFIG ?= config/config.yaml

# ── Development ────────────────────────────────────────────────────────────────

# Install project and dev dependencies
install:
	uv sync

# Run the detector server
run:
	uv run condor --config $(CONFIG)

# Run unit tests
test:
	uv run pytest tests/ -v

# Run the test/benchmark client against a running server
test-client:
	uv run python scripts/test_client.py --config $(CONFIG)

# ── Observability ───────────────────────────────────────────────────────────────

# Lightweight local metrics: Prometheus scrape endpoint (no database needed).
# After installing, set observability.mode: "prometheus" in config.yaml.
install-observability-local:
	uv sync --extra observability-local

# Full OTLP export to HyperDX, Grafana Tempo, Jaeger, etc.
# After installing, set observability.mode: "otlp" in config.yaml.
install-observability-otlp:
	uv sync --extra observability-otlp

# ── Metrics TUI ─────────────────────────────────────────────────────────────────

# Install Textual and register the condor-tui entry point.
install-tui:
	uv sync --extra tui

# Launch the TUI against a locally-running server (native uv run).
# Socket: /tmp/condor-metrics.sock  (default, override with CONDOR_STATS_SOCKET=...).
run-tui:
	uv run condor-tui

# Launch the TUI on the HOST, reading the socket exposed by docker compose
# via the bind-mount at ./run/metrics.sock.  Requires: make install-tui.
tui-host:
	CONDOR_STATS_SOCKET=run/metrics.sock uv run condor-tui

# Launch the TUI INSIDE the running Docker Compose container.
# No extra installs needed — condor-tui is already in the image.
tui-docker:
	docker compose exec condor condor-tui

# ── Docker ─────────────────────────────────────────────────────────────────────

IMAGE_TENSORRT       ?= condor
IMAGE_TENSORRT_BUILD ?= condor:tensorrt-build

# Override models/config mount paths:
#   make docker-run-tensorrt MODELS_DIR=/data/models CONFIG_DIR=/data/config
MODELS_DIR  ?= $(PWD)/models
CONFIG_DIR   ?= $(PWD)/config
# Number of workers / ports to expose (base port 5555 through 5555+NUM_WORKERS-1).
# Override to match num_workers in config.yaml, e.g.:
#   make docker-run-tensorrt NUM_WORKERS=3
NUM_WORKERS ?= 1
BASE_PORT   ?= 5555
# Build a -p flag for each worker port: $(call port_flags,NUM_WORKERS,BASE_PORT)
port_flags = $(foreach i,$(shell seq 0 $(shell expr $(1) - 1)),-p $(shell expr $(2) + $(i)):$(shell expr $(2) + $(i)))

# ── TensorRT backend — lean inference image (default) ─────────────────────────
#
# Requires: NVIDIA driver on the host + Docker with NVIDIA Container Toolkit.
# NEVER run without --runtime nvidia.  NEVER install TensorRT on the host.
#
# Multi-stage build from nvcr.io/nvidia/cuda:13.1.1-base-ubuntu24.04.
# Copies only the TRT/cuDNN/cuBLAS runtime .so files needed for inference;
# builder-resource files (~1.82 GB) and unneeded CUDA math libs are excluded.
# trtexec is included for smoke-testing (see Dockerfile to remove it later).

docker-build:
	docker build \
	  -f docker/tensorrt/Dockerfile \
	  -t $(IMAGE_TENSORRT) \
	  .

# Force a clean rebuild — skips layer cache.
docker-rebuild:
	docker build \
	  --no-cache \
	  -f docker/tensorrt/Dockerfile \
	  -t $(IMAGE_TENSORRT) \
	  .

docker-run:
	docker run --rm -it --runtime nvidia \
	  $(call port_flags,$(NUM_WORKERS),$(BASE_PORT)) \
	  -v $(MODELS_DIR):/app/models \
	  -v $(CONFIG_DIR):/app/config \
	  $(IMAGE_TENSORRT)

docker-shell:
	docker run --rm -it --runtime nvidia \
	  --entrypoint bash \
	  $(IMAGE_TENSORRT)

docker-test:
	docker run --rm --runtime nvidia \
	  --entrypoint python \
	  $(IMAGE_TENSORRT) \
	  -m pytest tests/ -v

# ── TensorRT build image (full NGC base, includes engine builder) ──────────────
#
# Use this when you need to compile .engine files from ONNX models.
# Based on nvcr.io/nvidia/tensorrt:26.01-py3 (~10 GB); includes all TRT builder
# resources, nvcc, and the full CUDA toolkit.

docker-build-tensorrt-build:
	docker build \
	  -f docker/tensorrt/Dockerfile.build \
	  -t $(IMAGE_TENSORRT_BUILD) \
	  .

docker-run-tensorrt-build:
	docker run --rm -it --runtime nvidia \
	  $(call port_flags,$(NUM_WORKERS),$(BASE_PORT)) \
	  -v $(MODELS_DIR):/app/models \
	  -v $(CONFIG_DIR):/app/config \
	  $(IMAGE_TENSORRT_BUILD)

docker-shell-tensorrt-build:
	docker run --rm -it --runtime nvidia \
	  --entrypoint bash \
	  $(IMAGE_TENSORRT_BUILD)
