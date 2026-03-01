.PHONY: install run test test-client lint \
        docker-build-onnx docker-run-onnx \
        docker-build-openvino \
        docker-build-tensorrt docker-rebuild-tensorrt \
        docker-run-tensorrt docker-shell-tensorrt docker-test-tensorrt \
        install-openvino

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

# ── OpenVINO EP (onnxruntime-openvino) ────────────────────────────────────────
# Swaps the standard onnxruntime wheel for the OpenVINO-bundled one.
# Set provider: "openvino" in config/config.yaml afterwards.
install-openvino:
	uv pip uninstall onnxruntime --yes || true
	uv pip install onnxruntime-openvino

# ── Docker ─────────────────────────────────────────────────────────────────────

IMAGE_ONNX     ?= condor:onnxruntime
IMAGE_OPENVINO ?= condor:openvino
IMAGE_TENSORRT ?= condor:tensorrt

# ONNX Runtime (CPU + optional OpenVINO EP)
docker-build-onnx:
	docker build \
	  -f docker/onnxruntime/Dockerfile \
	  -t $(IMAGE_ONNX) \
	  .

docker-run-onnx:
	docker run --rm -it \
	  -p 5555:5555 \
	  -v $(PWD)/models:/app/models \
	  -v $(PWD)/config:/app/config \
	  $(IMAGE_ONNX)

# OpenVINO native backend (Phase 2)
docker-build-openvino:
	docker build \
	  -f docker/openvino/Dockerfile \
	  -t $(IMAGE_OPENVINO) \
	  .

# TensorRT backend
docker-build-tensorrt:
	docker build \
	  -f docker/tensorrt/Dockerfile \
	  -t $(IMAGE_TENSORRT) \
	  .

# Force a clean rebuild — pulls the latest NGC base image and skips layer cache.
docker-rebuild-tensorrt:
	docker build \
	  --no-cache \
	  --pull \
	  -f docker/tensorrt/Dockerfile \
	  -t $(IMAGE_TENSORRT) \
	  .

# Run the Condor server with the TRT backend.
# Override models/config dirs with MODELS_DIR and CONFIG_DIR variables:
#   make docker-run-tensorrt MODELS_DIR=/data/models CONFIG_DIR=/data/config
MODELS_DIR ?= $(PWD)/models
CONFIG_DIR  ?= $(PWD)/config

docker-run-tensorrt:
	docker run --rm -it --runtime nvidia \
	  -p 5555:5555 \
	  -v $(MODELS_DIR):/app/models \
	  -v $(CONFIG_DIR):/app/config \
	  $(IMAGE_TENSORRT)

# Interactive dev shell inside the TRT container (code is baked in).
docker-shell-tensorrt:
	docker run --rm -it --runtime nvidia \
	  --entrypoint bash \
	  $(IMAGE_TENSORRT)

# Run the full pytest suite against the baked-in code.
docker-test-tensorrt:
	docker run --rm --runtime nvidia \
	  --entrypoint python \
	  $(IMAGE_TENSORRT) \
	  -m pytest tests/ -v
