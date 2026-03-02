.PHONY: install run test test-client lint \
        docker-build-onnx docker-run-onnx \
        docker-build-onnx-cuda docker-run-onnx-cuda docker-shell-onnx-cuda docker-test-onnx-cuda \
        docker-build-openvino docker-run-openvino \
        docker-build-tensorrt docker-rebuild-tensorrt \
        docker-run-tensorrt docker-shell-tensorrt docker-test-tensorrt \
        install-openvino install-onnxruntime-openvino

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

# ── OpenVINO ───────────────────────────────────────────────────────────────────
# Install the native OpenVINO backend extra (provider: "openvino").
install-openvino:
	uv sync --extra openvino

# Install onnxruntime-openvino EP for use with provider: "onnx" +
# execution_provider: "OpenVINOExecutionProvider" in provider_options.
install-onnxruntime-openvino:
	uv pip uninstall onnxruntime --yes || true
	uv pip install onnxruntime-openvino

# ── Docker ─────────────────────────────────────────────────────────────────────

IMAGE_ONNX      ?= condor:onnxruntime
IMAGE_ONNX_CUDA ?= condor:onnxruntime-cuda
IMAGE_OPENVINO  ?= condor:openvino
IMAGE_TENSORRT  ?= condor:tensorrt

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

# ── ONNX Runtime (CPU + optional OpenVINO EP) ─────────────────────────────────

docker-build-onnx:
	docker build \
	  -f docker/onnxruntime/Dockerfile \
	  -t $(IMAGE_ONNX) \
	  .

docker-run-onnx:
	docker run --rm -it \
	  $(call port_flags,$(NUM_WORKERS),$(BASE_PORT)) \
	  -v $(PWD)/models:/app/models \
	  -v $(PWD)/config:/app/config \
	  $(IMAGE_ONNX)

# ── ONNX Runtime CUDA EP ───────────────────────────────────────────────────────
#
# Requires: NVIDIA driver on the host + Docker with NVIDIA Container Toolkit.
# NEVER run without --gpus all.  NEVER install onnxruntime-gpu on the host.

docker-build-onnx-cuda:
	docker build \
	  -f docker/onnxruntime-cuda/Dockerfile \
	  -t $(IMAGE_ONNX_CUDA) \
	  .

docker-run-onnx-cuda:
	docker run --rm -it --gpus all \
	  $(call port_flags,$(NUM_WORKERS),$(BASE_PORT)) \
	  -v $(MODELS_DIR):/app/models \
	  -v $(CONFIG_DIR):/app/config \
	  $(IMAGE_ONNX_CUDA)

docker-shell-onnx-cuda:
	docker run --rm -it --gpus all \
	  --entrypoint bash \
	  $(IMAGE_ONNX_CUDA)

docker-test-onnx-cuda:
	docker run --rm --gpus all \
	  --entrypoint python \
	  $(IMAGE_ONNX_CUDA) \
	  -m pytest tests/ -v

# ── Native OpenVINO backend ────────────────────────────────────────────────────

docker-build-openvino:
	docker build \
	  -f docker/openvino/Dockerfile \
	  -t $(IMAGE_OPENVINO) \
	  .

docker-run-openvino:
	docker run --rm -it \
	  $(call port_flags,$(NUM_WORKERS),$(BASE_PORT)) \
	  -v $(PWD)/models:/app/models \
	  -v $(PWD)/config:/app/config \
	  $(IMAGE_OPENVINO)

# ── TensorRT backend ───────────────────────────────────────────────────────────
#
# Requires: NVIDIA driver on the host + Docker with NVIDIA Container Toolkit.
# NEVER run without --runtime nvidia.  NEVER install TensorRT on the host.

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

docker-run-tensorrt:
	docker run --rm -it --runtime nvidia \
	  $(call port_flags,$(NUM_WORKERS),$(BASE_PORT)) \
	  -v $(MODELS_DIR):/app/models \
	  -v $(CONFIG_DIR):/app/config \
	  $(IMAGE_TENSORRT)

docker-shell-tensorrt:
	docker run --rm -it --runtime nvidia \
	  --entrypoint bash \
	  $(IMAGE_TENSORRT)

docker-test-tensorrt:
	docker run --rm --runtime nvidia \
	  --entrypoint python \
	  $(IMAGE_TENSORRT) \
	  -m pytest tests/ -v
