# ─── Condor — ONNX Runtime CUDA Execution Provider ───────────────────────────
#
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  HOST SYSTEM RULE — NON-NEGOTIABLE                                         ║
# ║                                                                            ║
# ║  NEVER install onnxruntime-gpu, CUDA toolkit, cuDNN, or any NVIDIA        ║
# ║  stack on the HOST beyond the kernel driver + nvidia-container-toolkit.   ║
# ║  ALL GPU-accelerated ONNX inference runs INSIDE this container.           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Uses onnxruntime-gpu which bundles CUDA 12.x runtime libraries in the wheel,
# so the runtime image remains python:3.12-slim — no CUDA base image needed.
# GPU access is provided by the NVIDIA Container Toolkit via --gpus all.
#
# Model files are standard .onnx — the same files used for CPU inference.
# Eligible ops run on GPU; unsupported ops fall back to CPUExecutionProvider.
#
# Build:
#   make docker-build-onnx-cuda
#
# Run:
#   make docker-run-onnx-cuda
#   # equivalent to:
#   docker run --rm -it --gpus all \
#     -p 5555:5555 \
#     -v /path/to/models:/app/models \
#     -v /path/to/config:/app/config \
#     condor:onnxruntime-cuda
#
# See config/examples/onnx-cuda.yaml for the matching server config.
# ─────────────────────────────────────────────────────────────────────────────

ARG PYTHON_VERSION=3.12

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -Ls https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy project metadata first (layer-cache friendly).
COPY pyproject.toml uv.lock* ./

# Replace the standard onnxruntime wheel with onnxruntime-gpu.
# These two packages conflict — only one may be installed at a time.
RUN sed -i 's|"onnxruntime>=.*"|"onnxruntime-gpu>=1.19"|g' pyproject.toml

# Install dependencies into a local venv.
RUN uv sync --no-dev --no-install-project

# Copy the rest of the project and install it.
COPY condor/ ./condor/
RUN uv sync --no-dev

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
# onnxruntime-gpu bundles CUDA 12.x runtime libs in the wheel (libcudart,
# libcublas, libcurand, libcudnn, etc.).  No CUDA base image is required;
# the NVIDIA Container Toolkit exposes the GPU device at runtime.
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="condor"
LABEL org.opencontainers.image.description="Frigate ZMQ remote detector — ONNX Runtime CUDA EP"

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the virtualenv (includes onnxruntime-gpu with bundled CUDA libs).
COPY --from=builder /build/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code.
COPY --from=builder /build/condor /app/condor

# Default config and models directories (override with bind mounts at runtime).
RUN mkdir -p /app/config /app/models
COPY config/examples/onnx-cuda.yaml /app/config/config.yaml

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 5555-5559

ENTRYPOINT ["condor"]
CMD ["--config", "/app/config/config.yaml"]
