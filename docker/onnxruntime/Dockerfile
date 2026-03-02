# ─── Condor — ONNX Runtime backend ───────────────────────────────────────────
#
# Supports:
#   - CPU execution provider (default)
#   - Intel OpenVINO execution provider (set provider: "openvino" in config)
#
# Build:
#   docker build -f docker/onnxruntime/Dockerfile -t condor:onnxruntime .
#
# Run:
#   docker run --rm -it \
#     -p 5555:5555 \
#     -v /path/to/models:/app/models \
#     -v /path/to/config:/app/config \
#     condor:onnxruntime
#
# For OpenVINO EP, override the onnxruntime wheel at build time:
#   docker build \
#     --build-arg ONNX_PACKAGE=onnxruntime-openvino \
#     -f docker/onnxruntime/Dockerfile \
#     -t condor:openvino-ep .
# ─────────────────────────────────────────────────────────────────────────────

ARG PYTHON_VERSION=3.12
ARG ONNX_PACKAGE=onnxruntime

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS builder

ARG ONNX_PACKAGE

WORKDIR /build

# System build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -Ls https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy project metadata first (layer-cache friendly)
COPY pyproject.toml uv.lock* ./

# Swap onnxruntime wheel if a different package is requested
RUN if [ "$ONNX_PACKAGE" != "onnxruntime" ]; then \
        sed -i 's|"onnxruntime>=.*"|"'"$ONNX_PACKAGE"'"|g' pyproject.toml; \
    fi

# Install dependencies into a local venv
RUN uv sync --no-dev --no-install-project

# Copy the rest of the project
COPY condor/ ./condor/

# Install the project itself
RUN uv sync --no-dev

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="condor"
LABEL org.opencontainers.image.description="Frigate ZMQ remote detector — ONNX Runtime backend"

# Runtime system libraries required by onnxruntime and opencv-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the virtualenv from the builder stage
COPY --from=builder /build/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code
COPY --from=builder /build/condor /app/condor

# Default config and models directories (override with bind mounts at runtime)
RUN mkdir -p /app/config /app/models
COPY config/config.yaml /app/config/config.yaml

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 5555-5559

ENTRYPOINT ["condor"]
CMD ["--config", "/app/config/config.yaml"]
