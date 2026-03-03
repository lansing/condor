# ─── Condor — Native OpenVINO backend ────────────────────────────────────────
#
# Uses the openvino Python package directly (not onnxruntime-openvino EP).
# Supports CPU, GPU (Intel integrated / Arc), AUTO, NPU device targets.
#
# Build:
#   docker build -f docker/openvino/Dockerfile -t condor:openvino .
#
# Run:
#   docker run --rm -it \
#     -p 5555:5555 \
#     -v /path/to/models:/app/models \
#     -v /path/to/config:/app/config \
#     condor:openvino
#
# For Intel GPU (i915 driver) access, add:
#   --device /dev/dri --group-add video
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

# Copy project metadata first (layer-cache friendly)
COPY pyproject.toml uv.lock* ./

# Install core + openvino extra into a local venv
RUN uv sync --no-dev --no-install-project --extra openvino --extra tui

# Copy the rest of the project
COPY condor/ ./condor/

# Install the project itself
RUN uv sync --no-dev --extra openvino --extra tui

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="condor"
LABEL org.opencontainers.image.description="Frigate ZMQ remote detector — native OpenVINO backend"

# Runtime libraries required by OpenVINO and opencv-headless
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
