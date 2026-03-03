# ─── Condor — TensorRT backend ────────────────────────────────────────────────
#
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  HOST SYSTEM RULE — NON-NEGOTIABLE                                         ║
# ║                                                                            ║
# ║  NEVER install, modify, or interact with NVIDIA drivers, CUDA toolkit,    ║
# ║  cuDNN, TensorRT, cuda-python, or any GPU/NVIDIA stack on the HOST.       ║
# ║  The host needs ONLY the NVIDIA kernel driver + nvidia-container-toolkit.  ║
# ║  ALL TensorRT work runs INSIDE this container via --runtime nvidia.        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Base image: nvcr.io/nvidia/tensorrt:26.01-py3  (NVIDIA NGC registry)
#   Provides: TensorRT Python bindings, cuda-python, CUDA runtime + driver API,
#             cuDNN, libnvinfer plugins, Python 3.12
#
# Build:
#   make docker-build-tensorrt
#   make docker-rebuild-tensorrt   # force --no-cache
#
# See docs/requirements/TENSORRT.md §2 for full dev workflow.
# ─────────────────────────────────────────────────────────────────────────────

FROM nvcr.io/nvidia/tensorrt:26.01-py3

LABEL org.opencontainers.image.title="condor"
LABEL org.opencontainers.image.description="Frigate ZMQ remote detector — TensorRT backend"

# curl: needed to bootstrap uv.
# libglib2.0-0: runtime dep of opencv-python-headless.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv — the only package manager used in this project.
RUN curl -Ls https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create a virtualenv that inherits the base image's system site-packages so
# that tensorrt and cuda-python (pre-installed in the NGC image) are
# importable without reinstalling or duplicating them.
RUN uv venv --system-site-packages /opt/condor-venv
ENV VIRTUAL_ENV=/opt/condor-venv
# uv pip commands respect VIRTUAL_ENV; uv sync (a project command) uses
# UV_PROJECT_ENVIRONMENT instead — both must point to the same venv.
ENV UV_PROJECT_ENVIRONMENT=/opt/condor-venv
ENV PATH="/opt/condor-venv/bin:$PATH"

WORKDIR /app

# Copy project metadata (layer-cache friendly: dep install is re-run only when
# pyproject.toml changes, not when application code changes).
COPY pyproject.toml ./

# onnxruntime is not used by the TensorRT backend; remove it so uv sync does
# not install a ~300 MB wheel that serves no purpose in this image.
# tensorrt and cuda-python are provided by the NGC base via --system-site-packages.
# All other deps — including opentelemetry-api/sdk — are installed from pyproject.toml.
# No lockfile is used here because pyproject.toml is modified above; uv will
# resolve and pin a fresh lockfile inside the build context.
RUN sed -i '/"onnxruntime>=/d' pyproject.toml \
    && uv sync --no-dev --no-install-project --extra tui

# Copy application code and install the condor package itself.
COPY condor/ ./condor/
RUN uv sync --no-dev --extra tui

# Default config (override at runtime with -v /path/to/config:/app/config).
RUN mkdir -p /app/config /app/models
COPY config/config.yaml /app/config/config.yaml

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 5555-5559

ENTRYPOINT ["condor"]
CMD ["--config", "/app/config/config.yaml"]
