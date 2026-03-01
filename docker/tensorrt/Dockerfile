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
ENV PATH="/opt/condor-venv/bin:$PATH"

WORKDIR /app

# Install project Python dependencies.
# Intentionally excluded:
#   onnxruntime  — not used in the TensorRT backend.
#   tensorrt     — inherited from base image via --system-site-packages.
#   cuda-python  — inherited from base image via --system-site-packages.
COPY pyproject.toml ./
RUN uv pip install \
        "pyzmq>=26.0" \
        "aiofiles>=24.0" \
        "numpy>=2.0" \
        "pydantic>=2.0" \
        "pyyaml>=6.0" \
        "opencv-python-headless>=4.10"

# Install the condor package itself (--no-deps: deps already installed above).
COPY condor/ ./condor/
RUN uv pip install --no-deps .

# Default config (override at runtime with -v /path/to/config:/app/config).
RUN mkdir -p /app/config /app/models
COPY config/config.yaml /app/config/config.yaml

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 5555

ENTRYPOINT ["condor"]
CMD ["--config", "/app/config/config.yaml"]
