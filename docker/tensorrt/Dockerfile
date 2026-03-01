# ─── Condor — TensorRT backend (Phase 2 — placeholder) ───────────────────────
#
# This Dockerfile will be implemented in Phase 2 when TensorRTBackend is added.
#
# Planned base image: nvcr.io/nvidia/tensorrt:<version>-py3
# Requires: NVIDIA GPU, nvidia-container-toolkit on the host.
#
# Build (Phase 2):
#   docker build -f docker/tensorrt/Dockerfile -t condor:tensorrt .
#
# Run (Phase 2):
#   docker run --rm --gpus all \
#     -p 5555:5555 \
#     -v /path/to/models:/app/models \
#     -v /path/to/config:/app/config \
#     condor:tensorrt
# ─────────────────────────────────────────────────────────────────────────────

# TODO (Phase 2): replace with the appropriate TensorRT base image, e.g.:
# FROM nvcr.io/nvidia/tensorrt:24.01-py3 AS builder

FROM python:3.12-slim AS placeholder

RUN echo "TensorRT backend not yet implemented (Phase 2)." && exit 1
