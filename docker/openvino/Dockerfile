# ─── Condor — OpenVINO native backend (Phase 2 — placeholder) ────────────────
#
# This Dockerfile is for a future native OpenVINO backend (Phase 2).
# For the Phase 1 OpenVINO EP (via onnxruntime-openvino), use the ONNX Runtime
# Dockerfile with --build-arg ONNX_PACKAGE=onnxruntime-openvino instead.
#
# Build (Phase 2):
#   docker build -f docker/openvino/Dockerfile -t condor:openvino .
#
# Run (Phase 2):
#   docker run --rm -it \
#     -p 5555:5555 \
#     -v /path/to/models:/app/models \
#     -v /path/to/config:/app/config \
#     condor:openvino
# ─────────────────────────────────────────────────────────────────────────────

# TODO (Phase 2): replace with the Intel OpenVINO base image, e.g.:
# FROM openvino/ubuntu22_runtime:2024.1.0 AS builder

FROM python:3.12-slim AS placeholder

RUN echo "Native OpenVINO backend not yet implemented (Phase 2)." && exit 1
