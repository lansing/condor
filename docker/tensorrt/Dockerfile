# ─── Condor — TensorRT backend ────────────────────────────────────────────────
#
# Multi-stage build:
#   Stage 1 (trt-libs): nvcr.io/nvidia/tensorrt:26.01-py3  [already local]
#     Acts as source only.  Provides TRT runtime .so files, cuDNN .so files,
#     CUDA math libs, and the cuda-python / tensorrt Python packages.
#
#   Stage 2 (final):    nvcr.io/nvidia/cuda:13.1.1-base-ubuntu24.04
#     Minimal CUDA base: just libcudart + cuda-compat (~500 MB).
#     No cuDNN, no cuBLAS, no Python.  Everything is selectively copied from
#     Stage 1 — only what TRT inference actually needs for object detection:
#       TRT runtime .so   — libnvinfer + plugin + lean + dispatch
#       CUDA math libs    — cuBLAS + cuBLASLt (GEMM for all OD models incl.
#                           transformer-based), libnvrtc + libnvJitLink (TRT
#                           JIT-compiles some kernels at engine load time)
#       cuDNN runtime     — conv, BN, activations, attention
#     The ~1.82 GB of TRT builder-resource .so files are deliberately excluded
#     (libnvinfer_builder_resource_*.so) — they are only consulted by the TRT
#     engine builder and are never touched during inference on pre-built engines.
#     Also excluded: libcufft, libcusparse, libcusolver, libcurand, NPP (~941 MB
#     combined) — none are on the execution path for any OD model family.
#
# trtexec is included in this image for inference smoke-testing.
# Once confidence is established, it (and libnvonnxparser) can be removed.
#
# Build:
#   make docker-build-tensorrt
# ─────────────────────────────────────────────────────────────────────────────

FROM nvcr.io/nvidia/tensorrt:26.01-py3 AS trt-libs

# ─── Final lean image ─────────────────────────────────────────────────────────
FROM nvcr.io/nvidia/cuda:13.1.1-base-ubuntu24.04

LABEL org.opencontainers.image.title="condor"
LABEL org.opencontainers.image.description="TensorRT sidecar for Frigate — lean runtime"
LABEL condor.builder.image="nvcr.io/nvidia/tensorrt:26.01-py3"

# ── TRT runtime .so files ──────────────────────────────────────────────────────
# Only the inference-time subset is copied.
# The 8 builder-resource files (libnvinfer_builder_resource_*.so, ~1.82 GB)
# are intentionally omitted.
#
# Each COPY line is a separate layer so Docker can cache individual files.
COPY --from=trt-libs \
    /usr/lib/x86_64-linux-gnu/libnvinfer.so.10.14.1 \
    /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.10.14.1 \
    /usr/lib/x86_64-linux-gnu/libnvinfer_lean.so.10.14.1 \
    /usr/lib/x86_64-linux-gnu/libnvinfer_dispatch.so.10.14.1 \
    /usr/lib/x86_64-linux-gnu/libnvinfer_vc_plugin.so.10.14.1 \
    /usr/lib/x86_64-linux-gnu/

# libnvonnxparser: required by trtexec even in --loadEngine mode.
# Remove this line (and the trtexec COPY below) once smoke-testing is done.
COPY --from=trt-libs \
    /usr/lib/x86_64-linux-gnu/libnvonnxparser.so.10.14.1 \
    /usr/lib/x86_64-linux-gnu/

# ── cuDNN runtime .so files ────────────────────────────────────────────────────
# The base image does NOT include cuDNN; copy all runtime components from
# Stage 1.  Inference-only; no -dev/-headers packages.
COPY --from=trt-libs \
    /usr/lib/x86_64-linux-gnu/libcudnn.so.9.17.1 \
    /usr/lib/x86_64-linux-gnu/libcudnn_cnn.so.9.17.1 \
    /usr/lib/x86_64-linux-gnu/libcudnn_ops.so.9.17.1 \
    /usr/lib/x86_64-linux-gnu/libcudnn_graph.so.9.17.1 \
    /usr/lib/x86_64-linux-gnu/libcudnn_heuristic.so.9.17.1 \
    /usr/lib/x86_64-linux-gnu/libcudnn_engines_runtime_compiled.so.9.17.1 \
    /usr/lib/x86_64-linux-gnu/libcudnn_adv.so.9.17.1 \
    /usr/lib/x86_64-linux-gnu/libcudnn_engines_precompiled.so.9.17.1 \
    /usr/lib/x86_64-linux-gnu/

# ── CUDA math libs ─────────────────────────────────────────────────────────────
# cuBLAS + cuBLASLt: required for all GEMM / matmul operations — used by every
#   OD model family (YOLO conv heads, transformer attention in RT-DETR, etc.).
# libnvrtc + libnvJitLink + libnvrtc-builtins: TRT JIT-compiles certain kernels
#   at engine-load time (fused attention, dynamic-shape tactic selection).
#   Safe to include; without these some transformer-based engines fail to load.
# These live in /usr/local/cuda/lib64/ which is on LD_LIBRARY_PATH in the base.
COPY --from=trt-libs \
    /usr/local/cuda/lib64/libcublas.so.13.2.1.1 \
    /usr/local/cuda/lib64/libcublasLt.so.13.2.1.1 \
    /usr/local/cuda/lib64/libnvrtc.so.13.1.115 \
    /usr/local/cuda/lib64/libnvrtc-builtins.so.13.1.115 \
    /usr/local/cuda/lib64/libnvJitLink.so.13.1.115 \
    /usr/local/cuda/lib64/

# Create versioned symlinks for all copied .so files and rebuild the linker cache.
RUN ln -s libnvinfer.so.10.14.1           /usr/lib/x86_64-linux-gnu/libnvinfer.so.10 \
    && ln -s libnvinfer_plugin.so.10.14.1  /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.10 \
    && ln -s libnvinfer_lean.so.10.14.1    /usr/lib/x86_64-linux-gnu/libnvinfer_lean.so.10 \
    && ln -s libnvinfer_dispatch.so.10.14.1 /usr/lib/x86_64-linux-gnu/libnvinfer_dispatch.so.10 \
    && ln -s libnvinfer_vc_plugin.so.10.14.1 /usr/lib/x86_64-linux-gnu/libnvinfer_vc_plugin.so.10 \
    && ln -s libnvonnxparser.so.10.14.1    /usr/lib/x86_64-linux-gnu/libnvonnxparser.so.10 \
    && ln -s libcudnn.so.9.17.1            /usr/lib/x86_64-linux-gnu/libcudnn.so.9 \
    && ln -s libcudnn_cnn.so.9.17.1        /usr/lib/x86_64-linux-gnu/libcudnn_cnn.so.9 \
    && ln -s libcudnn_ops.so.9.17.1        /usr/lib/x86_64-linux-gnu/libcudnn_ops.so.9 \
    && ln -s libcudnn_graph.so.9.17.1      /usr/lib/x86_64-linux-gnu/libcudnn_graph.so.9 \
    && ln -s libcudnn_heuristic.so.9.17.1  /usr/lib/x86_64-linux-gnu/libcudnn_heuristic.so.9 \
    && ln -s libcudnn_engines_runtime_compiled.so.9.17.1 \
             /usr/lib/x86_64-linux-gnu/libcudnn_engines_runtime_compiled.so.9 \
    && ln -s libcudnn_adv.so.9.17.1        /usr/lib/x86_64-linux-gnu/libcudnn_adv.so.9 \
    && ln -s libcudnn_engines_precompiled.so.9.17.1 \
             /usr/lib/x86_64-linux-gnu/libcudnn_engines_precompiled.so.9 \
    && ln -s libcublas.so.13.2.1.1         /usr/local/cuda/lib64/libcublas.so.13 \
    && ln -s libcublasLt.so.13.2.1.1       /usr/local/cuda/lib64/libcublasLt.so.13 \
    && ln -s libnvrtc.so.13.1.115          /usr/local/cuda/lib64/libnvrtc.so.13 \
    && ln -s libnvrtc-builtins.so.13.1.115 /usr/local/cuda/lib64/libnvrtc-builtins.so.13.1 \
    && ln -s libnvJitLink.so.13.1.115      /usr/local/cuda/lib64/libnvJitLink.so.13 \
    && ldconfig

# ── trtexec — for inference smoke-testing (remove when no longer needed) ──────
COPY --from=trt-libs /opt/tensorrt/bin/trtexec /usr/local/bin/trtexec

# ── System packages ────────────────────────────────────────────────────────────
# python3.12 + venv: matches the NGC source image (compiled against 3.12).
# libglib2.0-0: runtime dep of opencv-python-headless.
# curl: needed to bootstrap uv.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        python3.12 \
        python3.12-venv \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── cuda-python + tensorrt Python packages ────────────────────────────────────
# These are native extension packages compiled for Python 3.12 and linked
# against the system .so files copied above.  The cuda-python package provides
# 'from cuda.bindings import driver'; tensorrt provides 'import tensorrt as trt'.
# The nvidia/ wheel dir (376 MB, bundled CUDA runtime Python packages) is not
# copied — our backend uses only the CUDA driver API injected by the container
# runtime, not the bundled CUDA runtime wheels.
COPY --from=trt-libs \
    /usr/local/lib/python3.12/dist-packages/cuda \
    /usr/local/lib/python3.12/dist-packages/cuda

COPY --from=trt-libs \
    /usr/local/lib/python3.12/dist-packages/tensorrt \
    /usr/local/lib/python3.12/dist-packages/tensorrt

# ── uv + condor venv ──────────────────────────────────────────────────────────
RUN curl -Ls https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# The venv inherits system site-packages so that the tensorrt and cuda-python
# packages copied above (in /usr/local/lib/python3.12/dist-packages/) are
# importable without reinstalling or duplicating them.
RUN uv venv --python python3.12 --system-site-packages /opt/condor-venv
ENV VIRTUAL_ENV=/opt/condor-venv
ENV UV_PROJECT_ENVIRONMENT=/opt/condor-venv
ENV PATH="/opt/condor-venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml ./

# onnxruntime is not used by the TensorRT backend.
# tensorrt and cuda-python are provided via system-site-packages above.
RUN sed -i '/"onnxruntime>=/d' pyproject.toml \
    && uv sync --no-dev --no-install-project --extra tui

COPY condor/ ./condor/
RUN uv sync --no-dev --extra tui

RUN mkdir -p /app/config /app/models
COPY config/config.yaml /app/config/config.yaml

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TRT_VERSION=10.14.1.48+cuda13.0

# NVIDIA_VISIBLE_DEVICES, NVIDIA_DRIVER_CAPABILITIES, NVIDIA_REQUIRE_CUDA,
# LD_LIBRARY_PATH, and CUDA_VERSION are inherited from the cuda:base image.

EXPOSE 5555-5559

ENTRYPOINT ["condor"]
CMD ["--config", "/app/config/config.yaml"]
