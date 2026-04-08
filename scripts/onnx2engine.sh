#!/usr/bin/env bash
# onnx2engine — Convert an ONNX model to a TensorRT engine file.
#
# Uses the same TensorRT version as your local condor image so the resulting
# engine is guaranteed to be compatible with condor at runtime.
#
# Usage (recommended — works with interactive prompts):
#   curl -fsSL https://raw.githubusercontent.com/lansing/condor/master/scripts/onnx2engine.sh \
#       | bash -s -- model.onnx
#
# Or download and run:
#   curl -fsSL https://raw.githubusercontent.com/lansing/condor/master/scripts/onnx2engine.sh \
#       -o /tmp/onnx2engine.sh && bash /tmp/onnx2engine.sh model.onnx
set -euo pipefail

# ── Constants ──────────────────────────────────────────────────────────────────

CONDOR_IMAGE="${CONDOR_IMAGE:-ghcr.io/lansing/condor:latest}"
BUILDER_LABEL="condor.builder.image"

_BOLD="\033[1m"
_DIM="\033[2m"
_CYAN="\033[36m"
_GREEN="\033[32m"
_YELLOW="\033[33m"
_RED="\033[31m"
_RESET="\033[0m"

_c() { [ -t 1 ] && printf "%b%s%b" "$1" "$2" "$_RESET" || printf "%s" "$2"; }

# ── Usage ──────────────────────────────────────────────────────────────────────

usage() {
    cat <<EOF

$(_c "$_BOLD" "onnx2engine") — Convert ONNX → TensorRT engine

$(_c "$_BOLD" "Usage:")
  $(basename "$0") [OPTIONS] model.onnx [output.engine]

$(_c "$_BOLD" "Arguments:")
  model.onnx       Input ONNX model file
  output.engine    Output engine file  (default: same path, .engine extension)

$(_c "$_BOLD" "Options:")
  --no-fp16        Disable FP16 precision (build FP32 engine instead)
  --               Pass remaining arguments directly to trtexec
  -h, --help       Show this help

$(_c "$_BOLD" "Environment:")
  CONDOR_IMAGE     Override the condor image to inspect for TensorRT version
                   (default: $CONDOR_IMAGE)

$(_c "$_BOLD" "Examples:")
  $(basename "$0") models/model.onnx
  $(basename "$0") models/model.onnx models/model.engine
  $(basename "$0") models/model.onnx -- --minShapes=images:1x3x320x320 \\
                                        --maxShapes=images:1x3x640x640

EOF
}

# ── Argument parsing ───────────────────────────────────────────────────────────

FP16=true
ONNX_FILE=""
ENGINE_FILE=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)  usage; exit 0 ;;
        --no-fp16)  FP16=false; shift ;;
        --)         shift; EXTRA_ARGS=("$@"); break ;;
        -*)         echo "Unknown option: $1"; usage; exit 1 ;;
        *)
            if   [[ -z "$ONNX_FILE"   ]]; then ONNX_FILE="$1"
            elif [[ -z "$ENGINE_FILE" ]]; then ENGINE_FILE="$1"
            else echo "Unexpected argument: $1"; usage; exit 1
            fi
            shift ;;
    esac
done

if [[ -z "$ONNX_FILE" ]]; then
    echo "Error: ONNX input file is required." >&2
    usage; exit 1
fi

# ── Resolve paths ──────────────────────────────────────────────────────────────

ONNX_ABS=$(realpath "$ONNX_FILE" 2>/dev/null || true)
if [[ -z "$ONNX_ABS" || ! -f "$ONNX_ABS" ]]; then
    echo "Error: ONNX file not found: $ONNX_FILE" >&2
    exit 1
fi

if [[ -z "$ENGINE_FILE" ]]; then
    ENGINE_ABS="${ONNX_ABS%.onnx}.engine"
else
    # realpath -m accepts non-existent paths
    ENGINE_ABS=$(realpath -m "$ENGINE_FILE")
fi

ONNX_DIR=$(dirname "$ONNX_ABS")
ENGINE_DIR=$(dirname "$ENGINE_ABS")
mkdir -p "$ENGINE_DIR"

# ── Check condor image ─────────────────────────────────────────────────────────

if ! docker image inspect "$CONDOR_IMAGE" >/dev/null 2>&1; then
    cat >&2 <<EOF

$(_c "$_RED" "Error:") condor image not found locally: $(_c "$_DIM" "$CONDOR_IMAGE")

The converter reads the condor image's metadata to determine which TensorRT
version to use for building the engine.  An engine built with a different
TensorRT version than condor uses will be rejected at runtime.

Pull the condor image first:
  $(_c "$_DIM" "docker pull $CONDOR_IMAGE")

Or install condor first — see the README for instructions.
If you are using a locally built image, set:
  $(_c "$_DIM" "CONDOR_IMAGE=condor:latest $(basename "$0") model.onnx")

EOF
    exit 1
fi

# ── Read builder image from condor label ───────────────────────────────────────

BUILDER_IMAGE=$(docker inspect "$CONDOR_IMAGE" \
    --format "{{ index .Config.Labels \"$BUILDER_LABEL\" }}" 2>/dev/null)

if [[ -z "$BUILDER_IMAGE" ]]; then
    cat >&2 <<EOF

$(_c "$_RED" "Error:") Could not read label '$BUILDER_LABEL' from $(_c "$_DIM" "$CONDOR_IMAGE").

The condor image appears to be missing this label — it may be outdated.
Update to a newer condor image:
  $(_c "$_DIM" "docker pull $CONDOR_IMAGE")

EOF
    exit 1
fi

# ── Summary ────────────────────────────────────────────────────────────────────

echo ""
echo "  $(_c "$_BOLD" "condor image:")   $(_c "$_DIM" "$CONDOR_IMAGE")"
echo "  $(_c "$_BOLD" "TRT builder:")    $(_c "$_DIM" "$BUILDER_IMAGE")"
echo "  $(_c "$_BOLD" "ONNX input:")     $(_c "$_DIM" "$ONNX_ABS")"
echo "  $(_c "$_BOLD" "Engine output:")  $(_c "$_DIM" "$ENGINE_ABS")"
[[ "$FP16" == "true" ]] \
    && echo "  $(_c "$_BOLD" "Precision:")     FP16 (--fp16)" \
    || echo "  $(_c "$_BOLD" "Precision:")     FP32"
echo ""

# ── Mount setup ────────────────────────────────────────────────────────────────
# Use one /workspace mount when ONNX and engine are in the same directory,
# otherwise mount input read-only at /input and output at /output.

if [[ "$ONNX_DIR" == "$ENGINE_DIR" ]]; then
    MOUNTS=(-v "$ONNX_DIR:/workspace")
    ONNX_CONTAINER="/workspace/$(basename "$ONNX_ABS")"
    ENGINE_CONTAINER="/workspace/$(basename "$ENGINE_ABS")"
else
    MOUNTS=(-v "$ONNX_DIR:/input:ro" -v "$ENGINE_DIR:/output")
    ONNX_CONTAINER="/input/$(basename "$ONNX_ABS")"
    ENGINE_CONTAINER="/output/$(basename "$ENGINE_ABS")"
fi

# ── Build trtexec command ──────────────────────────────────────────────────────

TRTEXEC_ARGS=(
    "--onnx=$ONNX_CONTAINER"
    "--saveEngine=$ENGINE_CONTAINER"
)
[[ "$FP16" == "true" ]] && TRTEXEC_ARGS+=("--fp16")
TRTEXEC_ARGS+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")

# ── Run conversion ─────────────────────────────────────────────────────────────

docker run --rm --gpus all \
    "${MOUNTS[@]}" \
    "$BUILDER_IMAGE" \
    trtexec "${TRTEXEC_ARGS[@]}"

echo ""
echo "  $(_c "$_GREEN" "✓") Engine saved to: $(_c "$_DIM" "$ENGINE_ABS")"
echo ""
