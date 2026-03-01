#!/usr/bin/env python3
"""Frigate ZMQ Remote Detector — test / benchmark client.

Simulates the Frigate NVR client using the full protocol:
  1. Model availability check  (model_request)
  2. Model data transfer        (model_data)   — only if step 1 says unavailable
  3. Inference request          (tensor + header)

Uses ``sample_image.jpg`` as the test image and the YOLOv10 wildlife model.
Class labels are read from ``models/md.classes.txt``.

Usage
-----
  # Start the server first:
  make run

  # Then in another terminal:
  uv run python scripts/test_client.py [--endpoint tcp://localhost:5555]
                                       [--model MDV6-yolov10-c_float16_320.onnx]
                                       [--image sample_image.jpg]
                                       [--runs N]
                                       [--verbose]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import zmq


# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------

def letterbox(
    image: np.ndarray,
    new_shape: tuple[int, int] = (320, 320),
    color: tuple[int, int, int] = (114, 114, 114),
) -> np.ndarray:
    """Resize *image* to *new_shape* with letterbox padding."""
    h, w = image.shape[:2]
    nh, nw = new_shape

    scale = min(nw / w, nh / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (nw - new_w) / 2
    pad_h = (nh - new_h) / 2
    top    = int(round(pad_h - 0.1))
    bottom = int(round(pad_h + 0.1))
    left   = int(round(pad_w - 0.1))
    right  = int(round(pad_w + 0.1))

    return cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )


def preprocess(image_path: str | Path, input_size: tuple[int, int] = (320, 320)) -> np.ndarray:
    """Load and preprocess image → float32 NCHW tensor in [0, 1]."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = letterbox(img, new_shape=input_size)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)       # HWC → CHW
    img = np.expand_dims(img, 0)       # CHW → NCHW (1, C, H, W)
    return img


# ---------------------------------------------------------------------------
# Protocol helpers
# ---------------------------------------------------------------------------

def send_model_request(sock: zmq.Socket, model_name: str) -> dict:
    req = {"model_request": True, "model_name": model_name}
    sock.send_multipart([json.dumps(req).encode()])
    frames = sock.recv_multipart()
    return json.loads(frames[0])


def send_model_data(sock: zmq.Socket, model_name: str, model_path: Path) -> dict:
    data = model_path.read_bytes()
    req = {"model_data": True, "model_name": model_name}
    sock.send_multipart([json.dumps(req).encode(), data])
    frames = sock.recv_multipart()
    return json.loads(frames[0])


def send_inference(
    sock: zmq.Socket, tensor: np.ndarray, model_type: str = "yolo-generic"
) -> tuple[np.ndarray, float]:
    """Send an inference request; returns ``(result_array, round_trip_ms)``."""
    header = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "model_type": model_type,
    }
    t0 = time.perf_counter()
    sock.send_multipart([json.dumps(header).encode(), tensor.tobytes(order="C")])
    frames = sock.recv_multipart()
    rtt = (time.perf_counter() - t0) * 1000.0

    resp_header = json.loads(frames[0])
    result = np.frombuffer(frames[1], dtype=np.float32).reshape(resp_header["shape"])
    return result, rtt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Frigate ZMQ detector test client")
    p.add_argument("--endpoint", default="tcp://localhost:5555")
    p.add_argument("--model",    default="MDV6-yolov10-c_float16_320.onnx")
    p.add_argument("--image",    default="sample_image.jpg")
    p.add_argument("--input-size", type=int, default=320,
                   help="Square input resolution to resize the image to (default: 320).")
    p.add_argument("--runs",     type=int, default=1,
                   help="Number of inference runs for benchmarking (default: 1).")
    p.add_argument("--verbose",  action="store_true")
    return p.parse_args()


def load_class_names(models_dir: Path = Path("models")) -> list[str]:
    classes_file = models_dir / "md.classes.txt"
    if classes_file.exists():
        return [line.strip() for line in classes_file.read_text().splitlines() if line.strip()]
    return []


def print_detections(result: np.ndarray, class_names: list[str]) -> None:
    found = False
    for i, row in enumerate(result):
        class_id, score, ymin, xmin, ymax, xmax = row
        if score <= 0.0:
            continue
        found = True
        cname = (
            class_names[int(class_id)]
            if int(class_id) < len(class_names)
            else f"class_{int(class_id)}"
        )
        print(
            f"  [{i:2d}] {cname:<12s}  score={score:.4f}  "
            f"box=[{ymin:.3f}, {xmin:.3f}, {ymax:.3f}, {xmax:.3f}]"
        )
    if not found:
        print("  (no detections above threshold)")


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "models" / args.model
    image_path = project_root / args.image
    class_names = load_class_names(project_root / "models")

    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    # ----------------------------------------------------------------
    # ZMQ connection
    # ----------------------------------------------------------------
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 30_000)   # 30 s — model loading can be slow
    sock.setsockopt(zmq.SNDTIMEO, 30_000)
    sock.connect(args.endpoint)
    print(f"Connected to {args.endpoint}")

    try:
        # ----------------------------------------------------------------
        # Stage 1 — model negotiation
        # ----------------------------------------------------------------
        print(f"\n{'─'*60}")
        print("Stage 1A — Model availability check")
        resp = send_model_request(sock, args.model)
        print(f"  Response: {resp}")

        if not resp.get("model_available") or not resp.get("model_loaded"):
            if not model_path.exists():
                print(
                    f"ERROR: Model not found locally either: {model_path}",
                    file=sys.stderr,
                )
                sys.exit(1)

            print("\nStage 1B — Model data transfer")
            resp = send_model_data(sock, args.model, model_path)
            print(f"  Response: {resp}")

            if not resp.get("model_loaded"):
                print("ERROR: Server failed to load model.", file=sys.stderr)
                sys.exit(1)

        # ----------------------------------------------------------------
        # Pre-process image once
        # ----------------------------------------------------------------
        tensor = preprocess(image_path, input_size=(args.input_size, args.input_size))
        print(f"\nInput tensor: shape={list(tensor.shape)}  dtype={tensor.dtype}")

        # ----------------------------------------------------------------
        # Stage 2 — inference
        # ----------------------------------------------------------------
        print(f"\n{'─'*60}")
        print("Stage 2 — Inference")

        rtts: list[float] = []
        for run in range(args.runs):
            result, rtt = send_inference(sock, tensor)
            rtts.append(rtt)
            if args.verbose or run == 0:
                print(f"\n  Run {run + 1}/{args.runs}  RTT={rtt:.1f}ms")
                print_detections(result, class_names)

        if args.runs > 1:
            print(
                f"\nBenchmark ({args.runs} runs): "
                f"min={min(rtts):.1f}ms  "
                f"avg={sum(rtts)/len(rtts):.1f}ms  "
                f"max={max(rtts):.1f}ms"
            )

    finally:
        sock.close()
        ctx.term()


if __name__ == "__main__":
    main()
