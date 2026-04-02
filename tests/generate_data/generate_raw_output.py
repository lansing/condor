#!/usr/bin/env python3
"""Generate raw model output for post-processor testing.

Model is expected to have:
    - Input tensor format: float32 dtype, NCHW (batch, channels, height, width)
    - Typical input size: 320x320 or 640x640

Usage:
    python generate_raw_output.py --model /path/to/model.onnx --image /path/to/image.jpg [--output /path/to/output.npz]

Example:
    python generate_raw_output.py --model yolov9.onnx --image sample_image.jpg --output tests/yolov9_raw_output.npz
"""

import argparse
import numpy as np
import onnxruntime as ort
import cv2


def preprocess_image(image_path, target_size=320):
    """Preprocess image using letterbox resize."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    orig_h, orig_w = img.shape[:2]

    gain = target_size / max(orig_h, orig_w)
    new_w = int(orig_w * gain)
    new_h = int(orig_h * gain)

    resized = cv2.resize(img, (new_w, new_h))

    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2

    padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

    blob = padded.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)
    blob = np.expand_dims(blob, axis=0)

    return blob, (orig_h, orig_w)


def main():
    parser = argparse.ArgumentParser(
        description="Generate raw model output for testing"
    )
    parser.add_argument("--model", required=True, help="Path to ONNX model file")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--output",
        default=None,
        help="Output npz file path (default: <model_name>_raw_output.npz)",
    )
    parser.add_argument(
        "--input-size", type=int, default=320, help="Model input size (default: 320)"
    )
    args = parser.parse_args()

    print(f"Loading model from: {args.model}")

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    print(f"Input name: {input_name}")
    input_shape = session.get_inputs()[0].shape
    print(f"Input shape: {input_shape}")

    outputs = session.get_outputs()
    for out in outputs:
        print(f"Output: {out.name}, shape: {out.shape}")

    print(f"\nPreprocessing image: {args.image}")
    blob, orig_dims = preprocess_image(args.image, args.input_size)
    print(f"Input blob shape: {blob.shape}")
    print(f"Original image dims: {orig_dims}")

    print("\nRunning inference...")
    raw_output = session.run(None, {input_name: blob})

    print(f"\nNumber of outputs: {len(raw_output)}")
    for i, out in enumerate(raw_output):
        print(f"Output {i} shape: {out.shape}, dtype: {out.dtype}")
        print(f"Output {i} min: {out.min():.4f}, max: {out.max():.4f}")

    if args.output:
        output_path = args.output
    else:
        model_name = args.model.split("/")[-1].replace(".onnx", "")
        output_path = f"tests/{model_name}_raw_output.npz"

    np.savez_compressed(output_path, raw_output=raw_output[0])
    print(f"\nSaved raw output to: {output_path}")

    print(f"\nRaw output shape: {raw_output[0].shape}")
    print("Done.")


if __name__ == "__main__":
    main()
