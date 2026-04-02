# Generating Test Data for Post-Processor Tests

This directory contains scripts and data used to generate real model outputs for testing post-processors.

## Why Generate Real Model Output?

Post-processor tests should use real model output rather than synthetic data because:

1. **Correct format verification**: Ensures the post-processor handles the actual output format from real ONNX models
2. **Edge case coverage**: Real models have quirks in their output that synthetic data may not capture
3. **Confidence in correctness**: Testing with real inference output provides higher confidence that the post-processor works correctly with actual models

## Model Requirements

The model is expected to have:
- **Input tensor format**: float32 dtype, NCHW (batch, channels, height, width)
- **Typical input size**: 320x320 or 640x640

## Setup

To run the inference script, you need a Python environment with ONNXRuntime:

```bash
# Create a new virtual environment
python -m venv onnx_test

# Activate it
source onnx_test/bin/activate

# Install dependencies
pip install onnxruntime numpy opencv-python-headless pillow
```

Alternatively, using uv:
```bash
uv venv onnx_test
source onnx_test/bin/activate
uv pip install onnxruntime numpy opencv-python-headless pillow
```

## Files

- `generate_raw_output.py` - Generic script to run inference with any ONNX model
- `sample_image.jpg` - Test image containing an animal

## Usage

```bash
source onnx_test/bin/activate
python generate_raw_output.py --model /path/to/model.onnx --image sample_image.jpg
```

This will generate `tests/<model_name>_raw_output.npz`.

### Options

- `--model`: Path to ONNX model file (required)
- `--image`: Path to input image (required)
- `--output`: Output npz file path (optional, defaults to `tests/<model_name>_raw_output.npz`)
- `--input-size`: Model input size (default: 320)

### Examples

Generate raw output for a YoloV9 model:
```bash
python generate_raw_output.py \
    --model yolov9.onnx \
    --image sample_image.jpg \
    --output tests/yolov9_raw_output.npz
```

Generate raw output for a YoloV10 model:
```bash
python generate_raw_output.py \
    --model yolov10.onnx \
    --image sample_image.jpg \
    --output tests/yolov10_raw_output.npz
```

## Adding Tests for New Model Types

1. Run the generation script to produce `tests/<model_name>_raw_output.npz`
2. Create test file `tests/test_<model_name>_post_processor.py` that loads and uses this data

## Notes

- ONNXRuntime is only needed for generating test data, not for running the post-processor tests themselves
- The post-processor tests only need numpy and the post-processor code
- Real inference should ideally be run in a GPU-enabled environment, but CPU works for generating test data
