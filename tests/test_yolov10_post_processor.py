"""Unit tests for YoloV10PostProcessor using real model output.

Run with:
    uv run pytest tests/test_yolov10_post_processor.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from condor.post_process.yolov10 import YoloV10PostProcessor


@pytest.fixture
def real_yolov10_output():
    """Load real model output from MDV6-yolov10-c_float16_320.onnx on sample_image.jpg."""
    data = np.load("/home/max/Code/condor/tests/yolov10_raw_output.npz")
    return data["raw_output"]


@pytest.mark.asyncio
async def test_post_processor_with_real_yolov10_model_output(real_yolov10_output):
    """Test with actual output from MDV6-yolov10-c_float16_320.onnx model.

    The model was run on sample_image.jpg and produces a detection
    of an animal (class 0) with confidence ~0.85.
    """
    pp = YoloV10PostProcessor(confidence_threshold=0.5, max_detections=20)

    result = await pp.process([real_yolov10_output], (320, 320))

    assert result.shape == (20, 6)
    assert result.dtype == np.float32

    non_zero_rows = np.where(result[:, 1] > 0)[0]
    assert len(non_zero_rows) > 0, "Expected at least one detection"

    top_detection = result[0]
    class_id = int(top_detection[0])
    confidence = top_detection[1]

    assert class_id == 0, f"Expected class 0 (animal), got {class_id}"
    assert confidence > 0.8, f"Expected confidence > 0.8, got {confidence}"

    ymin, xmin, ymax, xmax = (
        top_detection[2],
        top_detection[3],
        top_detection[4],
        top_detection[5],
    )
    assert 0 <= xmin < xmax <= 1.0
    assert 0 <= ymin < ymax <= 1.0


@pytest.mark.asyncio
async def test_post_processor_returns_zeros_when_very_high_threshold(
    real_yolov10_output,
):
    """Test that with a very high threshold, no detections are returned."""
    pp = YoloV10PostProcessor(confidence_threshold=0.99)
    result = await pp.process([real_yolov10_output], (320, 320))
    assert result.shape == (20, 6)
    assert result.sum() == 0.0


@pytest.mark.asyncio
async def test_post_processor_filters_by_confidence(real_yolov10_output):
    """Test that confidence filtering works correctly."""
    pp = YoloV10PostProcessor(confidence_threshold=0.5, max_detections=20)

    result = await pp.process([real_yolov10_output], (320, 320))

    for row in result:
        if row[1] > 0:
            assert row[1] >= 0.5, f"Confidence {row[1]} should be >= 0.5"


@pytest.mark.asyncio
async def test_post_processor_normalises_coordinates(real_yolov10_output):
    """Test that coordinates are normalized to [0, 1] range."""
    pp = YoloV10PostProcessor(confidence_threshold=0.5, max_detections=20)

    result = await pp.process([real_yolov10_output], (320, 320))

    for row in result:
        if row[1] > 0:
            ymin, xmin, ymax, xmax = row[2], row[3], row[4], row[5]
            assert 0 <= ymin < ymax <= 1.0, (
                f"y coordinates not normalized: {ymin}, {ymax}"
            )
            assert 0 <= xmin < xmax <= 1.0, (
                f"x coordinates not normalized: {xmin}, {xmax}"
            )


@pytest.mark.asyncio
async def test_post_processor_respects_max_detections(real_yolov10_output):
    """Test that max_detections limit is respected."""
    pp = YoloV10PostProcessor(confidence_threshold=0.1, max_detections=5)

    result = await pp.process([real_yolov10_output], (320, 320))

    assert result.shape == (5, 6)


@pytest.mark.asyncio
async def test_post_processor_handles_float16_input(real_yolov10_output):
    """Test that float16 input is handled correctly."""
    pp = YoloV10PostProcessor(confidence_threshold=0.5, max_detections=20)

    raw_float16 = real_yolov10_output.astype(np.float16)

    result = await pp.process([raw_float16], (320, 320))

    assert result.dtype == np.float32
    non_zero_rows = np.where(result[:, 1] > 0)[0]
    assert len(non_zero_rows) > 0


@pytest.mark.asyncio
async def test_post_processor_clips_coordinates(real_yolov10_output):
    """Test that coordinates are clipped to [0, 1] range."""
    pp = YoloV10PostProcessor(confidence_threshold=0.5, max_detections=20)

    result = await pp.process([real_yolov10_output], (320, 320))

    for row in result:
        if row[1] > 0:
            ymin, xmin, ymax, xmax = row[2], row[3], row[4], row[5]
            assert xmin >= 0.0 and xmax <= 1.0
            assert ymin >= 0.0 and ymax <= 1.0
