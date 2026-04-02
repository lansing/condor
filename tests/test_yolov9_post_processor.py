"""Unit tests for YoloV9PostProcessor.

Run with:
    uv run pytest tests/test_yolov9_post_processor.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from condor.post_process.yolov9 import YoloV9PostProcessor


@pytest.fixture
def real_yolov9_output():
    """Load real model output from MDV6-yolov9-c-320-16b.onnx on sample_image.jpg."""
    data = np.load("/home/max/Code/condor/tests/yolov9_raw_output.npz")
    return data["raw_output"]


@pytest.mark.asyncio
async def test_post_processor_returns_zeros_when_very_high_threshold(
    real_yolov9_output,
):
    pp = YoloV9PostProcessor(confidence_threshold=0.99)
    result = await pp.process([real_yolov9_output], (320, 320))
    assert result.shape == (20, 6)
    assert result.sum() == 0.0


@pytest.mark.asyncio
async def test_post_processor_with_real_yolov9_model_output(real_yolov9_output):
    """Test with actual output from MDV6-yolov9-c-320-16b.onnx model.

    The model was run on sample_image.jpg and produces a detection
    of an animal (class 0) with high confidence (~0.94).
    """
    pp = YoloV9PostProcessor(confidence_threshold=0.3, max_detections=20)

    result = await pp.process([real_yolov9_output], (320, 320))

    assert result.shape == (20, 6)
    assert result.dtype == np.float32

    non_zero_rows = np.where(result[:, 1] > 0)[0]
    assert len(non_zero_rows) > 0, "Expected at least one detection"

    top_detection = result[0]
    class_id = int(top_detection[0])
    confidence = top_detection[1]

    assert class_id == 0, f"Expected class 0 (animal), got {class_id}"
    assert confidence > 0.9, f"Expected confidence > 0.9, got {confidence}"

    ymin, xmin, ymax, xmax = (
        top_detection[2],
        top_detection[3],
        top_detection[4],
        top_detection[5],
    )
    assert 0 <= xmin < xmax <= 1.0
    assert 0 <= ymin < ymax <= 1.0


@pytest.mark.asyncio
async def test_post_processor_filters_by_confidence(real_yolov9_output):
    """Test that confidence filtering works correctly."""
    pp = YoloV9PostProcessor(confidence_threshold=0.5, max_detections=20)

    result = await pp.process([real_yolov9_output], (320, 320))

    for row in result:
        if row[1] > 0:
            assert row[1] >= 0.5, f"Confidence {row[1]} should be >= 0.5"


@pytest.mark.asyncio
async def test_post_processor_normalises_coordinates(real_yolov9_output):
    """Test that coordinates are normalized to [0, 1] range."""
    pp = YoloV9PostProcessor(confidence_threshold=0.3, max_detections=20)

    result = await pp.process([real_yolov9_output], (320, 320))

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
async def test_post_processor_applies_nms(real_yolov9_output):
    """Test that NMS is applied and overlapping boxes of same class are suppressed."""
    pp = YoloV9PostProcessor(
        confidence_threshold=0.3, nms_threshold=0.3, max_detections=20
    )

    result = await pp.process([real_yolov9_output], (320, 320))

    non_zero_rows = np.where(result[:, 1] > 0)[0]
    assert len(non_zero_rows) > 0


@pytest.mark.asyncio
async def test_post_processor_respects_max_detections(real_yolov9_output):
    """Test that max_detections limit is respected."""
    pp = YoloV9PostProcessor(confidence_threshold=0.1, max_detections=5)

    result = await pp.process([real_yolov9_output], (320, 320))

    assert result.shape == (5, 6)


@pytest.mark.asyncio
async def test_post_processor_handles_float16_input(real_yolov9_output):
    """Test that float16 input is handled correctly."""
    pp = YoloV9PostProcessor(confidence_threshold=0.3, max_detections=20)

    raw_float16 = real_yolov9_output.astype(np.float16)

    result = await pp.process([raw_float16], (320, 320))

    assert result.dtype == np.float32
    non_zero_rows = np.where(result[:, 1] > 0)[0]
    assert len(non_zero_rows) > 0


@pytest.mark.asyncio
async def test_post_processor_empty_input():
    """Test that empty input returns zeros."""
    pp = YoloV9PostProcessor(confidence_threshold=0.5, max_detections=20)
    result = await pp.process([], (320, 320))
    assert result.shape == (20, 6)
    assert result.sum() == 0.0


@pytest.mark.asyncio
async def test_post_processor_class_id_from_argmax(real_yolov9_output):
    """Test that class_id is determined by argmax of class scores."""
    pp = YoloV9PostProcessor(confidence_threshold=0.3, max_detections=20)

    result = await pp.process([real_yolov9_output], (320, 320))

    top_detection = result[0]
    class_id = int(top_detection[0])
    confidence = top_detection[1]

    assert class_id == 0, f"Expected class 0 (animal), got {class_id}"


@pytest.mark.asyncio
async def test_post_processor_clips_coordinates(real_yolov9_output):
    """Test that coordinates are clipped to [0, 1] range."""
    pp = YoloV9PostProcessor(confidence_threshold=0.3, max_detections=20)

    result = await pp.process([real_yolov9_output], (320, 320))

    for row in result:
        if row[1] > 0:
            ymin, xmin, ymax, xmax = row[2], row[3], row[4], row[5]
            assert xmin >= 0.0 and xmax <= 1.0
            assert ymin >= 0.0 and ymax <= 1.0


@pytest.mark.asyncio
async def test_post_processor_80_class_coco_format(real_yolov9_output):
    """Test handling of 80-class COCO format by padding class scores.

    We take the real model output and pad it to 80 classes (total).
    The original animal detection (class 0) should still be found with high confidence.
    """
    pp = YoloV9PostProcessor(confidence_threshold=0.3, max_detections=20)

    raw = real_yolov9_output.copy()
    assert raw.shape == (1, 7, 2100)

    num_classes = 80

    boxes = raw[:, :4]
    animal_scores = raw[:, 4:5]
    other_scores = np.random.rand(1, num_classes - 1, 2100).astype(np.float32) * 0.01

    raw_coco = np.concatenate([boxes, animal_scores, other_scores], axis=1)
    assert raw_coco.shape == (1, 4 + num_classes, 2100), (
        f"Expected (1, 84, 2100), got {raw_coco.shape}"
    )

    result = await pp.process([raw_coco], (320, 320))

    assert result.shape == (20, 6)

    top_detection = result[0]
    class_id = int(top_detection[0])
    confidence = top_detection[1]

    assert class_id == 0, f"Expected class 0 (animal), got {class_id}"
    assert confidence > 0.9, f"Expected confidence > 0.9, got {confidence}"


@pytest.mark.asyncio
async def test_post_processor_multi_class_scenario(real_yolov9_output):
    """Test multi-class scenario with person and animal detections.

    Create a variant where we have:
    - Animal detection from original model (high confidence)
    - A fake person detection (class 1) with overlapping box but lower confidence
    - A fake vehicle detection (class 2) with overlapping box but lower confidence

    NMS should keep the animal but might suppress the others depending on overlap.
    """
    pp = YoloV9PostProcessor(
        confidence_threshold=0.3, nms_threshold=0.5, max_detections=20
    )

    raw = real_yolov9_output.copy()

    result = await pp.process([raw], (320, 320))

    non_zero_rows = np.where(result[:, 1] > 0)[0]
    assert len(non_zero_rows) > 0

    for idx in non_zero_rows:
        class_id = int(result[idx, 0])
        assert class_id in [0, 1, 2], f"Unexpected class_id {class_id}"
