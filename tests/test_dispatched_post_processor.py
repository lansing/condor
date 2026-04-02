"""Unit tests for DispatchedPostProcessor.

Run with:
    uv run pytest tests/test_dispatched_post_processor.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from condor.post_process.dispatcher import DispatchedPostProcessor


@pytest.fixture
def real_yolov9_output():
    data = np.load(Path(__file__).parent / "yolov9_raw_output.npz")
    return data["raw_output"]


@pytest.fixture
def real_yolov10_output():
    data = np.load(Path(__file__).parent / "yolov10_raw_output.npz")
    return data["raw_output"]


@pytest.mark.asyncio
async def test_dispatcher_detects_yolov10_from_real_output(real_yolov10_output):
    model_type = DispatchedPostProcessor.detect_output_type([real_yolov10_output])
    assert model_type == "yolov10"


@pytest.mark.asyncio
async def test_dispatcher_detects_yolov9_from_real_output(real_yolov9_output):
    model_type = DispatchedPostProcessor.detect_output_type([real_yolov9_output])
    assert model_type == "yolov9"


@pytest.mark.asyncio
async def test_dispatcher_processes_real_yolov10_output(real_yolov10_output):
    pp = DispatchedPostProcessor(confidence_threshold=0.5, max_detections=20)

    result = await pp.process([real_yolov10_output], (320, 320))

    assert result.shape == (20, 6)
    assert result.dtype == np.float32

    non_zero_rows = np.where(result[:, 1] > 0)[0]
    assert len(non_zero_rows) > 0

    top_detection = result[0]
    class_id = int(top_detection[0])
    confidence = top_detection[1]

    assert class_id == 0
    assert confidence > 0.8


@pytest.mark.asyncio
async def test_dispatcher_processes_real_yolov9_output(real_yolov9_output):
    pp = DispatchedPostProcessor(confidence_threshold=0.3, max_detections=20)

    result = await pp.process([real_yolov9_output], (320, 320))

    assert result.shape == (20, 6)
    assert result.dtype == np.float32

    non_zero_rows = np.where(result[:, 1] > 0)[0]
    assert len(non_zero_rows) > 0

    top_detection = result[0]
    class_id = int(top_detection[0])
    confidence = top_detection[1]

    assert class_id == 0
    assert confidence > 0.9


@pytest.mark.asyncio
async def test_dispatcher_handles_yolov10_3d_shape():
    pp = DispatchedPostProcessor(confidence_threshold=0.5, max_detections=20)
    raw = np.zeros((1, 300, 6), dtype=np.float32)
    raw[0, 0] = [10, 10, 50, 50, 0.9, 0]

    result = await pp.process([raw], (320, 320))
    assert result.shape == (20, 6)
    assert result[0, 1] == pytest.approx(0.9, abs=1e-5)


@pytest.mark.asyncio
async def test_dispatcher_handles_yolov10_2d_shape():
    pp = DispatchedPostProcessor(confidence_threshold=0.5, max_detections=20)
    raw = np.zeros((300, 6), dtype=np.float32)
    raw[0] = [10, 10, 50, 50, 0.9, 0]

    result = await pp.process([raw], (320, 320))
    assert result.shape == (20, 6)
    assert result[0, 1] == pytest.approx(0.9, abs=1e-5)


@pytest.mark.asyncio
async def test_dispatcher_handles_yolov9_3d_shape():
    pp = DispatchedPostProcessor(confidence_threshold=0.5, max_detections=20)
    raw = np.zeros((1, 7, 100), dtype=np.float32)
    raw[0, 0, 0] = 160
    raw[0, 1, 0] = 160
    raw[0, 2, 0] = 80
    raw[0, 3, 0] = 80
    raw[0, 4, 0] = 0.9

    result = await pp.process([raw], (320, 320))
    assert result.shape == (20, 6)


@pytest.mark.asyncio
async def test_dispatcher_handles_yolov9_80_class_format():
    pp = DispatchedPostProcessor(confidence_threshold=0.5, max_detections=20)
    raw = np.zeros((1, 84, 10), dtype=np.float32)
    raw[0, 0, 0] = 160
    raw[0, 1, 0] = 160
    raw[0, 2, 0] = 80
    raw[0, 3, 0] = 80
    raw[0, 4, 0] = 0.9

    result = await pp.process([raw], (320, 320))
    assert result.shape == (20, 6)
    assert result[0, 1] == pytest.approx(0.9, abs=1e-5)


@pytest.mark.asyncio
async def test_dispatcher_returns_zeros_for_unknown_output():
    pp = DispatchedPostProcessor(confidence_threshold=0.5, max_detections=20)
    raw = np.zeros((1, 3, 100), dtype=np.float32)

    result = await pp.process([raw], (320, 320))
    assert result.shape == (20, 6)
    assert result.sum() == 0.0


@pytest.mark.asyncio
async def test_dispatcher_returns_zeros_for_empty_input():
    pp = DispatchedPostProcessor(confidence_threshold=0.5, max_detections=20)

    result = await pp.process([], (320, 320))
    assert result.shape == (20, 6)
    assert result.sum() == 0.0


@pytest.mark.asyncio
async def test_dispatcher_respects_max_detections(
    real_yolov9_output, real_yolov10_output
):
    pp = DispatchedPostProcessor(confidence_threshold=0.1, max_detections=5)

    result_yolov9 = await pp.process([real_yolov9_output], (320, 320))
    assert result_yolov9.shape == (5, 6)

    result_yolov10 = await pp.process([real_yolov10_output], (320, 320))
    assert result_yolov10.shape == (5, 6)


@pytest.mark.asyncio
async def test_dispatcher_has_same_init_args_as_yolov10():
    pp = DispatchedPostProcessor(confidence_threshold=0.5, max_detections=10)
    assert pp.confidence_threshold == 0.5
    assert pp.max_detections == 10


@pytest.mark.asyncio
async def test_dispatcher_filters_by_confidence_yolov10(real_yolov10_output):
    pp = DispatchedPostProcessor(confidence_threshold=0.5, max_detections=20)

    result = await pp.process([real_yolov10_output], (320, 320))

    for row in result:
        if row[1] > 0:
            assert row[1] >= 0.5


@pytest.mark.asyncio
async def test_dispatcher_filters_by_confidence_yolov9(real_yolov9_output):
    pp = DispatchedPostProcessor(confidence_threshold=0.5, max_detections=20)

    result = await pp.process([real_yolov9_output], (320, 320))

    for row in result:
        if row[1] > 0:
            assert row[1] >= 0.5


@pytest.mark.asyncio
async def test_dispatcher_normalises_coordinates_yolov10(real_yolov10_output):
    pp = DispatchedPostProcessor(confidence_threshold=0.5, max_detections=20)

    result = await pp.process([real_yolov10_output], (320, 320))

    for row in result:
        if row[1] > 0:
            ymin, xmin, ymax, xmax = row[2], row[3], row[4], row[5]
            assert 0 <= ymin < ymax <= 1.0
            assert 0 <= xmin < xmax <= 1.0


@pytest.mark.asyncio
async def test_dispatcher_normalises_coordinates_yolov9(real_yolov9_output):
    pp = DispatchedPostProcessor(confidence_threshold=0.3, max_detections=20)

    result = await pp.process([real_yolov9_output], (320, 320))

    for row in result:
        if row[1] > 0:
            ymin, xmin, ymax, xmax = row[2], row[3], row[4], row[5]
            assert 0 <= ymin < ymax <= 1.0
            assert 0 <= xmin < xmax <= 1.0
