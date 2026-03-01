"""Async protocol and unit tests.

Run with:
    uv run pytest tests/ -v
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from condor.backends.base import ModelInfo
from condor.config.settings import AppConfig, InferenceConfig, PostProcessConfig, ServerConfig
from condor.post_process.yolov10 import YoloV10PostProcessor
from condor.server.zmq_handler import AsyncZMQHandler, _zeros_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(**overrides) -> AppConfig:
    return AppConfig(
        server=ServerConfig(endpoint="tcp://*:15555", models_dir="/tmp/test_models"),
        inference=InferenceConfig(provider="cpu"),
        post_process=PostProcessConfig(confidence_threshold=0.4, max_detections=20),
        **overrides,
    )


def make_model_info(input_dtype: str = "float32") -> ModelInfo:
    return ModelInfo(
        input_name="images",
        input_shape=[1, 3, 320, 320],
        input_dtype=input_dtype,
        output_names=["output0"],
        output_shapes=[[1, 300, 6]],
        output_dtypes=["float32"],
    )


def encode_header(data: dict) -> bytes:
    return json.dumps(data).encode()


# ---------------------------------------------------------------------------
# YoloV10PostProcessor unit tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_post_processor_returns_zeros_when_no_detections():
    pp = YoloV10PostProcessor(confidence_threshold=0.9)
    raw = np.zeros((1, 300, 6), dtype=np.float32)
    result = await pp.process([raw], (320, 320))
    assert result.shape == (20, 6)
    assert result.sum() == 0.0


@pytest.mark.asyncio
async def test_post_processor_filters_by_confidence():
    pp = YoloV10PostProcessor(confidence_threshold=0.5, max_detections=20)
    raw = np.zeros((1, 5, 6), dtype=np.float32)
    # Row 0: confidence 0.8 → kept
    raw[0, 0] = [10, 10, 50, 50, 0.8, 0]
    # Row 1: confidence 0.3 → filtered out
    raw[0, 1] = [20, 20, 60, 60, 0.3, 1]

    result = await pp.process([raw], (320, 320))
    assert result[0, 1] == pytest.approx(0.8, abs=1e-5)     # score preserved
    assert result[1, 1] == pytest.approx(0.0, abs=1e-5)     # second row is zeros


@pytest.mark.asyncio
async def test_post_processor_normalises_coordinates():
    pp = YoloV10PostProcessor(confidence_threshold=0.5, max_detections=20)
    raw = np.zeros((1, 1, 6), dtype=np.float32)
    # x1=32, y1=64, x2=160, y2=192, conf=0.9, class=0
    raw[0, 0] = [32, 64, 160, 192, 0.9, 0]

    result = await pp.process([raw], (320, 320))
    _, score, ymin, xmin, ymax, xmax = result[0]
    assert score == pytest.approx(0.9, abs=1e-5)
    assert ymin == pytest.approx(64 / 320, abs=1e-5)
    assert xmin == pytest.approx(32 / 320, abs=1e-5)
    assert ymax == pytest.approx(192 / 320, abs=1e-5)
    assert xmax == pytest.approx(160 / 320, abs=1e-5)


@pytest.mark.asyncio
async def test_post_processor_clips_coordinates():
    pp = YoloV10PostProcessor(confidence_threshold=0.5)
    raw = np.zeros((1, 1, 6), dtype=np.float32)
    # Boxes outside the image boundaries
    raw[0, 0] = [-10, -20, 400, 400, 0.9, 1]

    result = await pp.process([raw], (320, 320))
    _, _, ymin, xmin, ymax, xmax = result[0]
    assert xmin >= 0.0
    assert ymin >= 0.0
    assert xmax <= 1.0
    assert ymax <= 1.0


@pytest.mark.asyncio
async def test_post_processor_respects_max_detections():
    pp = YoloV10PostProcessor(confidence_threshold=0.0, max_detections=5)
    raw = np.zeros((1, 10, 6), dtype=np.float32)
    for i in range(10):
        raw[0, i] = [i * 10, i * 10, (i + 1) * 10, (i + 1) * 10, 0.9, 0]

    result = await pp.process([raw], (320, 320))
    assert result.shape == (5, 6)
    # Only first 5 rows should have detections
    assert all(result[:, 1] > 0)


@pytest.mark.asyncio
async def test_post_processor_handles_float16_input():
    pp = YoloV10PostProcessor(confidence_threshold=0.5)
    raw = np.zeros((1, 1, 6), dtype=np.float16)
    raw[0, 0] = [10, 10, 100, 100, 0.8, 0]

    result = await pp.process([raw], (320, 320))
    assert result.dtype == np.float32
    assert result[0, 1] == pytest.approx(0.8, abs=1e-3)


# ---------------------------------------------------------------------------
# AsyncZMQHandler dispatcher tests (without real ZMQ socket)
# ---------------------------------------------------------------------------

@pytest.fixture
def handler():
    """AsyncZMQHandler with mocked model manager and ZMQ socket."""
    cfg = make_config()
    h = AsyncZMQHandler(cfg)
    return h


@pytest.mark.asyncio
async def test_model_request_unknown_model(handler: AsyncZMQHandler):
    handler.manager.model_exists = MagicMock(return_value=False)

    frames = [encode_header({"model_request": True, "model_name": "unknown.onnx"})]
    response = await handler._dispatch(frames)

    resp = json.loads(response[0])
    assert resp["model_available"] is False
    assert resp["model_loaded"] is False


@pytest.mark.asyncio
async def test_model_request_existing_model_loads(handler: AsyncZMQHandler):
    handler.manager.model_exists = MagicMock(return_value=True)
    handler.manager.load_model = AsyncMock(return_value=True)
    handler.manager._active_model = None  # force reload path

    frames = [encode_header({"model_request": True, "model_name": "test.onnx"})]
    response = await handler._dispatch(frames)

    resp = json.loads(response[0])
    assert resp["model_available"] is True
    assert resp["model_loaded"] is True
    handler.manager.load_model.assert_awaited_once_with("test.onnx")


@pytest.mark.asyncio
async def test_model_request_already_active_skips_reload(handler: AsyncZMQHandler):
    handler.manager.model_exists = MagicMock(return_value=True)
    handler.manager.load_model = AsyncMock(return_value=True)
    handler.manager._active_model = "test.onnx"
    handler.manager._backend = MagicMock()  # simulate loaded backend

    frames = [encode_header({"model_request": True, "model_name": "test.onnx"})]
    await handler._dispatch(frames)

    # Should NOT call load_model because the model is already active
    handler.manager.load_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_inference_no_model_returns_zeros(handler: AsyncZMQHandler):
    handler.manager._backend = None

    tensor = np.zeros((1, 3, 320, 320), dtype=np.float32)
    inf_header = {"shape": [1, 3, 320, 320], "dtype": "float32", "model_type": "yolo-generic"}
    frames = [encode_header(inf_header), tensor.tobytes(order="C")]

    response = await handler._dispatch(frames)
    resp_header = json.loads(response[0])
    result = np.frombuffer(response[1], dtype=np.float32).reshape(resp_header["shape"])
    assert result.shape == (20, 6)
    assert result.sum() == 0.0


@pytest.mark.asyncio
async def test_inference_dtype_mismatch_returns_zeros(handler: AsyncZMQHandler):
    mock_backend = MagicMock()
    mock_backend.model_info = make_model_info(input_dtype="float32")
    handler.manager._backend = mock_backend

    # Send uint8 but model expects float32
    tensor = np.zeros((1, 3, 320, 320), dtype=np.uint8)
    inf_header = {"shape": [1, 3, 320, 320], "dtype": "uint8", "model_type": "yolo-generic"}
    frames = [encode_header(inf_header), tensor.tobytes(order="C")]

    response = await handler._dispatch(frames)
    resp_header = json.loads(response[0])
    result = np.frombuffer(response[1], dtype=np.float32).reshape(resp_header["shape"])
    assert result.sum() == 0.0


@pytest.mark.asyncio
async def test_inference_runs_and_returns_detections(handler: AsyncZMQHandler):
    # Build a mock raw output with one confident detection
    raw_output = np.zeros((1, 300, 6), dtype=np.float32)
    raw_output[0, 0] = [32, 64, 160, 192, 0.9, 0]  # animal at 0.9 confidence

    mock_backend = AsyncMock()
    mock_backend.model_info = make_model_info(input_dtype="float32")
    mock_backend.infer = AsyncMock(return_value=[raw_output])
    handler.manager._backend = mock_backend

    tensor = np.zeros((1, 3, 320, 320), dtype=np.float32)
    inf_header = {"shape": [1, 3, 320, 320], "dtype": "float32", "model_type": "yolo-generic"}
    frames = [encode_header(inf_header), tensor.tobytes(order="C")]

    response = await handler._dispatch(frames)
    resp_header = json.loads(response[0])
    result = np.frombuffer(response[1], dtype=np.float32).reshape(resp_header["shape"])

    assert result.shape == (20, 6)
    assert result[0, 1] == pytest.approx(0.9, abs=1e-5)   # confidence
    assert result[0, 0] == pytest.approx(0.0, abs=1e-5)   # class_id = 0 (animal)


@pytest.mark.asyncio
async def test_model_data_saves_and_loads(handler: AsyncZMQHandler):
    handler.manager.save_model = AsyncMock(return_value=True)
    handler.manager.load_model = AsyncMock(return_value=True)

    data_header = encode_header({"model_data": True, "model_name": "new_model.onnx"})
    fake_bytes = b"\x00" * 100
    frames = [data_header, fake_bytes]

    response = await handler._dispatch(frames)
    resp = json.loads(response[0])
    assert resp["model_saved"] is True
    assert resp["model_loaded"] is True
    handler.manager.save_model.assert_awaited_once_with("new_model.onnx", fake_bytes)
    handler.manager.load_model.assert_awaited_once_with("new_model.onnx")


@pytest.mark.asyncio
async def test_bad_header_returns_zeros(handler: AsyncZMQHandler):
    frames = [b"this is not json"]
    response = await handler._dispatch(frames)
    resp_header = json.loads(response[0])
    result = np.frombuffer(response[1], dtype=np.float32).reshape(resp_header["shape"])
    assert result.sum() == 0.0


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = AppConfig()
    assert cfg.server.endpoint == "tcp://*:5555"
    assert cfg.server.models_dir == "./models"
    assert cfg.inference.provider == "cpu"
    assert cfg.post_process.confidence_threshold == pytest.approx(0.4)
    assert cfg.post_process.max_detections == 20
    assert cfg.logging.level == "INFO"


def test_config_none_sections_use_defaults():
    """YAML sections that parse as None should fall back to defaults."""
    from condor.config.settings import AppConfig
    cfg = AppConfig.model_validate(
        {"server": None, "inference": None, "post_process": None, "logging": None}
    )
    assert cfg.server.endpoint == "tcp://*:5555"
    assert cfg.inference.provider == "cpu"
    assert cfg.post_process.confidence_threshold == pytest.approx(0.4)
