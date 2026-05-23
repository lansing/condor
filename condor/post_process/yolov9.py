"""YOLOv9 post-processor for yolo-generic model output.

YOLOv9 models output a single tensor of shape
``(1, num_attributes, num_predictions)`` where:
    - num_attributes = 4 (box xywh) + num_classes
    - For 3 classes: (1, 7, 33600)
    - For 80 classes (COCO): (1, 84, 8400)

Each prediction row is:
    [x_center, y_center, width, height, class_score_0, class_score_1, ...]

Post-processing steps:
    1. Transpose to (num_predictions, num_attributes)
    2. Extract boxes (xywh) and class scores
    3. Convert xywh to xyxy
    4. Compute confidence = max(class_scores) and class_id = argmax(class_scores)
    5. Filter by confidence_threshold
    6. Apply class-aware NMS using batched NMS trick
    7. Normalise box coordinates by input tensor dimensions (÷ width/height)
    8. Clip coordinates to [0.0, 1.0]
    9. Pack into a (max_detections, 6) float32 output as
       [class_id, score, ymin, xmin, ymax, xmax]
"""

from __future__ import annotations

import asyncio
import logging

import cv2
import numpy as np

from .base import BasePostProcessor

logger = logging.getLogger(__name__)


def xywh2xyxy_np(x: np.ndarray) -> np.ndarray:
    """Convert bounding box from (x, y, w, h) to (x1, y1, x2, y2) format."""
    y = np.empty_like(x)
    xy = x[..., :2]
    wh_half = x[..., 2:] / 2
    y[..., 0] = xy[..., 0] - wh_half[..., 0]
    y[..., 1] = xy[..., 1] - wh_half[..., 1]
    y[..., 2] = xy[..., 0] + wh_half[..., 0]
    y[..., 3] = xy[..., 1] + wh_half[..., 1]
    return y


class YoloV9PostProcessor(BasePostProcessor):
    """Post-processor for YOLOv9 ONNX models with yolo-generic output."""

    short_name = "V9"

    def __init__(
        self,
        confidence_threshold: float = 0.4,
        nms_threshold: float = 0.4,
        max_detections: int = 20,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections

    async def process(
        self,
        inference_output: list[np.ndarray],
        input_shape: tuple[int, int],
    ) -> np.ndarray:
        """Async entry point - delegates CPU work to a thread-pool executor."""
        return await asyncio.to_thread(
            self._process_sync, inference_output, input_shape
        )

    def _process_sync(
        self,
        inference_output: list[np.ndarray],
        input_shape: tuple[int, int],
    ) -> np.ndarray:
        result = np.zeros((self.max_detections, 6), dtype=np.float32)

        if not inference_output:
            logger.warning("YoloV9PostProcessor: empty inference output.")
            return result

        raw = inference_output[0]

        raw = raw.astype(np.float32)

        if raw.ndim == 3:
            if raw.shape[0] == 1:
                raw = raw[0].transpose(1, 0)
            else:
                logger.error(
                    "YoloV9PostProcessor: unexpected batch size %s; expected (1, ..., ...).",
                    raw.shape,
                )
                return result

        elif raw.ndim == 2:
            if raw.shape[0] < raw.shape[1]:
                raw = raw.transpose(1, 0)

        if raw.ndim != 2 or raw.shape[1] < 5:
            logger.error(
                "YoloV9PostProcessor: unexpected output shape %s; expected (N, 5+).",
                raw.shape,
            )
            return result

        input_h, input_w = input_shape

        boxes_xywh = raw[:, :4]
        class_scores = raw[:, 4:]

        if class_scores.size == 0:
            logger.warning("YoloV9PostProcessor: no class scores found.")
            return result

        confidences = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        mask = confidences >= self.confidence_threshold
        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if boxes_xywh.shape[0] == 0:
            return result

        boxes_xyxy = xywh2xyxy_np(boxes_xywh)

        max_coordinate = np.max(boxes_xyxy)
        offsets = class_ids * (max_coordinate + 1)
        boxes_for_nms = boxes_xyxy + offsets[:, None]

        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes_for_nms.astype(np.float32).tolist(),
            scores=confidences.astype(np.float32).tolist(),
            score_threshold=self.confidence_threshold,
            nms_threshold=self.nms_threshold,
            top_k=self.max_detections,
        )

        if len(indices) == 0:
            return result

        indices = np.asarray(indices).flatten()
        keep_indices = indices

        for i, idx in enumerate(keep_indices):
            if i >= self.max_detections:
                break
            x1, y1, x2, y2 = boxes_xyxy[idx]
            result[i] = [
                float(class_ids[idx]),
                float(confidences[idx]),
                max(0.0, min(1.0, float(y1) / input_h)),
                max(0.0, min(1.0, float(x1) / input_w)),
                max(0.0, min(1.0, float(y2) / input_h)),
                max(0.0, min(1.0, float(x2) / input_w)),
            ]

        n_dets = len(keep_indices)
        if n_dets:
            logger.debug(
                "YoloV9PostProcessor: %d detection(s) above threshold %.2f.",
                n_dets,
                self.confidence_threshold,
            )

        return result
