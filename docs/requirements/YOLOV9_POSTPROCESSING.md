# YOLOv9 Post-Processing Implementation Requirements

## Context

Condor is an object detection inference provider for Frigate NVR. It handles post-processing of model output to produce a format expected by Frigate. Currently, only YoloV10 model output is supported via `YoloV10PostProcessor`. This document outlines requirements for adding YoloV9 (yolo-generic) support.

## Existing Implementation Reference

- **YoloV10 Post-Processor**: `condor/post_process/yolov10.py`
- **Base Interface**: `condor/post_process/base.py`
- **Test Pattern**: `tests/test_protocol.py` (see `YoloV10PostProcessor` tests)

## YoloV9 Output Format

Based on analysis of Frigate's implementation in `frigate/frigate/util/model.py`, the pytorch-wildlife reference implementation, and Ultralytics export behavior:

### YoloV9 (yolo-generic) Raw Output Structure

YoloV9 models exported from Ultralytics produce output with the following characteristics:

**Single Output Tensor (pre-NMS, raw)**:
- Shape: `(1, num_attributes, num_predictions)`
- **IMPORTANT**: This is `(batch, attributes, predictions)` NOT `(batch, predictions, attributes)`
- For 3 classes: `(1, 7, 33600)` where 7 = 4 (box xywh) + 3 (class scores)
- For 80 classes (COCO): `(1, 84, 8400)` where 84 = 4 (box xywh) + 80 (class scores)
- Attributes: `[x_center, y_center, width, height, class_score_0, class_score_1, ...]`

**Shape Handling**:
- If shape is `(1, num_attributes, num_predictions)`: transpose to `(num_predictions, num_attributes)`
- If shape is `(num_predictions, num_attributes)`: already in correct format
- If `shape[0] < shape[1]` and dim is 2: transpose

**Multi-part Output (pre-NMS, stride-based, less common)**:
- 3 output tensors, each corresponding to a different stride (8, 16, 32)
- Each tensor shape: `(batch, num_anchors, 85, grid_h, grid_w)` for 80 classes
- 85 = 4 (box) + 1 (objectness) + 80 (classes)

### Key Differences from YoloV10

| Aspect | YoloV10 | YoloV9 (yolo-generic) |
|--------|---------|------------------------|
| Box format | xyxy (already in final format) | xywh (center + dimensions) |
| Confidence | Combined confidence in column 4 | Max of class scores column 4+ |
| Class ID | Direct class ID in column 5 | Argmax of class score columns |
| NMS | Internal (already NMS'd) | Requires external NMS |
| Attributes | 6 (x1,y1,x2,y2,conf,class) | 5+ (x,y,w,h,class_scores...) |

## Requirements

### Key Implementation Notes from Reference

**Batched NMS Trick**: For class-aware NMS (preventing boxes of different classes from suppressing each other), the reference implementation uses an offsetting approach:
```python
max_coordinate = np.max(boxes_xyxy)
offsets = class_ids * (max_coordinate + 1)
boxes_for_nms = boxes_xyxy + offsets[:, None]
```
This allows using `cv2.dnn.NMSBoxes` (which is class-agnostic) while achieving class-aware NMS.

**Scaling Approach**: The pytorch-wildlife reference uses letterbox-aware scaling with `ratio_pad` from preprocessing. However, Frigate's protocol only passes `input_shape` (not `ratio_pad`), so condor must use simple scaling by input dimensions (like yolov10 does):
```python
ymin = y1 / input_h  # simple scaling, no letterbox adjustment
```

### 1. New Post-Processor Class

Create `condor/post_process/yolov9.py` with a `YoloV9PostProcessor` class:

```python
class YoloV9PostProcessor(BasePostProcessor):
    """Post-processor for YOLOv9 ONNX models with yolo-generic output."""
    
    def __init__(
        self,
        confidence_threshold: float = 0.4,
        nms_threshold: float = 0.4,
        max_detections: int = 20,
    ) -> None:
        ...
```

### 2. Supported Output Formats

The post-processor MUST handle:

1. **Single tensor with NMS (simplified)**:
   - Shape: `(1, N, num_classes + 4)` or `(N, num_classes + 4)`
   - Attributes: `[x_center, y_center, width, height, class_score_0, ...]`
   - Process: transpose if needed → filter by confidence → xywh to xyxy → NMS

2. **Multi-part tensor (full post-processing)**:
   - 3 tensors from different FPN levels (strides 8, 16, 32)
   - Each with anchor-based predictions
   - Process: decode boxes using anchors and strides → filter → NMS

### 3. Post-Processing Steps

For single-tensor NMS style output:

1. **Input**: Raw tensor from inference, shape `(1, num_attributes, num_predictions)`
2. **Transpose**: `raw_output[0].transpose(1, 0)` to get `(num_predictions, num_attributes)`
3. **Alternative transpose check**: If `shape[0] < shape[1]` and ndim==2, transpose
4. **Extract boxes**: `boxes_xywh = predictions[:, :4]`
5. **Extract class scores**: `class_scores = predictions[:, 4:]`
6. **xywh to xyxy conversion**:
   ```
   x1 = x_center - width/2
   y1 = y_center - height/2
   x2 = x_center + width/2
   y2 = y_center + height/2
   ```
7. **Final confidence and class ID**: 
   - `confidences = max(class_scores, axis=1)`
   - `class_ids = argmax(class_scores, axis=1)`
8. **Confidence filtering**: Keep rows where `confidences >= confidence_threshold`
9. **Class-aware NMS**: Apply batched NMS using offset trick:
   ```python
   max_coordinate = np.max(boxes_xyxy)
   offsets = class_ids * (max_coordinate + 1)
   boxes_for_nms = boxes_xyxy + offsets[:, None]
   indices = cv2.dnn.NMSBoxes(boxes_for_nms, confidences, confidence_threshold, nms_threshold)
   ```
10. **Normalize**: Scale box coordinates by input dimensions and clip to [0, 1]
11. **Format output**: `[class_id, score, ymin, xmin, ymax, xmax]` per row

### 4. Output Format

The output MUST match the Frigate format:

```
shape: (max_detections, 6)
dtype: float32
format per row: [class_id, score, ymin, xmin, ymax, xmax]
- class_id: integer class index
- score: confidence (0.0 to 1.0)
- coordinates: normalized to [0.0, 1.0] relative to input dimensions
```

### 5. Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| confidence_threshold | 0.4 | Minimum confidence to keep a detection |
| nms_threshold | 0.4 | IOU threshold for NMS |
| max_detections | 20 | Maximum number of detections to return |

### 6. Error Handling

- Empty inference output → return zeros
- Unexpected output shape → log error, return zeros
- Handle float16, float32, and float64 inputs → normalize to float32

## Unit Test Requirements

### Test File: `tests/test_yolov9_post_processor.py`

1. **test_post_processor_returns_zeros_when_no_detections**
   - Input: empty or all-low-confidence output
   - Expected: all zeros array (20, 6)

2. **test_post_processor_filters_by_confidence**
   - Input: mixed confidence detections
   - Expected: only detections above threshold

3. **test_post_processor_normalises_coordinates**
   - Input: known box coordinates
   - Expected: coordinates normalized to [0,1] range

4. **test_post_processor_xywh_to_xyxy_conversion**
   - Input: xywh format boxes
   - Expected: xyxy format in output

5. **test_post_processor_applies_nms**
   - Input: overlapping boxes with same class
   - Expected: only highest confidence box kept

6. **test_post_processor_handles_different_output_shapes**
   - Test: (1, N, 5+classes), (N, 5+classes), transposed variants
   - Important: YoloV9 is `(1, num_attributes, N)` needing transpose, NOT `(1, N, num_attributes)`

7. **test_post_processor_handles_80_class_coco_format**
   - Input: shape `(1, 84, 8400)` (80 classes + 4 box attributes)
   - Verify correct parsing of 84 attributes

8. **test_post_processor_respects_max_detections**
   - Input: many detections above threshold
   - Expected: only top N by confidence

9. **test_post_processor_handles_float16_input**
   - Input: float16 tensor
   - Expected: float32 output

10. **test_post_processor_multi_class_nms**
    - Input: boxes from different classes
    - Expected: NMS applied per-class (class-aware)

11. **test_post_processor_empty_input**
    - Input: empty list or empty array
    - Expected: zeros returned with warning log

## Auto-Detection Requirements (Future)

### Problem Statement

Currently, condor uses the `model_type` from Frigate's inference request header (`"yolo-generic"`, `"yolov10"`, etc.) to select the appropriate post-processor. This requires Frigate to specify the model type correctly.

However, we can optionally auto-detect the model output type based on tensor shape characteristics, which would:
1. Allow model-agnostic inference
2. Provide fallback when model_type is not specified
3. Enable easier integration with custom models

### Auto-Detection Heuristics

Based on output tensor shape characteristics:

| Shape | Likely Model Type |
|-------|------------------|
| `(1, N, 6)` where N varies | YoloV10 (NMS-ready, already xyxy) |
| `(1, num_attributes, N)` where num_attributes = 4+num_classes | YoloV9 single-output (needs transpose) |
| `(N, 6)` | YoloV10 without batch dim |
| `(N, 4+num_classes)` | YoloV9 without batch dim (needs transpose if not matching) |
| List of 3 tensors with shapes like `(1, 3, 85, H, W)` | YoloV9 multi-part |

### Implementation (Future Phase)

Create `condor/post_process/registry.py`:

```python
class PostProcessorRegistry:
    """Registry for post-processors with auto-detection."""
    
    @staticmethod
    def detect_output_type(outputs: list[np.ndarray]) -> str:
        """Detect model type from output tensor shapes.
        
        Returns: 'yolov10', 'yolov9-single', 'yolov9-multi', or 'unknown'
        """
        
    @staticmethod
    def create_post_processor(
        model_type: str | None,
        outputs: list[np.ndarray],
        **kwargs
    ) -> BasePostProcessor:
        """Create appropriate post-processor based on type or auto-detection."""
```

### Auto-Detection Rules

1. **YoloV10 detection**:
   - Single tensor
   - Shape `(1, N, 6)` where N is number of detections
   - 3rd dimension = 6

2. **YoloV9 single-output detection**:
   - Single tensor
   - Shape `(1, N, 5+num_classes)` where num_classes >= 1
   - Not matching YoloV10 6-attribute pattern

3. **YoloV9 multi-output detection**:
   - List of 3 tensors
   - Each tensor has 4+ dimensions with small spatial grids

## Implementation Tasks

### Phase 1: YoloV9 Post-Processor (this effort)
- [ ] Create `condor/post_process/yolov9.py`
- [ ] Implement single-output post-processing path
- [ ] Implement multi-output post-processing path
- [ ] Add unit tests in `tests/test_yolov9_post_processor.py`
- [ ] Update `condor/post_process/__init__.py` exports

### Phase 2: Auto-Detection (future)
- [ ] Create `condor/post_process/registry.py`
- [ ] Implement `detect_output_type()` heuristics
- [ ] Implement `create_post_processor()` factory
- [ ] Add unit tests for auto-detection
- [ ] Update server to use registry

## References

- Existing YoloV10 implementation: `condor/post_process/yolov10.py`
- Frigate YoloV9 post-processing: `frigate/frigate/util/model.py:229` (`post_process_yolo`)
- Frigate multi-part YoloV9: `frigate/frigate/util/model.py:103` (`__post_process_multipart_yolo`)
- Frigate NMS YoloV9: `frigate/frigate/util/model.py:187` (`__post_process_nms_yolo`)
- Protocol documentation: `docs/requirements/REMOTE_DETECTOR_PROTO.md`
- Reference YoloV9 implementation: `/home/max/Code/pytorch-wildlife-onnx/PytorchWildlife_Export/postprocessors/yolov_postprocessor.py`
- Reference utility functions: `/home/max/Code/pytorch-wildlife-onnx/PytorchWildlife_Export/postprocessors/util.py`
