# Model Post-processing for YOLOv9 and YOLOv10 ONNX Models

This document details the post-processing steps implemented for YOLOv9 and YOLOv10 ONNX models exported from the Ultralytics framework. The goal is to "fully own" the post-inference workflow, meaning converting the raw tensor output from an ONNX Runtime inference session into structured detection predictions (bounding boxes, confidence, class IDs) using our own code, rather than relying on external libraries for this specific step. Our custom post-processing logic is designed to closely replicate the Ultralytics' internal methodology to ensure identical or virtually identical final predictions.

## 1. Preprocessing (`ONNXInferenceSession.preprocess_image`)

Before any model inference, the input image must be prepared to match the model's expected format. Both YOLOv9 and YOLOv10 models, when used within the Ultralytics framework, employ a "LetterBox" strategy for image preprocessing. Our `ONNXInferenceSession.preprocess_image` method replicates this.

**Steps:**
1.  **Read Image:** The image is read using OpenCV (BGR format) and then converted to RGB.
2.  **LetterBox Resizing:** The image is resized to fit the model's input dimensions (`new_shape`) while preserving its aspect ratio. This involves calculating a `gain` (scale ratio) and new unpadded dimensions (`new_unpad`).
3.  **Padding:** If the resized image doesn't perfectly fill the `new_shape`, padding (typically with a gray value of 114) is applied to center the image within the `new_shape`. The padding amounts (`pad_x`, `pad_y`) are calculated.
4.  **Normalization:** The padded image's pixel values (which are now in HWC, RGB format) are converted from `[0, 255]` to `[0.0, 1.0]` by dividing by 255.0.
5.  **Transpose:** The image is transposed from HWC to BCHW (`Batch, Channels, Height, Width`) format, and a batch dimension is added.

**Output of Preprocessing:**
The `preprocess_image` method returns:
*   The preprocessed image as a NumPy array (BCHW format).
*   The original image dimensions (`original_dims`).
*   A `ratio_pad` tuple `((gain, gain), (pad_x, pad_y))` containing the scaling factor and padding amounts. These values are crucial for accurately scaling the detected bounding boxes back to the original image coordinates during post-processing.

## 2. Post-processing for YOLOv9 Models (`YOLOvPostProcessor`)

YOLOv9 models, when exported as raw (pre-NMS) ONNX outputs from Ultralytics, typically produce a single output tensor (`output0`).

### Raw Model Output Structure (YOLOv9): `(batch_size, num_attributes, num_predictions)`

For a single image (`batch_size = 1`), the raw output tensor from a YOLOv9 model (e.g., `MDV6-yolov9-c`) has the shape: `(1, 7, 33600)`.
This means there are `33600` potential predictions, and each prediction has `7` attributes. These 7 attributes are interpreted as:

*   `prediction[0:4]`: Bounding box coordinates in `xywh` format (`x_center`, `y_center`, `width`, `height`), normalized to the model's input image size.
*   `prediction[4:7]`: Three class scores (e.g., for `animal`, `person`, `vehicle`). These are often logits or raw scores that have already incorporated the objectness confidence.

### Post-processing Steps in `YOLOvPostProcessor`:

Our `YOLOvPostProcessor` closely mimics `ultralytics.utils.nms.non_max_suppression` for this output format:

1.  **Extract Predictions:** The raw output tensor `(1, 7, 33600)` is first processed to extract `predictions` (shape `(num_predictions, 7)`).
2.  **Separate Components:**
    *   `boxes_xywh`: Bounding box coordinates (`x_center`, `y_center`, `width`, `height`).
    *   `class_scores`: Scores for each class.
3.  **Initial Confidence Filtering:** Predictions are filtered based on a `confidence_threshold` by checking if the maximum class score for a prediction is above the threshold. This reduces the number of candidates for NMS.
4.  **`xywh` to `xyxy` Conversion:** The bounding boxes are converted from `xywh` format to `xyxy` (`x1`, `y1`, `x2`, `y2`) format using our custom `xywh2xyxy_np` function.
5.  **Final Confidence and Class ID:**
    *   The final confidence for each box is taken as the maximum score among its class scores.
    *   The class ID is the index of the class with the highest score.
6.  **Scaling Bounding Boxes (`scale_boxes_np`):** The bounding boxes (which are currently scaled to the model's input dimensions) are transformed back to the original image dimensions. This step accurately undoes the `LetterBox` preprocessing by accounting for the `gain` (scaling ratio) and `pad_x`, `pad_y` values obtained during preprocessing.
7.  **Non-Maximum Suppression (NMS):** Our custom `non_max_suppression_np` function is applied. This reimplements `ultralytics`'s NMS strategy, including:
    *   **Batched NMS Trick:** Offsetting boxes by their class ID to enable class-aware NMS in a single pass using `cv2.dnn.NMSBoxes`.
    *   Filtering by `confidence_threshold` and `iou_threshold`.
8.  **Format Detections:** The final filtered and scaled detections are formatted into a list of dictionaries, each containing `box` (xyxy), `confidence`, `class_id`, and `class_name`.

## 3. Post-processing for YOLOv10 Models (`YOLOv10PostProcessor`)

YOLOv10 models, when exported as raw (pre-NMS) ONNX outputs from Ultralytics, have a different output structure compared to YOLOv9.

### Raw Model Output Structure (YOLOv10): `(batch_size, num_detections, 6)`

For a single image (`batch_size = 1`), the raw output tensor from a YOLOv10 model (e.g., `MDV6-yolov10-e`) has the shape: `(1, 300, 6)`.
This means there are `300` potential detections (a fixed maximum number), and each prediction already contains `6` attributes. These 6 attributes are interpreted as:

*   `prediction[0:4]`: Bounding box coordinates in `xyxy` format (`x1`, `y1`, `x2`, `y2`), scaled to the model's input image size.
*   `prediction[4]`: The confidence score (already combined objectness and class score).
*   `prediction[5]`: The class ID.

This `(1, 300, 6)` output is effectively an "NMS-ready" output, meaning the model might have internal filtering or a simplified output head.

### Post-processing Steps in `YOLOv10PostProcessor`:

Our `YOLOv10PostProcessor` handles this simpler output structure:

1.  **Extract Detections:** The raw output tensor `(1, 300, 6)` directly provides `detections_raw` (shape `(num_detections, 6)`).
2.  **Filter by Confidence:** Detections are filtered based on the `confidence_threshold` applied to the `confidence_score` (`prediction[4]`).
3.  **Scaling Bounding Boxes (`scale_boxes_np`):** The bounding boxes (which are already in `xyxy` format and scaled to the model's input dimensions) are transformed back to the original image dimensions using `scale_boxes_np`.
4.  **Format Detections:** The final filtered and scaled detections are formatted into a list of dictionaries.

## 4. Comparison: YOLOv9 vs. YOLOv10 Post-processing

The most significant difference between YOLOv9 and YOLOv10 models (when exported raw from Ultralytics) lies in their **raw ONNX output structure**, which directly impacts the post-processing workflow:

*   **YOLOv9 (`(1, 7, 33600)`):**
    *   **Output:** Pre-NMS, with bounding boxes in `xywh` format and separate class scores (3 classes).
    *   **Post-processing Complexity:** Requires a more involved NMS pipeline, including `xywh` to `xyxy` conversion, initial candidate filtering, calculating final confidence from class scores, and applying a robust NMS algorithm (like our `non_max_suppression_np`) with batched NMS.
*   **YOLOv10 (`(1, 300, 6)`):**
    *   **Output:** Post-NMS, or at least NMS-ready, with bounding boxes already in `xyxy` format, combined confidence, and class ID. The number of detections is fixed (`300`).
    *   **Post-processing Complexity:** Significantly simpler. Primarily involves filtering by confidence threshold and scaling the bounding boxes back to original image dimensions. The NMS logic is either handled internally by the model itself or by `ultralytics` during export, resulting in this pre-filtered output.

**Preprocessing Comparison:**
Both YOLOv9 and YOLOv10 models utilize the same `LetterBox`-like preprocessing strategy implemented in `ONNXInferenceSession.preprocess_image`. This ensures that the input images are prepared identically for both architectures.

## 5. Implementation Details

To ensure "full ownership" and matching results, our custom post-processing classes (`YOLOvPostProcessor`, `YOLOv10PostProcessor`) do not call any Ultralytics code directly. Instead, they implement the necessary utility functions (`xywh2xyxy_np`, `scale_boxes_np`, `clip_boxes_np`, `non_max_suppression_np`) in pure NumPy/OpenCV, replicating the algorithms found in `ultralytics.utils.ops` and `ultralytics.utils.nms`. This approach guarantees that our pipeline is independent while producing results identical to the Ultralytics framework.
