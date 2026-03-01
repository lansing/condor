# Frigate Remote Object Detector Protocol (ZMQ)

This document specifies the communication protocol between Frigate NVR and a remote object detector server using ZeroMQ (ZMQ).

## Overview

Frigate supports a `zmq` detector type that offloads object detection to a remote server. The protocol uses a **Request/Response (REQ/REP)** pattern over ZMQ. The detector server acts as the **REP** (Reply) node, binding to an endpoint, while Frigate acts as the **REQ** (Request) node, connecting to that endpoint.

Endpoints can be either Unix Domain Sockets (IPC) or TCP sockets.

## Frigate Configuration

To use a remote ZMQ detector, Frigate is configured as follows:

```yaml
detectors:
  remote_zmq:
    type: zmq
    endpoint: tcp://<server_ip>:<port>  # or ipc:///tmp/cache/zmq_detector
    request_timeout_ms: 200            # Optional: timeout for requests
```

## Protocol Stages

The protocol consists of three main stages: **Model Management**, **Inference**, and **Handling Configuration Changes**.

### 1. Model Management (Handshake)

Before starting inference, Frigate ensures the required model is available and loaded on the remote server. This stage establishes the **Active Model** for the connection.

**Crucial Assumption:** Once a model is successfully negotiated (via Availability Check or Data Transfer), all subsequent Inference Requests on that connection are implied to be for that specific model. Frigate does **not** include the model name in individual inference requests.

#### A. Model Availability Check
Frigate sends a single-frame multipart message containing a JSON header.

**Request:**
- Frame 0 (JSON):
  ```json
  {
    "model_request": true,
    "model_name": "model_filename.onnx"
  }
  ```

**Expected Response (JSON):**
- Frame 0 (JSON):
  ```json
  {
    "model_available": true,
    "model_loaded": true
  }
  ```
If `model_available` or `model_loaded` is `false`, Frigate will proceed to transfer the model.

#### B. Model Data Transfer
If the model is not ready, Frigate sends the model file content.

**Request:**
- Frame 0 (JSON):
  ```json
  {
    "model_data": true,
    "model_name": "model_filename.onnx"
  }
  ```
- Frame 1 (Raw Bytes): The complete content of the model file.

**Expected Response (JSON):**
- Frame 0 (JSON):
  ```json
  {
    "model_saved": true,
    "model_loaded": true
  }
  ```

### 2. Inference

Once the model is ready, Frigate sends image tensors for detection.

#### A. Inference Request
Frigate sends a two-frame multipart message.

**Request:**
- Frame 0 (JSON):
  ```json
  {
    "shape": [1, 3, 320, 320],
    "dtype": "uint8",
    "model_type": "yolo-generic"
  }
  ```
  *Note: The request applies to the Active Model established in Stage 1.*
- Frame 1 (Raw Bytes): The raw image tensor in C-order (row-major).

#### B. Inference Response
The server must respond with detection results. Frigate supports two formats for the response.

**Response Format 1 (Multipart):**
- Frame 0 (JSON):
  ```json
  {
    "shape": [20, 6],
    "dtype": "float32"
  }
  ```
- Frame 1 (Raw Bytes): 20x6 float32 array (480 bytes) containing detection data.

**Response Format 2 (Single Frame):**
- Frame 0 (Raw Bytes): 20x6 float32 array (480 bytes) containing detection data.

**Detection Data Format:**
The 20x6 array contains up to 20 detections. Each detection is represented by 6 float32 values:
`[class_id, score, ymin, xmin, ymax, xmax]`
- `class_id`: The numerical index of the detected object class.
- `score`: The confidence score (0.0 to 1.0).
- `ymin, xmin, ymax, xmax`: Normalized bounding box coordinates (0.0 to 1.0).

### 3. Handling Configuration Changes

If the model configuration (e.g., model path or type) changes on the Frigate side, Frigate restarts its detector plugin process. This results in:
1.  **A New Handshake:** The new Frigate detector instance will immediately send a new `model_request` (Stage 1).
2.  **State Update:** The detector server must update its **Active Model** to match the new `model_name` before processing further inference requests.

## Technical Requirements for Detector Server

To be compatible with Frigate, a detector server must meet the following requirements:

1.  **ZMQ Socket:** Must use a `zmq.REP` socket and `bind` to the configured endpoint.
2.  **State Management (Active Model):**
    - Must track the currently loaded "Active Model" for the connection.
    - Must assume all inference requests target the most recently negotiated model via Stage 1.
3.  **Message Handling:**
    - Must be capable of receiving multipart messages (`recv_multipart`).
    - Must distinguish between Model Management and Inference requests by inspecting the JSON header in Frame 0.
4.  **Model Management:**
    - Must implement a mechanism to store and load models provided by Frigate.
    - Must respond with appropriate JSON status codes (`model_available`, `model_loaded`, etc.).
5.  **Inference Execution:**
    - Must parse the input tensor based on the provided `shape` and `dtype`.
    - Must run the **Active Model** and post-process results into the `[20, 6]` float32 format.
    - Bounding boxes must be normalized to `[0, 1]` relative to the input image dimensions.
6.  **Performance and Robustness:**
    - Should handle requests within the `request_timeout_ms` (typically 200ms).
    - Should return a zero-filled `[20, 6]` array on error or if no detections are found.
    - Must handle socket resets or restarts gracefully.
