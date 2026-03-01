# Frigate Remote Detector Server - Initial Engineering Requirements

This document outlines the architecture, functional requirements, and implementation plan for a Python-based remote object detector server compatible with Frigate NVR.

## 1. Overview

The goal is to build a robust, extensible, and high-performance object detection server that implements the Frigate ZMQ Remote Detector Protocol. The server will be designed with an **asynchronous, non-blocking architecture** to support high concurrency and various inference backends (TensorRT, OpenVINO, ONNX) managed via a modern Python 3.12 stack.

## 2. Architecture Design

The system will follow a modular, **asyncio-based** architecture with a clear separation of concerns. All operations involving network I/O or offloaded computation (inference/post-processing) must be non-blocking.

### 2.1 Core Components

1.  **Server Core (`server` package):**
    -   **Async ZMQ Handler:** Uses `zmq.asyncio` to manage the ZMQ `REP` socket without blocking the event loop.
    -   **Async Protocol Dispatcher:** Parses incoming messages and routes them using `await`.
    -   **State Machine:** Tracks the "Active Model" state.

2.  **Model Manager (`model_manager` package):**
    -   **Async Cache System:** Handles model file persistence using `aiofiles` or similar non-blocking I/O.
    -   **Loader:** Orchestrates the loading of the specific runtime backend.

3.  **Inference Engine (`backends` package):**
    -   **Async Plugin Interface:** A standard abstract base class (`BaseBackend`) using `async` methods.
        -   `async load(model_path: str, config: dict) -> None`
        -   `async infer(input_tensor: np.ndarray) -> Any` (Considered I/O as it offloads to GPU/NPU or remote API).
        -   `async cleanup() -> None`
    -   **Implementations:** `TensorRTBackend`, `OpenVINOBackend`, `OnnxRuntimeBackend`.

4.  **Post-Processing (`post_process` package):**
    -   **Async Interface:** A standard abstract base class (`BasePostProcessor`).
        -   `async process(inference_output: Any, input_shape: Tuple[int, int]) -> np.ndarray` (Asynchronous to support offloaded or complex decoding).
    -   **Implementations:** `YoloV10PostProcessor`.

5.  **Configuration (`config` package):**
    -   Uses `pydantic` for typed, validated configuration.

### 2.2 Technology Stack

-   **Language:** Python 3.12+
-   **Concurrency:** `asyncio` (Standard library)
-   **Dependency Management:** `uv`
-   **Networking:** `pyzmq` (with `zmq.asyncio`)
-   **Async File I/O:** `aiofiles`
-   **Data Handling:** `numpy`
-   **Validation:** `pydantic`
-   **Testing:** `pytest`, `pytest-asyncio`

## 3. Functional Requirements

### 3.1 Asynchronous Execution
-   **Non-blocking Event Loop:** The main thread must run an `asyncio` event loop. No blocking calls (e.g., `time.sleep`, synchronous `socket.recv`, or heavy CPU-bound tasks without `run_in_executor`) are allowed in the main loop.
-   **Concurrency Support:** The server must be able to handle subsequent requests (e.g., model management heartbeats) while an inference task is being awaited.

### 3.2 Protocol Compliance
-   **Socket:** Bind using `zmq.asyncio.Context`.
-   **Inference:** `await` the result of the backend inference and post-processing before sending the response.

### 3.3 Inference Backends
-   **Offloading:** Inference is treated as an asynchronous operation. For backends like TensorRT, calls should be wrapped to ensure they don't block the event loop if the driver/library call is synchronous.
-   **Resource Management:** Explicit `await cleanup()` during model swaps to ensure VRAM is freed before the new model is loaded.
-   **Config:**: The config yaml file can set up the inference provider. For example, onnxruntime might need to have a preferred execution provider specified. Other backends might need to have a GPU specified. Etc.

### 3.4 Post-Processing
-   **YOLOv10:** Implement as an `async` task. If the decoding is CPU-heavy, it should be executed in a thread pool executor to maintain event loop responsiveness.

## 4. Non-Functional Requirements

-   **High Concurrency:** The architecture must assume the backend could be a remote API or a shared hardware resource with variable latency.
-   **Performance:** Minimize Python overhead; utilize `numpy` and efficient async patterns.
-   **Stability:** Ensure graceful shutdown of async tasks and ZMQ sockets.

## 5. Implementation Phases

### Phase 1: Async Framework & ONNX Backend
-   Set up project structure with `uv`, `Makefile`, and `pytest-asyncio`.
-   Implement the **Async ZMQ Server** using `zmq.asyncio`.
-   Implement the `ModelManager` with async file operations.
-   Define `BaseBackend` and `BasePostProcessor` as `async` interfaces.
-   Implement `OnnxRuntimeBackend` (Async-wrapped). Introduce support for onnx openvino EP for this phase. Use this EP for testing.
-   Implement `YoloV10PostProcessor` (Async-wrapped).
-   Use models/MDV6-yolov10-c_float16_320.onnx which is a yolov10 model, for testing. It has 320x320 inputs in normalized float format, nchw tensor format. See models/md.classes.txt for the class names.
-   Implement a test/benchmark client script that will simulate the frigate "client" with a sample image. Use sample_image.jpg, which is an image of an animal, the model should successfully detect it if inputs and post-processing is done right.
-   **Goal:** A fully asynchronous server passing protocol checks. 

### Phase 2: Advanced Backends
-   Implement `TensorRTBackend` and `OpenVINOBackend` with proper async integration (e.g., using `asyncio.to_thread` for synchronous C-bindings if necessary).

### Phase 3: Robustness & Benchmarking
-   Add comprehensive async unit tests.
-   Perform concurrency testing to ensure the server remains responsive during heavy inference loads.
