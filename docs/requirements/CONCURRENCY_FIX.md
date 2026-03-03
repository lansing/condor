# Concurrency Fix — Frigate Sync Contract & Multi-Worker Design

## 1. Frigate code review — confirming the hypothesis

### 1.1 Each ZMQ detector is strictly synchronous

`frigate/detectors/plugins/zmq_ipc.py` — `ZmqIpcDetector.detect_raw()`:

```python
self._socket.send_multipart([header_bytes, payload_bytes])
reply_frames = self._socket.recv_multipart()          # blocking
detections = self._decode_response(reply_frames)
return detections
```

One `zmq.REQ` socket, strict blocking send → recv. There is no concurrency within a
single detector instance. ZMQ's REQ socket enforces the alternation: a second `send`
before the corresponding `recv` is a protocol error.

### 1.2 Each detector runs in its own sequential process

`frigate/object_detection/base.py` — `DetectorRunner.run()`:

```python
while not self.stop_event.is_set():
    connection_id = self.detection_queue.get(timeout=1)   # wait for a frame
    ...
    detections = object_detector.detect_raw(input_frame)  # blocks until reply
    ...
    self.outputs[connection_id]["np"][:] = detections[:]
    detector_publisher.publish(connection_id)              # signal camera process
```

One `DetectorRunner` process, one blocking loop, no concurrency whatsoever within
that runner.

### 1.3 Multiple detectors share one camera queue (load balancing)

`frigate/app.py`:

```python
self.detection_queue: Queue = mp.Queue()              # ONE shared queue

for name, detector_config in self.config.detectors.items():
    self.detectors[name] = ObjectDetectProcess(
        ..., self.detection_queue, ...                # every detector pulls from the same queue
    )
```

All camera processes push frames into one `detection_queue`. All `DetectorRunner`
processes compete for frames from that queue — simple multi-consumer load balancing.
Each winning runner processes exactly one frame at a time, then loops.

### 1.4 The 200 ms timeout is the wall

`zmq_ipc.py`:

```python
self._socket.setsockopt(zmq.RCVTIMEO, self._request_timeout_ms)   # default: 200 ms
```

On timeout: the socket is closed and recreated, the model re-initialised (another
two round-trips), and zeros are returned for that frame.

### 1.5 Result ordering is guaranteed per connection — no protocol bug

ZMQ's REQ/REP contract guarantees that Detector A's `recv` returns the reply to
Detector A's `send`. Replies can never cross connections. The "out of sync" symptom
is **not** a protocol ordering bug.

---

## 2. Root cause: temporal drift from serial queuing

With N Frigate ZMQ detectors all connecting to Condor's **single** REP socket,
requests serialise through it. ZMQ REP can only accept the next message after it
has sent the reply to the current one.

Approximate per-request timing on our TRT setup:

| Step | ~time |
|---|---|
| ZMQ recv + JSON parse | 1 ms |
| Tensor reconstruct | 0.5 ms |
| H→D copy | 1 ms |
| TRT `execute_v2` | 10–15 ms |
| D→H copy | 1 ms |
| Post-process | 1 ms |
| ZMQ send | 1 ms |
| **Total** | **~16–21 ms** |

With N=1 Frigate detector this is fine — 16–21 ms RTT, well within 200 ms.

With N=2 Frigate detectors, Detector B queues behind A:

```
t=0    Detector A → Condor REP socket  (A queued)
t=0    Detector B → Condor REP socket  (B queued behind A)
t=0    Server recvs A, starts inference
t=18   Server replies to A  ✓  (A waited 18 ms)
t=18   Server recvs B, starts inference
t=36   Server replies to B  ✓  (B waited 36 ms from its send at t=0)
```

Detector B's latency is 2× single-worker latency.  With N=3: 3×, etc.

A camera running object detection at 10 fps emits a new frame every 100 ms.
If the detection round trip is 36 ms for 2 detectors, or 54 ms for 3, Condor is
still responsive.  But that extra latency translates directly into stale
annotations: at 30 fps video with 36 ms detection lag, boxes lag ~1 frame; at
54 ms they lag ~1.6 frames.  On a moving animal/person this looks like the
boxes are always slightly behind — exactly the "out of sync" observation.

Additionally:
- Model reload (triggered by socket timeout): sends `model_request` + possibly
  `model_data` before any inference resumes. With a shared REP socket, this
  "out of band" round trip blocks all other detectors during that time.
- GPU is idle during ~40 % of each request's wall time (ZMQ I/O, host-side
  processing, data copies). With one worker there is no way to pipeline that
  idle time into useful GPU work.

---

## 3. Proposed solutions

### Option A — Multi-worker (single Condor process, N ports) ✅ recommended

**Architecture:**

```
config.yaml:
  num_workers: 3          ← new setting
  base_port: 5555         ← workers bind to 5555, 5556, 5557

Frigate config:
  detector0: { type: zmq, endpoint: tcp://condor:5555 }
  detector1: { type: zmq, endpoint: tcp://condor:5556 }
  detector2: { type: zmq, endpoint: tcp://condor:5557 }
```

Inside Condor, one process starts N workers:

```
Worker 0: asyncio loop in thread-0
  REP socket :5555
  TRT execution context 0  +  CUDA stream 0  +  I/O buffers 0

Worker 1: asyncio loop in thread-1
  REP socket :5556
  TRT execution context 1  +  CUDA stream 1  +  I/O buffers 1

Worker 2: asyncio loop in thread-2
  REP socket :5557
  TRT execution context 2  +  CUDA stream 2  +  I/O buffers 2

Shared (all workers):
  trt.ICudaEngine (deserialized once, read-only, safe to share)
  AsyncModelManager (mutex-protected load/unload)
  YoloV10PostProcessor (stateless, thread-safe)
```

Each worker is a `AsyncZMQHandler` variant running in its own `threading.Thread`
with its own `asyncio.run(...)`.  Within each worker the REQ/REP ordering
contract is still strictly maintained — the worker never starts the next recv
until it has sent the current reply.  Across workers they are fully concurrent.

**TRT concurrency notes:**
- A `trt.ICudaEngine` is immutable after deserialisation and safe to use from
  multiple threads as a factory.
- Each `create_execution_context()` creates independent state; concurrent calls
  on different contexts on different CUDA streams are safe and will overlap on
  the GPU's compute queue.
- Each worker needs its own `CUDA context` or shared context with careful
  `cuCtxPushCurrent/Pop` management. Simplest: one CUDA context per worker
  (maps cleanly to the existing `_load_sync` flow).

**GPU memory:**
- Engine weight memory: ~34 MB (shared conceptually, but with N separate CUDA
  contexts each context holds its own copy — TRT does not currently support true
  cross-context engine sharing). At N=3 this is ~102 MB, negligible on a modern
  GPU.
- Activation memory per context: proportional to the largest layer, not the full
  model — typically 37–50 MB for YOLOv10-e at batch=1.
- Total overhead for N=3: ~250–300 MB GPU RAM additional vs. N=1.

**Throughput improvement:**
- Three independent workers, each ~18 ms RTT, effectively deliver ~167 inferences/s.
- GPU is occupied during one worker's inference while other workers are on
  ZMQ/copy/post-process — the ~40 % idle time is filled.
- Each Frigate detector sees its own dedicated worker with no queuing behind peers.

**What changes in Condor:**
1. `config.yaml` / `ServerConfig` — `num_workers: N` setting.
2. `server/main.py` — spawn N worker threads, each running its own event loop
   and an `AsyncZMQHandler` with port `base_port + i`.
3. `backends/tensorrt_backend.py` — change `_load_sync` to accept an already-
   deserialised engine (passed in from outside), create its own execution context
   and CUDA stream from it.  The `AsyncModelManager` deserialises the engine once
   and hands a reference to each backend instance.
4. Dockerfile — `EXPOSE 5555-5559` (or however wide the range is).
5. `docker-run-tensorrt` Makefile target — `-p 5555-5559:5555-5559`.

---

### Option B — Multiple Condor containers (no code changes)

Run N separate `condor:tensorrt` containers, each on a different host port.
Frigate configures N ZMQ detectors pointing to `condor:5555`, `condor:5556`, etc.

```bash
# docker-compose or equivalent:
condor-trt-0:
  image: condor:tensorrt
  ports: ["5555:5555"]
  runtime: nvidia

condor-trt-1:
  image: condor:tensorrt
  ports: ["5556:5555"]
  runtime: nvidia

condor-trt-2:
  image: condor:tensorrt
  ports: ["5557:5555"]
  runtime: nvidia
```

Each container loads the model independently. No code changes.

**Tradeoffs vs. Option A:**

| | Option A (multi-worker) | Option B (multi-container) |
|---|---|---|
| Code changes | Moderate | None |
| GPU RAM (N=3) | ~250–300 MB extra | ~300 MB × N (full model per container) |
| Container management | 1 container | N containers |
| Port exposure | N ports, 1 container | N ports, N containers |
| Model hot-reload | Coordinated (single manager) | Independent per container |
| Failure isolation | One crash affects all workers | Independent |
| Operational simplicity | Single `make docker-run-tensorrt` | docker-compose or manual |

Option B is a viable stopgap — it can be deployed today with no code changes and
validate that the multi-worker approach actually helps before implementing Option A.

---

## 4. Recommendation

**Short term:** deploy Option B with 2 containers to confirm the hypothesis —
run 2 Condor containers on ports 5555 and 5556, configure Frigate with 2 ZMQ
detectors, and observe whether latency drops and the "out of sync" artefact
disappears.

**Long term:** implement Option A.  The implementation is well-scoped:
- The asyncio-per-thread worker pattern is straightforward (each thread is an
  independent, already-working `AsyncZMQHandler`).
- The TRT engine factory + N contexts pattern is a small refactor of
  `TensorRTBackend._load_sync`.
- No changes to the ZMQ protocol or Frigate side.

The multi-worker design also naturally extends to the ONNX backend: N workers
each with their own `ort.InferenceSession` provide the same concurrency benefit
for CPU/OpenVINO workloads.

---

## 5. What this does NOT change

- The ZMQ REQ/REP protocol with Frigate is unchanged. Each worker is still
  strictly synchronous from Frigate's perspective.
- The existing single-worker mode (N=1) is fully backward-compatible.
- No changes to Frigate configuration beyond adding additional `detectors:` entries.
