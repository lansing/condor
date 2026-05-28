# condor

TensorRT sidecar for [Frigate NVR](https://frigate.video). Efficienctly runs inference on a dedicated NVIDIA GPU and exposes it to Frigate via the ZMQ remote detector protocol. Provides significantly better FPS throughput and GPU utilization, with less CPU and host memory usage, compared to Frigate's built-in ONNX Runtime detector.

## Requirements

- [Frigate NVR](https://frigate.video) 0.17+
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- NVIDIA Driver 590+
- Python on your system _(for automated installer only)_

## Install

1. [Install](#install)
   - [Automated installer (recommended)](#automated-installer)
   - [Manual installation](#manual-installation) 
2. [Build a TensorRT Engine from your ONNX model](#build-a-tensorrt-engine-from-your-onnx-model)
3. Restart Frigate (use `docker compose down && docker compose up` for a full reload)
4. Observe Condor metrics using `condor` TUI. 

## FAQ

**Why would I use this instead of Frigate's built in detector?**

Condor provides an optimized, TensorRT-backed inference runtime with significantly better efficiency and throughput compared to what Frigate offers out of the box via its ONNX Runtime (CUDA EP) backend. For mid-range GPUs, expect to see 1.5-2x throughput compared to ONNX Runtime with CUDA EP.

Condor has a lower memory footprint (500-600 MB compared to 1+ GB for ONNX Runtime), and better efficiency on the CPU side.

---

**What GPUs are supported?**

NVIDIA GPUs from Turing (RTX 20 series) onward.

---

**What detector model architectures are supported?**

YOLOv9 and YOLOv10 are currently supported. Most likely, earlier YOLO variants would also work, as (AFAIK) their tensor input/output formats are compatible with YOLOv9.
 
If you'd like to see another architecture supported, please reach out.

---

**What model should I use with Condor?**

I recommend MegaDetector V6 in YOLOv10. This model provides excellent performance for a typical Frigate home setup. 

> Check out [pytorch-wildlife-onnx](https://github.com/lansing/pytorch-wildlife-onnx) for an easy way to export TensorRT (or ONNX) artifacts of this model. 

I get over 90 FPS max throughput on an RTX 3050 6GB, limited to 50 watts, using MegaDetector V6, YOLOv10 Extra, 640x640, int8 quantization, exported using pytorch-wildlife-onnx.

---

**How can I use Condor as my inference provider for Frigate?**

First, [convert your ONNX model to a TensorRT Engine](#build-a-tensorrt-engine-from-your-onnx-model). Then, [configure Frigate to use Condor](#install) as a remote (ZMQ) detector.

If you use MDV6 model via [my exporter project](https://github.com/lansing/pytorch-wildlife-onnx) as mentioned above, you can export directly to TensorRT engine.

---

**Besides efficiency, are there any other benefits to using Condor/TensorRT?**

A few to mention.

* YOLOv10 support: as of writing, Frigate does not support YOLOv10, but Condor does.
* int8 model support: ONNX Runtime CUDA EP has weak support for int8 quantized models. Condor can run quantized TRT engines without issue.
* TUI with fine-graned metrics across the inference lifecycle per frame.

---

**How many ZMQ detectors (Frigate side) / workers (Condor side) should I run?**

I recommend running two detectors/workers in order to max out GPU utilization and throughput. You'll still get all of TensorRT's efficiency with a single detector, but since Frigate's detector process is essentially single-threaded and synchronous, a single detector can never fully utilize any GPU (there will always be GPU idle as pre/postprocessing is done on CPU, as well as other overhead).

Note that running two detectors may result in slightly increased reported detector latency metrics when GPU utilization is high, compared to a single detector scenario. However, total throughput (how many detector FPS you get out of your GPU) will be higher, which is what we care about. 

**Why is TensorRT more efficient than ONNX Runtime/CUDA EP??** 

Condor is a ZMQ worker and orchestration wrapper around TensorRT. In short, TensorRT builds an engine (a compiled artifact derived from a specific ONNX or PyTorch model) that has been tuned to provide the best performance for your particular hardware. A model architecture describes a computational graph, but for any given architecture there exist numerous low-level approaches to executing that graph on hardware. TensorRT provides numerous building blocks, called kernels, that execute layers defined in the model architecture such that compute is better utilized compared to a more generic approach. Among other things, the implementation of fused layer kernels can greatly improve compute efficiency in many models. When building the engine, TensorRT experiments with various kernel implementations, layer fusion approaches, and scheduling regimes to find the combination that maximizes efficiency for a given hardware platform.

ONNX Runtime, with its CUDA EP, also executes your model using efficient kernels, but the scope of optimization is less aggressive compared to TensorRT. 

Condor also has improved efficiency on the CPU side of things by running multiple inference threads in a single process while synchronizing the GPU utilization (i.e. only one thread is waiting on GPU at a time, meanwhile the others can transfer data between host and device, post-process the model output, communicate with frigate etc). This results in very high GPU utilization and maximum FPS with minimum resources on the host side. You can expect to max out any GPU with about ~1 CPU core utilized, and about a half gig of host memory. Frigate's ONNX detector requires using multiple detector processes in order to max out GPU utilization, each of which comes with over a gig memory footprint. 
---

## Automated installer

Run the installer from your Frigate project root (the directory containing your `docker-compose.yml`). It will:

1. Add the `condor` service to `docker-compose.yml`
2. Write a starter `condor/config.yaml` into your Frigate config directory
3. Install a `condor` shell command that opens the TUI inside the running container
4. Add `zmq` detector entries to your Frigate `config.yaml`

All modified files are backed up (`.bak`) before being changed.

**Run The Install Script**

```bash
curl -fsSL https://raw.githubusercontent.com/lansing/condor/master/scripts/compose_install.py \
    -o /tmp/condor_install.py && python3 /tmp/condor_install.py
```

**Same, with uv:**

```bash
uv run https://raw.githubusercontent.com/lansing/condor/master/scripts/compose_install.py
```

### Installer Options

For a typical install you probably don't need to specify any of these, but in case you do:

```
--frigate-service NAME   Compose service name for Frigate
--models-dir PATH        Host path for model files
--bin-dir PATH           Directory to install the 'condor' launcher
--port N                 ZMQ base port (default: 5555)
--num-workers N          Number of condor workers (default: 1)
```

Other flags: `--dry-run`, `-y/--yes`

---

### Manual installation

Alternatively, you can wire everything in by hand. The steps below mirror exactly what the installer does.

#### 1. Add the condor service to `docker-compose.yml`

Add the following service block. Adjust the volume host paths (`./models`, `./config/condor`, `./run`) to match your layout:

```yaml
services:
  condor:
    image: ghcr.io/lansing/condor:latest
    runtime: nvidia
    restart: unless-stopped
    volumes:
      - ./models:/app/models        # host models dir → container
      - ./config/condor:/app/config # condor config dir → container
      - ./run:/run/condor           # stats socket bind-mount
    environment:
      - CONDOR_STATS_SOCKET=/run/condor/metrics.sock
    healthcheck:
      test:
        - CMD-SHELL
        - "python3 -c 'import socket,sys; s=socket.socket(); s.settimeout(2); sys.exit(s.connect_ex((\"localhost\",5555)))'"
      interval: 5s
      timeout: 3s
      retries: 12
      start_period: 15s
```

Also add a `depends_on` entry to your Frigate service so it waits for condor to be healthy before starting:

```yaml
  frigate:
    # ... your existing frigate config ...
    depends_on:
      condor:
        condition: service_healthy
```

If your Frigate service already has a `depends_on` list (rather than a map), convert it to the long-form map syntax first:

```yaml
    # Before (list form):
    depends_on:
      - mosquitto

    # After (map form, add condor):
    depends_on:
      mosquitto:
        condition: service_started
      condor:
        condition: service_healthy
```

Create the `run/` directory (used for the stats socket bind-mount):

```bash
mkdir -p run
```

#### 2. Write the condor config

Create `config/condor/config.yaml` (adjust paths and settings to match your hardware):

```yaml
# condor configuration
# Models are read from ./models on the host, mounted to /app/models inside the container.
# Frigate sends the model filename in each inference request.

server:
  base_port: 5555
  num_workers: 1        # increase to match your Frigate detector count
  models_dir: /app/models

inference:
  provider: tensorrt
  provider_options:
    device: 0           # CUDA device index (0 = first GPU)
  max_inference_concurrency: 1

post_process:
  confidence_threshold: 0.5
  max_detections: 20

observability:
  enabled: true
  mode: tui             # stats socket only; use 'condor' command to inspect
  service_name: condor
```

#### 3. Install the `condor` TUI launcher (optional)

This is a small shell script that runs `condor-tui` inside the container so you can monitor condor from any terminal without remembering the full `docker compose exec` command.

Create `~/.local/bin/condor` (or any directory on your `PATH`):

```sh
#!/bin/sh
# Edit COMPOSE_FILE if you move your project directory.
COMPOSE_FILE="/absolute/path/to/your/docker-compose.yml"
exec docker compose -f "$COMPOSE_FILE" exec condor condor-tui "$@"
```

Make it executable:

```bash
chmod 755 ~/.local/bin/condor
```

If `~/.local/bin` is not on your `PATH`, add it to your shell profile:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

#### 4. Add condor detector entries to Frigate's `config.yaml`

In your Frigate `config/config.yaml`, add a `detectors` section (or add to an existing one):

```yaml
detectors:
  condor:
    type: zmq
    endpoint: tcp://condor:5555
```

For multiple workers (matching `num_workers` in condor's config), add one entry per worker with incrementing ports:

```yaml
detectors:
  condor_0:
    type: zmq
    endpoint: tcp://condor:5555
  condor_1:
    type: zmq
    endpoint: tcp://condor:5556
```

#### 5. Start the stack

```bash
docker compose down && docker compose up -d
```

#### 6. Monitor condor

If you installed the TUI launcher:

```bash
condor
```

Otherwise:

```bash
docker compose exec condor condor-tui
```

---

## Build a TensorRT engine from your ONNX model

condor requires a TensorRT `.engine` file, and it cannot serve ONNX models directly.

>If you're looking for a first model to test Condor with, I recommend using [pytorch-wildlife-onnx](https://github.com/lansing/pytorch-wildlife-onnx) to export a MegaDetector V6 YOLOv10 model into TensorRT `.engine` format. This is what I use with Condor.

We offer some scripts below to convert an ONNX model using NVIDIA's TensorRT base image. The builder image is only needed once, when you convert your model. This size of this image (10+ GB) is one reason we don't include this functionality in Condor.

Once you have an engine file, update your Frigate `config.yaml` to reference it by filename. If you previously pointed Frigate at an ONNX file, change the extension:

```yaml
model:
  path: /models/your-model.engine  # was: your-model.onnx
```

### TensorRT version compatibility

A `.engine` file is compiled for a specific TensorRT version and GPU architecture. Generally speaking, it cannot be loaded by a different version of TensorRT, and it may not be portable between GPU generations. **Build your engine on the same machine that will run condor.**

As of this release, Condor uses **TensorRT 26.01** and `.engine` files must be built using this version of TensorRT, to be used with Condor.

### Convert an ONNX model

To simplify matters slightly, we provide the following `onnx2engine.sh` script. This script will examine the currently installed Condor container image to ensure the right TensorRT builder image is used, and the convert the specified ONNX model to TRT engine.

Run the following (condor must be installed and its image present locally), replacing the model path appropriately:

```bash
curl -fsSL https://raw.githubusercontent.com/lansing/condor/master/scripts/onnx2engine.sh \
    | bash -s -- models/your-model.onnx
```

This produces `models/your-model.engine` alongside the ONNX file. To specify a different output path:

```bash
curl -fsSL https://raw.githubusercontent.com/lansing/condor/master/scripts/onnx2engine.sh \
    | bash -s -- models/your-model.onnx models/your-model.engine
```

Options:
- `--no-fp16`: build an FP32 engine (FP16 is the default and recommended for most GPUs)
- `--` followed by any arguments: passed directly to `trtexec` (e.g. for dynamic shapes)

## AI Use Disclosure

Much of the code in this project was written using AI assistance. In particular, the TUI, ZMQ handler and installer utilities were essentially "vibe coded". The core inference orchestration (`tensorrt_backend.py`) was prototyped using AI assistance, then rewritten and optimized by hand. This README was written by a human.
