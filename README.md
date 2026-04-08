# condor

Remote TensorRT detector for [Frigate NVR](https://frigate.video). Runs inference on a dedicated GPU and exposes it to Frigate via the ZMQ remote detector protocol.

## Install

### Automated installer

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

Requires Python 3.10+ and Docker.

#### Installer assumptions

The installer auto-detects your setup by reading `docker-compose.yml`. It assumes:

| What | Assumption |
|---|---|
| Working directory | Your Frigate project root (contains `docker-compose.yml`) |
| Frigate service | A compose service whose `image:` name contains the word `frigate` |
| Models directory | A volume in the Frigate service mapped to `/models` inside the container (e.g. `./models:/models`) |
| Config directory | A volume mapped to `/config` inside the container (e.g. `./config:/config`); defaults to `./config` if not found |
| Frigate config file | `<config_dir>/config.yaml` (or `.yml`) |

If any assumption doesn't hold, the installer will print what it expected and exit with a helpful message. Use the flags below to override:

```
--frigate-service NAME   Compose service name for Frigate
--models-dir PATH        Host path for model files
--bin-dir PATH           Directory to install the 'condor' launcher
--port N                 ZMQ base port (default: 5555)
--num-workers N          Number of condor workers (default: 1)
```

Other flags: `--dry-run`, `-y/--yes`, `--no-backup`, `--force`, `--no-compose`, `--no-config`, `--no-tui`, `--no-detector`.

---

### Manual installation

If the installer's assumptions don't match your setup, you can wire everything in by hand. The steps below mirror exactly what the installer does.

#### Prerequisites

- Docker with the NVIDIA container runtime (`nvidia` runtime available in Docker)
- A Frigate deployment managed by `docker-compose.yml`
- A `models/` directory on the host containing your TensorRT engine files

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
