# condor

Remote TensorRT detector for [Frigate NVR](https://frigate.video). Runs inference on a dedicated GPU and exposes it to Frigate via the ZMQ remote detector protocol.

## Install

Run the installer from your Frigate directory. It will add condor to your `docker-compose.yml`, drop a starter config, install the `condor` TUI command, and patch your Frigate detector config:

```bash
curl -fsSL https://raw.githubusercontent.com/lansing/condor/master/scripts/compose_install.py | python3 -
```

Requires Python 3.10+ and Docker. No other dependencies needed — the installer bootstraps its own.
