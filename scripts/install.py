#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["ruamel.yaml>=0.18"]
# ///
"""
condor installer — convert your ONNX model to a TensorRT engine, then wire
condor into a Frigate docker-compose setup.

Usage (recommended):
    curl -fsSL https://raw.githubusercontent.com/lansing/condor/master/scripts/install.py \
        -o /tmp/condor_install.py && python3 /tmp/condor_install.py

Usage (with uv):
    uv run https://raw.githubusercontent.com/lansing/condor/master/scripts/install.py

Usage (from repo):
    python3 scripts/install.py

Note: 'uv run python scripts/install.py' does NOT work — use
'uv run scripts/install.py' instead (PEP-723 standalone script mode).

Installer assumptions (auto-detected from docker-compose.yml):
    - Run from your Frigate project root (directory containing docker-compose.yml)
    - Frigate service image name contains the word 'frigate'
    - Frigate mounts a models directory to /models inside the container
    - Frigate mounts a config directory to /config inside the container
    - Frigate's config file is at <config_dir>/config.yaml

Phases (all enabled by default):

    [0] convert   Convert ONNX model → TensorRT engine
    [1] compose   Add condor service + depends_on to docker-compose.yml
    [2] config    Write starter condor config.yaml
    [3] tui       Install 'condor' TUI launcher
    [4] detector  Patch Frigate config: add zmq detectors + update model.path

All file modifications are backed up before editing (filename.ext.bak).
"""

from __future__ import annotations

import argparse
import atexit
import importlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ── YAML backend bootstrap ─────────────────────────────────────────────────────

_YAML_BACKEND: str = ""
_YAML_TEMP_DIR: Optional[Path] = None


def _cleanup_yaml_temp() -> None:
    global _YAML_TEMP_DIR
    if _YAML_TEMP_DIR and _YAML_TEMP_DIR.exists():
        shutil.rmtree(_YAML_TEMP_DIR, ignore_errors=True)
        _YAML_TEMP_DIR = None


def _bootstrap_yaml() -> None:
    global _YAML_BACKEND, _YAML_TEMP_DIR

    try:
        import ruamel.yaml  # noqa: F401
        _YAML_BACKEND = "ruamel"
        return
    except ImportError:
        pass

    tmp = Path(tempfile.mkdtemp(prefix="condor_deps_"))
    _YAML_TEMP_DIR = tmp
    atexit.register(_cleanup_yaml_temp)

    print(
        f"  {_c(_CYAN, 'Note:')} Installing ruamel.yaml into a temporary directory for the\n"
        f"  installer's use only.  It will be removed when the installer exits.\n"
        f"    Location: {_path(tmp)}\n"
    )

    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install",
             "--quiet", "--target", str(tmp), "ruamel.yaml"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        sys.path.insert(0, str(tmp))
        importlib.invalidate_caches()
        import ruamel.yaml  # noqa: F401
        _YAML_BACKEND = "ruamel"
        return
    except Exception:
        _cleanup_yaml_temp()

    try:
        import yaml  # noqa: F401
        _YAML_BACKEND = "pyyaml"
        return
    except ImportError:
        pass

    print(
        "error: no YAML library available.\n"
        "  Install ruamel.yaml:  pip install ruamel.yaml\n"
        "  Or install uv and run: uv run https://raw.githubusercontent.com/"
        "lansing/condor/master/scripts/install.py",
        file=sys.stderr,
    )
    sys.exit(1)


# ── Visual constants ───────────────────────────────────────────────────────────

_TAGLINE = "condor — TensorRT sidecar for Frigate"
_HR      = "─" * 56
_BOLD    = "\033[1m"
_DIM     = "\033[2m"
_CYAN    = "\033[36m"
_GREEN   = "\033[32m"
_YELLOW  = "\033[33m"
_RED     = "\033[31m"
_RESET   = "\033[0m"

_SCRIPT_URL = (
    "https://raw.githubusercontent.com/lansing/condor/master/scripts/install.py"
)

# ── Installer constants ────────────────────────────────────────────────────────

TENSORRT_IMAGE      = "nvcr.io/nvidia/tensorrt:26.01-py3"
CONDOR_IMAGE        = "ghcr.io/lansing/condor:latest"
CONDOR_SERVICE_NAME = "condor"
CONDOR_ZMQ_PORT     = 5555
CONDOR_STATS_DIR    = "/run/condor"
CONDOR_STATS_SOCKET = "/run/condor/metrics.sock"
CONDOR_MODELS_DIR   = "/app/models"
CONDOR_CONFIG_DIR   = "/app/config"

_FRIGATE_IMAGE_RE = re.compile(r"frigate", re.IGNORECASE)


# ── YAML helpers ───────────────────────────────────────────────────────────────

def _yaml_load(path: Path) -> tuple[Any, dict]:
    if _YAML_BACKEND == "ruamel":
        from ruamel.yaml import YAML
        y = YAML()
        y.preserve_quotes = True
        y.width = 120
        with path.open() as f:
            return y, y.load(f)
    else:
        import yaml
        with path.open() as f:
            return yaml, yaml.safe_load(f)


def _yaml_dump(dumper: Any, data: dict, dest: Any) -> None:
    if _YAML_BACKEND == "ruamel":
        dumper.dump(data, dest)
    else:
        import yaml
        yaml.dump(data, dest, default_flow_style=False, allow_unicode=True)


def _new_map(d: dict | None = None) -> dict:
    if _YAML_BACKEND == "ruamel":
        from ruamel.yaml.comments import CommentedMap
        return CommentedMap(d or {})
    return dict(d or {})


def _new_seq(items: list | None = None) -> list:
    if _YAML_BACKEND == "ruamel":
        from ruamel.yaml.comments import CommentedSeq
        return CommentedSeq(items or [])
    return list(items or [])


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class InstallContext:
    compose_file: Path

    frigate_service_name: str
    models_dir: str
    frigate_config_dir: str

    bin_dir: str
    frigate_config_file: Path
    num_workers: int = 2

    condor_port: int = CONDOR_ZMQ_PORT
    dry_run: bool    = False
    backup: bool     = True
    yes: bool        = False
    force: bool      = False

    device: int           = 0
    onnx_file: Optional[Path] = None   # ONNX to convert (None = skip conversion)
    engine_file: Optional[Path] = None  # engine to use for model.path update
    fp16: bool = True

    @property
    def condor_config_host_dir(self) -> str:
        return os.path.join(self.frigate_config_dir, "condor")

    @property
    def run_dir(self) -> str:
        return str(self.compose_file.parent / "run")

    @property
    def condor_launcher_path(self) -> Path:
        return Path(self.bin_dir) / "condor"


# ── File helpers ───────────────────────────────────────────────────────────────

def _backup(path: Path, dry_run: bool) -> Optional[Path]:
    if not path.exists():
        return None
    bak = path.with_suffix(path.suffix + ".bak")
    if not dry_run:
        shutil.copy2(path, bak)
    return bak


# ── UI helpers ─────────────────────────────────────────────────────────────────

def _c(code: str, text: str) -> str:
    if sys.stdout.isatty():
        return f"{code}{text}{_RESET}"
    return text


def print_banner() -> None:
    print()
    print(_c(_BOLD, f"  {_TAGLINE}"))
    print(f"  {_HR}")
    print()


def _bullet(text: str) -> str:
    return f"    {_c(_CYAN, '✦')} {text}"


def _path(text: Any) -> str:
    return _c(_DIM, str(text))


def _open_tty() -> Optional[Any]:
    if sys.stdin.isatty():
        return None
    try:
        return open("/dev/tty", "r")
    except OSError:
        return None


def _prompt(question: str, default: str = "", tty: Optional[Any] = None) -> str:
    placeholder = f" [{_c(_BOLD, default)}]" if default else ""
    sys.stdout.write(f"    {question}{placeholder}: ")
    sys.stdout.flush()
    src = tty if tty is not None else sys.stdin
    try:
        line = src.readline()
        if not line:
            raise EOFError
        answer = line.strip()
    except EOFError:
        print(default)
        return default
    return answer or default


def _confirm(prompt: str = "Proceed?", tty: Optional[Any] = None) -> bool:
    src = tty if tty is not None else (sys.stdin if sys.stdin.isatty() else None)

    if src is None:
        print(
            f"\n  {_c(_RED, 'Cannot prompt for confirmation:')} stdin is not a terminal\n"
            "  and /dev/tty is unavailable.\n\n"
            "  Options:\n"
            "    • Download and run directly (recommended):\n"
            f"        curl -fsSL {_SCRIPT_URL} -o /tmp/condor_install.py\n"
            "        python3 /tmp/condor_install.py\n"
            "    • Pass -y / --yes to accept all defaults non-interactively.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    sys.stdout.write(f"\n  {_c(_BOLD, prompt)} [y/N]: ")
    sys.stdout.flush()
    try:
        line = src.readline()
        if not line:
            raise EOFError
        answer = line.strip().lower()
    except EOFError:
        print()
        print(f"  {_c(_YELLOW, 'Aborted.')} No input received.\n")
        sys.exit(0)

    return answer in ("y", "yes")


def _print_block(text: str, indent: str = "        ") -> None:
    for line in text.rstrip("\n").splitlines():
        print(_c(_DIM, indent + line))


def _pyyaml_warning() -> None:
    if _YAML_BACKEND == "pyyaml":
        print(_c(_YELLOW,
            "  ⚠  ruamel.yaml unavailable — falling back to PyYAML.\n"
            "     Comments in edited YAML files will be stripped.\n"
            "     To preserve comments: pip install ruamel.yaml\n"
        ))


# ── GPU detection ──────────────────────────────────────────────────────────────

def detect_gpus() -> list[dict]:
    """Return [{index, name}] for each NVIDIA GPU via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        gpus = []
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(", ", 1)
            if len(parts) == 2:
                gpus.append({"index": int(parts[0].strip()), "name": parts[1].strip()})
        return gpus
    except Exception:
        return []


def gpu_guard(device: int) -> None:
    """Print GPU selection info or warnings; exit if device index is invalid."""
    gpus = detect_gpus()

    if len(gpus) == 0:
        print(
            f"  {_c(_YELLOW, '⚠  No NVIDIA GPU detected.')} The installer will continue,\n"
            "     but condor and the engine conversion require a CUDA-capable GPU.\n"
            "     Make sure nvidia-smi is in your PATH and drivers are installed.\n"
        )
        return

    if device >= len(gpus):
        names = ", ".join(f"{g['index']}: {g['name']}" for g in gpus)
        print(
            f"  {_c(_RED, f'error: --device {device} is out of range.')} "
            f"Available GPUs: {names}",
            file=sys.stderr,
        )
        sys.exit(1)

    selected = gpus[device]

    if len(gpus) == 1:
        print(
            f"  {_c(_CYAN, 'GPU:')}  {_c(_BOLD, selected['name'])} "
            f"(device {selected['index']})\n"
        )
    else:
        others = ", ".join(
            f"{g['index']}: {g['name']}" for g in gpus if g["index"] != device
        )
        print(
            f"  {_c(_CYAN, 'GPU:')}  Using device {selected['index']}: "
            f"{_c(_BOLD, selected['name'])}\n"
            f"         Other available: {others}\n"
            f"         Pass {_c(_DIM, f'--device N')} to use a different GPU.\n"
        )


# ── Discovery ──────────────────────────────────────────────────────────────────

def _volume_parts(vol: Any) -> tuple[str, str] | None:
    if isinstance(vol, str):
        parts = vol.split(":")
        if len(parts) >= 2:
            return parts[0], parts[1]
    elif isinstance(vol, dict):
        src, tgt = vol.get("source", ""), vol.get("target", "")
        if src and tgt:
            return src, tgt
    return None


def detect_frigate_service(services: dict) -> str:
    for name, svc in services.items():
        if svc and isinstance(svc, dict):
            if _FRIGATE_IMAGE_RE.search(str(svc.get("image", ""))):
                return name
    raise RuntimeError(
        "Could not auto-detect the Frigate service. "
        "Use --frigate-service NAME to specify it."
    )


def detect_models_dir(svc: dict) -> str:
    for vol in svc.get("volumes", []):
        parts = _volume_parts(vol)
        if not parts:
            continue
        host, target = parts
        if target.rstrip("/") == "/models" or host.rstrip("/").endswith("models"):
            return host
    raise RuntimeError(
        "Could not auto-detect the models directory. "
        "Use --models-dir PATH to specify it."
    )


def detect_config_dir(svc: dict) -> str:
    for vol in svc.get("volumes", []):
        parts = _volume_parts(vol)
        if not parts:
            continue
        host, target = parts
        if target.rstrip("/") == "/config":
            return host
    return "./config"


def detect_frigate_config_file(frigate_config_dir: str, compose_file: Path) -> Path:
    cfg_dir = (compose_file.parent / frigate_config_dir).resolve()
    for name in ("config.yaml", "config.yml"):
        candidate = cfg_dir / name
        if candidate.exists():
            return candidate
    return cfg_dir / "config.yaml"


def detect_bin_dir() -> str:
    local_bin = Path.home() / ".local" / "bin"
    if local_bin.exists():
        return str(local_bin)
    return "/usr/local/bin"


# ── ONNX / engine discovery ────────────────────────────────────────────────────

def _resolve_model_path(models_dir_host: str, compose_file: Path,
                        frigate_cfg: Path) -> Optional[Path]:
    """Read model.path from frigate config, map /models/... → host path."""
    if not frigate_cfg.exists():
        return None
    try:
        _, data = _yaml_load(frigate_cfg)
        model_path = (data or {}).get("model", {})
        if isinstance(model_path, dict):
            model_path = model_path.get("path", "")
        else:
            model_path = ""
        if not model_path:
            return None
        # map /models/<file> → <models_dir_host>/<file>
        container_models = "/models/"
        if str(model_path).startswith(container_models):
            rel = str(model_path)[len(container_models):]
            host = (compose_file.parent / models_dir_host / rel).resolve()
            return host
    except Exception:
        pass
    return None


def find_onnx_from_frigate(models_dir_host: str, compose_file: Path,
                           frigate_cfg: Path) -> Optional[Path]:
    """Return the .onnx that frigate config currently points at, or None."""
    p = _resolve_model_path(models_dir_host, compose_file, frigate_cfg)
    if p and p.suffix == ".onnx" and p.exists():
        return p
    return None


def find_engine_files(models_dir_host: str, compose_file: Path,
                      preferred: Optional[Path] = None) -> list[Path]:
    """Glob *.engine in models_dir; preferred (if given) comes first."""
    mdir = (compose_file.parent / models_dir_host).resolve()
    engines = sorted(mdir.glob("*.engine"), key=lambda p: p.stat().st_mtime, reverse=True)
    if preferred and preferred in engines:
        engines.remove(preferred)
        engines.insert(0, preferred)
    return engines


def _container_model_path(host_engine: Path, models_dir_host: str,
                           compose_file: Path) -> str:
    """Map a host engine path back to its /models/... container path."""
    mdir = (compose_file.parent / models_dir_host).resolve()
    try:
        rel = host_engine.resolve().relative_to(mdir)
        return f"/models/{rel}"
    except ValueError:
        return f"/models/{host_engine.name}"


# ── Plan building ──────────────────────────────────────────────────────────────

def build_plan(args: argparse.Namespace, compose_file: Path,
               tty: Optional[Any] = None) -> InstallContext:
    _, data = _yaml_load(compose_file)
    services: dict = data.get("services") or {}
    if not services:
        sys.exit("error: no 'services' key found in compose file")

    # ── auto-detect Frigate layout ─────────────────────────────────────────────
    try:
        frigate_name = args.frigate_service or detect_frigate_service(services)
    except RuntimeError as e:
        print(f"\n  {_c(_RED, 'Auto-detection failed:')} {e}\n")
        sys.exit(1)

    frigate_svc = services[frigate_name]

    try:
        models_dir = args.models_dir or detect_models_dir(frigate_svc)
    except RuntimeError as e:
        print(f"\n  {_c(_RED, 'Auto-detection failed:')} {e}\n")
        sys.exit(1)

    config_dir = detect_config_dir(frigate_svc)
    frigate_cfg_default = detect_frigate_config_file(config_dir, compose_file)

    compose_dir      = compose_file.parent
    run_dir          = compose_dir / "run"
    run_rel          = "./run"
    condor_cfg_path  = Path(config_dir) / "condor" / "config.yaml"
    config_condor_rel = os.path.join(config_dir, "condor")
    if not config_condor_rel.startswith("."):
        config_condor_rel = "./" + config_condor_rel

    print(f"  {_c(_BOLD, 'The following steps will be performed:')}\n")

    # ── Phase [0]: ONNX → engine conversion ───────────────────────────────────
    onnx_file: Optional[Path]   = None
    engine_file: Optional[Path] = None

    if args.convert:
        print(f"  {_c(_BOLD, '[0] Engine conversion')}")

        if args.onnx:
            candidate = Path(args.onnx).resolve()
            if not candidate.exists():
                print(f"      {_c(_RED, 'error:')} ONNX file not found: {candidate}")
                sys.exit(1)
            onnx_file = candidate
        else:
            onnx_file = find_onnx_from_frigate(models_dir, compose_file, frigate_cfg_default)

        if onnx_file:
            expected_engine = onnx_file.with_suffix(".engine")
            print(_bullet(f"Convert {_path(onnx_file.name)} → {_path(expected_engine.name)}"))
            print(_bullet(f"TensorRT image: {_path(TENSORRT_IMAGE)}"))
            prec = "FP16" if args.fp16 else "FP32"
            print(_bullet(f"Precision: {prec}  |  GPU device: {args.device}"))
            if expected_engine.exists():
                print(f"      {_c(_YELLOW, '⚠')}  {expected_engine.name} already exists — "
                      "will be overwritten.")
            engine_file = expected_engine
        else:
            print(f"      {_c(_YELLOW, '⚠')}  No ONNX model found in Frigate config.")
            print(f"         Pass {_c(_DIM, '--onnx PATH')} to specify one, or")
            print(f"         {_c(_DIM, '--no-convert')} to skip conversion.")
            if not args.convert_only:
                print()
            else:
                sys.exit(1)
        print()

    # ── Engine selection (for install phases) ──────────────────────────────────
    if not args.convert_only:
        # Determine which engine the install phases will use
        if args.engine:
            engine_file = Path(args.engine).resolve()
            if not engine_file.exists():
                print(f"  {_c(_RED, 'error:')} engine file not found: {engine_file}")
                sys.exit(1)
        elif engine_file is None and not args.no_engine:
            # No conversion planned — scan for existing engines
            engines = find_engine_files(models_dir, compose_file)
            if not engines:
                print(
                    f"  {_c(_RED, '✗')}  No .engine file found in {_path(models_dir)}/\n\n"
                    "  condor requires a TensorRT engine file, not an ONNX model.\n"
                    "  Convert your ONNX model first:\n\n"
                    f"    {_c(_DIM, 'python3 /tmp/condor_install.py --convert-only')}\n\n"
                    "  Then re-run without --no-convert.\n"
                )
                sys.exit(1)
            elif len(engines) == 1:
                engine_file = engines[0]
                print(f"  {_c(_BOLD, 'Engine file:')}")
                print(_bullet(f"Using {_path(engine_file.name)}"))
                print()
            else:
                print(f"  {_c(_BOLD, 'Engine file')}  (select one):\n")
                for i, e in enumerate(engines, 1):
                    tag = "  [recommended — newest]" if i == 1 else ""
                    print(f"      {_c(_CYAN, str(i))}) {_path(e.name)}{tag}")
                print()
                if args.yes:
                    choice = 1
                    print(f"      Auto-selecting: {_path(engines[0].name)}")
                else:
                    raw = _prompt("      Select", "1", tty=tty)
                    try:
                        choice = int(raw)
                    except ValueError:
                        choice = 1
                    choice = max(1, min(choice, len(engines)))
                engine_file = engines[choice - 1]
                print()

    # ── Phase [1]: compose ─────────────────────────────────────────────────────
    if args.compose and not args.convert_only:
        print(f"  {_c(_BOLD, '[1] docker-compose.yml')}")
        print(_bullet(f"Add {_c(_CYAN, CONDOR_IMAGE)} service"))
        print(_bullet(
            f"Add {_c(_CYAN, 'depends_on: condor')} "
            f"(condition: service_healthy) to '{frigate_name}'"
        ))
        print(_bullet(f"Create {_path(run_dir)}/"))
        print()
        print(f"      Service block preview:")
        print()
        _print_block(f"""\
condor:
  image: {CONDOR_IMAGE}
  runtime: nvidia
  restart: unless-stopped
  volumes:
    - {models_dir}:{CONDOR_MODELS_DIR}
    - {config_condor_rel}:{CONDOR_CONFIG_DIR}
    - {run_rel}:{CONDOR_STATS_DIR}
  environment:
    - CONDOR_STATS_SOCKET={CONDOR_STATS_SOCKET}
  healthcheck:
    test: [CMD-SHELL, "python3 -c 'import socket,sys; ...tcp:{args.port}...'"]
    interval: 5s  timeout: 3s  retries: 12  start_period: 15s
""")
        print()

    # ── Phase [2]: condor config ───────────────────────────────────────────────
    if args.config and not args.convert_only:
        print(f"  {_c(_BOLD, '[2] Condor config')}")
        print(_bullet("Write starter condor/config.yaml"))
        print(f"      File:   {_path(condor_cfg_path)}")
        if condor_cfg_path.exists():
            print(f"      {_c(_YELLOW, '⚠')}  File already exists — "
                  "will be skipped (use --force to overwrite).")
        print()

    # ── Phase [3]: TUI launcher ────────────────────────────────────────────────
    bin_dir = args.bin_dir
    if args.tui and not args.convert_only:
        print(f"  {_c(_BOLD, '[3] TUI launcher')}")
        print(_bullet(f"Install {_c(_CYAN, 'condor')} command"))
        default_bin = bin_dir or detect_bin_dir()
        if args.yes:
            bin_dir = default_bin
            print(f"      Install to: {_path(Path(default_bin) / 'condor')}")
        else:
            bin_dir = _prompt("      Install to", default_bin, tty=tty)
        print()

    # ── Phase [4]: Frigate detector config ────────────────────────────────────
    frigate_cfg_file: Optional[Path] = None
    if args.detector and not args.convert_only:
        print(f"  {_c(_BOLD, '[4] Frigate detector config')}")
        if args.yes:
            frigate_cfg_file = frigate_cfg_default
            print(f"      File:   {_path(frigate_cfg_file)}")
        else:
            raw = _prompt("      Edit file", str(frigate_cfg_default), tty=tty)
            frigate_cfg_file = Path(raw)

        # Show existing non-zmq detectors that will be removed
        _fcfg_for_scan = frigate_cfg_file or frigate_cfg_default
        _existing_non_zmq: list[str] = []
        if _fcfg_for_scan.exists():
            try:
                _, _det_data = _yaml_load(_fcfg_for_scan)
                _dets = (_det_data or {}).get("detectors") or {}
                _existing_non_zmq = [
                    k for k, v in _dets.items()
                    if isinstance(v, dict) and v.get("type") != "zmq"
                ]
            except Exception:
                pass
        if _existing_non_zmq:
            print(_bullet(
                f"{_c(_YELLOW, 'Remove')} existing non-ZMQ detectors: "
                f"{_c(_BOLD, ', '.join(_existing_non_zmq))}"
            ))
        print(_bullet(
            f"Add zmq detector entries "
            f"(port {args.port}, {args.num_workers} worker"
            f"{'s' if args.num_workers > 1 else ''})"
        ))
        print()

        if engine_file and not args.no_engine:
            _fcfg = frigate_cfg_file or frigate_cfg_default
            try:
                _, _fdata = _yaml_load(_fcfg)
                current_str = str((_fdata or {}).get("model", {}).get("path", "(not set)"))
            except Exception:
                current_str = "(not set)"
            new_container_path = _container_model_path(engine_file, models_dir, compose_file)
            print(f"  {_c(_BOLD, '[4b] Frigate model path')}")
            print(_bullet(f"Update model.path in {_path((frigate_cfg_file or frigate_cfg_default).name)}"))
            print(f"        From: {_path(current_str)}")
            print(f"        To:   {_path(new_container_path)}")
            print()

    # ── Backup notice ──────────────────────────────────────────────────────────
    files_to_backup: list[tuple[Path, Path]] = []
    compose_bak = compose_file.with_suffix(compose_file.suffix + ".bak")

    if args.compose and compose_file.exists() and not args.convert_only:
        files_to_backup.append((compose_file, compose_bak))
    if args.tui and bin_dir:
        launcher = Path(bin_dir) / "condor"
        if launcher.exists():
            files_to_backup.append((launcher, launcher.with_name("condor.bak")))
    if args.config and condor_cfg_path.exists() and args.force and not args.convert_only:
        files_to_backup.append((
            condor_cfg_path,
            condor_cfg_path.with_suffix(condor_cfg_path.suffix + ".bak"),
        ))
    if args.detector and frigate_cfg_file and frigate_cfg_file.exists() and not args.convert_only:
        files_to_backup.append((
            frigate_cfg_file,
            frigate_cfg_file.with_suffix(frigate_cfg_file.suffix + ".bak"),
        ))

    if files_to_backup:
        print(f"  {_c(_YELLOW + _BOLD, '⚠  Backups')}  "
              f"The following files will be backed up before modification:\n")
        for src, bak in files_to_backup:
            print(f"    {_path(src)}")
            print(f"      → {_path(bak)}")
        print()
    elif not args.convert_only:
        print(f"  {_c(_DIM, 'No existing files will be overwritten (all new).')}\n")

    resolved_bin_dir    = bin_dir or detect_bin_dir()
    resolved_frigate_cfg = frigate_cfg_file or frigate_cfg_default

    return InstallContext(
        compose_file=compose_file,
        frigate_service_name=frigate_name,
        models_dir=models_dir,
        frigate_config_dir=config_dir,
        bin_dir=resolved_bin_dir,
        frigate_config_file=resolved_frigate_cfg,
        num_workers=args.num_workers,
        condor_port=args.port,
        dry_run=args.dry_run,
        backup=args.backup,
        yes=args.yes,
        force=args.force,
        device=args.device,
        onnx_file=onnx_file,
        engine_file=engine_file,
        fp16=args.fp16,
    )


# ── Phase [0]: ONNX → engine conversion ───────────────────────────────────────

def convert_onnx(ctx: InstallContext) -> None:
    print(f"\n  {_c(_BOLD, '[0]')} Converting ONNX model to TensorRT engine …\n")
    onnx = ctx.onnx_file
    engine = ctx.engine_file

    if onnx is None:
        print(f"      {_c(_YELLOW, '⚠')}  No ONNX file specified — skipping conversion.")
        return

    print(f"      Input:   {_path(onnx)}")
    print(f"      Output:  {_path(engine)}")
    print(f"      Image:   {_path(TENSORRT_IMAGE)}")
    prec = "FP16" if ctx.fp16 else "FP32"
    print(f"      Prec:    {prec}  |  GPU: device {ctx.device}")
    print()
    print(
        f"      {_c(_YELLOW, 'Note:')} Engine build time depends on your GPU:\n"
        "        Fast (RTX 3080/4080/4090 class):   3–6 min\n"
        "        Mid  (RTX 2080 Ti / A4000 class):  4–8 min\n"
        "        Slow (RTX A400 / mobile / older):  10–25 min\n"
        "      trtexec output will stream below.\n"
    )

    if ctx.dry_run:
        print("      DRY RUN — would run docker trtexec here.")
        return

    onnx_dir = onnx.parent
    engine_dir = engine.parent
    engine_dir.mkdir(parents=True, exist_ok=True)

    if onnx_dir == engine_dir:
        mounts = ["-v", f"{onnx_dir}:/workspace"]
        onnx_c   = f"/workspace/{onnx.name}"
        engine_c = f"/workspace/{engine.name}"
    else:
        mounts = ["-v", f"{onnx_dir}:/input:ro", "-v", f"{engine_dir}:/output"]
        onnx_c   = f"/input/{onnx.name}"
        engine_c = f"/output/{engine.name}"

    trt_args = [f"--onnx={onnx_c}", f"--saveEngine={engine_c}"]
    if ctx.fp16:
        trt_args.append("--fp16")

    cmd = [
        "docker", "run", "--rm",
        f"--gpus", f"device={ctx.device}",
        *mounts,
        TENSORRT_IMAGE,
        "trtexec", *trt_args,
    ]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n  {_c(_RED, 'error:')} trtexec failed (exit {result.returncode}).")
        sys.exit(result.returncode)

    if not engine.exists():
        print(f"\n  {_c(_RED, 'error:')} Engine file not found after conversion: {engine}")
        sys.exit(1)

    print(f"\n      {_c(_GREEN, '✓')} Engine saved: {_path(engine)}")


# ── Phase [1]: docker-compose ──────────────────────────────────────────────────

def _build_condor_service(ctx: InstallContext) -> dict:
    compose_dir = ctx.compose_file.parent
    run_rel     = "./" + os.path.relpath(ctx.run_dir, compose_dir).replace("\\", "/")
    config_rel  = ctx.condor_config_host_dir
    if not config_rel.startswith("."):
        config_rel = "./" + config_rel

    svc = _new_map()
    svc["image"]   = CONDOR_IMAGE
    svc["runtime"] = "nvidia"
    svc["restart"] = "unless-stopped"

    svc["volumes"] = _new_seq([
        f"{ctx.models_dir}:{CONDOR_MODELS_DIR}",
        f"{config_rel}:{CONDOR_CONFIG_DIR}",
        f"{run_rel}:{CONDOR_STATS_DIR}",
    ])
    svc["environment"] = _new_seq([
        f"CONDOR_STATS_SOCKET={CONDOR_STATS_SOCKET}",
    ])

    hc = _new_map()
    hc["test"] = _new_seq([
        "CMD-SHELL",
        (
            f"python3 -c 'import socket,sys; s=socket.socket();"
            f" s.settimeout(2); sys.exit(s.connect_ex((\"localhost\",{ctx.condor_port})))'"
        ),
    ])
    hc["interval"]     = "5s"
    hc["timeout"]      = "3s"
    hc["retries"]      = 12
    hc["start_period"] = "15s"
    svc["healthcheck"] = hc

    return svc


def _ensure_depends_on(frigate_svc: dict) -> None:
    dep_entry = _new_map({"condition": "service_healthy"})
    existing  = frigate_svc.get("depends_on")

    if existing is None:
        frigate_svc["depends_on"] = _new_map({CONDOR_SERVICE_NAME: dep_entry})
    elif isinstance(existing, list):
        if CONDOR_SERVICE_NAME not in existing:
            upgraded = _new_map({n: _new_map({"condition": "service_started"})
                                  for n in existing})
            upgraded[CONDOR_SERVICE_NAME] = dep_entry
            frigate_svc["depends_on"] = upgraded
    elif isinstance(existing, dict):
        if CONDOR_SERVICE_NAME not in existing:
            existing[CONDOR_SERVICE_NAME] = dep_entry


def install_compose(ctx: InstallContext) -> None:
    print(f"\n  {_c(_BOLD, '[1]')} Updating docker-compose.yml …")
    yaml_obj, data = _yaml_load(ctx.compose_file)
    services: dict = data.get("services") or {}

    if CONDOR_SERVICE_NAME in services:
        print(f"      {_c(_YELLOW, '⚠')}  '{CONDOR_SERVICE_NAME}' service already present — skipping.")
        return

    new_services = _new_map({CONDOR_SERVICE_NAME: _build_condor_service(ctx)})
    for k, v in services.items():
        new_services[k] = v
    data["services"] = new_services

    _ensure_depends_on(new_services[ctx.frigate_service_name])

    if ctx.dry_run:
        print("      DRY RUN — proposed condor service block:")
        dummy = _new_map({"services": _new_map(
            {CONDOR_SERVICE_NAME: _build_condor_service(ctx)}
        )})
        _yaml_dump(yaml_obj, dummy, sys.stdout)
        return

    bak = _backup(ctx.compose_file, ctx.dry_run)
    if bak:
        print(f"      {_c(_DIM, f'Backed up → {bak}')}")

    with ctx.compose_file.open("w") as f:
        _yaml_dump(yaml_obj, data, f)

    run_dir = Path(ctx.run_dir)
    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"      {_c(_GREEN, '✓')} Created {run_dir}/")

    print(f"      {_c(_GREEN, '✓')} condor service added")
    print(f"      {_c(_GREEN, '✓')} depends_on wired to '{ctx.frigate_service_name}'")


# ── Phase [2]: condor config ───────────────────────────────────────────────────

_CONDOR_CONFIG_TEMPLATE = textwrap.dedent("""\
    # condor configuration — generated by install.py
    # Edit this file to match your model and hardware.

    server:
      base_port: {port}
      num_workers: {num_workers}        # match your Frigate detector count
      models_dir: /app/models

    inference:
      provider: tensorrt
      provider_options:
        device: {device}           # CUDA device index
      max_inference_concurrency: 1

    post_process:
      confidence_threshold: 0.5
      max_detections: 20
""")


def install_config(ctx: InstallContext) -> None:
    print(f"\n  {_c(_BOLD, '[2]')} Writing condor config.yaml …")
    config_dir  = Path(ctx.condor_config_host_dir)
    config_file = config_dir / "config.yaml"

    if config_file.exists() and not ctx.force:
        print(f"      {_c(_YELLOW, '⚠')}  {config_file} already exists — skipping "
              "(use --force to overwrite).")
        return

    content = _CONDOR_CONFIG_TEMPLATE.format(
        port=ctx.condor_port,
        num_workers=ctx.num_workers,
        device=ctx.device,
    )

    if ctx.dry_run:
        print("      DRY RUN — would write:")
        print(textwrap.indent(content, "        "))
        return

    bak = _backup(config_file, ctx.dry_run)
    if bak:
        print(f"      {_c(_DIM, f'Backed up → {bak}')}")

    config_dir.mkdir(parents=True, exist_ok=True)
    config_file.write_text(content)
    print(f"      {_c(_GREEN, '✓')} Wrote {config_file}")


# ── Phase [3]: TUI launcher ────────────────────────────────────────────────────

_CONDOR_LAUNCHER_TEMPLATE = textwrap.dedent("""\
    #!/bin/sh
    # condor — attach to condor-tui inside the running container.
    # Generated by condor install.py — edit COMPOSE_FILE if you move things.
    COMPOSE_FILE="{compose_file}"
    exec docker compose -f "$COMPOSE_FILE" exec -it \\
      -e TERM="${{TERM:-xterm-256color}}" \\
      -e COLORTERM=truecolor \\
      condor condor-tui "$@"
""")


def install_tui(ctx: InstallContext) -> None:
    print(f"\n  {_c(_BOLD, '[3]')} Installing 'condor' TUI launcher …")
    launcher = ctx.condor_launcher_path
    bin_dir  = Path(ctx.bin_dir)

    content = _CONDOR_LAUNCHER_TEMPLATE.format(
        compose_file=str(ctx.compose_file.resolve()),
    )

    if ctx.dry_run:
        print(f"      DRY RUN — would write to {launcher}")
        return

    bak = _backup(launcher, ctx.dry_run)
    if bak:
        print(f"      {_c(_DIM, f'Backed up → {bak}')}")

    bin_dir.mkdir(parents=True, exist_ok=True)
    launcher.write_text(content)
    launcher.chmod(0o755)
    print(f"      {_c(_GREEN, '✓')} Wrote {launcher} (chmod 755)")

    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    resolved  = [str(Path(p).resolve()) for p in path_dirs if p]
    if str(bin_dir.resolve()) not in resolved:
        print(f"      {_c(_YELLOW, '⚠')}  {bin_dir} is not on your PATH.")
        print(f"         Add to your shell profile:")
        print(f"         {_c(_DIM, f'export PATH=\"{bin_dir}:$PATH\"')}")


# ── Phase [4]: Frigate detector config ────────────────────────────────────────

def install_detector(ctx: InstallContext) -> None:
    print(f"\n  {_c(_BOLD, '[4]')} Patching Frigate detector config …")
    cfg_file = ctx.frigate_config_file

    if not cfg_file.exists():
        print(f"      {_c(_YELLOW, '⚠')}  {cfg_file} not found — skipping.")
        return

    yaml_obj, data = _yaml_load(cfg_file)

    if "detectors" not in data or data["detectors"] is None:
        data["detectors"] = _new_map()

    detectors: dict = data["detectors"]

    # Remove existing non-zmq detectors (they can't load a .engine file)
    non_zmq = [k for k, v in detectors.items()
                if isinstance(v, dict) and v.get("type") != "zmq"]

    new_entries: dict[str, dict] = {}
    for i in range(ctx.num_workers):
        name  = f"condor_{i}" if ctx.num_workers > 1 else "condor"
        port  = ctx.condor_port + i
        entry = _new_map({"type": "zmq",
                           "endpoint": f"tcp://{CONDOR_SERVICE_NAME}:{port}"})
        new_entries[name] = entry

    conflicts = [k for k in new_entries if k in detectors]
    if conflicts:
        print(f"      {_c(_YELLOW, '⚠')}  condor entries already present: {conflicts} — skipping.")
        return

    if ctx.dry_run:
        if non_zmq:
            print(f"      DRY RUN — would remove detectors: {', '.join(non_zmq)}")
        print(f"      DRY RUN — would add zmq detector entries to {cfg_file}")
        return

    bak = _backup(cfg_file, ctx.dry_run)
    if bak:
        print(f"      {_c(_DIM, f'Backed up → {bak}')}")

    for k in non_zmq:
        del detectors[k]
        print(f"      {_c(_GREEN, '✓')} Removed detector '{k}'")

    for name, entry in new_entries.items():
        detectors[name] = entry

    with cfg_file.open("w") as f:
        _yaml_dump(yaml_obj, data, f)

    for name in new_entries:
        print(f"      {_c(_GREEN, '✓')} Added detector '{name}'")


# ── Phase [4b]: Update model.path ─────────────────────────────────────────────

def install_model_path(ctx: InstallContext) -> None:
    print(f"\n  {_c(_BOLD, '[4b]')} Updating Frigate model.path …")
    cfg_file = ctx.frigate_config_file
    engine   = ctx.engine_file

    if engine is None:
        print(f"      {_c(_YELLOW, '⚠')}  No engine file selected — skipping.")
        return

    if not cfg_file.exists():
        print(f"      {_c(_YELLOW, '⚠')}  {cfg_file} not found — skipping.")
        return

    new_container_path = _container_model_path(engine, ctx.models_dir, ctx.compose_file)

    yaml_obj, data = _yaml_load(cfg_file)
    if data is None:
        data = _new_map()

    model_section = data.get("model")
    if model_section is None:
        model_section = _new_map()
        data["model"] = model_section

    current = model_section.get("path", "(not set)")
    if str(current) == new_container_path:
        print(f"      {_c(_DIM, 'model.path already set to')} {_path(new_container_path)} — skipping.")
        return
    if str(current).endswith(".engine") and not ctx.force:
        print(f"      {_c(_YELLOW, '⚠')}  model.path already points at an engine: {_path(current)}")
        print(f"         Use --force to overwrite.")
        return

    if ctx.dry_run:
        print(f"      DRY RUN — would change model.path:")
        print(f"        From: {_path(current)}")
        print(f"        To:   {_path(new_container_path)}")
        return

    model_section["path"] = new_container_path

    with cfg_file.open("w") as f:
        _yaml_dump(yaml_obj, data, f)

    print(f"      {_c(_GREEN, '✓')} model.path updated")
    print(f"        From: {_path(current)}")
    print(f"        To:   {_path(new_container_path)}")


# ── Post-install summary ───────────────────────────────────────────────────────

def print_summary(ctx: InstallContext, phases_run: dict) -> None:
    print(f"\n  {_HR}")
    print(f"  {_c(_BOLD, 'Changes made:')}\n")

    if phases_run.get("convert") and ctx.engine_file:
        print(f"    {_c(_GREEN, '✓')} {_path(ctx.engine_file)}")
        print(f"         TensorRT engine built from {ctx.onnx_file.name if ctx.onnx_file else 'ONNX'}")
    if phases_run.get("compose"):
        print(f"    {_c(_GREEN, '✓')} {_path(ctx.compose_file)}")
        print(f"         condor service added; depends_on wired")
    if phases_run.get("config"):
        config_file = Path(ctx.condor_config_host_dir) / "config.yaml"
        print(f"    {_c(_GREEN, '✓')} {_path(config_file)}")
        print(f"         starter condor config written")
    if phases_run.get("tui"):
        print(f"    {_c(_GREEN, '✓')} {_path(ctx.condor_launcher_path)}")
        print(f"         'condor' TUI launcher installed")
    if phases_run.get("detector"):
        print(f"    {_c(_GREEN, '✓')} {_path(ctx.frigate_config_file)}")
        print(f"         zmq detector entries added")
    if phases_run.get("model_path"):
        print(f"    {_c(_GREEN, '✓')} {_path(ctx.frigate_config_file)}")
        print(f"         model.path updated to {ctx.engine_file.name if ctx.engine_file else ''}")

    print(f"\n  {_HR}")
    print(f"  {_c(_BOLD, 'Next steps:')}\n")

    steps = []
    steps.append(
        f"Restart the stack:\n"
        f"         {_c(_DIM, f'docker compose -f {ctx.compose_file} down')}\n"
        f"         {_c(_DIM, f'docker compose -f {ctx.compose_file} up -d')}"
    )
    if phases_run.get("tui"):
        steps.append(
            f"Monitor condor:\n"
            f"         {_c(_DIM, 'condor')}"
        )
    else:
        steps.append(
            f"Monitor condor:\n"
            f"         {_c(_DIM, 'docker compose exec condor condor-tui')}"
        )

    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step}\n")

    print(f"  {_HR}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("compose_file", nargs="?", default=None)

    conv = p.add_argument_group("conversion options (phase [0])")
    conv.add_argument("--convert-only", action="store_true",
                      help="Only convert ONNX → engine; skip condor install")
    conv.add_argument("--no-convert", action="store_false", dest="convert",
                      help="Skip conversion; go straight to condor install")
    conv.add_argument("--onnx", metavar="PATH",
                      help="ONNX file to convert (auto-detected from Frigate config if omitted)")
    conv.add_argument("--no-fp16", dest="fp16", action="store_false",
                      help="Build FP32 engine (default: FP16)")
    p.set_defaults(convert=True, fp16=True)

    phases = p.add_argument_group("install phase control (all enabled by default)")
    phases.add_argument("--no-compose",  dest="compose",  action="store_false")
    phases.add_argument("--no-config",   dest="config",   action="store_false")
    phases.add_argument("--no-tui",      dest="tui",      action="store_false")
    phases.add_argument("--no-detector", dest="detector", action="store_false")
    phases.add_argument("--no-engine",   dest="no_engine", action="store_true",
                        help="Skip engine auto-detection and model.path update")
    p.set_defaults(compose=True, config=True, tui=True, detector=True, no_engine=False)

    over = p.add_argument_group("detection overrides")
    over.add_argument("--frigate-service", metavar="NAME")
    over.add_argument("--models-dir",      metavar="PATH")
    over.add_argument("--bin-dir",         metavar="PATH")
    over.add_argument("--engine",          metavar="PATH",
                      help="Engine file to use (skips auto-detection)")
    over.add_argument("--port",            type=int, default=CONDOR_ZMQ_PORT, metavar="N")
    over.add_argument("--num-workers",     type=int, default=2, metavar="N")
    over.add_argument("--device",          type=int, default=0, metavar="N",
                      help="CUDA device index for engine build and condor (default: 0)")

    p.add_argument("--dry-run",   action="store_true")
    p.add_argument("-y", "--yes", action="store_true")
    p.add_argument("--no-backup", dest="backup", action="store_false")
    p.add_argument("--force",     action="store_true")

    return p


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # Locate compose file
    if args.compose_file:
        compose_file = Path(args.compose_file).resolve()
        if not compose_file.exists():
            sys.exit(f"error: compose file not found: {compose_file}")
    else:
        for name in ("docker-compose.yml", "docker-compose.yaml",
                     "compose.yml", "compose.yaml"):
            candidate = Path.cwd() / name
            if candidate.exists():
                compose_file = candidate
                break
        else:
            sys.exit(
                "error: no compose file found in current directory.\n"
                "       Tried: docker-compose.yml, docker-compose.yaml, "
                "compose.yml, compose.yaml\n"
                "       Run from your Frigate project root."
            )

    print_banner()
    _bootstrap_yaml()
    _pyyaml_warning()

    # GPU guard — runs before anything else
    gpu_guard(args.device)

    tty = _open_tty()
    try:
        ctx = build_plan(args, compose_file, tty=tty)

        if not args.dry_run and not args.yes:
            prompt = "Proceed with conversion only?" if args.convert_only else "Proceed with installation?"
            if not _confirm(prompt, tty=tty):
                print(f"\n  {_c(_YELLOW, 'Aborted.')} No files were changed.\n")
                sys.exit(0)

        print()
        phases_run: dict[str, bool] = {}

        if args.convert and ctx.onnx_file:
            convert_onnx(ctx)
            phases_run["convert"] = True

        if args.convert_only:
            print(f"\n  {_c(_GREEN, '✓')} Conversion complete.")
            if ctx.engine_file:
                print(f"     Engine: {_path(ctx.engine_file)}")
            print(f"\n  Run without {_c(_DIM, '--convert-only')} to complete the condor install.\n")
            return

        if args.compose:
            install_compose(ctx)
            phases_run["compose"] = True

        if args.config:
            install_config(ctx)
            phases_run["config"] = True

        if args.tui:
            install_tui(ctx)
            phases_run["tui"] = True

        if args.detector:
            install_detector(ctx)
            phases_run["detector"] = True

        if args.detector and ctx.engine_file and not args.no_engine:
            install_model_path(ctx)
            phases_run["model_path"] = True

        if args.dry_run:
            print(f"\n  {_c(_YELLOW, 'Dry run complete — no files were written.')}\n")
        else:
            print_summary(ctx, phases_run)

    finally:
        if tty:
            tty.close()


if __name__ == "__main__":
    main()
