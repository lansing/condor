#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["ruamel.yaml>=0.18"]
# ///
"""
condor installer — wire condor into a Frigate docker-compose setup.

Usage (with uv — preferred, handles dependencies automatically):
    uv run https://raw.githubusercontent.com/lansing/condor/master/scripts/compose_install.py

Usage (with plain python3 — self-bootstraps ruamel.yaml via pip):
    curl -fsSL https://raw.githubusercontent.com/lansing/condor/master/scripts/compose_install.py | python3 -
    python3 scripts/compose_install.py         # from repo

Note: 'uv run python scripts/compose_install.py' does NOT work — it runs the
script inside the project venv (which lacks ruamel.yaml) instead of treating
it as a standalone PEP-723 script.  Use 'uv run scripts/compose_install.py'.

Phases (all enabled by default):

    [1] compose   Add condor service + depends_on to docker-compose.yml
    [2] config    Write starter condor config.yaml into <frigate-config>/condor/
    [3] tui       Install 'condor' command that opens condor-tui in the container
    [4] detector  Patch Frigate's config.yaml to add zmq detector entries

All file modifications are backed up before editing.
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# ── YAML backend bootstrap ─────────────────────────────────────────────────────
# Must run before any YAML imports.  Sets _YAML_BACKEND to "ruamel" or "pyyaml".

_YAML_BACKEND: str = ""


def _bootstrap_yaml() -> None:
    global _YAML_BACKEND

    # 1. Already importable (common when running inside the condor project venv,
    #    or the user already has it installed).
    try:
        import ruamel.yaml  # noqa: F401
        _YAML_BACKEND = "ruamel"
        return
    except ImportError:
        pass

    # 2. Try to install ruamel.yaml into a temp directory so we don't pollute
    #    the user's environment.  Requires pip (comes with every CPython).
    tmp = Path(tempfile.mkdtemp(prefix="condor_deps_"))
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install",
             "--quiet", "--target", str(tmp), "ruamel.yaml"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        sys.path.insert(0, str(tmp))
        importlib.invalidate_caches()
        import ruamel.yaml  # noqa: F401
        _YAML_BACKEND = "ruamel"
        return
    except Exception:
        pass

    # 3. Fall back to PyYAML (stdlib-adjacent; often pre-installed on Debian/Ubuntu).
    #    This works but strips comments from any YAML files we edit.
    try:
        import yaml  # noqa: F401
        _YAML_BACKEND = "pyyaml"
        return
    except ImportError:
        pass

    # 4. Nothing worked.
    print(
        "error: no YAML library available.\n"
        "  Install ruamel.yaml:  pip install ruamel.yaml\n"
        "  Or install uv and run: uv run https://raw.githubusercontent.com/"
        "lansing/condor/master/scripts/compose_install.py",
        file=sys.stderr,
    )
    sys.exit(1)


_bootstrap_yaml()


# ── Visual constants ───────────────────────────────────────────────────────────

_LOGO = """\
 ██████╗ ██████╗ ███╗   ██╗██████╗  ██████╗ ██████╗
██╔════╝██╔═══██╗████╗  ██║██╔══██╗██╔═══██╗██╔══██╗
██║     ██║   ██║██╔██╗ ██║██║  ██║██║   ██║██████╔╝
██║     ██║   ██║██║╚██╗██║██║  ██║██║   ██║██╔══██╗
╚██████╗╚██████╔╝██║ ╚████║██████╔╝╚██████╔╝██║  ██║
 ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═════╝  ╚═════╝ ╚═╝  ╚═╝"""

_TAGLINE = "Remote TensorRT detector for Frigate"
_HR     = "─" * 56
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_CYAN   = "\033[36m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_RESET  = "\033[0m"

# ── Installer constants ────────────────────────────────────────────────────────

CONDOR_IMAGE        = "ghcr.io/lansing/condor:latest"
CONDOR_SERVICE_NAME = "condor"
CONDOR_ZMQ_PORT     = 5555
CONDOR_STATS_DIR    = "/run/condor"
CONDOR_STATS_SOCKET = "/run/condor/metrics.sock"
CONDOR_MODELS_DIR   = "/app/models"
CONDOR_CONFIG_DIR   = "/app/config"

_FRIGATE_IMAGE_RE = re.compile(r"frigate", re.IGNORECASE)

# ── YAML backend abstraction ───────────────────────────────────────────────────
# All YAML I/O goes through these four functions so that the rest of the script
# is backend-agnostic.  ruamel preserves comments; PyYAML strips them (warned).


def _yaml_load(path: Path) -> tuple[Any, dict]:
    """Load a YAML file.  Returns (dumper_token, data)."""
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
    """Write data to an open file (or stdout) using the active backend."""
    if _YAML_BACKEND == "ruamel":
        dumper.dump(data, dest)
    else:
        import yaml
        yaml.dump(data, dest, default_flow_style=False, allow_unicode=True)


def _new_map(d: dict | None = None) -> dict:
    """Return a new ordered mapping, using CommentedMap when available."""
    if _YAML_BACKEND == "ruamel":
        from ruamel.yaml.comments import CommentedMap
        return CommentedMap(d or {})
    return dict(d or {})


def _new_seq(items: list | None = None) -> list:
    """Return a new sequence, using CommentedSeq when available."""
    if _YAML_BACKEND == "ruamel":
        from ruamel.yaml.comments import CommentedSeq
        return CommentedSeq(items or [])
    return list(items or [])


# ── Data classes ───────────────────────────────────────────────────────────────


@dataclass
class InstallContext:
    """All resolved paths and settings, confirmed interactively before execution."""

    compose_file: Path

    # Auto-detected from compose file.
    frigate_service_name: str
    models_dir: str           # host-side, e.g. "./models"
    frigate_config_dir: str   # host-side, e.g. "./config"

    # Resolved interactively or via CLI.
    bin_dir: str              # where to install the 'condor' launcher
    frigate_config_file: Path # frigate's config.yaml to patch for detectors
    num_workers: int = 1      # condor workers → number of zmq detector entries

    condor_port: int = CONDOR_ZMQ_PORT
    dry_run: bool    = False
    backup: bool     = True
    yes: bool        = False
    force: bool      = False

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
    """Copy path → path.bak.  Returns the backup path, or None if path doesn't exist."""
    if not path.exists():
        return None
    bak = path.with_suffix(path.suffix + ".bak")
    if not dry_run:
        shutil.copy2(path, bak)
    return bak


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
    """Prefer ~/.local/bin (non-invasive, no sudo); fall back to /usr/local/bin."""
    local_bin = Path.home() / ".local" / "bin"
    if local_bin.exists():
        return str(local_bin)
    return "/usr/local/bin"


# ── UI helpers ─────────────────────────────────────────────────────────────────


def _c(code: str, text: str) -> str:
    if sys.stdout.isatty():
        return f"{code}{text}{_RESET}"
    return text


def print_banner() -> None:
    print()
    print(_c(_CYAN, _LOGO))
    print()
    if _YAML_BACKEND == "pyyaml":
        print(_c(_YELLOW,
            "  ⚠  ruamel.yaml unavailable — falling back to PyYAML.\n"
            "     Comments in edited YAML files will be stripped.\n"
            "     To preserve comments: pip install ruamel.yaml\n"
        ))
    print(_c(_BOLD, f"  {_TAGLINE}"))
    print(f"  {_HR}")
    print()


def _bullet(text: str) -> str:
    return f"    {_c(_CYAN, '✦')} {text}"


def _path(text: Any) -> str:
    return _c(_DIM, str(text))


def _prompt(question: str, default: str = "") -> str:
    """Single-line prompt; returns default on Enter or EOFError (non-interactive)."""
    placeholder = f" [{_c(_BOLD, default)}]" if default else ""
    try:
        answer = input(f"    {question}{placeholder}: ").strip()
    except EOFError:
        print(default)
        return default
    return answer or default


def _confirm(prompt: str = "Proceed with installation?") -> bool:
    try:
        answer = input(f"\n  {_c(_BOLD, prompt)} [Y/n]: ").strip().lower()
    except EOFError:
        return True
    return answer in ("", "y", "yes")


# ── Interactive planning ───────────────────────────────────────────────────────


def build_plan(args: argparse.Namespace, compose_file: Path) -> InstallContext:
    """
    Auto-detect paths from the compose file, show the full plan with prompts
    for any user-configurable locations, and return a resolved InstallContext.
    """
    _, data = _yaml_load(compose_file)
    services: dict = data.get("services") or {}
    if not services:
        sys.exit("error: no 'services' key found in compose file")

    try:
        frigate_name = args.frigate_service or detect_frigate_service(services)
    except RuntimeError as e:
        sys.exit(f"error: {e}")

    frigate_svc = services[frigate_name]

    try:
        models_dir = args.models_dir or detect_models_dir(frigate_svc)
    except RuntimeError as e:
        sys.exit(f"error: {e}")

    config_dir = detect_config_dir(frigate_svc)

    # ── Plan display ───────────────────────────────────────────────────────────
    condor_config_path = Path(config_dir) / "condor" / "config.yaml"
    run_dir            = compose_file.parent / "run"
    condor_config_bak  = condor_config_path.with_suffix(
        condor_config_path.suffix + ".bak"
    ) if condor_config_path.exists() else None
    compose_bak = compose_file.with_suffix(compose_file.suffix + ".bak")

    print(f"  {_c(_BOLD, 'The following steps will be performed:')}\n")

    if args.compose:
        print(f"  {_c(_BOLD, '[1] docker-compose.yml')}")
        print(_bullet(f"Add {_c(_CYAN, CONDOR_IMAGE)} service"))
        print(_bullet(f"Add {_c(_CYAN, 'depends_on: condor')} "
                      f"(condition: service_healthy) to '{frigate_name}'"))
        print(_bullet(f"Create {_path(run_dir)}/ for stats socket bind-mount"))
        print(f"      File:   {_path(compose_file)}")
        print(f"      Backup: {_path(compose_bak)}")
        print()

    if args.config:
        print(f"  {_c(_BOLD, '[2] Condor config')}")
        print(_bullet("Write starter config.yaml with sensible defaults"))
        print(f"      File:   {_path(condor_config_path)}")
        if condor_config_bak:
            print(f"      Backup: {_path(condor_config_bak)}")
        else:
            print(f"      {_c(_DIM, '(new file — no backup needed)')}")
        print()

    # Phase 3: bin dir prompt.
    bin_dir = args.bin_dir
    if args.tui:
        print(f"  {_c(_BOLD, '[3] TUI launcher')}")
        print(_bullet(
            f"Install {_c(_CYAN, 'condor')} command "
            f"— opens condor-tui inside the running container"
        ))
        default_bin = bin_dir or detect_bin_dir()
        if args.yes or not sys.stdout.isatty():
            bin_dir = default_bin
            print(f"      Install to: {_path(Path(bin_dir) / 'condor')}")
        else:
            bin_dir = _prompt("      Install to", default_bin)
            launcher = Path(bin_dir) / "condor"
            if launcher.exists():
                print(f"      File:   {_path(launcher)}")
                print(f"      Backup: {_path(launcher.with_name('condor.bak'))}")
            else:
                print(f"      File:   {_path(launcher)}")
        print()

    # Phase 4: frigate config file prompt.
    frigate_cfg_file: Optional[Path] = None
    if args.detector:
        frigate_cfg_default = detect_frigate_config_file(config_dir, compose_file)
        print(f"  {_c(_BOLD, '[4] Frigate detector config')}")
        print(_bullet(
            f"Add zmq detector entries pointing to condor "
            f"(port {args.port}, {args.num_workers} worker"
            f"{'s' if args.num_workers > 1 else ''})"
        ))
        if args.yes or not sys.stdout.isatty():
            frigate_cfg_file = frigate_cfg_default
            print(f"      File:   {_path(frigate_cfg_file)}")
        else:
            raw = _prompt("      Edit file", str(frigate_cfg_default))
            frigate_cfg_file = Path(raw)
            if frigate_cfg_file.exists():
                print(f"      Backup: {_path(frigate_cfg_file.with_suffix(frigate_cfg_file.suffix + '.bak'))}")
            else:
                print(f"      {_c(_YELLOW, '⚠  file not found — will skip if missing at run time')}")
        print()

    return InstallContext(
        compose_file=compose_file,
        frigate_service_name=frigate_name,
        models_dir=models_dir,
        frigate_config_dir=config_dir,
        bin_dir=bin_dir or detect_bin_dir(),
        frigate_config_file=frigate_cfg_file or detect_frigate_config_file(
            config_dir, compose_file
        ),
        num_workers=args.num_workers,
        condor_port=args.port,
        dry_run=args.dry_run,
        backup=args.backup,
        yes=args.yes,
        force=args.force,
    )


# ── Phase 1: docker-compose ────────────────────────────────────────────────────


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
    dep_entry         = _new_map({"condition": "service_healthy"})
    existing          = frigate_svc.get("depends_on")

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

    # Prepend condor before all existing services.
    new_services = _new_map({CONDOR_SERVICE_NAME: _build_condor_service(ctx)})
    for k, v in services.items():
        new_services[k] = v
    data["services"] = new_services

    _ensure_depends_on(new_services[ctx.frigate_service_name])

    if ctx.dry_run:
        print("      DRY RUN — proposed condor service block:")
        print()
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


# ── Phase 2: condor config ─────────────────────────────────────────────────────

_CONDOR_CONFIG_TEMPLATE = textwrap.dedent("""\
    # condor configuration — generated by compose_install.py
    # Edit this file to match your model and hardware.
    #
    # Models are read from {models_dir} on the host, mounted to /app/models inside
    # the container. Frigate sends the model filename in each inference request.

    server:
      base_port: {port}
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
        models_dir=ctx.models_dir,
        port=ctx.condor_port,
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


# ── Phase 3: TUI launcher ──────────────────────────────────────────────────────

_CONDOR_LAUNCHER_TEMPLATE = textwrap.dedent("""\
    #!/bin/sh
    # condor — attach to condor-tui inside the running Frigate/condor container.
    # Generated by condor compose_install.py — edit COMPOSE_FILE if you move things.
    COMPOSE_FILE="{compose_file}"
    exec docker compose -f "$COMPOSE_FILE" exec condor condor-tui "$@"
""")


def install_tui(ctx: InstallContext) -> None:
    print(f"\n  {_c(_BOLD, '[3]')} Installing 'condor' TUI launcher …")
    launcher = ctx.condor_launcher_path
    bin_dir  = Path(ctx.bin_dir)

    content = _CONDOR_LAUNCHER_TEMPLATE.format(
        compose_file=str(ctx.compose_file.resolve()),
    )

    if ctx.dry_run:
        print(f"      DRY RUN — would write to {launcher}:")
        print(textwrap.indent(content, "        "))
        return

    bak = _backup(launcher, ctx.dry_run)
    if bak:
        print(f"      {_c(_DIM, f'Backed up → {bak}')}")

    bin_dir.mkdir(parents=True, exist_ok=True)
    launcher.write_text(content)
    launcher.chmod(0o755)
    print(f"      {_c(_GREEN, '✓')} Wrote {launcher} (chmod 755)")

    # Warn if the chosen dir isn't on PATH.
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    resolved  = [str(Path(p).resolve()) for p in path_dirs if p]
    if str(bin_dir.resolve()) not in resolved:
        print(f"      {_c(_YELLOW, '⚠')}  {bin_dir} is not on your PATH.")
        print(f"         Add to your shell profile:")
        export_line = f'export PATH="{bin_dir}:$PATH"'
        print(f"         {_c(_DIM, export_line)}")


# ── Phase 4: Frigate detector config ──────────────────────────────────────────


def install_detector(ctx: InstallContext) -> None:
    print(f"\n  {_c(_BOLD, '[4]')} Patching Frigate detector config …")
    cfg_file = ctx.frigate_config_file

    if not cfg_file.exists():
        print(f"      {_c(_YELLOW, '⚠')}  {cfg_file} not found — skipping.")
        print("         Create it first, then re-run with --no-compose --no-config --no-tui.")
        return

    yaml_obj, data = _yaml_load(cfg_file)

    if "detectors" not in data or data["detectors"] is None:
        data["detectors"] = _new_map()

    detectors: dict = data["detectors"]

    # Build new detector entries (one per worker).
    new_entries: dict[str, dict] = {}
    for i in range(ctx.num_workers):
        name  = f"condor_{i}" if ctx.num_workers > 1 else "condor"
        port  = ctx.condor_port + i
        entry = _new_map({"type": "zmq",
                           "endpoint": f"tcp://{CONDOR_SERVICE_NAME}:{port}"})
        new_entries[name] = entry

    conflicts = [k for k in new_entries if k in detectors]
    if conflicts:
        print(f"      {_c(_YELLOW, '⚠')}  Entries already present: {conflicts} — skipping.")
        print("         Remove them manually and re-run to replace.")
        return

    if ctx.dry_run:
        print(f"      DRY RUN — would add to detectors in {cfg_file}:")
        for name, entry in new_entries.items():
            print(f"        {name}:")
            for k, v in entry.items():
                print(f"          {k}: {v}")
        return

    bak = _backup(cfg_file, ctx.dry_run)
    if bak:
        print(f"      {_c(_DIM, f'Backed up → {bak}')}")

    for name, entry in new_entries.items():
        detectors[name] = entry

    with cfg_file.open("w") as f:
        _yaml_dump(yaml_obj, data, f)

    for name in new_entries:
        print(f"      {_c(_GREEN, '✓')} Added detector '{name}'")


# ── Post-install summary ───────────────────────────────────────────────────────


def print_next_steps(ctx: InstallContext, phases: dict) -> None:
    steps = []

    if phases.get("detector"):
        steps.append(
            "Verify the detector entries in your Frigate config "
            f"({ctx.frigate_config_file.name}) and confirm the model path is set."
        )

    steps.append(
        f"Start the stack:\n"
        f"         {_c(_DIM, f'docker compose -f {ctx.compose_file} up -d')}"
    )

    if phases.get("tui"):
        steps.append(
            f"Monitor condor (from any terminal):\n"
            f"         {_c(_DIM, 'condor')}"
        )
    else:
        steps.append(
            f"Monitor condor (from the compose directory):\n"
            f"         {_c(_DIM, 'docker compose exec condor condor-tui')}"
        )

    print(f"\n  {_HR}")
    print(f"  {_c(_BOLD, 'Next steps:')}\n")
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step}\n")
    print(f"  {_HR}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "compose_file", nargs="?", default="docker-compose.yml",
        help="Path to docker-compose.yml (default: ./docker-compose.yml)",
    )

    phases = p.add_argument_group("phase control (all enabled by default)")
    phases.add_argument("--no-compose",  dest="compose",  action="store_false",
                        help="Skip phase 1: compose file modification")
    phases.add_argument("--no-config",   dest="config",   action="store_false",
                        help="Skip phase 2: condor config.yaml generation")
    phases.add_argument("--no-tui",      dest="tui",      action="store_false",
                        help="Skip phase 3: 'condor' TUI launcher")
    phases.add_argument("--no-detector", dest="detector", action="store_false",
                        help="Skip phase 4: Frigate detector config patch")
    p.set_defaults(compose=True, config=True, tui=True, detector=True)

    over = p.add_argument_group("detection overrides")
    over.add_argument("--frigate-service", metavar="NAME",
                      help="Compose service name for Frigate (auto-detected)")
    over.add_argument("--models-dir", metavar="PATH",
                      help="Host path for model files (auto-detected from volumes)")
    over.add_argument("--bin-dir", metavar="PATH",
                      help="Directory to install the 'condor' launcher (prompted if omitted)")
    over.add_argument("--port", type=int, default=CONDOR_ZMQ_PORT, metavar="N",
                      help=f"ZMQ port (default: {CONDOR_ZMQ_PORT})")
    over.add_argument("--num-workers", type=int, default=1, metavar="N",
                      help="Number of condor workers / zmq detector entries (default: 1)")

    p.add_argument("--dry-run", action="store_true",
                   help="Show what would change without writing any files")
    p.add_argument("-y", "--yes", action="store_true",
                   help="Accept all detected defaults without prompting")
    p.add_argument("--no-backup", dest="backup", action="store_false",
                   help="Skip backing up files before modifying them")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing files (e.g. condor config.yaml)")

    return p


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    compose_file = Path(args.compose_file).resolve()
    if not compose_file.exists():
        sys.exit(f"error: compose file not found: {compose_file}")

    print_banner()

    ctx = build_plan(args, compose_file)

    if not args.dry_run and not args.yes:
        if not _confirm():
            print(f"\n  {_c(_YELLOW, 'Aborted.')} No files were changed.\n")
            sys.exit(0)

    print()
    phases_run: dict[str, bool] = {}

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

    if not args.dry_run:
        print_next_steps(ctx, phases_run)
    else:
        print(f"\n  {_c(_YELLOW, 'Dry run complete — no files were written.')}\n")


if __name__ == "__main__":
    main()
