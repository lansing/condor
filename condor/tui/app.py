"""Condor metrics TUI — 90s BBS ANSI-art style.

Connects to the stats Unix socket at /tmp/condor-metrics.sock and displays
live metrics from the running Condor server.

Usage:
    condor-tui
    uv run condor-tui

Requires:
    uv sync --extra tui
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, ScrollableContainer
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Footer, Sparkline, Static

from ..stats import SOCKET_PATH as _DEFAULT_SOCKET_PATH
from .art import CONDOR_LOGO, build_combined_logo, get_provider_logo, get_bird_frame

# Allow override via env var so the host TUI can reach a socket that is
# bind-mounted from a running Docker container (see docker-compose.yaml).
SOCKET_PATH = os.environ.get("CONDOR_STATS_SOCKET", _DEFAULT_SOCKET_PATH)

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _ms(d: dict | None, field: str = "avg") -> str:
    if d is None:
        return "  ---"
    return f"{d[field]:6.1f}"


def _fmt_ms_row(d: dict | None) -> str:
    if d is None:
        return "    ---  ---  ---"
    return f"{d['avg']:6.1f}  {d['min']:6.1f}  {d['max']:6.1f}"


# ---------------------------------------------------------------------------
# Header widget
# ---------------------------------------------------------------------------


class HeaderWidget(Static):
    """Combined CONDOR logo (left) and provider logo (right) with flying bird animation."""

    DEFAULT_CSS = """
    HeaderWidget {
        height: 8;
        border: heavy $success;
        padding: 0 1;
        color: $success;
        background: $background;
    }
    """

    provider: reactive[str] = reactive("", layout=False)

    def __init__(self) -> None:
        super().__init__()
        self._bird_x = 5
        self._bird_y = 1
        self._bird_x_direction = 1
        self._bird_y_direction = 1
        self._animation_tick = 0

    def on_mount(self) -> None:
        """Start the animation loop."""
        self._start_animation()

    @work(exclusive=True)
    async def _start_animation(self) -> None:
        """Continuously animate the bird."""
        while True:
            self._animation_tick += 1

            # Move bird every 2 frames
            if self._animation_tick % 2 == 0:
                self._bird_x += self._bird_x_direction
                self._bird_y += self._bird_y_direction

                # Bounce off edges
                max_x = 20  # Bird moves in the gap between logos
                max_y = 5
                if self._bird_x <= 0 or self._bird_x >= max_x:
                    self._bird_x_direction *= -1
                if self._bird_y <= 0 or self._bird_y >= max_y:
                    self._bird_y_direction *= -1

            self.refresh()
            await asyncio.sleep(0.05)  # ~20 FPS

    def render(self) -> str:
        condor_lines = CONDOR_LOGO.split("\n")
        provider_lines = get_provider_logo(self.provider).split("\n")

        bird_frame = (self._animation_tick // 3) % 4

        return build_combined_logo(
            condor_lines,
            provider_lines,
            bird_x=self._bird_x,
            bird_y=self._bird_y,
            bird_frame=bird_frame,
        )

    def watch_provider(self) -> None:
        """Refresh when provider changes."""
        self.refresh()


# ---------------------------------------------------------------------------
# Status bar
# ---------------------------------------------------------------------------


class StatusBar(Static):
    DEFAULT_CSS = """
    StatusBar {
        height: 3;
        border: heavy $primary;
        color: $primary;
        content-align: center middle;
        background: $background;
        padding: 0 1;
    }
    """


# ---------------------------------------------------------------------------
# Sparkline panels
# ---------------------------------------------------------------------------


class GraphPanel(Widget):
    """Labeled sparkline panel with a live max y-scale label."""

    DEFAULT_CSS = """
    GraphPanel {
        height: 1fr;
        border: heavy $accent;
        padding: 0 1;
        background: $background;
    }
    GraphPanel > .title {
        height: 1;
        color: $accent;
        text-style: bold;
    }
    GraphPanel > Sparkline {
        height: 1fr;
    }
    GraphPanel > .summary {
        height: 1;
        color: $text-muted;
    }
    Sparkline > .sparkline--min-color {
        color: $success-darken-3;
    }
    Sparkline > .sparkline--max-color {
        color: $success;
    }
    """

    def __init__(self, title: str, unit: str, widget_id: str) -> None:
        super().__init__(id=widget_id)
        self._title = title
        self._unit = unit
        self._title_id = f"{widget_id}-title"
        self._spark_id = f"{widget_id}-spark"
        self._summary_id = f"{widget_id}-summary"

    def compose(self) -> ComposeResult:
        yield Static(f" ▶ {self._title}", id=self._title_id, classes="title")
        yield Sparkline([], id=self._spark_id, summary_function=max)
        yield Static("", id=self._summary_id, classes="summary")

    def update_data(self, data: list[float], summary: str) -> None:
        if not data:
            return
        # Show the current max value as a y-axis scale hint in the title
        peak = max(data)
        scale = f"{peak:.0f}"
        self.query_one(f"#{self._title_id}", Static).update(
            f" ▶ {self._title}  [dim]↑ {scale}[/dim]"
        )
        self.query_one(f"#{self._spark_id}", Sparkline).data = data
        self.query_one(f"#{self._summary_id}", Static).update(summary)


# ---------------------------------------------------------------------------
# Per-worker panel
# ---------------------------------------------------------------------------


class WorkerPanel(Static):
    """Displays stats for one worker thread."""

    DEFAULT_CSS = """
    WorkerPanel {
        width: 1fr;
        height: 100%;
        border: double $success;
        padding: 0 1;
        background: $background;
        color: $text;
    }
    """

    def __init__(self, worker_id: int, port: int) -> None:
        super().__init__(id=f"worker-panel-{worker_id}")
        self._worker_id = worker_id
        self._port = port
        self._data: dict = {}
        self._trt_data: dict = {}

    def update_data(self, wdata: dict, snapshot: dict) -> None:
        self._data = wdata
        self._trt_data = snapshot
        self.refresh()

    def render(self) -> str:  # type: ignore[override]
        d = self._data
        g = self._trt_data

        req = d.get("requests_total", 0)
        inf = d.get("inference_total", 0)
        rps = d.get("req_per_sec", 0.0)
        e2e = d.get("e2e_ms")
        infer = d.get("infer_ms")
        pp = d.get("postprocess_ms")
        h2d = g.get("global_trt_h2d_ms")

        lines = [
            f"[bold cyan]WORKER {self._worker_id}[/bold cyan]  [dim]:{self._port}[/dim]",
            "─" * 80,
            f"  Req   [green]{req:>7,}[/green]  [yellow]{rps:5.1f} rps[/yellow]",
            f"  Infer [green]{inf:>7,}[/green]",
            "─" * 80,
            "  [dim]          avg    min    max[/dim]",
            f"  E2E   [white]{_fmt_ms_row(e2e)}[/white] ms",
            f"  Infer [white]{_fmt_ms_row(infer)}[/white] ms",
            f"  PostP [white]{_fmt_ms_row(pp)}[/white] ms",
        ]
        if h2d is not None:
            lines.append(f"  H2D   [white]{_fmt_ms_row(h2d)}[/white] ms")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Global stats panel
# ---------------------------------------------------------------------------


class GlobalPanel(Static):
    """Displays global metrics: concurrent inferences, TRT timing."""

    DEFAULT_CSS = """
    GlobalPanel {
        width: 1fr;
        height: 100%;
        border: double $warning;
        padding: 0 1;
        background: $background;
        color: $text;
    }
    """

    def __init__(self) -> None:
        super().__init__(id="global-panel")
        self._data: dict = {}

    def update_data(self, snapshot: dict) -> None:
        self._data = snapshot
        self.refresh()

    def render(self) -> str:  # type: ignore[override]
        d = self._data
        concurrent = d.get("inference_concurrent", 0)
        rps = d.get("global_throughput_rps", 0.0)
        e2e = d.get("global_e2e_ms")
        sem = d.get("global_sem_wait_ms")
        h2d = d.get("global_trt_h2d_ms")
        exe = d.get("global_trt_execute_ms")
        d2h = d.get("global_trt_d2h_ms")

        lines = [
            "[bold yellow]GLOBAL METRICS[/bold yellow]",
            "─" * 80,
            f"  Throughput [green]{rps:7.2f} rps[/green]",
            f"  Concurrent [cyan]{concurrent:>4}[/cyan]",
            "─" * 80,
            "  [dim]          avg    min    max[/dim]",
            f"  E2E   [white]{_fmt_ms_row(e2e)}[/white] ms",
        ]

        if sem is not None:
            lines.append(f"  SemWait[white]{_fmt_ms_row(sem)}[/white] ms")

        if h2d is not None or exe is not None:
            lines += [
                "─" * 80,
                "  [bold magenta]TensorRT Timing[/bold magenta]",
            ]
            if h2d is not None:
                lines.append(f"  H2D    [white]{_fmt_ms_row(h2d)}[/white] ms")
            if exe is not None:
                lines.append(f"  Execute[white]{_fmt_ms_row(exe)}[/white] ms")
            if d2h is not None:
                lines.append(f"  D2H    [white]{_fmt_ms_row(d2h)}[/white] ms")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------


class CondorTUI(App[None]):
    """Condor metrics TUI."""

    TITLE = "CONDOR — Frigate Remote Detector"

    CSS = """
    Screen {
        background: #0d0d0d;
        layers: base;
    }

    #graphs-row {
        height: 1fr;
    }

    #graphs-row GraphPanel {
        width: 1fr;
        height: 1fr;
    }

    #graphs-row #latency-panel {
        border: heavy $success;
    }

    #graphs-row #latency-panel > .title {
        color: $success;
    }

    #graphs-row #throughput-panel {
        border: heavy $accent;
    }

    #graphs-row #throughput-panel > .title {
        color: $accent;
    }

    #workers-row {
        height: 1fr;
        layout: horizontal;
    }

    Footer {
        background: #111111;
        color: $text-muted;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._snapshot: dict = {}
        self._layout_ready = False
        self._num_workers = 0

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield HeaderWidget()
        yield StatusBar("● CONNECTING…", id="status-bar")
        with Horizontal(id="graphs-row"):
            yield GraphPanel("E2E LATENCY", "ms", "latency-panel")
            yield GraphPanel("THROUGHPUT", "req/s", "throughput-panel")
        with Horizontal(id="workers-row"):
            pass  # Worker panels mounted dynamically on first snapshot
        yield Footer()

    # ------------------------------------------------------------------
    # Stats reader worker
    # ------------------------------------------------------------------

    @work(exclusive=True)
    async def _read_stats(self) -> None:
        """Async worker: connects to the stats socket and reads snapshots."""
        while True:
            try:
                reader, writer = await asyncio.open_unix_connection(SOCKET_PATH)
                async for line in reader:
                    text = line.decode(errors="replace").strip()
                    if not text:
                        continue
                    try:
                        data = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    await self._update_ui(data)
                writer.close()
            except (ConnectionRefusedError, FileNotFoundError, OSError):
                self._update_disconnected()
                await asyncio.sleep(2.0)

    def on_mount(self) -> None:
        self._read_stats()

    # ------------------------------------------------------------------
    # UI updates
    # ------------------------------------------------------------------

    def _update_disconnected(self) -> None:
        try:
            self.query_one("#status-bar", StatusBar).update(
                "[bold red]● DISCONNECTED[/bold red]  "
                f"[dim]waiting for condor server at {SOCKET_PATH}…[/dim]"
            )
        except Exception:
            pass

    async def _update_ui(self, data: dict) -> None:
        self._snapshot = data
        cfg = data.get("config", {})
        provider = cfg.get("provider", "")
        num_workers = cfg.get("num_workers", 1)
        base_port = cfg.get("base_port", 5555)

        # Update header provider logo
        header = self.query_one(HeaderWidget)
        if header.provider != provider:
            header.provider = provider

        # Update status bar
        uptime = data.get("uptime_s", 0.0)
        workers_active = data.get("active_workers", 0)
        model_raw = data.get("active_model", "")
        model = Path(model_raw).stem if model_raw else "(none)"
        concurrent = data.get("inference_concurrent", 0)
        rps = data.get("global_throughput_rps", 0.0)

        status_text = (
            f"[bold green]● ONLINE[/bold green]  "
            f"⏱ [cyan]{_fmt_time(uptime)}[/cyan]  "
            f"⚙ [yellow]{workers_active}/{num_workers} workers[/yellow]  "
            f"📦 [white]{model}[/white]  "
            f"[magenta]{provider or '—'}[/magenta]  "
            f"[green]{rps:.1f} rps[/green]"
        )
        self.query_one("#status-bar", StatusBar).update(status_text)

        # Update sparklines
        lat_data = data.get("sparkline_latency", [])
        tput_data = data.get("sparkline_throughput", [])

        lat_summary = ""
        if lat_data:
            lat_summary = (
                f"  now {lat_data[-1]:.1f}  "
                f"avg {sum(lat_data)/len(lat_data):.1f}  "
                f"peak {max(lat_data):.1f}"
            )
        self.query_one("#latency-panel", GraphPanel).update_data(lat_data, lat_summary)

        tput_summary = ""
        if tput_data:
            tput_summary = (
                f"  now {tput_data[-1]:.1f}  "
                f"avg {sum(tput_data)/len(tput_data):.1f}  "
                f"peak {max(tput_data):.1f}"
            )
        self.query_one("#throughput-panel", GraphPanel).update_data(
            tput_data, tput_summary
        )

        # Create worker panels on first snapshot (or if worker count changes)
        workers = data.get("workers", {})
        if not self._layout_ready or self._num_workers != num_workers:
            await self._create_worker_panels(num_workers, base_port)
            self._layout_ready = True
            self._num_workers = num_workers

        # Update per-worker panels — pass full snapshot so workers can read
        # global TRT timing (H2D etc.) which has no per-worker breakdown.
        for wid_str, wdata in workers.items():
            try:
                wid = int(wid_str)
                panel = self.query_one(f"#worker-panel-{wid}", WorkerPanel)
                panel.update_data(wdata, data)
            except Exception:
                pass

        # Update global panel
        try:
            self.query_one("#global-panel", GlobalPanel).update_data(data)
        except Exception:
            pass

    async def _create_worker_panels(self, num_workers: int, base_port: int) -> None:
        """Mount worker panels into the workers-row container."""
        container = self.query_one("#workers-row", Horizontal)
        for child in list(container.children):
            await child.remove()

        for i in range(num_workers):
            await container.mount(WorkerPanel(i, base_port + i))
        await container.mount(GlobalPanel())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    app = CondorTUI()
    app.run()


if __name__ == "__main__":
    main()
