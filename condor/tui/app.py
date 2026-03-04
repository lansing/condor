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
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Sparkline, Static

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
        """Animation disabled temporarily."""
        pass

    def render(self) -> str:
        condor_lines = CONDOR_LOGO.split("\n")
        provider_lines = get_provider_logo(self.provider).split("\n")

        return build_combined_logo(
            condor_lines,
            provider_lines,
            total_width=self.size.width,
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

    # Metric keys that can be None when the rolling window is empty.
    _WORKER_STICKY = ("e2e_ms", "infer_ms", "postprocess_ms")
    _GLOBAL_STICKY = (
        "global_trt_host_copy_ms", "global_trt_h2d_ms", "global_sem_wait_ms",
        "global_trt_execute_ms", "global_trt_d2h_ms",
    )

    def __init__(self, worker_id: int, port: int) -> None:
        super().__init__(id=f"worker-panel-{worker_id}")
        self._worker_id = worker_id
        self._port = port
        self._data: dict = {}
        self._trt_data: dict = {}
        self._frozen: dict = {}  # last-known non-None values

    def update_data(self, wdata: dict, snapshot: dict) -> None:
        # Freeze metric values: update cache on non-None, substitute on None.
        merged_w = dict(wdata)
        merged_g = dict(snapshot)
        for key in self._WORKER_STICKY:
            if wdata.get(key) is not None:
                self._frozen[key] = wdata[key]
            elif key in self._frozen:
                merged_w[key] = self._frozen[key]
        for key in self._GLOBAL_STICKY:
            if snapshot.get(key) is not None:
                self._frozen[key] = snapshot[key]
            elif key in self._frozen:
                merged_g[key] = self._frozen[key]
        self._data = merged_w
        self._trt_data = merged_g
        self.refresh()

    def render(self) -> str:  # type: ignore[override]
        d = self._data
        g = self._trt_data

        inf = d.get("inference_total", 0)
        rps = d.get("req_per_sec", 0.0)
        e2e = d.get("e2e_ms")
        mcpy = g.get("global_trt_host_copy_ms")
        h2d = g.get("global_trt_h2d_ms")
        sem = g.get("global_sem_wait_ms")
        infer = g.get("global_trt_execute_ms")
        d2h = g.get("global_trt_d2h_ms")
        pp = d.get("postprocess_ms")

        lines = [
            f"[bold cyan]WORKER {self._worker_id}[/bold cyan]  [dim]:{self._port}[/dim]  [yellow]{rps:5.1f} rps[/yellow]",
            f"  Inf   [green]{inf:>7,}[/green]",
            "  [dim]          avg    min    max[/dim]",
            f"  E2E   [white]{_fmt_ms_row(e2e)}[/white] ms",
            f"  MCpy  [white]{_fmt_ms_row(mcpy)}[/white] ms",
            f"  H2D   [white]{_fmt_ms_row(h2d)}[/white] ms",
            f"  SWait [white]{_fmt_ms_row(sem)}[/white] ms",
            f"  Exec  [white]{_fmt_ms_row(infer)}[/white] ms",
            f"  D2H   [white]{_fmt_ms_row(d2h)}[/white] ms",
            f"  PostP [white]{_fmt_ms_row(pp)}[/white] ms",
        ]
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

    _STICKY = (
        "global_e2e_ms", "global_trt_host_copy_ms", "global_trt_h2d_ms",
        "global_sem_wait_ms", "global_trt_execute_ms", "global_trt_d2h_ms",
        "global_postprocess_ms",
    )

    def __init__(self) -> None:
        super().__init__(id="global-panel")
        self._data: dict = {}
        self._frozen: dict = {}

    def update_data(self, snapshot: dict) -> None:
        merged = dict(snapshot)
        for key in self._STICKY:
            if snapshot.get(key) is not None:
                self._frozen[key] = snapshot[key]
            elif key in self._frozen:
                merged[key] = self._frozen[key]
        self._data = merged
        self.refresh()

    def render(self) -> str:  # type: ignore[override]
        d = self._data
        rps = d.get("global_throughput_rps", 0.0)
        e2e = d.get("global_e2e_ms")
        mcpy = d.get("global_trt_host_copy_ms")
        h2d = d.get("global_trt_h2d_ms")
        sem = d.get("global_sem_wait_ms")
        infer = d.get("global_trt_execute_ms")
        d2h = d.get("global_trt_d2h_ms")
        pp = d.get("global_postprocess_ms")

        lines = [
            f"[bold yellow]GLOBAL METRICS[/bold yellow]  [green]{rps:7.2f} rps[/green]",
            "  [dim]          avg    min    max[/dim]",
            f"  E2E   [white]{_fmt_ms_row(e2e)}[/white] ms",
            f"  MCpy  [white]{_fmt_ms_row(mcpy)}[/white] ms",
            f"  H2D   [white]{_fmt_ms_row(h2d)}[/white] ms",
            f"  SWait [white]{_fmt_ms_row(sem)}[/white] ms",
            f"  Exec  [white]{_fmt_ms_row(infer)}[/white] ms",
            f"  D2H   [white]{_fmt_ms_row(d2h)}[/white] ms",
            f"  PostP [white]{_fmt_ms_row(pp)}[/white] ms",
        ]

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tick selector dialog
# ---------------------------------------------------------------------------


class TickSelectorScreen(ModalScreen):
    """Modal dialog for choosing seconds-per-tick."""

    DEFAULT_CSS = """
    TickSelectorScreen {
        align: center middle;
    }
    #tick-dialog {
        width: 44;
        height: auto;
        border: heavy $accent;
        background: $surface;
        padding: 1 2;
    }
    .dlg-title {
        text-style: bold;
        color: $accent;
        padding-bottom: 1;
    }
    .dlg-opt {
        color: $text;
    }
    .dlg-hint {
        color: $text-muted;
        padding-top: 1;
    }
    """

    BINDINGS = [
        ("1", "pick_1", "1s/tick"),
        ("2", "pick_2", "2s/tick"),
        ("5", "pick_5", "5s/tick"),
        ("0", "pick_10", "10s/tick"),
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(self, current: int) -> None:
        super().__init__()
        self._current = current

    def compose(self) -> ComposeResult:
        with Static(id="tick-dialog"):
            yield Static("SET TICK DURATION", classes="dlg-title")
            yield Static("  [bold]1[/bold]  →  1 second per tick",   classes="dlg-opt")
            yield Static("  [bold]2[/bold]  →  2 seconds per tick",  classes="dlg-opt")
            yield Static("  [bold]5[/bold]  →  5 seconds per tick",  classes="dlg-opt")
            yield Static("  [bold]0[/bold]  →  10 seconds per tick", classes="dlg-opt")
            yield Static(
                f"  [dim]current: {self._current}s/tick — ESC to cancel[/dim]",
                classes="dlg-hint",
            )

    def action_pick_1(self)  -> None: self.dismiss(1)
    def action_pick_2(self)  -> None: self.dismiss(2)
    def action_pick_5(self)  -> None: self.dismiss(5)
    def action_pick_10(self) -> None: self.dismiss(10)
    def action_cancel(self)  -> None: self.dismiss(None)


# ---------------------------------------------------------------------------
# Custom footer
# ---------------------------------------------------------------------------


class AppFooter(Static):
    """Footer row: key hints + current tick rate."""

    DEFAULT_CSS = """
    AppFooter {
        height: 1;
        dock: bottom;
        background: #111111;
        color: $text-muted;
        padding: 0 1;
    }
    """

    seconds_per_tick: reactive[int] = reactive(2)

    def render(self) -> str:
        spt = self.seconds_per_tick
        return (
            f"[bold white]q[/bold white] Quit  "
            f"[bold white]t[/bold white] Set Tick  "
            f"[dim cyan]{spt}s/tick[/dim cyan]"
        )


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
        ("t", "set_tick", "Set Tick"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._snapshot: dict = {}
        self._layout_ready = False
        self._num_workers = 0
        self._seconds_per_tick: int = 2
        self._num_ticks: int = 60
        self._stats_writer: asyncio.StreamWriter | None = None

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
        yield AppFooter()

    # ------------------------------------------------------------------
    # Stats reader worker
    # ------------------------------------------------------------------

    @work(exclusive=True)
    async def _read_stats(self) -> None:
        """Async worker: connects to the stats socket and reads snapshots."""
        while True:
            try:
                reader, writer = await asyncio.open_unix_connection(SOCKET_PATH)
                self._stats_writer = writer
                # Tell the server about our current time config immediately.
                await self._send_time_config()
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
                pass
            finally:
                self._stats_writer = None
            self._update_disconnected()
            await asyncio.sleep(2.0)

    def on_mount(self) -> None:
        self._read_stats()

    # ------------------------------------------------------------------
    # Tick config
    # ------------------------------------------------------------------

    async def _send_time_config(self) -> None:
        """Push current window_s / sparkline_len config to the server."""
        w = self._stats_writer
        if w is None or w.is_closing():
            return
        window_s = self._num_ticks * self._seconds_per_tick
        msg = json.dumps({"window_s": window_s, "sparkline_len": self._num_ticks}) + "\n"
        try:
            w.write(msg.encode())
            await w.drain()
        except Exception:
            pass

    def action_set_tick(self) -> None:
        # push_screen_wait requires a worker context — delegate immediately.
        self._open_tick_dialog()

    @work
    async def _open_tick_dialog(self) -> None:
        result = await self.push_screen_wait(TickSelectorScreen(self._seconds_per_tick))
        if result is not None:
            self._seconds_per_tick = result
            self.query_one(AppFooter).seconds_per_tick = result
            await self._send_time_config()

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
            f"[magenta]{provider or '—'}[/magenta]"
        )
        self.query_one("#status-bar", StatusBar).update(status_text)

        # Derive num_ticks from the actual sparkline widget width so the
        # graph X-axis and metric rolling windows stay in sync.
        try:
            panel = self.query_one("#latency-panel", GraphPanel)
            spark = panel.query_one(f"#{panel._spark_id}", Sparkline)
            w = spark.size.width
            if w > 0 and w != self._num_ticks:
                self._num_ticks = w
                await self._send_time_config()
        except Exception:
            pass

        # Update sparklines — trim or left-pad to exactly _num_ticks points so
        # the X-scale is always consistent regardless of uptime or window changes.
        n = self._num_ticks
        lat_data = list(data.get("sparkline_latency", []))
        tput_data = list(data.get("sparkline_throughput", []))
        if len(lat_data) > n:
            lat_data = lat_data[-n:]
        elif len(lat_data) < n:
            lat_data = [0.0] * (n - len(lat_data)) + lat_data
        if len(tput_data) > n:
            tput_data = tput_data[-n:]
        elif len(tput_data) < n:
            tput_data = [0.0] * (n - len(tput_data)) + tput_data

        lat_summary = ""
        if any(v > 0 for v in lat_data):
            nonzero = [v for v in lat_data if v > 0]
            lat_summary = (
                f"  now {lat_data[-1]:.1f}  "
                f"avg {sum(nonzero)/len(nonzero):.1f}  "
                f"peak {max(lat_data):.1f}"
            )
        self.query_one("#latency-panel", GraphPanel).update_data(lat_data, lat_summary)

        tput_summary = ""
        if any(v > 0 for v in tput_data):
            nonzero = [v for v in tput_data if v > 0]
            tput_summary = (
                f"  now {tput_data[-1]:.1f}  "
                f"avg {sum(nonzero)/len(nonzero):.1f}  "
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
