from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from ..stats import SOCKET_PATH as _DEFAULT_SOCKET_PATH
from .art import _trunc, _vis
from .palette import STAGE_ORDER, Palette, available_palettes, load_palette

SOCKET_PATH = os.environ.get("CONDOR_STATS_SOCKET", _DEFAULT_SOCKET_PATH)
_THEME_FILE = Path(SOCKET_PATH).parent / "condor-theme"


def _load_saved_theme(names: list[str]) -> tuple[int, Palette]:
    """Return (index, palette) from the saved theme file, or Broica as default."""
    try:
        name = _THEME_FILE.read_text().strip()
        if name in names:
            return names.index(name), load_palette(name)
    except OSError:
        pass
    idx = names.index("Broica") if "Broica" in names else 0
    return idx, load_palette(names[idx])


def _save_theme(name: str) -> None:
    try:
        _THEME_FILE.write_text(name)
    except OSError:
        pass

STAGE_LABELS: dict[str, str] = {
    "mcpy": "Host memory copy",
    "h2d": "Host → Device (H2D)",
    "swait": "GPU queue wait",
    "exec": "TRT engine execute",
    "d2h": "Device → Host (D2H)",
    "pp": "Post-process",
}
STAGE_ABBREV: dict[str, str] = {
    "mcpy": "MCpy",
    "h2d": "H2D",
    "swait": "SWait",
    "exec": "Exec",
    "d2h": "D2H",
    "pp": "PostP",
}

_BLOCK = "█"
_BASELINE_CHAR = "▁"
_BASELINE_COLOR = "_baseline"  # sentinel — not a real Rich colour


def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _fmt_ms_row(d: dict) -> str:
    return f"{d['avg']:6.1f}  {d['p99']:6.1f}"


def _alloc_rows(vals: dict[str, float], bar_h: int) -> dict[str, int]:
    """Allocate bar_h rows to stages proportionally (descending-first greedy)."""
    D = sum(vals.values())
    if D == 0 or bar_h == 0:
        return {s: 0 for s in vals}
    sorted_stages = sorted(vals.items(), key=lambda x: x[1], reverse=True)
    rows: dict[str, int] = {s: 0 for s in vals}
    used = 0
    for stage, v in sorted_stages:
        r = min(round(v / D * bar_h), bar_h - used)
        rows[stage] = r
        used += r
        if used >= bar_h:
            break
    if used < bar_h and sorted_stages:
        rows[sorted_stages[0][0]] += bar_h - used
    return rows


def _build_column(
    vals: dict[str, float],
    bar_h: int,
    e2e: float,
    peak: float,
    stage_colors: dict[str, str],
) -> list[str]:
    col: list[str] = [""] * bar_h
    if bar_h == 0:
        return col
    if e2e <= 0 or peak <= 0:
        col[bar_h - 1] = _BASELINE_COLOR
        return col
    bar_total = max(1, min(bar_h, round(e2e / peak * bar_h)))
    start_row = bar_h - bar_total
    D = sum(vals.values())
    if D == 0:
        for r in range(start_row, bar_h):
            col[r] = stage_colors["exec"]
        return col
    alloc = _alloc_rows(vals, bar_total)
    cur_row = start_row
    for stage in STAGE_ORDER:
        n = alloc.get(stage, 0)
        if n > 0:
            color = stage_colors[stage]
            for r in range(cur_row, min(cur_row + n, bar_h)):
                col[r] = color
            cur_row += n
    return col


def _render_bar_row(row: list[str]) -> str:
    """Convert a list of colour strings to a Rich-markup line of block chars."""
    if not row:
        return ""
    parts: list[str] = []
    i = 0
    while i < len(row):
        color = row[i]
        j = i + 1
        while j < len(row) and row[j] == color:
            j += 1
        span = j - i
        if color == _BASELINE_COLOR:
            parts.append(f"[dim]{_BASELINE_CHAR * span}[/dim]")
        elif color:
            parts.append(f"[{color}]{_BLOCK * span}[/{color}]")
        else:
            parts.append(" " * span)
        i = j
    return "".join(parts)


# ---------------------------------------------------------------------------
# Gradient bar chart (throughput / GPU utilization)
# ---------------------------------------------------------------------------


class _GradientBars(Static):
    """Vertical bar chart where each row is coloured by its height fraction.

    Bottom row → palette.gradient_low, top row → palette.gradient_high.
    The caller supplies the normalisation peak so throughput and GPU can each
    use their own scale (session-max and 100.0 respectively).
    """

    DEFAULT_CSS = """
    _GradientBars { width: 1fr; height: 1fr; }
    """

    def __init__(self) -> None:
        super().__init__("")
        self._data: list[float] = []
        self._peak: float = 0.0

    def update(self, data: list[float], peak: float) -> None:
        self._data = data
        self._peak = peak
        self.refresh()

    def render(self) -> str:  # type: ignore[override]
        data = self._data
        peak = self._peak
        bar_h = max(1, self.size.height)
        bar_w = max(1, self.size.width)

        if not data or peak <= 0:
            row: list[str] = [_BASELINE_COLOR] * bar_w
            return _render_bar_row(row)

        p = self.app.palette
        # Precompute one gradient colour per row (bottom=low, top=high).
        row_colors = [
            p.gradient_color((bar_h - 1 - r) / max(bar_h - 1, 1))
            for r in range(bar_h)
        ]

        n_cols = min(bar_w, len(data))
        offset = len(data) - n_cols
        n_blank = bar_w - n_cols

        grid: list[list[str]] = [[""] * bar_w for _ in range(bar_h)]
        for col in range(n_blank):
            grid[bar_h - 1][col] = _BASELINE_COLOR

        for col_idx in range(n_cols):
            col = n_blank + col_idx
            val = data[offset + col_idx]
            if val <= 0:
                grid[bar_h - 1][col] = _BASELINE_COLOR
                continue
            bar_total = max(1, min(bar_h, round(val / peak * bar_h)))
            start_row = bar_h - bar_total
            for row_idx in range(start_row, bar_h):
                grid[row_idx][col] = row_colors[row_idx]

        return "\n".join(_render_bar_row(row) for row in grid)


# ---------------------------------------------------------------------------
# Status banner
# ---------------------------------------------------------------------------


class StatusBanner(Static):
    DEFAULT_CSS = """
    StatusBanner {
        height: 3;
        border: heavy $condor-border;
        color: $condor-text;
        content-align: left middle;
        background: $condor-info-bg;
        padding: 0 1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._uptime = 0.0
        self._workers_active = 0
        self._num_workers = 0
        self._model = ""
        self._postprocessor = ""
        self._state = "connecting"

    def update_status(
        self,
        uptime: float,
        workers_active: int,
        num_workers: int,
        model: str,
        postprocessor: str = "",
    ) -> None:
        self._uptime = uptime
        self._workers_active = workers_active
        self._num_workers = num_workers
        self._model = model
        self._postprocessor = postprocessor
        self._state = "online"
        self.refresh()

    def update_disconnected(self) -> None:
        self._state = "disconnected"
        self.refresh()

    def render(self) -> str:
        if self._state == "connecting":
            return "[dim]● CONNECTING…[/dim]"
        if self._state == "disconnected":
            return (
                "[bold red]● DISCONNECTED[/bold red]  "
                f"[dim]waiting for condor server at {SOCKET_PATH}…[/dim]"
            )
        p = self.app.palette
        c_online  = p.gradient_high
        c_uptime  = p.stage_color("h2d")
        c_workers = p.stage_color("exec")
        c_pp      = p.stage_color("swait")
        prefix = (
            f"[bold {c_online}]● ONLINE[/]  "
            f"Up [{c_uptime}]{_fmt_time(self._uptime)}[/]  "
            f"[{c_workers}]{self._num_workers} workers[/]  "
        )
        model_budget = self.size.width - _vis(prefix)
        model = _trunc(self._model, model_budget)
        if self._postprocessor:
            return f"{prefix}[white]{model}[/white] ([{c_pp}]{self._postprocessor}[/])"
        return f"{prefix}[white]{model}[/white]"


# ---------------------------------------------------------------------------
# Stacked-bar latency panel
# ---------------------------------------------------------------------------


class _LatencyBars(Static):
    DEFAULT_CSS = """
    _LatencyBars { width: 2fr; height: 1fr; }
    """

    def __init__(self) -> None:
        super().__init__("")
        self._lat_data: list[float] = []
        self._stages: dict[str, list[float]] = {}

    def update(self, lat_data: list[float], stages: dict[str, list[float]]) -> None:
        self._lat_data = lat_data
        self._stages = stages
        self.refresh()

    def render(self) -> str:  # type: ignore[override]
        lat = self._lat_data
        stages = self._stages

        bar_h = max(1, self.size.height - 1)
        bar_w = max(1, self.size.width)
        peak = max(lat) if lat else 0.0

        p = self.app.palette
        gh = p.gradient_high
        title = f" [bold white]▶ E2E LATENCY[/]  [{gh}]↑ {peak:.0f}[/]"

        if not lat:
            return title

        stage_colors = p.stage_color_map()
        n_cols = min(bar_w, len(lat))
        offset = len(lat) - n_cols
        lat_slice = lat[offset:]
        n_blank = bar_w - n_cols

        grid: list[list[str]] = [[""] * bar_w for _ in range(bar_h)]
        for col in range(n_blank):
            grid[bar_h - 1][col] = _BASELINE_COLOR
        for col_idx in range(n_cols):
            col = n_blank + col_idx
            t_idx = offset + col_idx
            vals: dict[str, float] = {
                stage: (
                    stages.get(stage, [])[t_idx]
                    if t_idx < len(stages.get(stage, []))
                    else 0.0
                )
                for stage in STAGE_ORDER
            }
            col_colors = _build_column(vals, bar_h, lat_slice[col_idx], peak, stage_colors)
            for row in range(bar_h):
                grid[row][col] = col_colors[row]

        lines = [title]
        for row in grid:
            lines.append(_render_bar_row(row))
        return "\n".join(lines)


class StackedBarPanel(Widget):
    DEFAULT_CSS = """
    StackedBarPanel {
        width: 1fr;
        height: 1fr;
        border: heavy $condor-border;
        padding: 0 1;
        background: $condor-bg;
        color: $condor-text;
    }
    StackedBarPanel > Horizontal {
        height: 1fr;
    }
    StackedBarPanel > .summary {
        height: 1;
        color: $condor-text-muted;
    }
    """

    def __init__(self) -> None:
        super().__init__(id="latency-panel")

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield _LatencyBars()
            yield LegendPanel()
        yield Static("", classes="summary")

    def update_data(
        self,
        lat_data: list[float],
        stages: dict[str, list[float]],
        summary: str,
    ) -> None:
        self.query_one(_LatencyBars).update(lat_data, stages)
        self.query_one(".summary", Static).update(summary)


class GraphPanel(Widget):
    """Labeled gradient-bar panel with a live max y-scale label."""

    DEFAULT_CSS = """
    GraphPanel {
        height: 1fr;
        border: heavy $condor-border;
        padding: 0 1;
        background: $condor-bg;
    }
    GraphPanel > .title {
        height: 1;
        color: $condor-text;
        text-style: bold;
    }
    GraphPanel > .summary {
        height: 1;
        color: $condor-text-muted;
    }
    """

    def __init__(self, title: str, unit: str, widget_id: str) -> None:
        super().__init__(id=widget_id)
        self._title = title
        self._unit = unit
        self._title_id = f"{widget_id}-title"
        self._summary_id = f"{widget_id}-summary"
        self._session_peak: float = 0.0

    def compose(self) -> ComposeResult:
        yield Static(f" ▶ {self._title}", id=self._title_id, classes="title")
        yield _GradientBars()
        yield Static("", id=self._summary_id, classes="summary")

    def update_data(self, data: list[float], summary: str) -> None:
        if not data:
            return
        window_peak = max(data)
        self._session_peak = max(self._session_peak, window_peak)
        gh = self.app.palette.gradient_high
        self.query_one(f"#{self._title_id}", Static).update(
            f" [bold white]▶ {self._title}[/]  [{gh}]↑ {window_peak:.0f}[/]"
        )
        self.query_one(_GradientBars).update(data, self._session_peak)
        self.query_one(f"#{self._summary_id}", Static).update(summary)


class WorkerPanel(Static):
    """Displays stats for one worker thread."""

    DEFAULT_CSS = """
    WorkerPanel {
        width: 1fr;
        height: 100%;
        border: double $condor-border;
        padding: 0 1;
        background: $condor-bg;
        color: $condor-text;
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

    _ZERO = {"avg": 0.0, "p99": 0.0}

    def render(self) -> str:  # type: ignore[override]
        d = self._data
        g = self._trt_data
        p = self.app.palette

        inf = d.get("inference_total", 0)
        rps = d.get("req_per_sec", 0.0)
        e2e = d.get("e2e_ms", self._ZERO)
        mcpy = g.get("global_trt_host_copy_ms", self._ZERO)
        h2d = g.get("global_trt_h2d_ms", self._ZERO)
        sem = g.get("global_sem_wait_ms", self._ZERO)
        infer = g.get("global_trt_execute_ms", self._ZERO)
        d2h = g.get("global_trt_d2h_ms", self._ZERO)
        pp = d.get("postprocess_ms", self._ZERO)

        gh = p.gradient_high
        mc, h2, sw, ex, d2, pp_ = (p.stage_color(s) for s in STAGE_ORDER)
        lines = [
            f"[bold white]WORKER {self._worker_id}[/]  [dim]:{self._port}[/dim]"
            f"  [{gh}]{rps:5.1f}/s[/]  [white]{inf:>7,}[/white]",
            "  [dim]         avg     p99[/dim]",
            f"  E2E   [white]{_fmt_ms_row(e2e)}[/white] ms",
            f"  [{mc}]MCpy[/{mc}]  [white]{_fmt_ms_row(mcpy)}[/white] ms",
            f"  [{h2}]H2D[/{h2}]   [white]{_fmt_ms_row(h2d)}[/white] ms",
            f"  [{sw}]SWait[/{sw}] [white]{_fmt_ms_row(sem)}[/white] ms",
            f"  [{ex}]Exec[/{ex}]  [white]{_fmt_ms_row(infer)}[/white] ms",
            f"  [{d2}]D2H[/{d2}]   [white]{_fmt_ms_row(d2h)}[/white] ms",
            f"  [{pp_}]PostP[/{pp_}] [white]{_fmt_ms_row(pp)}[/white] ms",
        ]
        return "\n".join(lines)


class GlobalPanel(Static):

    DEFAULT_CSS = """
    GlobalPanel {
        width: 1fr;
        height: 100%;
        border: double $condor-border;
        padding: 0 1;
        background: $condor-bg;
        color: $condor-text;
    }
    """

    def __init__(self) -> None:
        super().__init__(id="global-panel")
        self._data: dict = {}

    def update_data(self, snapshot: dict) -> None:
        self._data = snapshot
        self.refresh()

    _ZERO = {"avg": 0.0, "p99": 0.0}

    def render(self) -> str:  # type: ignore[override]
        d = self._data
        p = self.app.palette

        rps = d.get("global_throughput_rps", 0.0)
        e2e = d.get("global_e2e_ms", self._ZERO)
        mcpy = d.get("global_trt_host_copy_ms", self._ZERO)
        h2d = d.get("global_trt_h2d_ms", self._ZERO)
        sem = d.get("global_sem_wait_ms", self._ZERO)
        infer = d.get("global_trt_execute_ms", self._ZERO)
        d2h = d.get("global_trt_d2h_ms", self._ZERO)
        pp = d.get("global_postprocess_ms", self._ZERO)

        gh = p.gradient_high
        mc, h2, sw, ex, d2, pp_ = (p.stage_color(s) for s in STAGE_ORDER)
        lines = [
            f"[bold white]STAGE LATENCY[/]  [{gh}]{rps:7.2f} rps[/]",
            "  [dim]         avg     p99[/dim]",
            f"  E2E   [white]{_fmt_ms_row(e2e)}[/white] ms",
            f"  [{mc}]MCpy[/{mc}]  [white]{_fmt_ms_row(mcpy)}[/white] ms",
            f"  [{h2}]H2D[/{h2}]   [white]{_fmt_ms_row(h2d)}[/white] ms",
            f"  [{sw}]SWait[/{sw}] [white]{_fmt_ms_row(sem)}[/white] ms",
            f"  [{ex}]Exec[/{ex}]  [white]{_fmt_ms_row(infer)}[/white] ms",
            f"  [{d2}]D2H[/{d2}]   [white]{_fmt_ms_row(d2h)}[/white] ms",
            f"  [{pp_}]PostP[/{pp_}] [white]{_fmt_ms_row(pp)}[/white] ms",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# GPU panel
# ---------------------------------------------------------------------------


class GpuPanel(Widget):
    DEFAULT_CSS = """
    GpuPanel {
        width: 1fr;
        height: 100%;
        border: double $condor-border;
        padding: 0 1;
        background: $condor-bg;
        color: $condor-text;
    }
    GpuPanel > .gpu-header {
        height: 1;
        color: $condor-text;
        text-style: bold;
    }
    GpuPanel > .gpu-summary {
        height: 1;
        color: $condor-text-muted;
    }
    GpuPanel > .gpu-power {
        height: 1;
        color: $condor-text-muted;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("GPU  —", classes="gpu-header")
        yield _GradientBars()
        yield Static("", classes="gpu-summary")
        yield Static("", classes="gpu-power")

    def update_data(self, gpu: dict) -> None:
        name = gpu.get("name", "Unknown")
        index = gpu.get("index", 0)
        temp_c = gpu.get("temp_c", 0)
        util_pct = gpu.get("util_pct", 0.0)
        power_w = gpu.get("power_w", 0.0)
        power_limit_w = gpu.get("power_limit_w", 0.0)
        mem_used_mb = gpu.get("mem_used_mb", 0.0)
        mem_total_mb = gpu.get("mem_total_mb", 0.0)
        sparkline_data = gpu.get("sparkline", [])

        p = self.app.palette
        temp_color = p.gradient_color((temp_c - 30) / 50.0)  # 30°C=low, 80°C=high
        self.query_one(".gpu-header", Static).update(
            f"[bold white]GPU {index}[/]  [dim]{name}[/dim]  [{temp_color}]{temp_c}°C[/]"
        )
        self.query_one(_GradientBars).update(sparkline_data, 100.0)

        nonzero = [v for v in sparkline_data if v > 0]
        summary = (
            f"  now {util_pct:.0f}%  "
            f"avg {sum(nonzero) / len(nonzero):.0f}%  "
            f"peak {max(sparkline_data):.0f}%"
            if nonzero
            else ""
        )
        self.query_one(".gpu-summary", Static).update(summary)

        if power_limit_w > 0:
            pct = min(1.0, power_w / power_limit_w)
            pwr_color = p.gradient_color(pct)
            bar_w = 10
            filled = round(pct * bar_w)
            bar = "█" * filled + "░" * (bar_w - filled)
            mem_gb = f"{mem_used_mb / 1024:.1f}/{mem_total_mb / 1024:.1f}GB"
            pwr_text = (
                f"  Pwr [{pwr_color}]{power_w:.0f}W[/]/{power_limit_w:.0f}W "
                f"[{pwr_color}]{bar}[/]  Mem {mem_gb}"
            )
        else:
            pwr_text = f"  Pwr {power_w:.0f}W"
        self.query_one(".gpu-power", Static).update(pwr_text)


class LegendPanel(Static):
    DEFAULT_CSS = """
    LegendPanel {
        width: 1fr;
        height: 1fr;
        border-left: heavy $condor-border;
        padding: 0 1;
        background: $condor-bg;
        display: none;
    }
    """

    def render(self) -> str:  # type: ignore[override]
        p = self.app.palette
        gh = p.gradient_high
        lines = [f"[bold {gh}] ▶ STAGE LEGEND[/]", ""]
        for stage in STAGE_ORDER:
            color = p.stage_color(stage)
            lines.append(
                f"  [{color}]██[/{color}]  [{color}]{STAGE_ABBREV[stage]}[/{color}]"
            )
        return "\n".join(lines)



class AppFooter(Static):
    """Footer row: key hints + current tick rate."""

    DEFAULT_CSS = """
    AppFooter {
        height: 1;
        dock: bottom;
        background: $condor-bg;
        color: $condor-text-muted;
        padding: 0 1;
    }
    """

    workers_visible: reactive[bool] = reactive(False)

    def render(self) -> str:
        p = self.app.palette
        gh = p.gradient_high
        w_label = "GPU" if self.workers_visible else "Workers"
        return (
            f"[bold {gh}]q[/] Quit  "
            f"[bold {gh}]l[/] Legend  "
            f"[bold {gh}]w[/] {w_label}  "
            f"[bold {gh}]m[/]/[bold {gh}]n[/] Theme [{p.text_muted}]{p.name}[/]"
        )


class CondorTUI(App[None]):

    TITLE = "CONDOR — Frigate Remote Detector"

    CSS = """
    Screen {
        background: $condor-bg;
        layers: base;
    }

    #graphs-row {
        height: 1fr;
    }

    #graphs-row GraphPanel {
        width: 1fr;
        height: 1fr;
    }

    #workers-row {
        height: 11;
        layout: horizontal;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit"),
        ("l", "legend", "Legend"),
        ("w", "workers", "Workers"),
        ("m", "prev_theme", "Prev Theme"),
        ("n", "next_theme", "Next Theme"),
    ]

    def __init__(self) -> None:
        self._palette_names: list[str] = available_palettes()
        self._palette_idx, self._palette = _load_saved_theme(self._palette_names)
        super().__init__()
        self._snapshot: dict = {}
        self._layout_ready = False
        self._num_workers = 0
        self._stats_writer: asyncio.StreamWriter | None = None

    @property
    def palette(self) -> Palette:
        return self._palette

    def set_palette(self, palette: Palette) -> None:
        """Swap the active color palette and refresh all CSS and widgets."""
        self._palette = palette
        self.refresh_css()
        for widget in self.query("*"):
            widget.refresh()

    def action_prev_theme(self) -> None:
        self._palette_idx = (self._palette_idx - 1) % len(self._palette_names)
        name = self._palette_names[self._palette_idx]
        self.set_palette(load_palette(name))
        _save_theme(name)

    def action_next_theme(self) -> None:
        self._palette_idx = (self._palette_idx + 1) % len(self._palette_names)
        name = self._palette_names[self._palette_idx]
        self.set_palette(load_palette(name))
        _save_theme(name)

    def get_css_variables(self) -> dict[str, str]:
        base = super().get_css_variables()
        base.update(self._palette.css_variables())
        return base

    def compose(self) -> ComposeResult:
        yield StatusBanner()
        with Horizontal(id="graphs-row"):
            yield StackedBarPanel()
            yield GraphPanel("THROUGHPUT", "req/s", "throughput-panel")
        with Horizontal(id="workers-row"):
            yield GlobalPanel()
            yield GpuPanel()
        yield AppFooter()

    @work(exclusive=True)
    async def _read_stats(self) -> None:
        while True:
            try:
                reader, writer = await asyncio.open_unix_connection(SOCKET_PATH)
                self._stats_writer = writer
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

    def action_legend(self) -> None:
        legend = self.query_one(LegendPanel)
        legend.display = not legend.display

    def action_workers(self) -> None:
        gpu_panel = self.query_one(GpuPanel)
        show_workers = gpu_panel.display
        gpu_panel.display = not show_workers
        for wp in self.query(WorkerPanel):
            wp.display = show_workers
        self.query_one(AppFooter).workers_visible = show_workers

    def _update_disconnected(self) -> None:
        try:
            self.query_one(StatusBanner).update_disconnected()
        except Exception:
            pass

    async def _update_ui(self, data: dict) -> None:
        self._snapshot = data
        cfg = data.get("config", {})
        num_workers = cfg.get("num_workers", 1)
        base_port = cfg.get("base_port", 5555)

        uptime = data.get("uptime_s", 0.0)
        workers_active = data.get("active_workers", 0)
        model_raw = data.get("active_model", "")
        model = Path(model_raw).stem if model_raw else "(none)"

        self.query_one(StatusBanner).update_status(
            uptime,
            workers_active,
            num_workers,
            model,
            data.get("active_postprocessor", ""),
        )

        try:
            n = self.query_one("#latency-panel", StackedBarPanel).size.width or 200
        except Exception:
            n = 200

        lat_data = list(data.get("sparkline_latency", []))[-n:]
        tput_data = list(data.get("sparkline_throughput", []))[-n:]

        lat_summary = ""
        if any(v > 0 for v in lat_data):
            nonzero = [v for v in lat_data if v > 0]
            lat_summary = (
                f"  now {lat_data[-1]:.1f}  "
                f"avg {sum(nonzero) / len(nonzero):.1f}  "
                f"peak {max(lat_data):.1f}"
            )

        stages_raw = data.get("sparkline_stages", {})
        stages: dict[str, list[float]] = {
            stage: list(stages_raw.get(stage, []))[-n:] for stage in STAGE_ORDER
        }

        self.query_one("#latency-panel", StackedBarPanel).update_data(
            lat_data, stages, lat_summary
        )

        tput_summary = ""
        if any(v > 0 for v in tput_data):
            nonzero = [v for v in tput_data if v > 0]
            tput_summary = (
                f"  now {tput_data[-1]:.1f}  "
                f"avg {sum(nonzero) / len(nonzero):.1f}  "
                f"peak {max(tput_data):.1f}"
            )
        self.query_one("#throughput-panel", GraphPanel).update_data(tput_data, tput_summary)

        workers = data.get("workers", {})
        if not self._layout_ready or self._num_workers != num_workers:
            await self._create_worker_panels(num_workers, base_port)
            self._layout_ready = True
            self._num_workers = num_workers

        for wid_str, wdata in workers.items():
            try:
                wid = int(wid_str)
                panel = self.query_one(f"#worker-panel-{wid}", WorkerPanel)
                panel.update_data(wdata, data)
            except Exception:
                pass

        try:
            self.query_one("#global-panel", GlobalPanel).update_data(data)
        except Exception:
            pass

        gpu_data = data.get("gpu")
        if gpu_data:
            spark = list(gpu_data.get("sparkline", []))[-n:]
            try:
                self.query_one(GpuPanel).update_data({**gpu_data, "sparkline": spark})
            except Exception:
                pass

    async def _create_worker_panels(self, num_workers: int, base_port: int) -> None:
        container = self.query_one("#workers-row", Horizontal)
        for child in list(container.children):
            if isinstance(child, WorkerPanel):
                await child.remove()
        workers_visible = self.query_one(AppFooter).workers_visible
        gpu_panel = self.query_one(GpuPanel)
        for i in range(num_workers):
            wp = WorkerPanel(i, base_port + i)
            wp.display = workers_visible
            await container.mount(wp, before=gpu_panel)


def main() -> None:
    app = CondorTUI()
    app.run()


if __name__ == "__main__":
    main()
