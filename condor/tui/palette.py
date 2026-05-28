from __future__ import annotations

import colorsys
from dataclasses import dataclass
from pathlib import Path

# Canonical stage processing order.
STAGE_ORDER: list[str] = ["mcpy", "h2d", "swait", "exec", "d2h", "pp"]

_PALETTES_DIR = Path(__file__).parent / "palettes"


def _dp_to_hex(line: str) -> str:
    """Convert a 16-bit NsCDE .dp color line '#RRRRGGGGBBBB' → '#rrggbb'."""
    s = line.strip().lstrip("#")
    return f"#{int(s[0:4], 16) >> 8:02x}{int(s[4:8], 16) >> 8:02x}{int(s[8:12], 16) >> 8:02x}"


def _complement(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
    hue, lum, sat = colorsys.rgb_to_hls(r, g, b)
    r2, g2, b2 = colorsys.hls_to_rgb((hue + 0.5) % 1.0, lum, sat)
    return f"#{round(r2 * 255):02x}{round(g2 * 255):02x}{round(b2 * 255):02x}"


def _darken(hex_color: str, factor: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
    hue, lum, sat = colorsys.rgb_to_hls(r, g, b)
    r2, g2, b2 = colorsys.hls_to_rgb(hue, lum * factor, sat)
    return f"#{round(r2 * 255):02x}{round(g2 * 255):02x}{round(b2 * 255):02x}"


@dataclass(frozen=True)
class Palette:
    name: str
    border: str        # single border color for all panels
    background: str    # app/screen background
    info_bar_bg: str   # status banner background
    panel_bg: str      # widget panel background
    text: str          # primary text
    text_muted: str    # dim/secondary text
    gradient_low: str  # sparkline/bar gradient cool end
    gradient_high: str # sparkline/bar gradient warm end
    # Colors aligned with STAGE_ORDER: mcpy, h2d, swait, exec, d2h, pp
    stage_colors: tuple[str, ...]

    def stage_color(self, stage: str) -> str:
        return self.stage_colors[STAGE_ORDER.index(stage)]

    def stage_color_map(self) -> dict[str, str]:
        return dict(zip(STAGE_ORDER, self.stage_colors))

    def gradient_color(self, fraction: float) -> str:
        """Interpolate between gradient_low (0.0) and gradient_high (1.0)."""
        t = max(0.0, min(1.0, fraction))
        lo = self.gradient_low.lstrip("#")
        hi = self.gradient_high.lstrip("#")
        r = round(int(lo[0:2], 16) + t * (int(hi[0:2], 16) - int(lo[0:2], 16)))
        g = round(int(lo[2:4], 16) + t * (int(hi[2:4], 16) - int(lo[2:4], 16)))
        b = round(int(lo[4:6], 16) + t * (int(hi[4:6], 16) - int(lo[4:6], 16)))
        return f"#{r:02x}{g:02x}{b:02x}"

    def css_variables(self) -> dict[str, str]:
        """Inject palette colors as Textual CSS variables ($condor-*)."""
        return {
            "condor-border": self.border,
            "condor-bg": self.background,
            "condor-info-bg": self.info_bar_bg,
            "condor-panel-bg": self.panel_bg,
            "condor-text": self.text,
            "condor-text-muted": self.text_muted,
            "condor-grad-low": self.gradient_low,
            "condor-grad-high": self.gradient_high,
        }

    @classmethod
    def from_dp(cls, path: Path) -> "Palette":
        """Parse an NsCDE .dp file and derive TUI roles from its 8 slots.

        Slot → role mapping:
          0 → gradient high anchor
          1 → (unused — original CDE inactive border)
          2 → (unused — original CDE workspace backdrop)
          3 → panel / container background; also darkened 0.30× for screen bg
          4 → stage base: mcpy
          5 → stage base: h2d  (also gradient low anchor)
          6 → stage base: swait
          7 → stage base: exec
        Complements (+180° hue) are derived for d2h and pp.
        Border is slot-0 darkened to 55% luminance.
        App background and status-bar background share the same dark colour
        derived from slot-3 at 30% luminance.
        """
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        s = [_dp_to_hex(ln) for ln in lines[:8]]
        dark_bg = _darken(s[3], 0.30)
        return cls(
            name=path.stem,
            border=_darken(s[0], 0.55),
            background=dark_bg,
            info_bar_bg=dark_bg,
            panel_bg=s[3],
            text="#ffffff",
            text_muted="#aaaaaa",
            gradient_low=s[5],
            gradient_high=s[0],
            stage_colors=(
                s[4],              # mcpy
                s[5],              # h2d
                s[6],              # swait
                s[7],              # exec
                _complement(s[5]), # d2h
                _complement(s[4]), # pp
            ),
        )


def load_palette(name: str = "Broica") -> Palette:
    return Palette.from_dp(_PALETTES_DIR / f"{name}.dp")


def available_palettes() -> list[str]:
    return sorted(p.stem for p in _PALETTES_DIR.glob("*.dp"))
