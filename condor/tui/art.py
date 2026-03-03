"""ASCII art assets for the Condor TUI.

All logos use Unicode box-drawing and block characters for a 90s BBS aesthetic.
Every logo is exactly 6 lines tall to match the CONDOR header logo.
"""

from __future__ import annotations

import random
from typing import NamedTuple

# ---------------------------------------------------------------------------
# CONDOR block logo (6 lines)
# ---------------------------------------------------------------------------

CONDOR_LOGO = """\
 ██████╗ ██████╗ ███╗   ██╗██████╗  ██████╗ ██████╗
██╔════╝██╔═══██╗████╗  ██║██╔══██╗██╔═══██╗██╔══██╗
██║     ██║   ██║██╔██╗ ██║██║  ██║██║   ██║██████╔╝
██║     ██║   ██║██║╚██╗██║██║  ██║██║   ██║██╔══██╗
╚██████╗╚██████╔╝██║ ╚████║██████╔╝╚██████╔╝██║  ██║
 ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═════╝  ╚═════╝ ╚═╝  ╚═╝"""

# ---------------------------------------------------------------------------
# Provider logos — all exactly 6 lines, subtitle embedded on the last line
# ---------------------------------------------------------------------------

_TRT_LOGO = """\
████████╗██████╗ ████████╗
╚══██╔══╝██╔══██╗╚══██╔══╝
   ██║   ██████╔╝   ██║
   ██║   ██╔══██╗   ██║
   ╚═╝   ╚═╝  ╚═╝   ╚═╝
   TensorRT Backend"""

_ONNX_LOGO = """\
 ██████╗ ███╗  ██╗███╗  ██╗██╗  ██╗
██╔═══██╗████╗ ██║████╗ ██║╚██╗██╔╝
██║   ██║██╔██╗██║██╔██╗██║ ╚███╔╝
╚██████╔╝██║╚████║██║╚████║ ██╔██╗
 ╚═════╝ ╚═╝ ╚═══╝╚═╝ ╚═══╝╚═╝ ╚═╝
 ONNX Runtime Backend"""

_OV_LOGO = """\
 ██████╗ ██╗   ██╗
██╔═══██╗██║   ██║
██║   ██║██║   ██║
╚██████╔╝╚██╗ ██╔╝
 ╚═════╝  ╚████╔╝
OpenVINO™ Runtime"""

_CPU_LOGO = """\
 ██████╗██████╗ ██╗   ██╗
██╔════╝██╔══██╗██║   ██║
██║     ██████╔╝██║   ██║
██║     ██╔═══╝ ██║   ██║
╚██████╗██║     ╚██████╔╝
CPU / ONNX Runtime"""

_GENERIC_LOGO = """\
╔══════════════════════╗
║  INFERENCE BACKEND   ║
║                      ║
║  condor · v0.1.0     ║
╚══════════════════════╝
"""


def get_provider_logo(provider: str) -> str:
    """Return the 6-line logo string for the given provider name."""
    p = provider.lower()
    if p == "tensorrt":
        return _TRT_LOGO
    if p == "onnx":
        return _ONNX_LOGO
    if p == "openvino":
        return _OV_LOGO
    if p in ("cpu", ""):
        return _CPU_LOGO
    return _GENERIC_LOGO


# ---------------------------------------------------------------------------
# Bird animation
# ---------------------------------------------------------------------------

class BirdFrame(NamedTuple):
    """A single frame of the bird animation with wings."""
    left_wing: str
    body: str
    right_wing: str


# Bird frames for flapping animation
BIRD_FRAMES = [
    BirdFrame("╱", "◉", "╲"),  # wings up
    BirdFrame("╱ ", "◉", " ╲"),  # wings spread
    BirdFrame(" ╲", "◉", "╱ "),  # wings down
    BirdFrame("╲ ", "◉", " ╱"),  # wings folding
]


def get_bird_frame(frame_index: int) -> str:
    """Return the bird character for the given frame index."""
    bird = BIRD_FRAMES[frame_index % len(BIRD_FRAMES)]
    return f"{bird.left_wing}{bird.body}{bird.right_wing}"


def build_combined_logo(condor_lines: list[str], provider_lines: list[str], bird_x: int | None = None, bird_y: int | None = None, bird_frame: int = 0) -> str:
    """Combine CONDOR logo (left) and provider logo (right) into a single box."""
    # Fixed width for consistent layout
    total_width = 90
    condor_width = 54

    combined_lines = []

    for i in range(max(len(condor_lines), len(provider_lines))):
        condor_line = condor_lines[i] if i < len(condor_lines) else ""
        provider_line = provider_lines[i] if i < len(provider_lines) else ""

        # Left-align CONDOR, right-align provider
        condor_padded = condor_line.ljust(condor_width)
        provider_padded = provider_line.rjust(total_width - condor_width)

        combined = condor_padded + provider_padded

        combined_lines.append(combined)

    return "\n".join(combined_lines)
