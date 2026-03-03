"""ASCII art assets for the Condor TUI.

All logos use Unicode box-drawing and block characters for a 90s BBS aesthetic.
Every logo is exactly 6 lines tall to match the CONDOR header logo.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# CONDOR block logo (6 lines)
# ---------------------------------------------------------------------------

CONDOR_LOGO = """\
 ██████╗ ██████╗ ███╗  ██╗██████╗  ██████╗ ██████╗
██╔════╝██╔═══██╗████╗ ██║██╔══██╗██╔═══██╗██╔══██╗
██║     ██║   ██║██╔██╗██║██║  ██║██║   ██║██████╔╝
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
