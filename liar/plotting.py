"""Shared publication-figure colors sampled from Matplotlib's ``turbo`` map.

The fixed positions span the full rainbow while keeping semantic roles stable
across every figure generator.  Axes and grids remain neutral; subdued anchor
violet is reused for baselines and random-control reference marks.
"""
from __future__ import annotations

from matplotlib import colormaps
from matplotlib.colors import to_hex


TURBO_POSITIONS = {
    "anchor": 0.00,      # deep violet: unprojected vector / dark anchor
    "caa": 0.10,         # blue: CAA family
    "perp": 0.18,        # bright blue: projected-vector condition
    "positive": 0.40,    # green: fill-only positive emphasis
    "selected": 0.60,    # yellow: selected operating point
    "mass_mean": 0.78,   # orange: mass-mean family
    "parallel": 0.88,    # warm red: parallel component
    "gate": 0.98,        # deep red: incoherent region / rejection mark
}


def _sample_turbo(position: float) -> str:
    return to_hex(colormaps["turbo"](position), keep_alpha=False).upper()


TURBO = {name: _sample_turbo(position) for name, position in TURBO_POSITIONS.items()}

INK = "#1F2430"
MUTED = "#6F768A"
GRID = "#E1E3EA"
NEUTRAL = "#7A828F"

FAMILY_COLOR = {"dec": TURBO["caa"], "mm": TURBO["mass_mean"]}
FAMILY_MARKER = {"dec": "o", "mm": "s"}
