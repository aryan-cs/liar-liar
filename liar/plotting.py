"""Shared publication styling for Matplotlib figures.

The fixed positions span the full rainbow while keeping semantic roles stable
across every figure generator.  Axes and grids remain neutral; subdued anchor
violet is reused for baselines and random-control reference marks.

The NeurIPS paper uses Nimbus Roman, the Times-compatible face selected by the
venue style, with Computer Modern mathematics.  Matplotlib's Times face has
the same publication typography and metrics; the fallbacks keep regeneration
portable when that exact system font is unavailable.  TrueType embedding
avoids the Type 3 glyphs produced by Matplotlib's default PDF configuration.
"""
from __future__ import annotations

from matplotlib import colormaps, font_manager
from matplotlib.colors import to_hex
from matplotlib.font_manager import FontProperties


PAPER_SERIF_CANDIDATES = (
    "Nimbus Roman No9 L",
    "Nimbus Roman",
    "Times",
    "Times New Roman",
    "TeX Gyre TermesX",
    "TeX Gyre Termes",
    "Liberation Serif",
)


def _resolve_paper_serif() -> str:
    """Select a Times-compatible face without silently falling back to DejaVu."""
    for family in PAPER_SERIF_CANDIDATES:
        try:
            font_manager.findfont(FontProperties(family=family), fallback_to_default=False)
        except ValueError:
            continue
        return family
    raise RuntimeError(
        "No Times-compatible serif font is installed; install Nimbus Roman, "
        "Times, TeX Gyre Termes, or Liberation Serif before making figures."
    )


PAPER_SERIF = _resolve_paper_serif()

PAPER_FONT_RC = {
    "font.family": "serif",
    "font.serif": [PAPER_SERIF],
    "mathtext.fontset": "cm",
    "mathtext.default": "it",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


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
