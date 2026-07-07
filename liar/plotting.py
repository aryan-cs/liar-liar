"""Shared publication styling for Matplotlib figures.

CAA and mass-mean use a fixed sky-purple pair so family identity remains
distinct from outcome or quality.  The remaining fixed positions span the
``turbo`` rainbow while keeping intervention and state roles stable across
every figure generator.  Axes and grids remain neutral; subdued anchor violet
is reused for baselines and random-control reference marks.

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
from matplotlib.patches import Rectangle


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
    "cool_accent": 0.10,  # legacy cool categorical accent
    "perp": 0.18,        # bright blue: projected-vector condition
    "positive": 0.40,    # green: fill-only positive emphasis
    "selected": 0.60,    # yellow: selected operating point
    "warm_accent": 0.78,  # legacy warm categorical accent
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
HATCH_COLOR = "#FFFFFF"

FAMILY_COLOR = {
    "dec": "#56B4E9",  # sky: CAA
    "mm": "#AA3377",   # purple: mass-mean
}
FAMILY_MARKER = {"dec": "o", "mm": "s"}


def add_white_hatch_overlay(ax, bars, hatch: str) -> None:
    """Overlay white vector hatches without replacing a bar's dark keyline."""
    if not hatch:
        return
    for bar in bars:
        overlay = Rectangle(
            (bar.get_x(), bar.get_y()),
            bar.get_width(),
            bar.get_height(),
            facecolor="none",
            edgecolor=HATCH_COLOR,
            linewidth=0,
            hatch=hatch,
            label="_nolegend_",
            zorder=bar.get_zorder() + 0.1,
        )
        overlay.set_in_layout(False)
        ax.add_patch(overlay)
        outline = Rectangle(
            (bar.get_x(), bar.get_y()),
            bar.get_width(),
            bar.get_height(),
            facecolor="none",
            edgecolor=bar.get_edgecolor(),
            linewidth=bar.get_linewidth(),
            alpha=bar.get_alpha(),
            label="_nolegend_",
            zorder=bar.get_zorder() + 0.2,
        )
        outline.set_in_layout(False)
        ax.add_patch(outline)
