"""Figure style shared by all experiments.

Colours come from a validated palette: categorical slots are used in a fixed
order (never cycled), lattice sizes - an ordered variable - use a single-hue
blue ramp whose steps were checked for monotone, visibly separated lightness
(at most five sizes per panel), and exact reference curves are drawn in
secondary ink so they never compete with the data.  Grids and axes are solid
hairlines one step off the surface.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

__all__ = [
    "SURFACE",
    "INK",
    "INK_2",
    "MUTED",
    "GRID",
    "AXIS",
    "CATEGORICAL",
    "ORDINAL",
    "setup",
    "ordinal_colors",
    "figure",
    "errorbar",
    "reference_line",
    "label_end",
    "plain_log_ticks",
    "save",
]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# categorical slots in validated order: blue, orange, aqua, yellow, magenta, green, violet, red
CATEGORICAL = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]

# single-hue ordinal ramp (light -> dark); every subset keeps the lightness gaps
ORDINAL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]


def setup() -> None:
    """Install the rcParams (idempotent)."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlecolor": INK,
            "axes.labelsize": 9.5,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "xtick.minor.size": 0,
            "ytick.minor.size": 0,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "legend.labelcolor": INK_2,
            "legend.handlelength": 1.6,
            "lines.linewidth": 1.6,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "lines.markersize": 6.0,
            "errorbar.capsize": 0,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def ordinal_colors(n: int) -> list[str]:
    """``n <= 5`` ramp colours, spread evenly from light to dark."""
    if not 1 <= n <= len(ORDINAL):
        raise ValueError(
            f"the ordinal ramp supports 1..{len(ORDINAL)} series; facet or subset the sizes"
        )
    if n == 1:
        return [ORDINAL[2]]
    idx = np.round(np.linspace(0, len(ORDINAL) - 1, n)).astype(int)
    return [ORDINAL[i] for i in idx]


def figure(ncols: int = 1, nrows: int = 1, width: float = 3.4, height: float = 2.7, **kw):
    """Figure with ``nrows x ncols`` panels of the given size (inches)."""
    setup()
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(width * ncols, height * nrows), constrained_layout=True, **kw
    )
    return fig, axes


def errorbar(ax, x, y, dy=None, *, color, label=None, line: bool = True, marker: str = "o", **kw):
    """Data series: 1.6 pt line, markers with a surface-coloured ring, thin error bars."""
    style = dict(
        color=color,
        marker=marker,
        markersize=5.5,
        markeredgecolor=SURFACE,
        markeredgewidth=1.1,
        elinewidth=1.0,
        ecolor=color,
        label=label,
        linestyle="-" if line else "none",
        linewidth=1.6 if line else 0,
        zorder=3,
    )
    style.update(kw)
    return ax.errorbar(x, y, yerr=dy, **style)


def reference_line(ax, *, x=None, y=None, label: str | None = None, text_pos: str = "top"):
    """Exact or literature value drawn as a secondary-ink hairline with a short label."""
    if x is not None:
        ax.axvline(x, color=INK_2, linewidth=0.9, zorder=2)
        if label:
            ymin, ymax = ax.get_ylim()
            yt = ymax if text_pos == "top" else ymin
            va = "top" if text_pos == "top" else "bottom"
            ax.annotate(
                label,
                (x, yt),
                xytext=(3, -2 if va == "top" else 2),
                textcoords="offset points",
                fontsize=8,
                color=INK_2,
                va=va,
                ha="left",
            )
    if y is not None:
        ax.axhline(y, color=INK_2, linewidth=0.9, zorder=2)
        if label:
            xmin, xmax = ax.get_xlim()
            ax.annotate(
                label,
                (xmax, y),
                xytext=(-2, 3),
                textcoords="offset points",
                fontsize=8,
                color=INK_2,
                ha="right",
                va="bottom",
            )


def plain_log_ticks(ax, axis: str = "y") -> None:
    """Plain-number labels (0.6, 20, 500 ...) on a logarithmic axis.

    Tick density adapts to the number of decades shown, so a narrow range
    still gets labels.  Call after the data are plotted.
    """
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    for name in ("x", "y") if axis == "both" else (axis,):
        a = ax.xaxis if name == "x" else ax.yaxis
        lo, hi = ax.get_xlim() if name == "x" else ax.get_ylim()
        if lo <= 0 or hi <= 0:
            continue
        decades = np.log10(hi / lo)
        if getattr(a.get_transform(), "base", 10) == 2:  # powers of two: keep ticks, plain labels
            a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            continue
        subs = tuple(range(1, 10)) if decades < 1 else ((1, 2, 5) if decades < 2.5 else (1,))
        a.set_major_locator(LogLocator(base=10, subs=subs))
        a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        a.set_minor_formatter(NullFormatter())


def label_end(ax, x, y, text: str, *, dx: float = 4, dy: float = 0, color=INK_2) -> None:
    """Direct label at the end of a series (text stays in ink, never the series colour)."""
    ax.annotate(
        text,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=8,
        color=color,
        va="center",
        ha="left",
    )


def save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
