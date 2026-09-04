"""Shared chart styling so every figure in `reports/figures` matches."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # scripts run headless
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

FIGURES = Path(__file__).resolve().parents[2] / "reports" / "figures"

INK = "#12403c"
ACCENT = "#f2a0ab"
TRend = "#c2453f"
GRID = "#d8dee6"


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    })


def regression_scatter(ax, x, y, *, xlabel, ylabel, title, label=None,
                       color=INK, annotate=True):
    """Scatter with an OLS trend line and the fit statistics printed on it.

    Returns the `scipy.stats.linregress` result so callers can tabulate it.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]

    ax.scatter(x, y, s=26, alpha=0.75, color=color, edgecolor="none", label=label)
    fit = stats.linregress(x, y)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, fit.intercept + fit.slope * xs, color=TRend, lw=1.8, ls="--")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if annotate:
        ax.annotate(
            f"slope {fit.slope:.4g}\n$R^2$ {fit.rvalue**2:.3f}\np {fit.pvalue:.3g}\nn {len(x)}",
            xy=(0.03, 0.03), xycoords="axes fraction", fontsize=8,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRID, alpha=0.9),
        )
    return fit


def save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path)
    plt.close(fig)
    return path
