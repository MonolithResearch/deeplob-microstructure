"""
Monolith Research brand styling for matplotlib charts.

Palette (use these EXACT hex values everywhere):
    Ink   #0E0C14  near-black
    Plum  #A78AA3  accent
    Paper #F3EFE8  light
"""
from __future__ import annotations

INK = "#0E0C14"
PLUM = "#A78AA3"
PAPER = "#F3EFE8"
PLUM_DARK = "#7A6177"
BRAND_NAME = "Monolith Research"
BRAND_TAGLINE = "EVIDENCE OVER NARRATIVE"


def apply_brand(plt) -> None:
    """Apply the Monolith palette + typography to a matplotlib pyplot module."""
    plt.rcParams.update({
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "grid.color": PLUM,
        "grid.alpha": 0.25,
        "axes.grid": True,
        "font.family": "serif",
        "font.serif": ["Cormorant Garamond", "Garamond", "Georgia", "serif"],
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "figure.titlesize": 17,
        "figure.titleweight": "bold",
    })


def brand_footer(fig) -> None:
    """Stamp the sigma mark + tagline on a figure (bottom-right)."""
    fig.text(0.995, 0.01, f"Σ  {BRAND_NAME} · {BRAND_TAGLINE}",
             ha="right", va="bottom", color=PLUM_DARK, fontsize=8, style="italic")
