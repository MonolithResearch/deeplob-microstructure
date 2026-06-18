"""
Render the authoritative real-data results chart for the README.

These are the REAL numbers from the Strategy-16 research on genuine CME 6E
(EUR futures) Databento mbp-10 LOB data — not the synthetic demo. They are
hard-coded here (no raw data is shipped) and must match docs/RESULTS.md and the
README exactly.

    In-sample Sharpe            : 10.81   (the foresight)
    Net Sharpe at 3x cost       : -14.63  (after crossing the spread)
    Trades                      : 2,264
    Deflated Sharpe Ratio (DSR) : ~ 0

Output: docs/results/real_data_foresight_vs_cost.png
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import brand

# ── the only numbers permitted (from the real CME 6E research) ────────────────
IS_SHARPE = 10.81
NET_SHARPE_3X = -14.63
N_TRADES = 2264


def main():
    brand.apply_brand(plt)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    labels = ["In-sample\n(the foresight)", "Out-of-sample net\n(3x cost, 2,264 trades)"]
    values = [IS_SHARPE, NET_SHARPE_3X]
    colors = [brand.PLUM, brand.INK]

    bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor=brand.INK, linewidth=1.2)
    ax.axhline(0, color=brand.PLUM_DARK, lw=1.0)

    for b, v in zip(bars, values):
        ax.annotate(f"{v:+.2f}",
                    (b.get_x() + b.get_width() / 2, v),
                    ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=14, fontweight="bold", color=brand.INK,
                    xytext=(0, 6 if v >= 0 else -8), textcoords="offset points")

    ax.set_ylabel("Annualised Sharpe ratio")
    ax.set_title("DeepLOB + OFI on real CME 6E LOB: foresight is real, and that is the trap")
    ax.set_ylim(min(values) - 3, max(values) + 3)
    ax.text(0.5, 0.5,
            "Same signal, same data.\n"
            "The 25.4-Sharpe gap between the two bars\n"
            "is the cost of crossing the spread 2,264 times.\n"
            "Deflated Sharpe (DSR) ~ 0 after deflation.",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color=brand.PLUM_DARK, style="italic",
            bbox=dict(boxstyle="round,pad=0.5", fc=brand.PAPER, ec=brand.PLUM, alpha=0.9))

    brand.brand_footer(fig)
    fig.tight_layout()
    out = ROOT / "docs" / "results" / "real_data_foresight_vs_cost.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
