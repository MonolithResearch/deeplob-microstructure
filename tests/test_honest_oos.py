import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.synthetic_lob import generate_synthetic_lob
from src.honest_oos import run_honest_oos, compute_ofi_arrays


def test_ofi_first_bar_and_gaps_are_masked():
    lob = generate_synthetic_lob(n_bars=2000, seed=1)
    mid, half_spread_bp, ofi_lvl, ts, spread = compute_ofi_arrays(lob)
    # first bar has no previous snapshot
    assert np.isnan(ofi_lvl[0]).all()
    # half-spread is non-negative everywhere it is finite
    finite = np.isfinite(half_spread_bp)
    assert (half_spread_bp[finite] >= 0).all()


def test_honest_oos_runs_and_reports_a_verdict():
    # small but enough to clear the 200-trade IS floor with a permissive grid
    lob = generate_synthetic_lob(n_bars=40_000, seed=3, ofi_signal=0.0)
    res = run_honest_oos(
        lob,
        thresholds=[0.5, 1.0, 1.5],
        z_wins=[120],
        smooth_wins=[1, 3],
        horizons=[1, 3, 6],
        min_trades_is=100,
        verbose=False,
    )
    assert res["verdict"] in {"PASS", "PROBATION", "FAIL"}
    assert res["oos_summary"]["n_trades"] >= 0
    # cost bar must be a real positive number (you always cross the spread)
    assert res["oos_summary"]["avg_cost_bp"] > 0
