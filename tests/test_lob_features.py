import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lob_features import FEATURE_NAMES, compute_features, compute_labels, per_level_ofi


def _synthetic_lob(rows: int = 160, levels: int = 10) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=rows, freq="5s", tz="UTC")
    mid = 1.10 + np.arange(rows) * 1e-6
    data = {}
    for level in range(levels):
        spread = 1e-5 * (level + 1)
        data[f"bid_px_{level:02d}"] = mid - spread
        data[f"ask_px_{level:02d}"] = mid + spread
        data[f"bid_sz_{level:02d}"] = 10 + level + (np.arange(rows) % 5)
        data[f"ask_sz_{level:02d}"] = 11 + level + (np.arange(rows) % 7)
    return pd.DataFrame(data, index=idx)


def test_features_are_finite_when_deep_levels_are_missing():
    lob = _synthetic_lob()
    lob.loc[lob.index[20:40], ["bid_px_09", "ask_px_09"]] = np.nan
    lob.loc[lob.index[30:50], ["bid_sz_08", "ask_sz_08"]] = np.nan
    features = compute_features(lob)
    assert not features[FEATURE_NAMES].isna().any().any()
    assert np.isfinite(features[FEATURE_NAMES].to_numpy()).all()


def test_top_of_book_price_signs():
    features = compute_features(_synthetic_lob())
    assert (features["bid_p_1"] < 0).all()   # bids below mid
    assert (features["ask_p_1"] > 0).all()   # asks above mid


def test_labels_mark_directional_edges():
    idx = pd.date_range("2026-01-01", periods=20, freq="5s", tz="UTC")
    mid = pd.Series(np.linspace(1.0, 1.01, len(idx)), index=idx)
    spread = pd.Series(0.1, index=idx)
    labels = compute_labels(mid, spread, horizon=3, spread_multiplier=0.1)
    assert labels.iloc[10] == 2   # rising mid -> UP
    assert labels.iloc[-1] == 1   # no forward horizon -> FLAT


def test_per_level_ofi_no_lookahead_first_bar_is_nan():
    lob = _synthetic_lob()
    ofi = per_level_ofi(lob)
    assert len(ofi) == 10
    # first bar has no previous snapshot -> NaN on every level
    for series in ofi:
        assert np.isnan(series.iloc[0])
