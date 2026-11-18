import numpy as np
import pandas as pd
import pytest

from src import honest_oos
from src.lob_features import compute_features, compute_labels
from src.synthetic_lob import generate_synthetic_lob


@pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
def test_clock_units_and_gap_mask(unit):
    lob = generate_synthetic_lob(n_bars=100)
    lob = lob.drop(lob.index[50])
    lob.index = lob.index.as_unit(unit)
    _, _, ofi, seconds, _ = honest_oos.compute_ofi_arrays(lob)
    assert seconds[0] == lob.index[0].timestamp()
    np.testing.assert_array_equal(np.diff(seconds)[48:51], [5, 10, 5])
    assert np.isnan(ofi[50]).all()


def test_fixed_horizon_never_crosses_a_missing_bar():
    mid = np.ones(4)
    z = np.array([2.0, np.nan, np.nan, np.nan])
    exits, *_ = honest_oos.run_trades(z, mid, mid, np.array([0., 5., 15., 20.]),
                                      3, 1., 1., 15., 0, 4)
    assert len(exits) == 0


def test_model_inputs_require_a_contiguous_clock():
    lob = generate_synthetic_lob(n_bars=100)
    with pytest.raises(ValueError, match="clock"):
        compute_features(lob.drop(lob.index[50]))


def test_label_horizon_requires_a_contiguous_clock():
    lob = generate_synthetic_lob(n_bars=100)
    features = compute_features(lob)
    gappy = features.drop(features.index[50])
    with pytest.raises(ValueError, match="clock"):
        compute_labels(gappy["mid"], gappy["spread_bps"])


def test_one_round_trip_after_nan_warmup_closes_on_last_bar():
    z = np.array([np.nan, 2., 2., 2., 2.])
    exits, gross, net, cost, _ = honest_oos.run_trades(
        z, np.ones(5), np.full(5, 1.), np.arange(5.) * 5,
        3, 1., 1., 15., 0, 5)
    np.testing.assert_array_equal([exits, gross, net, cost], [[20.], [0.], [-3.], [3.]])


def test_signal_warms_up_again_after_a_gap():
    lob = generate_synthetic_lob(n_bars=200)
    lob = lob.drop(lob.index[100])
    result = honest_oos.run_honest_oos(
        lob, is_frac=0.5, depths=[1], smooth_wins=[1], horizons=[1],
        z_wins=[20], thresholds=[0.], min_trades_is=1, verbose=False)
    exits = result["oos_trades"]["exit_ts"]
    assert len(exits) > 0
    assert exits.min() >= lob.index[111].timestamp()


@pytest.mark.parametrize("direction", [-1, 1])
def test_marks_reconcile_across_midnight_and_drive_daily_sharpe(monkeypatch, direction):
    # One trade spanning midnight, then a flat full day and a final partial day.
    n_is = 20
    n_oos = 17285
    lob = generate_synthetic_lob(n_bars=n_is + n_oos, session_gap_every=0)
    split_time = pd.Timestamp("2026-01-01 23:59:50", tz="UTC")
    lob.index = pd.date_range(split_time - pd.Timedelta(seconds=5 * n_is),
                              periods=len(lob), freq="5s")
    mid = np.full(len(lob), 100.0)
    mid[n_is+1:n_is+4] = [101., 99., 100.5]
    lob["bid_px_00"] = mid
    lob["ask_px_00"] = mid
    z = np.zeros(len(lob))
    z[[0, n_is]] = 2. * direction
    monkeypatch.setattr(honest_oos, "rolling_z", lambda series, win: z.copy())
    result = honest_oos.run_honest_oos(
        lob, is_frac=(n_is + 0.5) / len(lob), commission_bp_rt=1.,
        min_trades_is=1, depths=[1], smooth_wins=[1], horizons=[3],
        z_wins=[8], thresholds=[1.], verbose=False)
    daily = np.array([100. * direction, -50. * direction - 1., 0.])
    stats = result["oos_summary"]
    assert stats["sharpe_net"] == pytest.approx(daily.mean() / daily.std(ddof=1) * np.sqrt(252))
    marks = result["oos_marks"]
    assert marks.index.equals(lob.index[n_is:])
    gross = np.array([0., 100., -200., 150.]) * direction
    np.testing.assert_allclose(marks.gross_bp.iloc[:4], gross)
    np.testing.assert_allclose(marks.net_bp.iloc[:4], gross - [0., 0., 0., 1.])
    assert marks.net_bp.sum() == pytest.approx(result["oos_trades"]["net_bp"].sum())
