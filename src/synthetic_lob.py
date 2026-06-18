"""
Small SYNTHETIC limit-order-book generator.

This exists so the whole pipeline (features -> model -> honest backtest) is
runnable WITHOUT any licensed market data. The real research used Databento
mbp-10 snapshots on CME 6E (EUR futures); that raw data is proprietary and is
NOT included in this repository. This generator emits the identical column
schema (bid_px_00..09, ask_px_00..09, bid_sz_00..09, ask_sz_00..09) on a regular
5-second grid so every downstream module consumes it unchanged.

It is a TOY. The mid follows a driftless random walk with a tiny, deliberately
weak order-flow-imbalance term, plus realistic-looking microstructure (a few-tick
spread, depth decaying away from the touch, occasional session gaps). It is good
enough to exercise the code paths and to reproduce the qualitative lesson —
in-sample fitting finds "signal" that does not survive the spread — but it is not
a model of any real market and must not be read as one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import N_LOB_LEVELS

N_LEVELS = N_LOB_LEVELS


def generate_synthetic_lob(
    n_bars: int = 60_000,
    seed: int = 7,
    start: str = "2026-01-05 00:00:00",
    bar_seconds: int = 5,
    tick: float = 0.00005,
    base_price: float = 1.0850,
    base_size: float = 40.0,
    ofi_signal: float = 0.0,
    session_gap_every: int = 17_280,   # ~1 trading day of 5s bars; inserts a gap
) -> pd.DataFrame:
    """
    Build a synthetic 10-level LOB snapshot DataFrame.

    Parameters
    ----------
    n_bars : number of 5-second snapshots.
    ofi_signal : strength of a (toy) order-flow -> next-return coupling. The
        default 0.0 makes the mid a pure random walk, which is the honest null:
        even then the IS sweep will manufacture an in-sample "edge" that the
        OOS slice and the spread destroy. Set a small positive value (e.g. 0.02)
        to inject a faint genuine edge and watch it still fail to clear the cost
        bar at short horizons.
    session_gap_every : insert a multi-minute timestamp gap this often, so the
        no-lookahead gap-masking logic is exercised.
    """
    rng = np.random.default_rng(seed)

    # ── timestamps with occasional session gaps ──────────────────────────────
    base = pd.Timestamp(start, tz="UTC")
    steps = np.full(n_bars, bar_seconds, dtype=np.int64)
    if session_gap_every and session_gap_every < n_bars:
        gap_idx = np.arange(session_gap_every, n_bars, session_gap_every)
        steps[gap_idx] = 3600  # one-hour gap stands in for a session break
    offsets = np.concatenate([[0], np.cumsum(steps[:-1])])
    index = base + pd.to_timedelta(offsets, unit="s")

    # ── mid: driftless random walk with a faint OFI coupling ──────────────────
    flow = rng.standard_normal(n_bars)              # signed order-flow proxy
    noise = rng.standard_normal(n_bars)
    # next-bar return is mostly noise; ofi_signal couples a little of THIS bar's
    # flow into the NEXT bar's return (so a predictor *could* see something).
    ret = np.empty(n_bars)
    ret[0] = 0.0
    ret[1:] = (ofi_signal * flow[:-1] + noise[1:]) * tick
    mid = base_price + np.cumsum(ret)

    # ── spread: 1-3 ticks, wider on high-|flow| bars ──────────────────────────
    spread_ticks = 1 + (np.abs(flow) > 1.3).astype(int) + (np.abs(flow) > 2.2).astype(int)
    half_spread = spread_ticks * tick / 2.0

    data: dict[str, np.ndarray] = {}
    for lvl in range(N_LEVELS):
        # prices step away from the touch by one tick per level
        data[f"bid_px_{lvl:02d}"] = mid - half_spread - lvl * tick
        data[f"ask_px_{lvl:02d}"] = mid + half_spread + lvl * tick

        # depth decays away from the touch; flow tilts the book (more bid size on
        # buy pressure). log-normal-ish positive integer sizes.
        decay = np.exp(-0.25 * lvl)
        tilt_bid = 1.0 + 0.15 * np.tanh(flow)
        tilt_ask = 1.0 - 0.15 * np.tanh(flow)
        bsz = base_size * decay * tilt_bid * np.exp(0.2 * rng.standard_normal(n_bars))
        asz = base_size * decay * tilt_ask * np.exp(0.2 * rng.standard_normal(n_bars))
        data[f"bid_sz_{lvl:02d}"] = np.maximum(1, np.round(bsz)).astype(np.float64)
        data[f"ask_sz_{lvl:02d}"] = np.maximum(1, np.round(asz)).astype(np.float64)

    cols = (
        [f"bid_px_{i:02d}" for i in range(N_LEVELS)]
        + [f"ask_px_{i:02d}" for i in range(N_LEVELS)]
        + [f"bid_sz_{i:02d}" for i in range(N_LEVELS)]
        + [f"ask_sz_{i:02d}" for i in range(N_LEVELS)]
    )
    return pd.DataFrame({c: data[c] for c in cols}, index=index)


if __name__ == "__main__":
    df = generate_synthetic_lob(n_bars=2000)
    print(df.shape)
    print(df.iloc[:3, :4])
