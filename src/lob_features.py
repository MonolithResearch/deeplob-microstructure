"""
Limit-order-book feature engineering + multi-level Order-Flow-Imbalance (OFI).

Produces the 40-channel DeepLOB input from 10-level LOB snapshots:
    10 bid prices + 10 ask prices  (deviation from mid, in basis points)
    10 bid sizes  + 10 ask sizes   (log1p-scaled)

It also computes ``ofi_z`` (the rolling z-score of aggregate OFI) used by the
backtest signal gate. ``ofi_z`` is NOT a model input channel.

References
----------
Cont, Kukanov & Stoikov (2014), "The Price Impact of Order Book Events",
    Journal of Financial Econometrics 12(1), 47-88 — the per-level OFI definition.
Zhang, Zohren & Roberts (2018), arXiv:1808.03668 — the DeepLOB normalisation.

NO-LOOKAHEAD CONTRACT
---------------------
Per-level OFI at snapshot t uses only snapshots t-1 and t (a one-step shift).
The rolling z-score uses a causal, right-aligned window. There is no use of any
information from t+1 onward in any feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BAR_SECONDS, OFI_Z_WINDOW, N_LOB_LEVELS

N_LEVELS = N_LOB_LEVELS

# The 40 model-input feature names, in channel order.
FEATURE_NAMES = (
    [f"bid_p_{i}" for i in range(1, N_LEVELS + 1)]
    + [f"ask_p_{i}" for i in range(1, N_LEVELS + 1)]
    + [f"bid_sz_{i}" for i in range(1, N_LEVELS + 1)]
    + [f"ask_sz_{i}" for i in range(1, N_LEVELS + 1)]
)
N_FEATURES = len(FEATURE_NAMES)  # 40


# Raw snapshot column names (the schema produced by the synthetic data generator
# and, in the private project, by the Databento loader).
BID_PX = [f"bid_px_{i:02d}" for i in range(N_LEVELS)]
ASK_PX = [f"ask_px_{i:02d}" for i in range(N_LEVELS)]
BID_SZ = [f"bid_sz_{i:02d}" for i in range(N_LEVELS)]
ASK_SZ = [f"ask_sz_{i:02d}" for i in range(N_LEVELS)]


def clock_starts(index: pd.DatetimeIndex) -> np.ndarray:
    """Row offsets where a new contiguous sequence of snapshots begins."""
    if (not isinstance(index, pd.DatetimeIndex) or index.hasnans
            or not index.is_unique or not index.is_monotonic_increasing):
        raise ValueError("bar clock requires unique, increasing timestamps")
    return np.flatnonzero(index.to_series().diff().ne(pd.Timedelta(seconds=BAR_SECONDS)))


def _require_contiguous(index: pd.DatetimeIndex) -> None:
    if len(clock_starts(index)) > 1:
        raise ValueError("model bar clock has a gap; supply one contiguous segment")


def per_level_ofi(lob: pd.DataFrame) -> list[pd.Series]:
    """
    Per-level Order-Flow-Imbalance following Cont, Kukanov & Stoikov (2014).

    For each level l, with ``prev`` = one-snapshot shift:
        bid_ofi_l = bsz            if bid_px rose
                  = -bsz_prev      if bid_px fell
                  =  bsz - bsz_prev if bid_px unchanged
        ask_ofi_l = asz            if ask_px fell
                  = -asz_prev      if ask_px rose
                  =  asz - asz_prev if ask_px unchanged
        ofi_l     = bid_ofi_l - ask_ofi_l   (positive => net buy pressure)

    Returns a list of N_LEVELS pd.Series aligned to ``lob.index``.
    """
    ofi_levels: list[pd.Series] = []
    for bpc, bsc, apc, asc in zip(BID_PX, BID_SZ, ASK_PX, ASK_SZ):
        bp_now = lob[bpc]
        bp_prev = bp_now.shift(1)
        bsz_now = lob[bsc].clip(lower=0)
        bsz_prev = bsz_now.shift(1)

        ap_now = lob[apc]
        ap_prev = ap_now.shift(1)
        asz_now = lob[asc].clip(lower=0)
        asz_prev = asz_now.shift(1)

        bid_ofi = np.where(
            bp_now > bp_prev, bsz_now,
            np.where(bp_now < bp_prev, -bsz_prev, bsz_now - bsz_prev),
        )
        ask_ofi = np.where(
            ap_now < ap_prev, asz_now,
            np.where(ap_now > ap_prev, -asz_prev, asz_now - asz_prev),
        )
        ofi_levels.append(pd.Series(bid_ofi - ask_ofi, index=lob.index))
    return ofi_levels


def compute_features(lob: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 40 model features (plus ``mid``, ``spread_bps``, ``ofi_z``)
    from a LOB snapshot DataFrame.

    Parameters
    ----------
    lob : DataFrame with columns bid_px_00..09, ask_px_00..09,
          bid_sz_00..09, ask_sz_00..09, indexed by a DatetimeIndex.

    Returns
    -------
    DataFrame containing FEATURE_NAMES + ['mid', 'spread_bps', 'ofi_z'].
    The extra three columns are used by the backtest, not fed to the model.
    """
    _require_contiguous(lob.index)
    out = pd.DataFrame(index=lob.index)

    mid = (lob[BID_PX[0]] + lob[ASK_PX[0]]) / 2
    out["mid"] = mid

    spread_bps = (lob[ASK_PX[0]] - lob[BID_PX[0]]) / mid * 10_000
    out["spread_bps"] = spread_bps.clip(lower=0)

    # Price channels: deviation from mid in bps (bids negative, asks positive).
    for i, (bc, ac) in enumerate(zip(BID_PX, ASK_PX)):
        lvl = i + 1
        out[f"bid_p_{lvl}"] = (lob[bc] - mid) / mid * 10_000
        out[f"ask_p_{lvl}"] = (lob[ac] - mid) / mid * 10_000

    # Size channels: log1p of resting size.
    for i, (bc, ac) in enumerate(zip(BID_SZ, ASK_SZ)):
        lvl = i + 1
        out[f"bid_sz_{lvl}"] = np.log1p(lob[bc].clip(lower=0))
        out[f"ask_sz_{lvl}"] = np.log1p(lob[ac].clip(lower=0))

    # Aggregate OFI z-score (causal) for the backtest gate.
    ofi_levels = per_level_ofi(lob)
    ofi_agg = sum(ofi_levels) / N_LEVELS
    roll_mean = ofi_agg.rolling(OFI_Z_WINDOW, min_periods=OFI_Z_WINDOW // 2).mean()
    roll_std = ofi_agg.rolling(OFI_Z_WINDOW, min_periods=OFI_Z_WINDOW // 2).std()
    out["ofi_z"] = ((ofi_agg - roll_mean) / roll_std.replace(0, np.nan)).fillna(0)

    out = out.replace([np.inf, -np.inf], np.nan)
    out[FEATURE_NAMES] = out[FEATURE_NAMES].fillna(0.0)
    out = out.dropna(subset=["mid", "bid_p_1", "ask_p_1"])
    return out


def compute_labels(
    mid: pd.Series,
    spread_bps: pd.Series,
    horizon: int = 30,
    spread_multiplier: float = 1.0,
) -> pd.Series:
    """
    3-class direction labels at ``horizon`` bars ahead: 0=DOWN, 1=FLAT, 2=UP.

    The neutral band is ``spread_multiplier x rolling_avg_spread`` (in return
    units), so a move must clear the typical transaction cost to be labelled
    directional. This keeps the FLAT class honest and prevents the model from
    "predicting" sub-spread noise.
    """
    _require_contiguous(mid.index)
    future_ret = mid.shift(-horizon) / mid - 1
    avg_spread_ret = (spread_bps / 10_000).rolling(120, min_periods=10).mean()
    threshold = spread_multiplier * avg_spread_ret

    labels = pd.Series(1, index=mid.index, name="label")  # FLAT default
    labels[future_ret > threshold] = 2   # UP
    labels[future_ret < -threshold] = 0  # DOWN
    return labels
