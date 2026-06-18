"""
Sliding-window dataset and walk-forward splitting for DeepLOB training.

The window builder uses a single vectorised numpy fancy-index op rather than
2,048 per-batch ``__getitem__`` calls — the lazy approach was ~6x slower on CPU.

Each training sample:
    X : float32 (WINDOW_SIZE, N_FEATURES)  — window of features ending at t-1
    y : int64   scalar                     — direction label at t (+HORIZON)
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import (
    WINDOW_SIZE,
    BATCH_SIZE,
    TRAIN_MONTHS,
    VAL_MONTHS,
    TEST_MONTHS,
    STEP_MONTHS,
)


def build_windows(
    feat_df: pd.DataFrame,
    label_series: pd.Series,
    window: int = WINDOW_SIZE,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Materialise (N, window, F) windows. Z-scores features on the supplied slice.
    Use on small slices (smoke tests / short evaluations) only.
    """
    idx = feat_df.index.intersection(label_series.index)
    feat = feat_df.loc[idx].to_numpy(np.float32)   # (T, F)
    labs = label_series.loc[idx].to_numpy(np.int64)

    mean = feat.mean(axis=0, keepdims=True)
    std = feat.std(axis=0, keepdims=True) + 1e-8
    feat = (feat - mean) / std

    n = len(feat) - window
    if n <= 0:
        return (
            np.empty((0, window, feat.shape[1]), np.float32),
            np.empty(0, dtype=np.int64),
            idx[:0],
        )

    X = np.lib.stride_tricks.sliding_window_view(feat, (window, feat.shape[1]))
    X = X[:n, 0, :, :]            # (n, window, F)
    y = labs[window:]            # (n,)
    timestamps = idx[window:]
    return X.astype(np.float32), y, timestamps


def build_timestamps(
    feat_df: pd.DataFrame,
    label_series: pd.Series,
    window: int = WINDOW_SIZE,
) -> pd.DatetimeIndex:
    """Window-end timestamps without materialising X."""
    idx = feat_df.index.intersection(label_series.index)
    return idx[window:]


def walk_forward_splits(
    feat_df: pd.DataFrame,
    label_series: pd.Series,
    train_months: int = TRAIN_MONTHS,
    val_months: int = VAL_MONTHS,
    test_months: int = TEST_MONTHS,
    step_months: int = STEP_MONTHS,
) -> list[dict]:
    """
    Generate anchored-rolling walk-forward (train / val / test) splits by month.

    Returns dicts: {"train": (feat, lab), "val": ..., "test": ...,
                    "test_start": Timestamp, "test_end": Timestamp}.
    """
    idx = feat_df.index.intersection(label_series.index)
    idx_naive = idx.tz_localize(None) if idx.tz is not None else idx
    start = idx_naive.min().to_period("M")
    end = idx_naive.max().to_period("M")

    splits: list[dict] = []
    cursor = start
    while True:
        train_end = cursor + train_months - 1
        val_end = train_end + val_months
        test_end = val_end + test_months
        if test_end > end:
            break

        def _slice(p_start, p_end):
            s = p_start.to_timestamp()
            e = (p_end + 1).to_timestamp() - pd.Timedelta(seconds=1)
            if idx.tz is not None:
                s = s.tz_localize("UTC")
                e = e.tz_localize("UTC")
            mask = (idx >= s) & (idx <= e)
            return feat_df.loc[idx[mask]], label_series.loc[idx[mask]]

        tr = _slice(cursor, train_end)
        va = _slice(train_end + 1, val_end)
        te = _slice(val_end + 1, test_end)

        if len(tr[1]) > 0 and len(va[1]) > 0 and len(te[1]) > 0:
            test_start = (val_end + 1).to_timestamp()
            test_stop = (test_end + 1).to_timestamp()
            if idx.tz is not None:
                test_start = test_start.tz_localize("UTC")
                test_stop = test_stop.tz_localize("UTC")
            splits.append({
                "train": tr, "val": va, "test": te,
                "test_start": test_start, "test_end": test_stop,
            })
        cursor += step_months
    return splits


def make_loader(
    features: np.ndarray,   # (T, F) raw feature array — NOT pre-windowed
    labels: np.ndarray,     # (T,) raw label array
    window: int = WINDOW_SIZE,
    stride: int = 1,
    shuffle: bool = True,
    batch_size: int = BATCH_SIZE,
) -> DataLoader:
    """
    Build a DataLoader by vectorised fancy indexing into a (N, window, F) tensor.

    stride > 1 subsamples window start points (10x fewer windows => ~10x faster
    epochs on CPU) while keeping diverse coverage.
    """
    T = len(features)
    n = max(0, T - window)
    if n <= 0:
        X_t = torch.empty(0, window, features.shape[1])
        y_t = torch.empty(0, dtype=torch.long)
    else:
        starts = np.arange(0, n, stride, dtype=np.intp)              # (N,)
        idx = starts[:, None] + np.arange(window, dtype=np.intp)     # (N, window)
        X = features[idx]                                            # (N, window, F)
        y = labels[starts + window]
        X_t = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        y_t = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))

    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)
