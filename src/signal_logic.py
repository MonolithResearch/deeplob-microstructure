"""
Dual-gate signal policy: combine the DeepLOB probability with the OFI z-score.

The research strategy enters only when BOTH gates pass:
    1. model confidence:  max(P_UP, P_DOWN) > min_prob
    2. order-flow gate:    |ofi_z| > ofi_threshold
    3. agreement:          (P_UP > P_DOWN and ofi_z > 0) or (P_DOWN > P_UP and ofi_z < 0)

The default ``contrarian`` mode fades the agreed direction. The ``trend`` mode
takes the agreed direction. See docs/RESULTS.md for the empirical status.
"""
from __future__ import annotations

import math


def direction_from_prob_ofi(
    p_down: float,
    p_up: float,
    ofi_z: float,
    min_prob: float,
    ofi_threshold: float,
    mode: str = "contrarian",
) -> int:
    """Return +1 (long), -1 (short), or 0 (no trade)."""
    vals = (p_down, p_up, ofi_z, min_prob, ofi_threshold)
    if not all(math.isfinite(float(v)) for v in vals):
        return 0

    p_down = float(p_down)
    p_up = float(p_up)
    ofi_z = float(ofi_z)
    min_prob = float(min_prob)
    ofi_threshold = float(ofi_threshold)

    if max(p_up, p_down) <= min_prob or abs(ofi_z) <= ofi_threshold:
        return 0

    if p_up > p_down and ofi_z > 0:
        agreed = +1
    elif p_down > p_up and ofi_z < 0:
        agreed = -1
    else:
        return 0

    if mode == "contrarian":
        return -agreed
    if mode == "trend":
        return agreed
    raise ValueError(f"Unknown signal mode: {mode!r}")
