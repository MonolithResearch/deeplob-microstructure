"""
The honest, no-lookahead out-of-sample test for the OFI microstructure signal.

This is the heart of the case study. It does NOT try to make money. It overfits
the in-sample window across a grid to MAXIMISE in-sample net Sharpe, FREEZES one
config, and evaluates it out-of-sample ONCE. The decisive question is whether the
gross directional edge (in bp) can pay the REAL spread-crossing cost bar.

It is engineered to make a real edge look as good as honestly possible and still
report exactly where it fails.

NO-LOOKAHEAD CONTRACT
  - Multi-level OFI at bar t uses snapshots t-1 and t only (Cont et al. 2014).
    Smoothing and z-score windows are causal (right-aligned rolling).
  - The position is decided at bar t from features <= t, filled at mid[t], held
    H bars, exited at mid[t+H]. Forward P&L is masked if the t -> t+H interval
    crosses a break in the five-second clock.
  - Cost = real top-of-book half-spread at entry + half-spread at exit (you cross
    the spread both ways) + a fixed commission. Charged per round trip.
  - Holds do not overlap. Open trades are marked at each observed mid.

This module works on any LOB snapshot DataFrame with the standard column schema
(bid_px_00..09, ask_px_00..09, bid_sz_00..09, ask_sz_00..09) — the same schema the
synthetic generator emits, so the demo runs with no licensed data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:  # numba optional — fall back to a pure-python loop
    _HAVE_NUMBA = False

from .config import (
    N_LOB_LEVELS, BAR_SECONDS,
    GAP_SECONDS,
    COMMISSION_BP_RT,
    IS_FRAC,
)
from .lob_features import clock_starts

N_LEVELS = N_LOB_LEVELS

# Default aggressive in-sample sweep grid.
DEPTHS = [1, 3, 5, 10]
SMOOTH_WINS = [1, 3, 6, 12]
HORIZONS = [1, 3, 6, 12]
Z_WINS = [120, 360]
THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load per-level OFI exactly like lob_features.per_level_ofi (Cont 2014).
# ─────────────────────────────────────────────────────────────────────────────
def compute_ofi_arrays(lob: pd.DataFrame):
    """
    Returns mid, half_spread_bp, ofi_lvl (T, N_LEVELS), ts_seconds, spread.
    NaNs are placed on the first bar after a time gap (stale previous snapshot).
    """
    bid_px = lob[[f"bid_px_{i:02d}" for i in range(N_LEVELS)]].to_numpy(np.float64)
    ask_px = lob[[f"ask_px_{i:02d}" for i in range(N_LEVELS)]].to_numpy(np.float64)
    bid_sz = lob[[f"bid_sz_{i:02d}" for i in range(N_LEVELS)]].clip(lower=0).to_numpy(np.float64)
    ask_sz = lob[[f"ask_sz_{i:02d}" for i in range(N_LEVELS)]].clip(lower=0).to_numpy(np.float64)

    best_bid = bid_px[:, 0]
    best_ask = ask_px[:, 0]
    mid = (best_bid + best_ask) / 2.0
    spread = best_ask - best_bid
    half_spread_bp = (spread / 2.0) / mid * 1e4

    starts = clock_starts(lob.index)
    ts = lob.index.as_unit("ns").asi8 / 1e9

    ofi_lvl = np.zeros((len(lob), N_LEVELS), np.float64)
    for l in range(N_LEVELS):
        bp = bid_px[:, l]; bs = bid_sz[:, l]
        ap = ask_px[:, l]; asz = ask_sz[:, l]
        bp_p = np.empty_like(bp); bp_p[0] = np.nan; bp_p[1:] = bp[:-1]
        bs_p = np.empty_like(bs); bs_p[0] = np.nan; bs_p[1:] = bs[:-1]
        ap_p = np.empty_like(ap); ap_p[0] = np.nan; ap_p[1:] = ap[:-1]
        as_p = np.empty_like(asz); as_p[0] = np.nan; as_p[1:] = asz[:-1]
        bid_ofi = np.where(bp > bp_p, bs, np.where(bp < bp_p, -bs_p, bs - bs_p))
        ask_ofi = np.where(ap < ap_p, asz, np.where(ap > ap_p, -as_p, asz - as_p))
        ofi_lvl[:, l] = bid_ofi - ask_ofi

    ofi_lvl[starts, :] = np.nan
    return mid, half_spread_bp, ofi_lvl, ts, spread


def cum_ofi(ofi_lvl, depth):
    return ofi_lvl[:, :depth].sum(axis=1)


def causal_smooth(x, win):
    if win <= 1:
        return x.copy()
    return pd.Series(x).rolling(win, min_periods=win).mean().to_numpy()


def rolling_z(x, win):
    s = pd.Series(x)
    m = s.rolling(win, min_periods=win // 2).mean()
    sd = s.rolling(win, min_periods=win // 2).std()
    return ((s - m) / sd.replace(0, np.nan)).to_numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Non-overlapping event backtest — per-trade arrays.
# ─────────────────────────────────────────────────────────────────────────────
def _run_trades_py(ofi_z, mid, half_spread_bp, ts, H, thr, commission_bp_rt,
                   gap_s, start_i, end_i, breaks=None):
    n = end_i - start_i
    gross_steps = np.zeros(n)
    net_steps = np.zeros(n)
    max_tr = n // H + 2
    out_ts = np.empty(max_tr); out_gross = np.empty(max_tr)
    out_net = np.empty(max_tr); out_cost = np.empty(max_tr)
    out_dir = np.empty(max_tr, np.int64)
    k = 0
    i = start_i
    while i < end_i - H:
        z = ofi_z[i]
        if not np.isfinite(z) or abs(z) <= thr:
            i += 1
            continue
        m0 = mid[i]; j = i + H; m1 = mid[j]
        if not (np.isfinite(m0) and np.isfinite(m1)) or m0 <= 0.0:
            i += 1
            continue
        valid_hold = True
        for u in range(i + 1, j + 1):
            step = ts[u] - ts[u - 1]
            if ((breaks is not None and breaks[u]) or step != BAR_SECONDS
                    or step > gap_s or not np.isfinite(mid[u]) or mid[u] <= 0):
                valid_hold = False
                break
        if not valid_hold:
            i += 1
            continue
        d = 1.0 if z > 0.0 else -1.0
        gross_bp = d * (m1 - m0) / m0 * 1e4
        hs0 = half_spread_bp[i]; hs1 = half_spread_bp[j]
        if not (np.isfinite(hs0) and np.isfinite(hs1)):
            i += 1
            continue
        hs0 = max(hs0, 0.0); hs1 = max(hs1, 0.0)
        cost_bp = hs0 + hs1 + commission_bp_rt
        out_ts[k] = ts[j]; out_gross[k] = gross_bp
        out_net[k] = gross_bp - cost_bp; out_cost[k] = cost_bp
        for u in range(i + 1, j + 1):
            change = d * (mid[u] - mid[u - 1]) / m0 * 1e4
            gross_steps[u - start_i] = change
            net_steps[u - start_i] = change
        net_steps[j - start_i] -= cost_bp
        out_dir[k] = int(d)
        k += 1
        i = j  # NON-OVERLAPPING
    return (out_ts[:k], out_gross[:k], out_net[:k], out_cost[:k], out_dir[:k],
            gross_steps, net_steps)


if _HAVE_NUMBA:
    _run_marked = njit(cache=True)(_run_trades_py)
else:
    _run_marked = _run_trades_py


def run_trades(ofi_z, mid, half_spread_bp, ts, H, thr, commission_bp_rt,
               gap_s, start_i, end_i):
    """Closed-trade ledger (exit seconds, gross bp, net bp, cost bp, direction)."""
    return _run_marked(ofi_z, mid, half_spread_bp, ts, H, thr, commission_bp_rt,
                       gap_s, start_i, end_i)[:5]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Daily Sharpe on the evaluation clock, including days without trades.
# ─────────────────────────────────────────────────────────────────────────────
def daily_sharpe(exit_ts_sec, pnl_bp, ppy: int = 252):
    if len(pnl_bp) < 3:
        return 0.0, 0
    if isinstance(exit_ts_sec, pd.DatetimeIndex):
        idx = pd.to_datetime(exit_ts_sec, utc=True)
    else:
        idx = pd.to_datetime(np.asarray(exit_ts_sec), unit="s", utc=True)
    s = pd.Series(pnl_bp, index=idx)
    daily = s.resample("1D").sum()
    if len(daily) < 3 or daily.std() == 0:
        return 0.0, len(daily)
    return float(daily.mean() / daily.std() * np.sqrt(ppy)), len(daily)


def summarize(exit_ts, gross, net, cost, *, marks=None):
    clock, gross_steps, net_steps = (exit_ts, gross, net) if marks is None else marks
    sh_net, ndays = daily_sharpe(clock, net_steps)
    sh_gross, _ = daily_sharpe(clock, gross_steps)
    n = len(net)
    return dict(
        n_trades=int(n),
        sharpe_net=sh_net,
        sharpe_gross=sh_gross,
        avg_gross_bp=float(np.mean(gross)) if n else 0.0,
        avg_cost_bp=float(np.mean(cost)) if n else 0.0,
        avg_net_bp=float(np.mean(net)) if n else 0.0,
        sum_net_bp=float(np.sum(net)) if n else 0.0,
        hit_rate=float(np.mean(net > 0)) if n else 0.0,
        trading_days=ndays,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Full IS-sweep / freeze / single-shot OOS.
# ─────────────────────────────────────────────────────────────────────────────
def run_honest_oos(
    lob: pd.DataFrame,
    is_frac: float = IS_FRAC,
    commission_bp_rt: float = COMMISSION_BP_RT,
    gap_s: float = GAP_SECONDS,
    min_trades_is: int = 200,
    depths=DEPTHS,
    smooth_wins=SMOOTH_WINS,
    horizons=HORIZONS,
    z_wins=Z_WINS,
    thresholds=THRESHOLDS,
    verbose: bool = True,
) -> dict:
    """
    Overfit IS net Sharpe over the grid, freeze the best config (>= min_trades_is
    trades), evaluate it ONCE out-of-sample. Returns a dict of IS/OOS stats plus
    the frozen config, the closed-trade ledger and per-bar OOS P&L increments.
    """
    def log(*a):
        if verbose:
            print(*a, flush=True)

    lob = lob.sort_index()
    mid, half_spread_bp, ofi_lvl, ts, spread = compute_ofi_arrays(lob)
    N = len(mid)
    starts = clock_starts(lob.index)
    breaks = np.zeros(N, dtype=bool)
    breaks[starts] = True
    segments = np.split(np.arange(N), starts[1:])
    split = int(N * is_frac)
    is_clock, oos_clock = lob.index[:split], lob.index[split:]
    log(f"[split] N={N}  IS=[0,{split})  OOS=[{split},{N})")

    cumofi = {d: cum_ofi(ofi_lvl, d) for d in depths}
    zcache: dict = {}
    rows = []
    best = None

    for d in depths:
        base = cumofi[d]
        for sw in smooth_wins:
            smooth_segments = [causal_smooth(base[part], sw) for part in segments]
            for zw in z_wins:
                z = np.concatenate([rolling_z(part, zw) for part in smooth_segments])
                zcache[(d, sw, zw)] = z
                for H in horizons:
                    for thr in thresholds:
                        et, g, nt, c, _, gross_steps, net_steps = _run_marked(
                            z, mid, half_spread_bp, ts, H, float(thr),
                            commission_bp_rt, gap_s, 0, split, breaks)
                        m = summarize(et, g, nt, c, marks=(is_clock, gross_steps, net_steps))
                        rows.append(dict(depth=d, smooth=sw, zwin=zw, horizon=H,
                                         thr=thr, **{f"is_{k}": v for k, v in m.items()}))
                        if m["n_trades"] >= min_trades_is:
                            if best is None or m["sharpe_net"] > best[1]:
                                best = ((d, sw, zw, H, thr), m["sharpe_net"], m)

    is_df = pd.DataFrame(rows).sort_values("is_sharpe_net", ascending=False)
    if best is None:
        raise RuntimeError(f"No IS config reached {min_trades_is} trades; widen the grid or use more data.")

    (d, sw, zw, H, thr), is_sh, is_rec = best
    log(f"[freeze] depth={d} smooth={sw} zwin={zw} horizon={H} thr={thr} "
        f"IS_net_Sharpe={is_sh:.4f} IS_trades={is_rec['n_trades']}")

    z = zcache[(d, sw, zw)]
    et, g, nt, c, dr, gross_steps, net_steps = _run_marked(
        z, mid, half_spread_bp, ts, H, float(thr), commission_bp_rt, gap_s, split, N, breaks)
    oos = summarize(et, g, nt, c, marks=(oos_clock, gross_steps, net_steps))
    gross_gt_cost = oos["avg_gross_bp"] > oos["avg_cost_bp"]

    if oos["sharpe_net"] > 0 and oos["n_trades"] >= min_trades_is and gross_gt_cost:
        verdict = "PASS"
    elif oos["sharpe_net"] > 0 and oos["n_trades"] > 0:
        verdict = "PROBATION"
    else:
        verdict = "FAIL"

    log(f"[OOS] net_Sharpe={oos['sharpe_net']:.4f} trades={oos['n_trades']} "
        f"avg_gross_bp={oos['avg_gross_bp']:.4f} avg_cost_bp={oos['avg_cost_bp']:.4f} "
        f"gross>cost={gross_gt_cost} => {verdict}")

    return dict(
        config=dict(depth=d, smooth=sw, zwin=zw, horizon=H, threshold=thr),
        is_summary=is_rec,
        oos_summary=oos,
        gross_gt_cost=gross_gt_cost,
        verdict=verdict,
        is_sweep=is_df,
        oos_marks=pd.DataFrame({"gross_bp": gross_steps, "net_bp": net_steps}, index=oos_clock),
        oos_trades=dict(exit_ts=et, gross_bp=g, net_bp=nt, cost_bp=c, direction=dr),
    )
