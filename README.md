<div align="center">

<img src="assets/sigma-mark-plum.svg" alt="Monolith Research" width="88" />

# DeepLOB + Order-Flow Imbalance

A reproducible order-book experiment in signal timing, spread costs and out-of-sample selection.

</div>

## Status and corrections

The reported real-data net Sharpe of **−14.63 over 2,264 trades at 3× cost**, with **DSR ≈ 0**, is withdrawn pending a corrected replay on licensed data. The original pipeline charged a round-trip cost twice and its processed five-second book contained clock gaps. The published perfect-foresight and cost comparisons have yet to be re-verified on a corrected clock. [Results and audit sources](docs/RESULTS.md)

## Run the synthetic experiment

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python demo/run_demo.py
```

The demo builds ten-level snapshots, constructs features and trains a short CNN–LSTM fold on the first contiguous segment. A separate OFI experiment selects settings in sample, freezes them and evaluates them once on the later slice. The OFI sweep uses book features directly. To run just the features, labels and sweep, use `python demo/run_demo.py --no-train`. The demo writes its chart to `docs/results/demo_oos_equity.png`. All runnable examples use synthetic data.

## How the OFI backtest accounts for a trade

- Consecutive snapshots must be five seconds apart within a position. An interruption resets the OFI smoothing and normalisation windows.
- A signal at a snapshot uses information through that snapshot. The modelled entry is its mid price. The exit is the mid a fixed number of bars later. Only holds that fit inside the evaluation slice enter the ledger.
- Gross P&L follows each mid-price change while the position is open. Net P&L deducts both half-spreads and the round-trip commission once, at exit. All changes are in basis points of entry notional.
- Non-overlapping trades can still be correlated. Daily Sharpe uses daily sums of marked P&L over the evaluation slice, including idle days.

Maintained by Bilal Malik. [Project profile](https://github.com/05bilalmalik-cmd)
