# Results

## Status and corrections

The historical real-data net Sharpe of **−14.63 over 2,264 out-of-sample trades at 3× cost**, with **DSR ≈ 0**, is withdrawn. The July 9 accounting audit found that the original engine charged a round-trip cost on both entry and exit. The July 10 clock check found missing five-second bars in the processed book. A corrected result requires replaying the licensed data. The direction and size of the revision remain unknown.

The published perfect-foresight and cost comparisons are historical measurements. For example, the reported perfect-predictor gross moves for one and twelve bars were about 0.22 and 0.84 bp, against about 1.4 bp for a round trip. These comparisons have yet to be re-verified after the clock finding. A row count alone cannot establish elapsed time across a gap.

Internal source records are `AUDIT_AND_PROGRESS_2026-07-09.md` (execution accounting) and `STRATEGY_LOGIC_HUNT_2026-07-10.md` (bar clock). The earlier public figures are preserved in the repository history at `030ce04`.

## What the public code reproduces

The public OFI event simulator already charged costs once, skipped nonfinite signals during warm-up and required an exit inside the test window. It has no volatility-based position sizing. The chart and daily Sharpe previously recognised an entire trade on its exit bar. Both now use bar-by-bar marks. The price clock now handles different pandas timestamp units and a missing snapshot interrupts a fixed-horizon hold.

On the default synthetic seed and 60,000 bars, using pandas 3.1.1, the baseline froze a one-bar hold and recorded 18,182 OOS trades. Repairing the clock changed the selected hold to six bars and produced 120 OOS trades. Average gross P&L changed from −0.0035 to +0.1027 bp per trade and average cost from 0.8954 to 0.9130 bp. Switching from exit-only to bar marks preserved those rounded figures and the synthetic FAIL verdict. The OOS slice covers two UTC dates, so the daily Sharpe routine returns its short-sample sentinel of zero. It cannot estimate a meaningful Sharpe on this slice.

Run `python demo/run_demo.py --no-train` for the synthetic comparison point. The long and short regression fixtures in [the accounting tests](../tests/test_accounting_clock.py) show a different consequence of marking on time. An open position moves before midnight and reverses after midnight. Gross and net changes land on their respective dates, and net marks sum to the closed-trade ledger.

## Model and feature references

- Zhang, Zohren and Roberts, *DeepLOB*. [Author preprint](https://arxiv.org/abs/1808.03668)
- Cont, Kukanov and Stoikov, *The Price Impact of Order Book Events*. [Article](https://doi.org/10.1093/jjfinec/nbt003)
