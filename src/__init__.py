"""
DeepLOB + Order-Flow-Imbalance microstructure pipeline.

Monolith Research — Evidence over narrative.

A working research-grade implementation of a DeepLOB-style CNN+LSTM classifier
and a multi-level Order-Flow-Imbalance (OFI) feature pipeline for limit-order-book
data, plus the HONEST no-lookahead out-of-sample backtest that shows why
in-sample foresight does not survive realistic transaction costs.

This package contains research code only. No broker/live-execution code, no
credentials, and no licensed market data are included.
"""

__all__ = [
    "config",
    "deeplob_model",
    "lob_features",
    "dataset",
    "trainer",
    "signal_logic",
    "honest_oos",
]
