"""
Strategy hyperparameters and Monolith Research brand constants — one place.

This is the public, research-only config. The original private project also held
broker credentials and data-vendor paths; those have been removed. Nothing in
this file reads a secret, an account number, or a licensed-data path.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── Monolith Research brand palette (use these EXACT hex values everywhere) ────
INK = "#0E0C14"     # near-black
PLUM = "#A78AA3"    # accent
PAPER = "#F3EFE8"   # light
PLUM_DARK = "#7A6177"   # muted plum used in the lockup
BRAND_NAME = "Monolith Research"
BRAND_TAGLINE = "Evidence over narrative"

# ── Data geometry ─────────────────────────────────────────────────────────────
# The real research used Databento mbp-10 (10-level) snapshots on CME 6E (EUR
# futures), resampled to a regular 5-second grid. Raw vendor data is NOT shipped;
# the demo uses a small synthetic generator with the identical column schema.
N_LOB_LEVELS = 10                 # price levels per side
BAR_SECONDS = 5                   # snapshot grid spacing

# ── Model input features ──────────────────────────────────────────────────────
WINDOW_SIZE = 100                 # bars fed into the model (100 x 5s = 8.3 min)
N_FEATURES = N_LOB_LEVELS * 4     # 40 = (bid_p, ask_p, bid_sz, ask_sz) per level

# ── OFI z-score gate ──────────────────────────────────────────────────────────
OFI_Z_THRESHOLD = 1.0             # only act when |ofi_z| exceeds this
OFI_Z_WINDOW = 120                # rolling normalisation window (bars)

# ── Label definition (3-class direction at a forward horizon) ─────────────────
HORIZON = 30                      # bars ahead the model predicts (30 x 5s = 2.5 min)
LABEL_SPREAD_MULTIPLIER = 1.0     # a "move" must exceed this x avg spread to count

# ── DeepLOB model ─────────────────────────────────────────────────────────────
CNN_CHANNELS = [32, 64, 64]
LSTM_HIDDEN = 64
DROPOUT = 0.3
N_CLASSES = 3                     # 0 = DOWN, 1 = FLAT, 2 = UP

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 2048                 # large batch for CPU efficiency
LEARNING_RATE = 6e-4             # linear-scaling-rule LR for the large batch
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 40
PATIENCE = 6                      # early stopping on val loss
CLASS_WEIGHT = True               # weight classes by inverse frequency

# Walk-forward windows (months)
TRAIN_MONTHS = 4
VAL_MONTHS = 1
TEST_MONTHS = 1
STEP_MONTHS = 1

# Window stride: use every Nth bar as a start point (10 => ~10x faster on CPU).
TRAIN_STRIDE = 10

# ── Honest backtest cost model ────────────────────────────────────────────────
# The decisive, honest cost is crossing the real top-of-book half-spread to enter
# and again to exit, plus a fixed commission. The headline result reported in the
# README uses a 3x multiple of the measured retail round-trip cost.
COMMISSION_BP_RT = 0.35           # round-trip commission in bp
GAP_SECONDS = 15.0                # a dt > this invalidates OFI continuity / fwd P&L
IS_FRAC = 0.65                    # 65% in-sample by time, 35% out-of-sample

# ── Inference confidence filter ───────────────────────────────────────────────
MIN_PROB = 0.38                   # max(P_UP, P_DOWN) gate; 3-class baseline = 0.33
COOLDOWN_BARS = 30                # bars to wait after an exit before re-entry
