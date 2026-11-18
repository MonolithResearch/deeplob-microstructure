"""
End-to-end demo — runs WITHOUT any licensed data.

Pipeline:
    1. generate a small synthetic 10-level LOB (toy random walk + faint flow term)
    2. compute the 40-channel DeepLOB features + the OFI z-score
    3. compute 3-class direction labels
    4. train ONE short DeepLOB fold (just to show the model trains & infers)
    5. run the HONEST no-lookahead OFI out-of-sample backtest
    6. render a Monolith-branded chart of the OOS equity (gross vs net)

The simulated returns illustrate the accounting. The status of the historical
licensed-data results is documented in docs/RESULTS.md.

Usage:
    py -3.14 demo/run_demo.py            # full demo (trains a tiny model)
    py -3.14 demo/run_demo.py --no-train # skip training, just the OFI backtest
    py -3.14 demo/run_demo.py --bars 40000 --ofi-signal 0.02
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make stdout UTF-8 so the brand glyphs print on Windows consoles (cp1252).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.synthetic_lob import generate_synthetic_lob
from src.lob_features import clock_starts, compute_features, compute_labels, FEATURE_NAMES
from src.honest_oos import run_honest_oos
from src import brand


def _train_one_fold(feat_df, labels, window, n_subset=8000):
    """Train a single short DeepLOB fold on a subset, just to prove it runs."""
    import torch
    from src.dataset import make_loader
    from src.deeplob_model import build_model, model_summary

    feat = feat_df[FEATURE_NAMES].to_numpy(np.float32)[:n_subset]
    lab = labels.reindex(feat_df.index).fillna(1).to_numpy(np.int64)[:n_subset]
    cut = int(len(feat) * 0.8)
    mean = feat[:cut].mean(0, keepdims=True)
    std = feat[:cut].std(0, keepdims=True) + 1e-8
    tr = ((feat[:cut] - mean) / std)
    va = ((feat[cut:] - mean) / std)

    model = build_model()
    print("  ", model_summary(model))
    train_loader = make_loader(tr, lab[:cut], window=window, stride=10, shuffle=True, batch_size=512)
    val_loader = make_loader(va, lab[cut:], window=window, stride=10, shuffle=False, batch_size=512)
    if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
        print("   (not enough windows to train; skipping)")
        return
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-4)
    crit = torch.nn.CrossEntropyLoss()
    for epoch in range(1, 4):  # a few epochs is plenty for a smoke run
        model.train()
        tot = 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(xb)
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb).argmax(1)
                correct += (pred == yb).sum().item()
                total += len(yb)
        print(f"   epoch {epoch}: train_loss={tot/len(train_loader.dataset):.4f} "
              f"val_acc={correct/max(total,1):.3f}")
    print("   DeepLOB trained + inferred OK on synthetic data.")


def main():
    ap = argparse.ArgumentParser(description="DeepLOB + OFI honest demo (no licensed data)")
    ap.add_argument("--bars", type=int, default=60_000)
    ap.add_argument("--ofi-signal", type=float, default=0.0,
                    help="toy flow->return coupling; 0.0 = pure random-walk null")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-train", action="store_true", help="skip the DeepLOB fold")
    ap.add_argument("--out", default=str(ROOT / "docs" / "results" / "demo_oos_equity.png"))
    args = ap.parse_args()

    print("=" * 72)
    print("  Monolith Research (Sigma) - DeepLOB + OFI microstructure demo")
    print("  Evidence over narrative")
    print("=" * 72)

    print(f"\n[1/5] Generating synthetic LOB ({args.bars:,} bars, ofi_signal={args.ofi_signal}) ...")
    lob = generate_synthetic_lob(n_bars=args.bars, seed=args.seed, ofi_signal=args.ofi_signal)
    print(f"      LOB shape {lob.shape}  span {lob.index.min()} -> {lob.index.max()}")

    print("\n[2/5] Computing model features on the first contiguous segment ...")
    starts = clock_starts(lob.index)
    first_stop = starts[1] if len(starts) > 1 else len(lob)
    feats = compute_features(lob.iloc[:first_stop])
    print(f"      feature frame {feats.shape}  "
          f"median spread bp = {float(feats['spread_bps'].median()):.3f}")

    print("\n[3/5] Computing 3-class direction labels ...")
    labels = compute_labels(feats["mid"], feats["spread_bps"], horizon=30, spread_multiplier=1.0)
    counts = labels.value_counts().sort_index()
    print(f"      label balance  DOWN={int(counts.get(0,0))}  "
          f"FLAT={int(counts.get(1,0))}  UP={int(counts.get(2,0))}")

    if not args.no_train:
        print("\n[4/5] Training one short DeepLOB fold on synthetic data ...")
        _train_one_fold(feats, labels, window=100)
    else:
        print("\n[4/5] Skipping DeepLOB training (--no-train).")

    print("\n[5/5] Running the HONEST no-lookahead OFI out-of-sample backtest ...")
    res = run_honest_oos(lob, verbose=True)
    oos = res["oos_summary"]
    print("\n  -- Honest OOS verdict ----------------------------------------------")
    print(f"     frozen config : {res['config']}")
    print(f"     OOS trades    : {oos['n_trades']}")
    print(f"     OOS net Sharpe: {oos['sharpe_net']:.4f}")
    print(f"     avg gross bp  : {oos['avg_gross_bp']:.4f}")
    print(f"     avg cost  bp  : {oos['avg_cost_bp']:.4f}   (the spread-crossing bar)")
    print(f"     gross > cost? : {res['gross_gt_cost']}")
    print(f"     VERDICT       : {res['verdict']}")
    print("  --------------------------------------------------------------------")
    print("  Synthetic results only. See docs/RESULTS.md for the historical correction.")

    _plot_equity(res, args.out)
    print(f"\n  Chart written -> {args.out}")


def _plot_equity(res, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    brand.apply_brand(plt)
    tr = res["oos_trades"]
    if len(tr["net_bp"]) == 0:
        print("   (no OOS trades to plot)")
        return
    eq = res["oos_marks"].cumsum()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(eq.index, eq["gross_bp"], color=brand.PLUM, lw=1.6,
            label="cumulative gross (bp)")
    ax.plot(eq.index, eq["net_bp"], color=brand.INK, lw=1.8,
            label="cumulative NET (bp) — after crossing the spread")
    ax.axhline(0, color=brand.PLUM_DARK, lw=0.8, ls="--", alpha=0.6)
    ax.set_title("Synthetic OFI out-of-sample P&L")
    ax.set_xlabel("snapshot time (UTC)")
    ax.set_ylabel("cumulative P&L (basis points)")
    ax.legend(loc="best", frameon=False)
    oos = res["oos_summary"]
    ax.text(0.01, 0.02,
            f"frozen config {res['config']}\n"
            f"OOS net Sharpe {oos['sharpe_net']:.2f} · "
            f"avg gross {oos['avg_gross_bp']:.3f}bp vs cost {oos['avg_cost_bp']:.3f}bp · "
            f"verdict {res['verdict']}",
            transform=ax.transAxes, fontsize=8, color=brand.PLUM_DARK, va="bottom",
            family="monospace")
    brand.brand_footer(fig)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
