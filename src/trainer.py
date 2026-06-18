"""
Walk-forward training loop for DeepLOB.

Per fold:
  1. z-score-normalise features on TRAIN-set statistics only (no leakage)
  2. build strided window loaders (no giant pre-materialised tensor)
  3. train AdamW + ReduceLROnPlateau + early stopping on val loss
  4. keep the best-val checkpoint
  5. run inference on the test slice for the backtest

Every fold asserts finiteness of features, normalised features, labels, and
probabilities — a non-finite value raises rather than silently poisoning a run.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .config import (
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MAX_EPOCHS,
    PATIENCE,
    CLASS_WEIGHT,
    N_CLASSES,
    WINDOW_SIZE,
    TRAIN_STRIDE,
)
from .deeplob_model import DeepLOB, build_model, model_summary
from .dataset import make_loader, walk_forward_splits, build_timestamps

log = logging.getLogger("trainer")


def _class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=N_CLASSES).astype(float)
    counts = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights /= weights.sum()
    return torch.tensor(weights * N_CLASSES, dtype=torch.float32)


def _assert_finite(name: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr)
    if not np.isfinite(arr).all():
        bad = int(arr.size - np.isfinite(arr).sum())
        raise ValueError(f"{name} contains {bad} non-finite value(s)")


def _assert_labels(name: str, labels: np.ndarray) -> None:
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {labels.shape}")
    bad = ~np.isin(labels, np.arange(N_CLASSES))
    if bad.any():
        raise ValueError(f"{name} has invalid class labels: {labels[bad][:5].tolist()}")


def train_fold(
    fold_idx: int,
    tr_feat: np.ndarray, tr_lab: np.ndarray,
    va_feat: np.ndarray, va_lab: np.ndarray,
    device: torch.device,
) -> tuple[DeepLOB, float, np.ndarray, np.ndarray]:
    """Train one fold. Returns (best_model, best_val_loss, train_mean, train_std)."""
    model = build_model().to(device)
    log.info("Fold %d | %s", fold_idx, model_summary(model))

    _assert_finite(f"fold {fold_idx} train features", tr_feat)
    _assert_finite(f"fold {fold_idx} val features", va_feat)
    _assert_labels(f"fold {fold_idx} train labels", tr_lab)
    _assert_labels(f"fold {fold_idx} val labels", va_lab)

    if len(tr_lab) <= WINDOW_SIZE or len(va_lab) <= WINDOW_SIZE:
        log.warning("Fold %d: not enough bars — skipping", fold_idx)
        m = np.zeros((1, tr_feat.shape[1]), np.float32)
        s = np.ones((1, tr_feat.shape[1]), np.float32)
        return model, float("inf"), m, s

    mean = tr_feat.mean(axis=0, keepdims=True).astype(np.float32)
    std = (tr_feat.std(axis=0, keepdims=True) + 1e-8).astype(np.float32)
    tr_norm = (tr_feat - mean) / std
    va_norm = (va_feat - mean) / std
    _assert_finite(f"fold {fold_idx} normalised train features", tr_norm)
    _assert_finite(f"fold {fold_idx} normalised val features", va_norm)

    val_stride = max(1, TRAIN_STRIDE // 2)
    train_loader = make_loader(tr_norm, tr_lab, stride=TRAIN_STRIDE, shuffle=True)
    val_loader = make_loader(va_norm, va_lab, stride=val_stride, shuffle=False, batch_size=BATCH_SIZE * 2)
    if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
        log.warning("Fold %d: empty dataset after windowing — skipping", fold_idx)
        return model, float("inf"), mean, std

    log.info("Fold %d | train=%d windows (stride=%d)  val=%d windows",
             fold_idx, len(train_loader.dataset), TRAIN_STRIDE, len(val_loader.dataset))

    cw = _class_weights(tr_lab).to(device) if CLASS_WEIGHT else None
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimiser = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimiser, patience=2, factor=0.5, min_lr=1e-6)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"fold {fold_idx} epoch {epoch}: non-finite train loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(train_loader.dataset)

        model.eval()
        va_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                batch_loss = criterion(out, yb)
                if not torch.isfinite(batch_loss):
                    raise FloatingPointError(f"fold {fold_idx} epoch {epoch}: non-finite val loss")
                va_loss += batch_loss.item() * len(xb)
                correct += (out.argmax(1) == yb).sum().item()
                total += len(yb)
        va_loss /= len(val_loader.dataset)
        va_acc = correct / max(total, 1)

        scheduler.step(va_loss)
        log.info("Fold %d | Ep %02d | tr=%.4f | va=%.4f | acc=%.3f | lr=%.2e",
                 fold_idx, epoch, tr_loss, va_loss, va_acc, optimiser.param_groups[0]["lr"])

        if va_loss < best_val_loss - 1e-5:
            best_val_loss = va_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                log.info("Fold %d | Early stop at epoch %d", fold_idx, epoch)
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_val_loss, mean, std


def predict_test(
    model: DeepLOB,
    te_feat: np.ndarray,
    te_lab: np.ndarray,
    train_mean: np.ndarray,
    train_std: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inference on the test slice using TRAIN normalisation. Returns (probs, preds, labels)."""
    _assert_finite("test features", te_feat)
    _assert_finite("train mean", train_mean)
    _assert_finite("train std", train_std)
    _assert_labels("test labels", te_lab)

    te_norm = (te_feat - train_mean) / (train_std + 1e-8)
    _assert_finite("normalised test features", te_norm)
    loader = make_loader(te_norm, te_lab, shuffle=False, batch_size=BATCH_SIZE * 2)
    if len(loader.dataset) == 0:
        return np.empty((0, 3)), np.empty(0), np.empty(0)

    probs, preds, labels = [], [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            p = torch.softmax(model(xb.to(device)), dim=-1).cpu().numpy()
            _assert_finite("test probabilities", p)
            probs.append(p)
            preds.append(p.argmax(axis=1))
            labels.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(preds), np.concatenate(labels)


def run_walk_forward(feat_df, label_series, device: torch.device | None = None,
                     max_folds: int | None = None) -> list[dict]:
    """Full walk-forward loop. Returns one result dict per fold."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    splits = walk_forward_splits(feat_df, label_series)
    if max_folds is not None:
        splits = splits[: max(0, int(max_folds))]

    results = []
    for i, split in enumerate(splits):
        log.info("=== Walk-Forward Fold %d / %d ===", i + 1, len(splits))
        tr_feat_df, tr_lab = split["train"]
        va_feat_df, va_lab = split["val"]
        te_feat_df, te_lab = split["test"]

        model, val_loss, tr_mean, tr_std = train_fold(
            i + 1,
            tr_feat_df.to_numpy(np.float32), tr_lab.to_numpy(np.int64),
            va_feat_df.to_numpy(np.float32), va_lab.to_numpy(np.int64),
            device,
        )
        probs, preds, labels = predict_test(
            model, te_feat_df.to_numpy(np.float32), te_lab.to_numpy(np.int64),
            tr_mean, tr_std, device,
        )
        results.append({
            "fold": i + 1,
            "test_start": split["test_start"],
            "test_end": split["test_end"],
            "model": model,
            "probs": probs, "preds": preds, "labels": labels,
            "timestamps": build_timestamps(te_feat_df, te_lab),
            "val_loss": val_loss,
        })
    return results
