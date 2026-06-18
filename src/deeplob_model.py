"""
DeepLOB-style CNN + LSTM classifier for limit-order-book data.

Reference: Zhang, Zohren & Roberts (2018), "DeepLOB: Deep Convolutional Neural
           Networks for Limit Order Books", arXiv:1808.03668.

The original DeepLOB consumes a 40-channel LOB representation
(10 levels x {bid_price, ask_price, bid_size, ask_size}) over a window of recent
snapshots and predicts the short-horizon direction of the mid price.

Architecture (this implementation):
    Input:  (batch, WINDOW_SIZE, N_FEATURES)            e.g. (B, 100, 40)
      |  Block 1: Conv1D(32, k=3) -> BN -> LeakyReLU  x2 -> MaxPool(2)
      |  Block 2: Conv1D(64, k=3) -> BN -> LeakyReLU  x2
      |  Block 3: Conv1D(64, k=3) -> BN -> LeakyReLU  x2
      |  LSTM(64 hidden, 1 layer)                       temporal aggregation
      |  Linear(64 -> 32) -> LeakyReLU -> Dropout(0.3)
      |  Linear(32 -> 3)                                logits
    Output: (batch, 3)  ->  softmax = [P(DOWN), P(FLAT), P(UP)]

Pass logits straight into nn.CrossEntropyLoss during training; apply softmax only
at inference via predict_proba().
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .config import (
    WINDOW_SIZE,
    N_FEATURES,
    CNN_CHANNELS,
    LSTM_HIDDEN,
    DROPOUT,
    N_CLASSES,
)


class _ConvBlock(nn.Module):
    """Two Conv1D -> BN -> LeakyReLU layers, with an optional trailing MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, pool: bool = False):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool1d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeepLOB(nn.Module):
    """
    DeepLOB CNN+LSTM classifier.

    Input shape:  (batch, WINDOW_SIZE, N_FEATURES)
    Output shape: (batch, N_CLASSES)   raw logits
    """

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        n_features: int = N_FEATURES,
        cnn_channels: list[int] = CNN_CHANNELS,
        lstm_hidden: int = LSTM_HIDDEN,
        dropout: float = DROPOUT,
        n_classes: int = N_CLASSES,
    ):
        super().__init__()
        self.window_size = window_size
        self.n_features = n_features

        # Conv blocks consume (B, C_in, L); we transpose (B, T, F) -> (B, F, T).
        ch = [n_features] + cnn_channels
        self.conv_blocks = nn.ModuleList()
        for i, (in_ch, out_ch) in enumerate(zip(ch[:-1], ch[1:])):
            pool = (i == 0)  # pool only after the first block
            self.conv_blocks.append(_ConvBlock(in_ch, out_ch, pool=pool))

        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = x.transpose(1, 2)                 # (B, F, T) for Conv1d
        for block in self.conv_blocks:
            x = block(x)                      # (B, C, T')
        x = x.transpose(1, 2)                 # (B, T', C) for LSTM
        _, (h, _) = self.lstm(x)              # h: (1, B, H)
        x = h.squeeze(0)                      # (B, H)
        return self.head(x)                   # (B, n_classes)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax class probabilities, shape (B, n_classes)."""
        self.eval()
        return torch.softmax(self(x), dim=-1)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: Path | str, **kwargs) -> "DeepLOB":
        model = cls(**kwargs)
        state = torch.load(path, map_location="cpu", weights_only=True)
        bad = sum((~torch.isfinite(v)).sum().item() for v in state.values())
        if bad:
            raise ValueError(f"Checkpoint {path} has {int(bad)} non-finite parameter value(s)")
        model.load_state_dict(state)
        model.eval()
        return model


def build_model() -> DeepLOB:
    return DeepLOB()


def model_summary(model: DeepLOB) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"DeepLOB: {total:,} params ({trainable:,} trainable)"
