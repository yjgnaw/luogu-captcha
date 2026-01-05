"""CRNN model for 35x90 Luogu captchas.

The generator in `generate.php` produces 4-character captchas with the charset
```
123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnpqrstuvwxyz
```
(9 digits without 0, 26 uppercase letters, and 25 lowercase letters without 'o').
This module provides a compact CRNN suitable for training with CTC loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
from torch import Tensor, nn

from .config import BLANK_INDEX, CHARSET, NUM_CLASSES


def get_charset() -> str:
    """Return the default charset used by the Luogu captcha generator."""

    return CHARSET


def text_to_labels(
    text: str, *, charset: str = CHARSET, device: torch.device | None = None
) -> Tensor:
    """Convert a ground-truth string into a 1D tensor of label indices.

    Args:
        text: The target string (length 4 for Luogu captchas).
        charset: The allowed characters; defaults to the Luogu charset.
        device: Optional device for the returned tensor.

    Returns:
        A ``torch.long`` tensor of shape ``(len(text),)``.
    """

    mapping = {ch: i for i, ch in enumerate(charset)}
    labels = [mapping[ch] for ch in text]
    return torch.tensor(labels, dtype=torch.long, device=device)


def labels_to_text(
    labels: Iterable[int], *, charset: str = CHARSET, blank_index: int = BLANK_INDEX
) -> str:
    """Convert a sequence of label indices back to string, skipping blanks."""

    chars: List[str] = []
    for idx in labels:
        if idx == blank_index:
            continue
        chars.append(charset[idx])
    return "".join(chars)


def ctc_greedy_decode(
    log_probs: Tensor, *, charset: str = CHARSET, blank_index: int = BLANK_INDEX
) -> List[str]:
    """Greedy CTC decoding.

    Args:
        log_probs: ``(T, B, C)`` log-probabilities from the model.
        charset: Charset used for decoding.
        blank_index: Index reserved for the CTC blank symbol.

    Returns:
        List of length ``B`` with decoded strings.
    """

    # (T, B, C) -> (T, B)
    best_path = log_probs.argmax(dim=2)
    batch_size = best_path.shape[1]
    decoded: List[str] = []

    for b in range(batch_size):
        prev = None
        chars: List[str] = []
        for idx in best_path[:, b].tolist():
            if idx == prev:
                continue  # collapse repeats
            if idx != blank_index:
                chars.append(charset[idx])
            prev = idx
        decoded.append("".join(chars))
    return decoded


@dataclass
class CRNNConfig:
    img_channels: int = 3
    hidden_size: int = 256
    num_lstm_layers: int = 2
    num_classes: int = NUM_CLASSES


class CRNN(nn.Module):
    """Compact CRNN tailored for 35x90 captchas (4 characters).

    Expected input: ``(batch, channels=3, height=35, width=90)`` tensors normalized
    to ``[0, 1]``. The network reduces height to 2 and produces a sequence length of
    ~22 steps along the width dimension, which is sufficient to decode 4 symbols
    with CTC.
    """

    def __init__(self, config: CRNNConfig | None = None):
        super().__init__()
        cfg = config or CRNNConfig()

        self.cnn = nn.Sequential(
            nn.Conv2d(cfg.img_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 35x90 -> 17x45
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 17x45 -> 8x22
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # 8x22 -> 4x22
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # 4x22 -> 2x22
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # After convs: feature shape (B, 512, 2, 22)
        self.rnn = nn.LSTM(
            input_size=512 * 2,  # channels * height
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_lstm_layers,
            bidirectional=True,
        )

        self.classifier = nn.Linear(cfg.hidden_size * 2, cfg.num_classes)
        self.log_softmax = nn.LogSoftmax(dim=2)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Compute log-probabilities ``(T, B, C)`` for CTC loss."""

        features = self.cnn(x)  # (B, C, H, W)
        b, c, h, w = features.shape

        # Prepare for RNN: (W, B, C*H)
        features = features.permute(3, 0, 1, 2).contiguous()
        features = features.view(w, b, c * h)

        recurrent, _ = self.rnn(features)
        logits = self.classifier(recurrent)
        return self.log_softmax(logits)

    @staticmethod
    def output_lengths(input_widths: Sequence[int]) -> List[int]:
        """Compute output sequence lengths after the CNN stack.

        The pooling keeps width unchanged except the first two 2x2 pools.
        Width transformation for a single value ``w``: ``w -> floor(w/2) -> floor(w/2)``.
        Remaining pooling has stride (2,1), so width stays the same.
        """

        return [w // 2 // 2 for w in input_widths]


__all__ = [
    "CHARSET",
    "BLANK_INDEX",
    "NUM_CLASSES",
    "CRNN",
    "CRNNConfig",
    "get_charset",
    "text_to_labels",
    "labels_to_text",
    "ctc_greedy_decode",
]
