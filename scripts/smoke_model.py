"""Quick smoke test for the CRNN model.

Runs a forward pass on random input shaped like Luogu captchas (3x35x90)
and performs greedy CTC decoding to verify the pipeline works end-to-end.
"""

from __future__ import annotations

import torch

from src.config import CHARSET, NUM_CLASSES
from src.model import CRNN, ctc_greedy_decode


def main() -> None:
    torch.manual_seed(0)

    batch_size = 2
    x = torch.randn(batch_size, 3, 35, 90)

    model = CRNN()
    with torch.no_grad():
        log_probs = model(x)

    decoded = ctc_greedy_decode(log_probs, charset=CHARSET)

    print(f"Log prob shape: {tuple(log_probs.shape)} (T, B, C={NUM_CLASSES})")
    for i, text in enumerate(decoded):
        print(f"Sample {i}: '{text}' (len={len(text)})")


if __name__ == "__main__":
    main()
