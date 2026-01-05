# src/

Core module for the Luogu captcha CRNN project.

## Modules

### `config.py`

Shared configuration constants used across the project.

**Exports:**

- `CHARSET` (str): Character set for 4-character captchas
  - 9 digits: 1-9 (0 excluded)
  - 26 uppercase letters: A-Z
  - 25 lowercase letters: a-z excluding 'o'
  - Total: 60 characters
- `BLANK_INDEX` (int): Index reserved for CTC blank symbol (=60)
- `NUM_CLASSES` (int): Total number of classes including blank (=61)

### `model.py`

CRNN model architecture and utility functions for captcha recognition.

**Classes:**

- `CRNN`: Main model class inheriting from `nn.Module`
  - Expects input: `(batch, 3, 35, 90)` normalized to [0, 1]
  - Outputs: `(T=22, batch, NUM_CLASSES)` log-probabilities for CTC loss
  - Configurable via `CRNNConfig`

- `CRNNConfig`: Configuration dataclass for CRNN
  - `img_channels` (int, default=3): Number of input channels (1 or 3)
  - `hidden_size` (int, default=256): BiLSTM hidden size
  - `num_lstm_layers` (int, default=2): Number of LSTM layers
  - `num_classes` (int, default=NUM_CLASSES): Output classes

**Functions:**

- `get_charset()` → str: Return the default charset
- `text_to_labels(text, charset=CHARSET, device=None)` → Tensor
  - Convert ground-truth string to 1D label tensor
  - Example: "ABC1" → tensor([0, 1, 2, 3])

- `labels_to_text(labels, charset=CHARSET, blank_index=BLANK_INDEX)` → str
  - Convert label indices to string, skipping blanks
  - Example: [0, 60, 1, 60, 2] → "AB1" (60 = blank)

- `ctc_greedy_decode(log_probs, charset=CHARSET, blank_index=BLANK_INDEX)` → List[str]
  - Greedy CTC decoding: argmax + collapse repeats + remove blanks
  - Input: (T, B, C) log-probabilities
  - Returns: List of B decoded strings

**Class Methods:**

- `CRNN.output_lengths(input_widths)` → List[int]
  - Compute output sequence lengths after CNN pooling
  - Input widths of 90 → output length of 22

## Architecture Overview

```
Input (B, 3, 35, 90)
    ↓
[CNN Backbone]
  Conv(3→64) + Pool → Conv(64→128) + Pool → Conv(128→256) × 2 + Pool
  → Conv(256→512) × 2 + Pool
    ↓
Features (B, 512, 2, 22)
    ↓
[Reshape to (22, B, 1024)]
    ↓
[BiLSTM]
  2 layers, 256 hidden
    ↓
[Linear Classifier]
  512 → 61
    ↓
Log-Softmax
    ↓
Output (T=22, B, 61)
```

## Usage Examples

```python
from src.model import CRNN, text_to_labels, ctc_greedy_decode
from src.config import CHARSET, BLANK_INDEX
import torch

# Create model
model = CRNN()
model.eval()

# Prepare batch
batch = torch.randn(4, 3, 35, 90)  # 4 images, 35×90 pixels, 3 channels

# Inference
with torch.no_grad():
    log_probs = model(batch)  # (22, 4, 61)

# Decode
predictions = ctc_greedy_decode(log_probs)
print(predictions)  # ['ABC1', 'XYZ9', ...]

# Training
import torch.nn as nn
criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)
ground_truth = ["ABC1", "XYZ9"]
targets = torch.cat([text_to_labels(gt) for gt in ground_truth])
target_lengths = torch.tensor([4, 4])
input_lengths = torch.full((4,), 22)  # sequence length from output_lengths

loss = criterion(log_probs, targets, input_lengths, target_lengths)
loss.backward()
```

## Notes

- Model weights are initialized with sensible defaults (Kaiming for Conv, Xavier for Linear)
- All layers use BatchNorm for stabilization
- ReLU activations with inplace=True for memory efficiency
- No dropout in current version; data augmentation is the main regularization
