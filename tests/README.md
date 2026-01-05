# tests/

Unit tests for the Luogu captcha CRNN model.

## Running Tests

Run all tests:

```bash
pytest -q
```

Run with verbose output:

```bash
pytest -v
```

Run a specific test:

```bash
pytest tests/test_model.py::test_forward_shape_and_normalization -v
```

Run with coverage:

```bash
pytest --cov=src tests/
```

## Test Organization

### `test_model.py`

Comprehensive test suite for `src.model` and `src.config`.

**Test Categories:**

1. **Architecture & Forward Pass**
   - `test_forward_shape_and_normalization`: Verify output shape (T, B, C) and probability normalization
   - `test_output_lengths`: Check CNN pooling output length calculation
   - `test_batch_consistency`: Ensure consistent behavior across different batch sizes

2. **Decoding & Conversion**
   - `test_ctc_greedy_decode_runs`: Greedy decoding produces valid output
   - `test_ctc_greedy_decode_collapse_repeats`: Repeated characters are collapsed properly
   - `test_text_to_labels`: String → label tensor conversion
   - `test_text_to_labels_with_device`: Device parameter handling
   - `test_labels_to_text`: Label tensor → string recovery (roundtrip)
   - `test_labels_to_text_with_blank`: Blank indices are properly removed

3. **Configuration & Mode**
   - `test_grayscale_config`: Grayscale (1-channel) model instantiation
   - `test_crnn_config_custom`: Custom config parameters
   - `test_crnn_train_eval_mode`: Train/eval mode switching

## Test Coverage

**Current:** 12 tests, all passing

| Category | Tests |
|----------|-------|
| Shape & normalization | 3 |
| Decoding & conversion | 5 |
| Configuration | 3 |
| Model behavior | 1 |

## Adding New Tests

When adding tests, follow these conventions:

1. **Naming**: `test_<feature>_<scenario>`
   - ✓ `test_text_to_labels`
   - ✓ `test_ctc_greedy_decode_collapse_repeats`

2. **Imports**: Import from `src` package:

   ```python
   from src.config import CHARSET, BLANK_INDEX, NUM_CLASSES
   from src.model import CRNN, CRNNConfig, ctc_greedy_decode, ...
   ```

3. **Assertions**: Use descriptive assertions:

   ```python
   assert log_probs.shape == (22, batch_size, NUM_CLASSES)
   assert all(ch in CHARSET for ch in prediction)
   ```

4. **Docstrings**: Add brief docstrings explaining what is tested:

   ```python
   def test_something():
       """Test that X behaves correctly when Y."""
   ```

## Dependencies

Tests require:

- `pytest>=7.0` (test runner)
- `torch>=2.2` (model framework)
- `torchvision>=0.17` (transforms)

Install with:

```bash
pip install -r ../requirements.txt
```

## CI/CD Integration

Tests can be integrated into GitHub Actions or other CI systems:

```yaml
- name: Run tests
  run: pytest -v --tb=short
```

All tests pass on Python 3.14+ with PyTorch 2.2+.
