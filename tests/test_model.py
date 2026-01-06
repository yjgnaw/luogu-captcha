import torch

from src.config import BLANK_INDEX, CHARSET, NUM_CLASSES
from src.model import (CRNN, CRNNConfig, ctc_greedy_decode, labels_to_text,
                       text_to_labels)


def test_forward_shape_and_normalization():
    batch_size = 2
    model = CRNN()
    x = torch.randn(batch_size, 3, 35, 90)

    with torch.no_grad():
        log_probs = model(x)

    # (T, B, C)
    assert log_probs.dim() == 3
    T, B, C = log_probs.shape
    assert B == batch_size
    assert C == NUM_CLASSES
    # 90 -> 45 -> 22 after two 2x2 pools
    assert T == 22

    probs = log_probs.exp().sum(dim=2)
    assert torch.allclose(probs, torch.ones_like(probs), atol=1e-4)


def test_output_lengths():
    widths = [90, 89, 1, 2, 3, 4]
    expected = [22, 22, 0, 0, 0, 1]
    assert CRNN.output_lengths(widths) == expected


def test_ctc_greedy_decode_runs():
    batch_size = 3
    seq_len = 22
    dummy_logits = torch.randn(seq_len, batch_size, NUM_CLASSES)
    log_probs = dummy_logits.log_softmax(dim=2)

    decoded = ctc_greedy_decode(log_probs)

    assert len(decoded) == batch_size
    # Ensure blanks are properly ignored and only charset characters appear
    for text in decoded:
        assert all(ch in CHARSET for ch in text)


def test_grayscale_config():
    # Ensure the model can be instantiated for grayscale images
    model = CRNN(CRNNConfig(img_channels=1))
    x = torch.randn(1, 1, 35, 90)
    with torch.no_grad():
        out = model(x)
    assert out.shape[1] == 1


def test_text_to_labels():
    """Test string to label tensor conversion."""
    text = "ABC1"
    labels = text_to_labels(text)
    assert labels.dtype == torch.long
    assert labels.shape == (4,)
    assert (labels >= 0).all() and (labels < len(CHARSET)).all()


def test_text_to_labels_with_device():
    """Test text_to_labels with device specification."""
    text = "XYZ9"
    device = torch.device("cpu")
    labels = text_to_labels(text, device=device)
    assert labels.device.type == device.type


def test_labels_to_text():
    """Test label tensor to string conversion."""
    text = "3K7F"
    labels = text_to_labels(text)
    recovered = labels_to_text(labels)
    assert recovered == text


def test_labels_to_text_with_blank():
    """Test that blanks are properly skipped during decoding."""
    # Create label sequence with blank indices interspersed
    label_seq = [0, BLANK_INDEX, 1, BLANK_INDEX, 2, BLANK_INDEX]
    recovered = labels_to_text(label_seq)
    assert BLANK_INDEX not in [ord(ch) for ch in recovered]


def test_ctc_greedy_decode_collapse_repeats():
    """Test that greedy decode collapses consecutive repeated characters."""
    # Create a simple log-prob that strongly predicts the same character repeatedly
    batch_size = 1
    seq_len = 10
    log_probs = torch.full((seq_len, batch_size, NUM_CLASSES), -100.0)
    # Set first character to high probability for repeated predictions
    log_probs[:, :, 0] = 0.0
    log_probs = torch.log_softmax(log_probs, dim=2)

    decoded = ctc_greedy_decode(log_probs)
    # Should collapse the repeated character
    assert len(decoded) == 1
    assert len(decoded[0]) == 1
    assert decoded[0] == CHARSET[0]


def test_crnn_train_eval_mode():
    """Test that CRNN properly switches between train and eval modes."""
    model = CRNN()
    assert model.training

    model.eval()
    assert not model.training

    model.train()
    assert model.training


def test_crnn_config_custom():
    """Test CRNN with custom config parameters."""
    config = CRNNConfig(
        img_channels=1,
        hidden_size=128,
        num_lstm_layers=1,
        num_classes=NUM_CLASSES,
    )
    model = CRNN(config)
    x = torch.randn(2, 1, 35, 90)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (22, 2, NUM_CLASSES)


def test_batch_consistency():
    """Test that batch processing gives consistent shapes."""
    model = CRNN()
    batch_sizes = [1, 4, 16, 32]

    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 3, 35, 90)
        with torch.no_grad():
            log_probs = model(x)

        assert log_probs.shape[0] == 22  # sequence length
        assert log_probs.shape[1] == batch_size
        assert log_probs.shape[2] == NUM_CLASSES
