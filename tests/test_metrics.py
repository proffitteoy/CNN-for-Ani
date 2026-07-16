import pytest
import torch

from cnn_for_ani.metrics import calculate_metrics
from cnn_for_ani.prediction import calculate_top_k_exact_accuracy, decode_logits, decode_top_k


def test_metrics_include_exact_captcha_accuracy() -> None:
    targets = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    predictions = torch.tensor([[1, 2, 3, 4], [5, 0, 7, 0]])
    logits = torch.full((2, 4, 10), -10.0)
    logits.scatter_(2, predictions.unsqueeze(-1), 10.0)

    metrics = calculate_metrics(logits, targets)

    assert metrics.char_accuracy == pytest.approx(0.75)
    assert metrics.exact_accuracy == pytest.approx(0.5)
    assert metrics.hamming_error == pytest.approx(1.0)


def test_prediction_confidence_uses_weakest_position() -> None:
    logits = torch.zeros(1, 4, 10)
    logits[0, 0, 9] = 12.0
    logits[0, 1, 2] = 11.0
    logits[0, 2, 7] = 1.0
    logits[0, 3, 1] = 10.0

    prediction = decode_logits(logits)
    per_position = logits.softmax(dim=-1).max(dim=-1).values

    assert prediction.digits.tolist() == [[9, 2, 7, 1]]
    assert prediction.confidence.item() == pytest.approx(per_position.min().item())


def test_top_k_decoding_returns_distinct_joint_sequences() -> None:
    logits = torch.full((2, 4, 10), -10.0)
    logits[0, :, 0] = 10.0
    logits[0, 3, 1] = 11.0
    logits[1, :, 0] = 10.0
    targets = torch.zeros(2, 4, dtype=torch.long)

    prediction = decode_top_k(logits, 3)
    accuracies = calculate_top_k_exact_accuracy(logits, targets, ks=(1, 2, 3))

    assert prediction.digits.shape == (2, 3, 4)
    assert len({tuple(sequence) for sequence in prediction.digits[0].tolist()}) == 3
    assert accuracies[1] == pytest.approx(0.5)
    assert accuracies[2] == pytest.approx(1.0)
    assert accuracies[3] == pytest.approx(1.0)
