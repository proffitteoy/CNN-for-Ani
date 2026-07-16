import torch
from torch.nn import functional as F

from cnn_for_ani.losses import mean_head_cross_entropy
from cnn_for_ani.model import CaptchaCNN


def test_mean_head_cross_entropy_backpropagates() -> None:
    model = CaptchaCNN()
    logits = model(torch.zeros(2, 1, 32, 96))
    targets = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])

    loss = mean_head_cross_entropy(logits, targets)
    loss.backward()

    assert loss.ndim == 0
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_weighted_smoothed_loss_matches_final_plan() -> None:
    logits = torch.randn(3, 4, 10)
    targets = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 0, 1]])
    weights = (1.0, 1.15, 1.15, 1.0)

    actual = mean_head_cross_entropy(
        logits,
        targets,
        position_weights=weights,
        label_smoothing=0.02,
    )
    expected = sum(
        weight * F.cross_entropy(logits[:, position], targets[:, position], label_smoothing=0.02)
        for position, weight in enumerate(weights)
    ) / sum(weights)

    assert torch.allclose(actual, expected)
