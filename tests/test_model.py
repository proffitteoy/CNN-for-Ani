import pytest
import torch

from cnn_for_ani.model import (
    CaptchaCNN,
    CaptchaEnsemble,
    FlattenCaptchaCNN,
    MediumCaptchaCNN,
    PositionDSCaptchaCNN,
    WidePositionCaptchaCNN,
    parameter_count,
)


def test_model_matches_documented_contract() -> None:
    model = CaptchaCNN()
    logits = model(torch.zeros(2, 1, 32, 96))

    assert logits.shape == (2, 4, 10)
    assert parameter_count(model) == 4_218


def test_ablation_models_match_output_and_parameter_contracts() -> None:
    images = torch.zeros(2, 1, 32, 96)

    flatten_model = FlattenCaptchaCNN()
    medium_model = MediumCaptchaCNN()

    assert flatten_model(images).shape == (2, 4, 10)
    assert medium_model(images).shape == (2, 4, 10)
    assert parameter_count(flatten_model) == 34_328
    assert parameter_count(medium_model) == 27_728

    wide_position_model = WidePositionCaptchaCNN()
    assert wide_position_model(images).shape == (2, 4, 10)
    assert parameter_count(wide_position_model) == 40_394

    final_model = PositionDSCaptchaCNN()
    assert final_model(images).shape == (2, 4, 10)
    assert 90_000 <= parameter_count(final_model) <= 120_000


def test_final_model_uses_independent_position_heads_and_horizontal_dilation() -> None:
    model = PositionDSCaptchaCNN()

    assert len(model.heads) == 4
    assert len({id(head) for head in model.heads}) == 4
    assert model.stage3[-1].depthwise.dilation == (1, 2)
    assert model.position_pool.output_size == (1, 4)


def test_model_rejects_unexpected_input_size() -> None:
    model = CaptchaCNN()

    with pytest.raises(ValueError, match="expected input shape"):
        model(torch.zeros(1, 1, 40, 120))


def test_ensemble_averages_member_logits() -> None:
    first = CaptchaCNN()
    second = CaptchaCNN()
    images = torch.zeros(1, 1, 32, 96)
    ensemble = CaptchaEnsemble([first, second])

    expected = torch.stack((first(images), second(images))).mean(dim=0)

    assert torch.equal(ensemble(images), expected)
