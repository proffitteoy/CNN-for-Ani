import pytest
import torch

from cnn_for_ani.dataset import LabeledCaptchaDataset, parse_label, parse_sample_id


def test_parse_label_preserves_leading_zero() -> None:
    assert torch.equal(parse_label("0834_000002.png"), torch.tensor([0, 8, 3, 4]))
    assert parse_sample_id("0834_000002.png") == "000002"


@pytest.mark.parametrize("filename", ["0834.png", "83_000002.png", "abcd_000002.png"])
def test_parse_label_rejects_invalid_filename(filename: str) -> None:
    with pytest.raises(ValueError, match="labeled filename"):
        parse_label(filename)


def test_dataset_ignores_directory_placeholder(tmp_path) -> None:
    (tmp_path / ".gitkeep").touch()

    dataset = LabeledCaptchaDataset(tmp_path)

    assert len(dataset) == 0


def test_dataset_can_replay_an_exact_filename_snapshot(tmp_path) -> None:
    from PIL import Image

    for filename in ("1234_first.png", "5678_second.png"):
        Image.new("L", (96, 32)).save(tmp_path / filename)

    dataset = LabeledCaptchaDataset(tmp_path, filenames=["5678_second.png"])

    assert [path.name for path in dataset.paths] == ["5678_second.png"]
