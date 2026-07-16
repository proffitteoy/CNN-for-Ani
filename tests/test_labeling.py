from pathlib import Path

import pytest
from PIL import Image

from cnn_for_ani.labeling import build_labeled_path, labeled_sample_ids, unlabeled_raw_images


def save_image(path: Path) -> None:
    Image.new("L", (2, 2), color=255).save(path)


def test_build_labeled_path_preserves_raw_sample_id() -> None:
    path = build_labeled_path(
        labeled_dir=Path("dataset/labeled"),
        raw_path=Path("dataset/raw/batch_a/000001.PNG"),
        label="0834",
    )

    assert str(path).replace("\\", "/") == "dataset/labeled/0834_000001.png"


def test_build_labeled_path_rejects_invalid_label(tmp_path) -> None:
    with pytest.raises(ValueError, match="four digits"):
        build_labeled_path(tmp_path, tmp_path / "000001.png", "834")


def test_unlabeled_raw_images_skips_existing_labeled_samples(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    labeled_dir = tmp_path / "labeled"
    (raw_dir / "batch_a").mkdir(parents=True)
    labeled_dir.mkdir()
    save_image(raw_dir / "batch_a" / "000001.png")
    save_image(raw_dir / "batch_a" / "000002.png")
    save_image(labeled_dir / "5271_000001.png")

    remaining = unlabeled_raw_images(raw_dir, labeled_dir)

    assert [path.name for path in remaining] == ["000002.png"]
    assert labeled_sample_ids(labeled_dir) == {"000001"}
