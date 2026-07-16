import csv
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from cnn_for_ani.final_pipeline import (
    ExponentialMovingAverage,
    FinalTrainingConfig,
    TrainingPausedError,
    _train_member,
    build_final_split,
    warmup_cosine_learning_rate,
)


def _write_batch(
    labeled_dir: Path,
    writer: csv.DictWriter,
    *,
    batch: str,
    source: str,
    timestamp: int,
    sample_count: int,
) -> None:
    for index in range(sample_count):
        sample_id = f"sample-{batch}-{timestamp + index:013d}-{index:04d}"
        filename = (
            f"{index % 10}{(index // 2) % 10}{(index // 3) % 10}{(index // 5) % 10}_{sample_id}.png"
        )
        (labeled_dir / filename).touch()
        writer.writerow(
            {
                "id": sample_id,
                "source": source,
                "width": 128,
                "height": 40,
                "batch": batch,
                "purpose": "training_raw",
                "relative_path": f"{batch}/{sample_id}.png",
                "sha256": "unused",
            }
        )


def test_final_split_balances_sources_and_covers_snapshot(tmp_path) -> None:
    labeled_dir = tmp_path / "labeled"
    labeled_dir.mkdir()
    metadata_path = tmp_path / "metadata.csv"
    fieldnames = (
        "id",
        "source",
        "width",
        "height",
        "batch",
        "purpose",
        "relative_path",
        "sha256",
    )
    with metadata_path.open("w", encoding="utf-8", newline="") as metadata_file:
        writer = csv.DictWriter(metadata_file, fieldnames=fieldnames)
        writer.writeheader()
        _write_batch(
            labeled_dir,
            writer,
            batch="batch_a",
            source="source_a",
            timestamp=1_700_000_000_000,
            sample_count=300,
        )
        _write_batch(
            labeled_dir,
            writer,
            batch="batch_b",
            source="source_b",
            timestamp=1_800_000_000_000,
            sample_count=300,
        )
        _write_batch(
            labeled_dir,
            writer,
            batch="batch_c",
            source="source_c",
            timestamp=1_900_000_000_000,
            sample_count=300,
        )

    split = build_final_split(labeled_dir, metadata_path)

    train = set(split["train_samples"])
    validation = set(split["validation_samples"])
    test = set(split["test_samples"])
    assert split["test_source_counts"] == {
        "source_a": 100,
        "source_b": 100,
        "source_c": 100,
    }
    assert all(len(filenames) == 100 for filenames in split["test_samples_by_source"].values())
    assert (len(train), len(validation), len(test)) == (540, 60, 300)
    assert not train & validation
    assert not train & test
    assert not validation & test
    assert len(train | validation | test) == 900


def test_warmup_cosine_schedule_reaches_both_documented_endpoints() -> None:
    arguments = {
        "total_updates": 25_000,
        "warmup_updates": 500,
        "learning_rate": 1.5e-3,
        "min_learning_rate": 1e-5,
    }

    first = warmup_cosine_learning_rate(1, **arguments)
    warm = warmup_cosine_learning_rate(500, **arguments)
    final = warmup_cosine_learning_rate(25_000, **arguments)

    assert first == pytest.approx(1.5e-3 / 500)
    assert warm == pytest.approx(1.5e-3)
    assert final == pytest.approx(1e-5)


def test_ema_updates_parameters_and_batch_norm_buffers() -> None:
    model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
    ema = ExponentialMovingAverage(model, decay=0.5)
    with torch.no_grad():
        model[0].weight.add_(2.0)
        model[1].running_mean.fill_(4.0)
        model[1].num_batches_tracked.fill_(3)

    ema.update(model)

    assert torch.allclose(ema.model[0].weight, model[0].weight - 1.0)
    assert torch.equal(ema.model[1].running_mean, torch.full((2,), 2.0))
    assert ema.model[1].num_batches_tracked.item() == 3


def test_interrupted_training_resumes_with_identical_final_weights(tmp_path) -> None:
    paths = []
    for index in range(8):
        path = tmp_path / f"{index:04d}_sample-{index}.png"
        Image.new("L", (96, 32), color=32 + index * 20).save(path)
        paths.append(path)
    continuous_dir = tmp_path / "continuous"
    resumed_dir = tmp_path / "resumed"
    continuous_dir.mkdir()
    resumed_dir.mkdir()
    config = FinalTrainingConfig(
        total_updates=4,
        min_updates=1,
        validation_interval=2,
        early_stopping_patience=10,
        hard_replay_start=2,
        batch_size=4,
        warmup_updates=1,
        checkpoint_interval=1,
        seeds=(42,),
    )

    continuous_model, continuous_result = _train_member(
        paths,
        None,
        continuous_dir,
        42,
        config,
        torch.device("cpu"),
    )
    with pytest.raises(TrainingPausedError, match="update 2"):
        _train_member(
            paths,
            None,
            resumed_dir,
            42,
            config,
            torch.device("cpu"),
            pause_after_update=2,
        )
    resumed_model, resumed_result = _train_member(
        paths,
        None,
        resumed_dir,
        42,
        config,
        torch.device("cpu"),
    )

    for name, parameter in continuous_model.state_dict().items():
        assert torch.equal(parameter, resumed_model.state_dict()[name])
    assert continuous_result["hard_replay"] == resumed_result["hard_replay"]
    assert resumed_result["trained_updates"] == 4
