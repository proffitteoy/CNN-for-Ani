"""Final Position-DS evaluation and production training pipeline."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, default_collate

from cnn_for_ani.augmentation import final_augment_image
from cnn_for_ani.dataset import LabeledCaptchaDataset, parse_label, parse_sample_id
from cnn_for_ani.error_analysis import confusion_matrix
from cnn_for_ani.losses import mean_head_cross_entropy
from cnn_for_ani.metrics import calculate_metrics
from cnn_for_ani.model import CaptchaEnsemble, build_model, parameter_count
from cnn_for_ani.prediction import calculate_top_k_exact_accuracy, decode_logits
from cnn_for_ani.preprocessing import preprocess_image
from cnn_for_ani.train_ensemble import export_ensemble_onnx
from cnn_for_ani.training import make_stratified_folds

FINAL_MODEL_NAME = "position_ds"
FINAL_SEEDS = (42, 3407, 20260716)
POSITION_WEIGHTS = (1.0, 1.15, 1.15, 1.0)
_TIMESTAMP_PATTERN = re.compile(r"-(?P<timestamp>\d{13})-\d+$")


def _progress(message: str) -> None:
    print(message, flush=True)
    log_path = os.environ.get("CNN_FOR_ANI_PROGRESS_LOG")
    if log_path:
        with Path(log_path).open("a", encoding="utf-8") as log_file:
            log_file.write(f"{datetime.now().astimezone().isoformat()} {message}\n")


@dataclass(frozen=True)
class FinalTrainingConfig:
    """Frozen optimization settings from ``docs/最终方案.md``."""

    total_updates: int = 25_000
    min_updates: int = 15_000
    validation_interval: int = 250
    early_stopping_patience: int = 4_000
    hard_replay_start: int = 17_500
    batch_size: int = 32
    learning_rate: float = 1.5e-3
    min_learning_rate: float = 1e-5
    warmup_updates: int = 500
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    ema_decay: float = 0.999
    label_smoothing: float = 0.02
    hard_sample_weight: float = 2.0
    checkpoint_interval: int = 250
    seeds: tuple[int, ...] = FINAL_SEEDS

    def validate(self) -> None:
        if self.total_updates < 1:
            raise ValueError("total_updates must be positive")
        if not 1 <= self.min_updates <= self.total_updates:
            raise ValueError("min_updates must be between 1 and total_updates")
        if not 1 <= self.validation_interval <= self.total_updates:
            raise ValueError("validation_interval must be between 1 and total_updates")
        if not 0 <= self.hard_replay_start < self.total_updates:
            raise ValueError("hard_replay_start must be in [0, total_updates)")
        if self.hard_sample_weight > 3.0 or self.hard_sample_weight < 1.0:
            raise ValueError("hard_sample_weight must be between 1 and 3")
        if not 1 <= self.checkpoint_interval <= self.total_updates:
            raise ValueError("checkpoint_interval must be between 1 and total_updates")
        if not self.seeds:
            raise ValueError("at least one seed is required")


class FinalTrainingDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, paths: list[Path], *, augment: bool) -> None:
        self.paths = paths
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[index]
        with Image.open(path) as image:
            prepared = final_augment_image(image) if self.augment else image.copy()
        return preprocess_image(prepared), parse_label(path)


class ExponentialMovingAverage:
    """EMA copy including BatchNorm parameters and buffers."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be between 0 and 1")
        self.model = copy.deepcopy(model).eval()
        self.decay = decay
        self.model.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_parameter, parameter in zip(
            self.model.parameters(), model.parameters(), strict=True
        ):
            ema_parameter.lerp_(parameter.detach(), 1.0 - self.decay)
        for ema_buffer, buffer in zip(self.model.buffers(), model.buffers(), strict=True):
            if torch.is_floating_point(ema_buffer):
                ema_buffer.lerp_(buffer.detach(), 1.0 - self.decay)
            else:
                ema_buffer.copy_(buffer)


def warmup_cosine_learning_rate(
    update: int,
    *,
    total_updates: int,
    warmup_updates: int,
    learning_rate: float,
    min_learning_rate: float,
) -> float:
    """Return the update-based warmup plus cosine-decay learning rate."""
    if not 1 <= update <= total_updates:
        raise ValueError("update must be between 1 and total_updates")
    if warmup_updates > 0 and update <= warmup_updates:
        return learning_rate * update / warmup_updates
    decay_updates = max(total_updates - warmup_updates, 1)
    progress = (update - warmup_updates) / decay_updates
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_learning_rate + (learning_rate - min_learning_rate) * cosine


def _stable_seed(text: str, seed: int) -> int:
    digest = hashlib.sha256(text.encode()).digest()
    return seed + int.from_bytes(digest[:4], "little")


def _sample_timestamp(sample_id: str) -> int:
    match = _TIMESTAMP_PATTERN.search(sample_id)
    return int(match.group("timestamp")) if match else 0


def _historical_sample_names(artifacts_dir: Path) -> set[str]:
    """Collect filenames explicitly referenced by completed earlier experiment manifests."""
    sample_keys = {
        "dataset_snapshot",
        "train_samples",
        "validation_samples",
        "holdout_samples",
        "test_samples",
    }
    filenames: set[str] = set()
    if not artifacts_dir.is_dir():
        return filenames
    for path in artifacts_dir.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        pending = [payload]
        while pending:
            item = pending.pop()
            if isinstance(item, dict):
                for key, value in item.items():
                    if key in sample_keys and isinstance(value, list):
                        filenames.update(name for name in value if isinstance(name, str))
                    else:
                        pending.append(value)
            elif isinstance(item, list):
                pending.extend(item)
    return filenames


def _select_digit_balanced_records(
    candidates: list[dict[str, Any]],
    count: int,
    *,
    target_records: list[dict[str, Any]],
    target_count: int,
    seed: int,
    initial_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Greedily select an exact-size subset while balancing every position's digit margins."""
    if count < 0 or count > len(candidates):
        raise ValueError("balanced subset count is outside the candidate range")
    selected = list(initial_records or [])
    if count == 0:
        return []
    labels = {record["filename"]: parse_label(record["path"]) for record in target_records}
    global_counts = torch.zeros(4, 10, dtype=torch.float64)
    selected_counts = torch.zeros(4, 10, dtype=torch.float64)
    for record in target_records:
        for position, digit in enumerate(labels[record["filename"]].tolist()):
            global_counts[position, digit] += 1
    for record in selected:
        for position, digit in enumerate(labels[record["filename"]].tolist()):
            selected_counts[position, digit] += 1
    targets = global_counts * (target_count / len(target_records))
    generator = torch.Generator().manual_seed(seed)
    tie_breakers = {
        record["filename"]: float(torch.rand((), generator=generator)) for record in candidates
    }
    remaining = list(candidates)
    additions: list[dict[str, Any]] = []
    for _ in range(count):

        def score(record: dict[str, Any]) -> tuple[float, float, str]:
            label = labels[record["filename"]]
            deficit = sum(
                (targets[position, digit].item() - selected_counts[position, digit].item())
                / max(targets[position, digit].item(), 1.0)
                for position, digit in enumerate(label.tolist())
            )
            return deficit, tie_breakers[record["filename"]], record["filename"]

        chosen = max(remaining, key=score)
        remaining.remove(chosen)
        additions.append(chosen)
        selected.append(chosen)
        for position, digit in enumerate(labels[chosen["filename"]].tolist()):
            selected_counts[position, digit] += 1
    return additions


def _select_source_balanced_test(
    records: list[dict[str, Any]],
    test_size: int,
    historical_samples: set[str],
    seed: int,
) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_source.setdefault(record["source"], []).append(record)
    if len(by_source) < 2:
        raise ValueError("final test set requires at least two data sources")
    base_quota, extra = divmod(test_size, len(by_source))
    source_order = sorted(
        by_source,
        key=lambda source: (
            -sum(record["filename"] not in historical_samples for record in by_source[source]),
            -len(by_source[source]),
            source,
        ),
    )
    quotas = {source: base_quota + (index < extra) for index, source in enumerate(source_order)}
    if any(quotas[source] > len(by_source[source]) for source in by_source):
        raise ValueError("a data source is too small for a source-balanced final test set")

    selected: list[dict[str, Any]] = []
    for source in sorted(by_source):
        source_records = by_source[source]
        quota = quotas[source]
        unseen = [
            record for record in source_records if record["filename"] not in historical_samples
        ]
        unseen_count = min(quota, len(unseen))
        source_selected = _select_digit_balanced_records(
            unseen,
            unseen_count,
            target_records=source_records,
            target_count=quota,
            seed=_stable_seed(f"test|{source}|unseen", seed),
        )
        remaining_count = quota - len(source_selected)
        if remaining_count:
            seen = [record for record in source_records if record["filename"] in historical_samples]
            source_selected.extend(
                _select_digit_balanced_records(
                    seen,
                    remaining_count,
                    target_records=source_records,
                    target_count=quota,
                    seed=_stable_seed(f"test|{source}|historical", seed),
                    initial_records=source_selected,
                )
            )
        selected.extend(source_selected)
    return selected


def build_final_split(
    labeled_dir: Path,
    metadata_path: Path,
    *,
    seed: int = 20260716,
    test_size: int | None = None,
    historical_samples: set[str] | None = None,
) -> dict[str, Any]:
    """Freeze a source-balanced test set, then stratify validation per source and batch."""
    dataset = LabeledCaptchaDataset(labeled_dir)
    with metadata_path.open(encoding="utf-8-sig", newline="") as metadata_file:
        rows = list(csv.DictReader(metadata_file))
    metadata_by_id = {row["id"]: row for row in rows}
    if len(metadata_by_id) != len(rows):
        raise ValueError("metadata contains duplicate sample ids")

    records: list[dict[str, Any]] = []
    for path in dataset.paths:
        sample_id = parse_sample_id(path)
        try:
            metadata = metadata_by_id[sample_id]
        except KeyError as error:
            raise ValueError(f"metadata is missing for labeled sample {path.name}") from error
        records.append(
            {
                "path": path,
                "filename": path.name,
                "sample_id": sample_id,
                "batch": metadata["batch"],
                "source": metadata["source"],
                "purpose": metadata["purpose"],
                "timestamp": _sample_timestamp(sample_id),
            }
        )

    minimum_test_size = max(300, math.ceil(0.15 * len(records)))
    selected_test_size = test_size or minimum_test_size
    if selected_test_size < minimum_test_size:
        raise ValueError(f"test_size must be at least {minimum_test_size}")
    historical_samples = historical_samples or set()
    test_records = _select_source_balanced_test(
        records,
        selected_test_size,
        historical_samples,
        seed,
    )
    test_names = {record["filename"] for record in test_records}
    remaining = [record for record in records if record["filename"] not in test_names]
    validation_names: set[str] = set()
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in remaining:
        groups.setdefault((record["source"], record["batch"]), []).append(record)
    for group_key, group_records in sorted(groups.items()):
        if len(group_records) < 10:
            continue
        labels = torch.stack([parse_label(record["path"]) for record in group_records])
        folds = make_stratified_folds(
            labels,
            fold_count=10,
            seed=_stable_seed("|".join(group_key), seed),
        )
        validation_names.update(group_records[index]["filename"] for index in folds[0])

    train_names = {
        record["filename"] for record in remaining if record["filename"] not in validation_names
    }
    if train_names & validation_names or train_names & test_names or validation_names & test_names:
        raise RuntimeError("final split is not disjoint")
    all_names = train_names | validation_names | test_names
    if all_names != {record["filename"] for record in records}:
        raise RuntimeError("final split does not cover the complete labeled snapshot")

    def ordered(names: set[str]) -> list[str]:
        return sorted(names)

    all_groups = {(record["source"], record["batch"]) for record in records}
    group_counts = {
        f"{source}/{batch}": sum(
            record["source"] == source and record["batch"] == batch for record in records
        )
        for source, batch in sorted(all_groups)
    }
    test_source_counts = {
        source: sum(record["source"] == source for record in test_records)
        for source in sorted({record["source"] for record in records})
    }
    test_samples_by_source = {
        source: sorted(record["filename"] for record in test_records if record["source"] == source)
        for source in test_source_counts
    }
    historical_overlap_by_source = {
        source: sum(
            record["source"] == source and record["filename"] in historical_samples
            for record in test_records
        )
        for source in test_source_counts
    }
    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "selection": (
            "equal-quota source stratification with approximate four-position digit balance; "
            "samples absent from historical experiment manifests are preferred"
        ),
        "minimum_test_size": minimum_test_size,
        "selected_test_size": selected_test_size,
        "sample_count": len(records),
        "train_samples": ordered(train_names),
        "validation_samples": ordered(validation_names),
        "test_samples": ordered(test_names),
        "train_size": len(train_names),
        "validation_size": len(validation_names),
        "test_size": len(test_names),
        "validation_strategy": (
            "position-wise approximate 10-fold stratification within each source/batch; "
            "fold 1 is validation"
        ),
        "test_isolation_note": (
            "The test set covers every source. Sources without enough never-used samples require "
            "historical experiment samples, so the recorded overlap must qualify final metrics."
        ),
        "test_source_counts": test_source_counts,
        "test_samples_by_source": test_samples_by_source,
        "historical_manifest_sample_count": len(historical_samples),
        "historical_test_overlap_count": sum(historical_overlap_by_source.values()),
        "historical_test_overlap_by_source": historical_overlap_by_source,
        "source_batch_counts": group_counts,
        "seed": seed,
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TrainingPausedError(RuntimeError):
    """Internal controlled interruption used to verify exact checkpoint recovery."""


def _capture_random_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_random_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all([cuda_state.cpu() for cuda_state in state["cuda"]])


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def _sample_training_batch(
    dataset: FinalTrainingDataset,
    weights: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.multinomial(
        weights,
        batch_size,
        replacement=True,
        generator=generator,
    ).tolist()
    return default_collate([dataset[index] for index in indices])


def _evaluation_loader(
    paths: list[Path], batch_size: int
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    return DataLoader(
        FinalTrainingDataset(paths, augment=False),
        batch_size=batch_size,
        shuffle=False,
    )


def _collect_logits(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    with torch.no_grad():
        for images, targets in loader:
            logits_batches.append(model(images.to(device)).cpu())
            target_batches.append(targets)
    return torch.cat(logits_batches), torch.cat(target_batches)


def _validation_result(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, float]:
    logits, targets = _collect_logits(model, loader, device)
    loss = mean_head_cross_entropy(
        logits,
        targets,
        position_weights=POSITION_WEIGHTS,
        label_smoothing=0.02,
    )
    return {"loss": loss.item(), **asdict(calculate_metrics(logits, targets))}


def _hard_sample_weights(
    model: nn.Module,
    paths: list[Path],
    config: FinalTrainingConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    logits, targets = _collect_logits(
        model,
        _evaluation_loader(paths, config.batch_size),
        device,
    )
    prediction = decode_logits(logits)
    matches = prediction.digits.eq(targets)
    exact_matches = matches.all(dim=1)
    correct_confidences = prediction.confidence[exact_matches]
    confidence_threshold = (
        float(torch.quantile(correct_confidences, 0.2)) if len(correct_confidences) else 1.0
    )
    hard_mask = (~exact_matches) | (exact_matches & (prediction.confidence <= confidence_threshold))
    weights = torch.ones(len(paths), dtype=torch.double)
    weights[hard_mask] = config.hard_sample_weight
    return weights, {
        "selection_update": config.hard_replay_start,
        "incorrect_count": int((~exact_matches).sum()),
        "low_confidence_correct_count": int(
            (exact_matches & (prediction.confidence <= confidence_threshold)).sum()
        ),
        "hard_sample_count": int(hard_mask.sum()),
        "low_confidence_quantile": 0.2,
        "confidence_threshold": confidence_threshold,
        "hard_sample_weight": config.hard_sample_weight,
    }


def _train_member(
    train_paths: list[Path],
    validation_paths: list[Path] | None,
    output_dir: Path,
    seed: int,
    config: FinalTrainingConfig,
    device: torch.device,
    *,
    total_updates: int | None = None,
    resume_checkpoint: Path | None = None,
    pause_after_update: int | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    _set_seed(seed)
    updates_to_run = total_updates or config.total_updates
    training_dataset = FinalTrainingDataset(train_paths, augment=True)
    model = build_model(FINAL_MODEL_NAME).to(device)
    ema = ExponentialMovingAverage(model, config.ema_decay)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    validation_loader = (
        _evaluation_loader(validation_paths, config.batch_size) if validation_paths else None
    )
    raw_best_path = output_dir / f"model_seed{seed}_raw_best.pt"
    ema_best_path = output_dir / f"model_seed{seed}_ema_best.pt"
    raw_final_path = output_dir / f"model_seed{seed}_raw_final.pt"
    ema_final_path = output_dir / f"model_seed{seed}_ema_final.pt"
    resume_path = resume_checkpoint or output_dir / f"model_seed{seed}.resume.pt"
    best_raw_exact = -1.0
    best_ema_exact = -1.0
    best_raw_update = 0
    best_ema_update = 0
    history: list[dict[str, Any]] = []
    hard_replay: dict[str, Any] | None = None
    weights = torch.ones(len(train_paths), dtype=torch.double)
    sampler_generator = torch.Generator().manual_seed(seed)
    stopped_update = updates_to_run
    interval_loss = 0.0
    interval_samples = 0
    start_update = 0
    stop_requested = False
    snapshot_sha256 = _snapshot_hash([path.name for path in train_paths])

    def save_resume_state(
        update: int,
        *,
        completed: bool,
        result: dict[str, Any] | None = None,
    ) -> None:
        _atomic_torch_save(
            {
                "resume_version": 1,
                "completed": completed,
                "model_name": FINAL_MODEL_NAME,
                "seed": seed,
                "total_updates": updates_to_run,
                "validation_enabled": validation_loader is not None,
                "train_snapshot_sha256": snapshot_sha256,
                "config": asdict(config),
                "update": update,
                "model": model.state_dict(),
                "ema_model": ema.model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_raw_exact": best_raw_exact,
                "best_ema_exact": best_ema_exact,
                "best_raw_update": best_raw_update,
                "best_ema_update": best_ema_update,
                "history": history,
                "hard_replay": hard_replay,
                "weights": weights,
                "interval_loss": interval_loss,
                "interval_samples": interval_samples,
                "stopped_update": stopped_update,
                "stop_requested": stop_requested,
                "sampler_generator_state": sampler_generator.get_state(),
                "random_state": _capture_random_state(),
                "result": result,
            },
            resume_path,
        )

    if resume_path.is_file():
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        expected = {
            "resume_version": 1,
            "model_name": FINAL_MODEL_NAME,
            "seed": seed,
            "total_updates": updates_to_run,
            "validation_enabled": validation_loader is not None,
            "train_snapshot_sha256": snapshot_sha256,
            "config": asdict(config),
        }
        mismatches = {
            key: (checkpoint.get(key), value)
            for key, value in expected.items()
            if checkpoint.get(key) != value
        }
        if mismatches:
            raise ValueError(f"resume checkpoint is incompatible: {mismatches}")
        model.load_state_dict(checkpoint["model"])
        ema.model.load_state_dict(checkpoint["ema_model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_raw_exact = float(checkpoint["best_raw_exact"])
        best_ema_exact = float(checkpoint["best_ema_exact"])
        best_raw_update = int(checkpoint["best_raw_update"])
        best_ema_update = int(checkpoint["best_ema_update"])
        history = checkpoint["history"]
        hard_replay = checkpoint["hard_replay"]
        weights = checkpoint["weights"].cpu()
        interval_loss = float(checkpoint["interval_loss"])
        interval_samples = int(checkpoint["interval_samples"])
        stopped_update = int(checkpoint["stopped_update"])
        stop_requested = bool(checkpoint["stop_requested"])
        start_update = int(checkpoint["update"])
        sampler_generator.set_state(checkpoint["sampler_generator_state"].cpu())
        _restore_random_state(checkpoint["random_state"])
        if checkpoint["completed"]:
            result = checkpoint["result"]
            if not isinstance(result, dict):
                raise ValueError("completed resume checkpoint is missing its member result")
            selected_model = build_model(FINAL_MODEL_NAME)
            selected_model.load_state_dict(
                torch.load(
                    output_dir / result["ema_checkpoint"],
                    map_location="cpu",
                    weights_only=True,
                )
            )
            _progress(f"seed={seed} already_complete update={start_update:05d}")
            return selected_model.eval(), result
        _progress(f"seed={seed} resumed_from_update={start_update:05d}")

    for update in range(start_update + 1, updates_to_run + 1) if not stop_requested else ():
        if update == config.hard_replay_start + 1:
            weights, hard_replay = _hard_sample_weights(ema.model, train_paths, config, device)
            sampler_generator.manual_seed(seed + config.hard_replay_start)
        images, targets = _sample_training_batch(
            training_dataset,
            weights,
            config.batch_size,
            sampler_generator,
        )
        images = images.to(device)
        targets = targets.to(device)
        learning_rate = warmup_cosine_learning_rate(
            update,
            total_updates=updates_to_run,
            warmup_updates=min(config.warmup_updates, updates_to_run - 1),
            learning_rate=config.learning_rate,
            min_learning_rate=config.min_learning_rate,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        model.train()
        loss = mean_head_cross_entropy(
            model(images),
            targets,
            position_weights=POSITION_WEIGHTS,
            label_smoothing=config.label_smoothing,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        ema.update(model)
        interval_loss += loss.item() * len(images)
        interval_samples += len(images)

        should_validate = validation_loader is not None and (
            update % config.validation_interval == 0 or update == updates_to_run
        )
        if should_validate:
            raw_result = _validation_result(model, validation_loader, device)
            ema_result = _validation_result(ema.model, validation_loader, device)
            if raw_result["exact_accuracy"] > best_raw_exact:
                best_raw_exact = raw_result["exact_accuracy"]
                best_raw_update = update
                torch.save(model.state_dict(), raw_best_path)
            if ema_result["exact_accuracy"] > best_ema_exact:
                best_ema_exact = ema_result["exact_accuracy"]
                best_ema_update = update
                torch.save(ema.model.state_dict(), ema_best_path)
            history.append(
                {
                    "update": update,
                    "learning_rate": learning_rate,
                    "train_augmented_loss": interval_loss / interval_samples,
                    "raw_validation": raw_result,
                    "ema_validation": ema_result,
                }
            )
            interval_loss = 0.0
            interval_samples = 0
            _progress(
                f"seed={seed} update={update:05d}/{updates_to_run} "
                f"raw_exact={raw_result['exact_accuracy']:.4f} "
                f"ema_exact={ema_result['exact_accuracy']:.4f}"
            )
            if (
                update >= config.min_updates
                and update - best_ema_update >= config.early_stopping_patience
            ):
                stopped_update = update
                stop_requested = True
        elif validation_loader is None and (
            update == 1 or update % 1_000 == 0 or update == updates_to_run
        ):
            _progress(
                f"seed={seed} update={update:05d}/{updates_to_run} "
                f"loss={interval_loss / interval_samples:.4f}"
            )
            interval_loss = 0.0
            interval_samples = 0

        should_checkpoint = (
            update % config.checkpoint_interval == 0
            or update in (updates_to_run, pause_after_update)
            or stop_requested
        )
        if should_checkpoint:
            save_resume_state(update, completed=False)
        if update == pause_after_update:
            raise TrainingPausedError(f"controlled pause after update {update}")
        if stop_requested:
            break

    torch.save(model.state_dict(), raw_final_path)
    torch.save(ema.model.state_dict(), ema_final_path)
    selected_ema_path = ema_best_path if validation_loader is not None else ema_final_path
    selected_raw_path = raw_best_path if validation_loader is not None else raw_final_path
    selected_model = build_model(FINAL_MODEL_NAME)
    selected_model.load_state_dict(
        torch.load(selected_ema_path, map_location="cpu", weights_only=True)
    )
    result = {
        "seed": seed,
        "trained_updates": stopped_update,
        "raw_best_update": best_raw_update if validation_loader is not None else stopped_update,
        "ema_best_update": best_ema_update if validation_loader is not None else stopped_update,
        "raw_best_exact_accuracy": best_raw_exact if validation_loader is not None else None,
        "ema_best_exact_accuracy": best_ema_exact if validation_loader is not None else None,
        "raw_checkpoint": selected_raw_path.name,
        "ema_checkpoint": selected_ema_path.name,
        "raw_final_checkpoint": raw_final_path.name,
        "ema_final_checkpoint": ema_final_path.name,
        "hard_replay": hard_replay,
        "history": history,
        "resume_checkpoint": resume_path.name,
    }
    save_resume_state(stopped_update, completed=True, result=result)
    return selected_model.eval(), result


def _detailed_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
    predictions = logits.argmax(dim=-1)
    matrix = confusion_matrix(targets, predictions)
    true_support = matrix.sum(dim=1)
    predicted_support = matrix.sum(dim=0)
    precision = matrix.diag().float() / predicted_support.clamp_min(1)
    recall = matrix.diag().float() / true_support.clamp_min(1)
    return {
        **asdict(calculate_metrics(logits, targets)),
        "top_k_exact_accuracy": {
            str(k): value
            for k, value in calculate_top_k_exact_accuracy(logits, targets, ks=(1, 2, 3, 5)).items()
        },
        "position_accuracies": predictions.eq(targets).float().mean(dim=0).tolist(),
        "digit_precision": precision.tolist(),
        "digit_recall": recall.tolist(),
        "digit_support": true_support.tolist(),
        "confusion_matrix": matrix.tolist(),
    }


def _metrics_by_source(
    logits: torch.Tensor,
    targets: torch.Tensor,
    test_samples: list[str],
    test_samples_by_source: dict[str, list[str]],
) -> dict[str, dict[str, Any]]:
    indices_by_name = {filename: index for index, filename in enumerate(test_samples)}
    return {
        source: _detailed_metrics(
            logits[[indices_by_name[filename] for filename in filenames]],
            targets[[indices_by_name[filename] for filename in filenames]],
        )
        for source, filenames in test_samples_by_source.items()
    }


def _snapshot_hash(filenames: list[str]) -> str:
    return hashlib.sha256("\n".join(filenames).encode()).hexdigest()


def _atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _config_from_payload(payload: dict[str, Any]) -> FinalTrainingConfig:
    values = dict(payload)
    values["seeds"] = tuple(values["seeds"])
    config = FinalTrainingConfig(**values)
    config.validate()
    return config


def _read_run_state(run_dir: Path) -> dict[str, Any]:
    state_path = run_dir / "run_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"resume run state does not exist: {state_path}")
    return json.loads(state_path.read_text(encoding="utf-8"))


def run_final_evaluation(
    labeled_dir: Path,
    metadata_path: Path,
    artifacts_dir: Path,
    config: FinalTrainingConfig,
    *,
    device_name: str | None = None,
    test_size: int | None = None,
    output_name: str | None = None,
    resume_dir: Path | None = None,
) -> Path:
    """Train three evaluation members, then consume the frozen test set once."""
    config.validate()
    if resume_dir is None:
        historical_samples = _historical_sample_names(artifacts_dir)
        split = build_final_split(
            labeled_dir,
            metadata_path,
            test_size=test_size,
            historical_samples=historical_samples,
        )
        run_name = output_name or datetime.now().strftime("captcha-final-eval_%Y%m%d_%H%M%S")
        output_dir = artifacts_dir / run_name
        output_dir.mkdir(parents=True, exist_ok=False)
        split_path = output_dir / "split.json"
        _atomic_json_save(split, split_path)
        run_state = {
            "resume_version": 1,
            "phase": "eval",
            "completed": False,
            "version": run_name,
            "created_at": datetime.now().astimezone().isoformat(),
            "labeled_dir": str(labeled_dir.resolve()),
            "metadata_path": str(metadata_path.resolve()),
            "artifacts_dir": str(artifacts_dir.resolve()),
            "split": split_path.name,
            "config": asdict(config),
        }
        _atomic_json_save(run_state, output_dir / "run_state.json")
    else:
        output_dir = resume_dir.resolve()
        run_state = _read_run_state(output_dir)
        if run_state.get("phase") != "eval":
            raise ValueError("resume directory is not an eval run")
        stored_config = _config_from_payload(run_state["config"])
        if asdict(stored_config) != asdict(config):
            raise ValueError("resume config does not match the original eval run")
        run_name = run_state["version"]
        split_path = output_dir / run_state["split"]
        split = json.loads(split_path.read_text(encoding="utf-8"))
        if run_state.get("completed"):
            report_path = output_dir / run_state["report"]
            if not report_path.is_file():
                raise FileNotFoundError("completed eval run is missing its report")
            _progress(f"eval_run_already_complete={report_path}")
            return report_path
        _progress(f"resuming_eval_run={output_dir}")
    train_paths = [labeled_dir / name for name in split["train_samples"]]
    validation_paths = [labeled_dir / name for name in split["validation_samples"]]
    test_paths = [labeled_dir / name for name in split["test_samples"]]
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))

    models: list[nn.Module] = []
    members: list[dict[str, Any]] = []
    for seed in config.seeds:
        model, member = _train_member(
            train_paths,
            validation_paths,
            output_dir,
            seed,
            config,
            device,
        )
        models.append(model)
        members.append(member)

    test_loader = _evaluation_loader(test_paths, config.batch_size)
    member_logits: list[torch.Tensor] = []
    test_targets: torch.Tensor | None = None
    for model, member in zip(models, members, strict=True):
        logits, targets = _collect_logits(model.to(device), test_loader, device)
        member_logits.append(logits)
        test_targets = targets
        member["test"] = _detailed_metrics(logits, targets)
        member["test_by_source"] = _metrics_by_source(
            logits,
            targets,
            split["test_samples"],
            split["test_samples_by_source"],
        )
        model.cpu()
    assert test_targets is not None
    ensemble_logits = torch.stack(member_logits).mean(dim=0)
    ensemble_test = _detailed_metrics(ensemble_logits, test_targets)
    ensemble_test_by_source = _metrics_by_source(
        ensemble_logits,
        test_targets,
        split["test_samples"],
        split["test_samples_by_source"],
    )
    ensemble = CaptchaEnsemble(models).eval()
    ensemble_path = output_dir / "ensemble.pt"
    torch.save(ensemble.state_dict(), ensemble_path)
    sample_images, _ = next(iter(test_loader))
    onnx_result = export_ensemble_onnx(
        ensemble,
        output_dir / "captcha-final-eval.onnx",
        sample_images[:8],
    )
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "version": run_name,
        "phase": "eval",
        "model_name": FINAL_MODEL_NAME,
        "member_parameter_count": parameter_count(models[0]),
        "ensemble_parameter_count": parameter_count(ensemble),
        "device": str(device),
        "config": asdict(config),
        "split": split_path.name,
        "snapshot_sha256": _snapshot_hash(
            split["train_samples"] + split["validation_samples"] + split["test_samples"]
        ),
        "members": members,
        "test_evaluation_count": 1,
        "ensemble_test": ensemble_test,
        "ensemble_test_by_source": ensemble_test_by_source,
        "release_threshold": 0.60,
        "release_passed": ensemble_test["exact_accuracy"] >= 0.60,
        "ideal_threshold": 0.65,
        "ensemble_checkpoint": ensemble_path.name,
        "onnx": onnx_result,
    }
    report_path = output_dir / "report.json"
    _atomic_json_save(report, report_path)
    run_state["completed"] = True
    run_state["completed_at"] = datetime.now().astimezone().isoformat()
    run_state["report"] = report_path.name
    _atomic_json_save(run_state, output_dir / "run_state.json")
    _progress(
        f"eval_done exact={ensemble_test['exact_accuracy']:.4f} "
        f"release_passed={report['release_passed']} report={report_path}"
    )
    return report_path


def run_final_production(
    eval_report_path: Path,
    labeled_dir: Path,
    artifacts_dir: Path,
    config: FinalTrainingConfig,
    *,
    device_name: str | None = None,
    output_name: str | None = None,
    allow_below_threshold: bool = False,
    resume_dir: Path | None = None,
) -> Path:
    """Retrain the selected recipe on every frozen sample without touching test metrics."""
    config.validate()
    eval_report = json.loads(eval_report_path.read_text(encoding="utf-8"))
    if not eval_report.get("release_passed", False) and not allow_below_threshold:
        raise ValueError(
            "evaluation did not reach the 60% ExactAcc@1 release threshold; "
            "production training is blocked"
        )
    split = json.loads((eval_report_path.parent / eval_report["split"]).read_text(encoding="utf-8"))
    all_names = sorted(split["train_samples"] + split["validation_samples"] + split["test_samples"])
    all_paths = [labeled_dir / name for name in all_names]
    if resume_dir is None:
        run_name = output_name or datetime.now().strftime("captcha-final-prod_%Y%m%d_%H%M%S")
        output_dir = artifacts_dir / run_name
        output_dir.mkdir(parents=True, exist_ok=False)
        run_state = {
            "resume_version": 1,
            "phase": "prod",
            "completed": False,
            "version": run_name,
            "created_at": datetime.now().astimezone().isoformat(),
            "labeled_dir": str(labeled_dir.resolve()),
            "artifacts_dir": str(artifacts_dir.resolve()),
            "source_eval_report": str(eval_report_path.resolve()),
            "allow_below_threshold": allow_below_threshold,
            "config": asdict(config),
        }
        _atomic_json_save(run_state, output_dir / "run_state.json")
    else:
        output_dir = resume_dir.resolve()
        run_state = _read_run_state(output_dir)
        if run_state.get("phase") != "prod":
            raise ValueError("resume directory is not a production run")
        stored_config = _config_from_payload(run_state["config"])
        if asdict(stored_config) != asdict(config):
            raise ValueError("resume config does not match the original production run")
        if Path(run_state["source_eval_report"]).resolve() != eval_report_path.resolve():
            raise ValueError("resume production run references a different eval report")
        run_name = run_state["version"]
        if run_state.get("completed"):
            report_path = output_dir / run_state["report"]
            if not report_path.is_file():
                raise FileNotFoundError("completed production run is missing its report")
            _progress(f"production_run_already_complete={report_path}")
            return report_path
        _progress(f"resuming_production_run={output_dir}")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    models: list[nn.Module] = []
    members: list[dict[str, Any]] = []
    eval_members = {int(member["seed"]): member for member in eval_report["members"]}
    for seed in config.seeds:
        selected_updates = int(eval_members[seed]["ema_best_update"])
        model, member = _train_member(
            all_paths,
            None,
            output_dir,
            seed,
            config,
            device,
            total_updates=selected_updates,
        )
        models.append(model)
        members.append(member)
    ensemble = CaptchaEnsemble(models).eval()
    ensemble_path = output_dir / "ensemble.pt"
    torch.save(ensemble.state_dict(), ensemble_path)
    sample_loader = _evaluation_loader(all_paths[:8], batch_size=8)
    sample_images, _ = next(iter(sample_loader))
    onnx_result = export_ensemble_onnx(
        ensemble,
        output_dir / "captcha.onnx",
        sample_images,
    )
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "dataset_snapshot": all_names,
        "sample_count": len(all_names),
        "snapshot_sha256": _snapshot_hash(all_names),
        "source_eval_report": str(eval_report_path),
    }
    manifest_path = output_dir / "manifest.json"
    _atomic_json_save(manifest, manifest_path)
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "version": run_name,
        "phase": "prod",
        "model_name": FINAL_MODEL_NAME,
        "member_parameter_count": parameter_count(models[0]),
        "ensemble_parameter_count": parameter_count(ensemble),
        "device": str(device),
        "config": asdict(config),
        "source_eval_report": str(eval_report_path),
        "accuracy_source": str(eval_report_path),
        "production_metrics": None,
        "production_metrics_note": (
            "Production uses the full frozen snapshot; credibility metrics remain in eval report."
        ),
        "members": members,
        "ensemble_checkpoint": ensemble_path.name,
        "manifest": manifest_path.name,
        "onnx": onnx_result,
    }
    report_path = output_dir / "report.json"
    _atomic_json_save(report, report_path)
    run_state["completed"] = True
    run_state["completed_at"] = datetime.now().astimezone().isoformat()
    run_state["report"] = report_path.name
    _atomic_json_save(run_state, output_dir / "run_state.json")
    _progress(f"prod_done samples={len(all_names)} report={report_path}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen final Position-DS pipeline.")
    parser.add_argument("--phase", choices=("eval", "prod", "all"), default="all")
    parser.add_argument("--labeled-dir", type=Path, default=Path("dataset/labeled"))
    parser.add_argument("--metadata", type=Path, default=Path("dataset/metadata.csv"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--eval-report", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--test-size", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--total-updates", type=int, default=25_000)
    parser.add_argument("--min-updates", type=int, default=15_000)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--early-stopping-patience", type=int, default=4_000)
    parser.add_argument("--hard-replay-start", type=int, default=17_500)
    parser.add_argument("--batch-size", type=int, choices=(32, 64), default=32)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--allow-below-threshold", action="store_true")
    parser.add_argument("--progress-log", type=Path)
    args = parser.parse_args()
    if args.progress_log is not None:
        os.environ["CNN_FOR_ANI_PROGRESS_LOG"] = str(args.progress_log.resolve())
    if args.resume is not None:
        resume_dir = args.resume.resolve()
        run_state = _read_run_state(resume_dir)
        config = _config_from_payload(run_state["config"])
        if run_state["phase"] == "eval":
            run_final_evaluation(
                Path(run_state["labeled_dir"]),
                Path(run_state["metadata_path"]),
                Path(run_state["artifacts_dir"]),
                config,
                device_name=args.device,
                resume_dir=resume_dir,
            )
        elif run_state["phase"] == "prod":
            run_final_production(
                Path(run_state["source_eval_report"]),
                Path(run_state["labeled_dir"]),
                Path(run_state["artifacts_dir"]),
                config,
                device_name=args.device,
                allow_below_threshold=bool(run_state.get("allow_below_threshold")),
                resume_dir=resume_dir,
            )
        else:
            parser.error(f"unsupported resume phase: {run_state['phase']!r}")
        return
    config = FinalTrainingConfig(
        total_updates=args.total_updates,
        min_updates=args.min_updates,
        validation_interval=args.validation_interval,
        early_stopping_patience=args.early_stopping_patience,
        hard_replay_start=args.hard_replay_start,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
    )
    eval_report = args.eval_report
    if args.phase in {"eval", "all"}:
        eval_report = run_final_evaluation(
            args.labeled_dir,
            args.metadata,
            args.artifacts_dir,
            config,
            device_name=args.device,
            test_size=args.test_size,
        )
    if args.phase in {"prod", "all"}:
        if eval_report is None:
            parser.error("--eval-report is required for --phase prod")
        run_final_production(
            eval_report,
            args.labeled_dir,
            args.artifacts_dir,
            config,
            device_name=args.device,
            allow_below_threshold=args.allow_below_threshold,
        )


if __name__ == "__main__":
    main()
