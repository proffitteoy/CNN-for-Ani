"""Reproducible five-fold training for the captcha CNN baseline."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from cnn_for_ani.augmentation import light_augment_image
from cnn_for_ani.dataset import LabeledCaptchaDataset, parse_label
from cnn_for_ani.losses import mean_head_cross_entropy
from cnn_for_ani.metrics import CaptchaMetrics, calculate_metrics
from cnn_for_ani.model import (
    CAPTCHA_LENGTH,
    MODEL_NAMES,
    NUM_CLASSES,
    build_model,
    parameter_count,
)
from cnn_for_ani.preprocessing import preprocess_image

DEFAULT_SEED = 20260715
AUGMENTATION_NAMES = ("light", "none")


class _LightAugmentedSubset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, dataset: LabeledCaptchaDataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.dataset.paths[self.indices[index]]
        with Image.open(path) as image:
            tensor = preprocess_image(light_augment_image(image))
        return tensor, parse_label(path)


def make_stratified_folds(
    labels: torch.Tensor,
    fold_count: int = 5,
    seed: int = DEFAULT_SEED,
) -> list[list[int]]:
    """Split indices while approximately balancing each position's digit margins."""
    if labels.ndim != 2 or labels.shape[1] != CAPTCHA_LENGTH:
        raise ValueError(f"expected labels shape [N, 4], got {tuple(labels.shape)}")
    if fold_count < 2 or fold_count > len(labels):
        raise ValueError("fold_count must be between 2 and the number of samples")
    if labels.numel() and (labels.min().item() < 0 or labels.max().item() >= NUM_CLASSES):
        raise ValueError("labels must contain digits from 0 to 9")

    generator = torch.Generator().manual_seed(seed)
    tie_breakers = torch.rand(len(labels), generator=generator)
    global_counts = torch.zeros(CAPTCHA_LENGTH, NUM_CLASSES, dtype=torch.long)
    for position in range(CAPTCHA_LENGTH):
        global_counts[position] = torch.bincount(labels[:, position], minlength=NUM_CLASSES)

    rarity = torch.zeros(len(labels), dtype=torch.float64)
    for index, label in enumerate(labels):
        rarity[index] = sum(
            1.0 / global_counts[position, digit].item()
            for position, digit in enumerate(label.tolist())
        )
    ordered_indices = sorted(
        range(len(labels)),
        key=lambda index: (-rarity[index].item(), tie_breakers[index].item()),
    )

    capacities = [len(labels) // fold_count] * fold_count
    for fold_index in range(len(labels) % fold_count):
        capacities[fold_index] += 1
    folds: list[list[int]] = [[] for _ in range(fold_count)]
    fold_counts = torch.zeros(fold_count, CAPTCHA_LENGTH, NUM_CLASSES, dtype=torch.long)

    for sample_index in ordered_indices:
        label = labels[sample_index]
        sample_digits = label.tolist()
        candidates = [
            fold_index
            for fold_index in range(fold_count)
            if len(folds[fold_index]) < capacities[fold_index]
        ]

        def score(fold_index: int, digits: list[int] = sample_digits) -> tuple[float, float]:
            digit_score = sum(
                fold_counts[fold_index, position, digit].item()
                / global_counts[position, digit].item()
                for position, digit in enumerate(digits)
            )
            size_score = len(folds[fold_index]) / capacities[fold_index]
            return digit_score, size_score

        selected_fold = min(candidates, key=lambda fold_index: (*score(fold_index), fold_index))
        folds[selected_fold].append(sample_index)
        for position, digit in enumerate(sample_digits):
            fold_counts[selected_fold, position, digit] += 1

    return [sorted(fold) for fold in folds]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[float, CaptchaMetrics]:
    model.eval()
    total_loss = 0.0
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = mean_head_cross_entropy(logits, targets)
            total_loss += loss.item() * len(images)
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

    metrics = calculate_metrics(torch.cat(all_logits), torch.cat(all_targets))
    return total_loss / len(loader.dataset), metrics


def should_stop_early(
    epoch: int,
    min_epochs: int,
    epochs_without_improvement: int,
    patience: int,
) -> bool:
    """Return whether validation-loss early stopping is currently allowed to fire."""
    return epoch >= min_epochs and epochs_without_improvement >= patience


def _train_fold(
    dataset: LabeledCaptchaDataset,
    train_indices: list[int],
    validation_indices: list[int],
    output_dir: Path,
    fold_index: int,
    device: torch.device,
    epochs: int,
    min_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    model_name: str,
    augmentation_name: str,
) -> dict[str, object]:
    _set_seed(seed + fold_index)
    model = build_model(model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.3,
        patience=10,
        min_lr=1e-5,
    )
    train_dataset: Dataset[tuple[torch.Tensor, torch.Tensor]]
    if augmentation_name == "light":
        train_dataset = _LightAugmentedSubset(dataset, train_indices)
    else:
        train_dataset = Subset(dataset, train_indices)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + fold_index),
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices),
        batch_size=batch_size,
        shuffle=False,
    )
    evaluation_train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=batch_size,
        shuffle=False,
    )

    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []
    checkpoint_path = output_dir / f"fold_{fold_index}.pt"
    final_checkpoint_path = output_dir / f"fold_{fold_index}_final.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = mean_head_cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss, train_metrics = _evaluate(model, evaluation_train_loader, device)
        validation_loss, validation_metrics = _evaluate(model, validation_loader, device)
        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_char_accuracy": train_metrics.char_accuracy,
            "validation_char_accuracy": validation_metrics.char_accuracy,
            "train_exact_accuracy": train_metrics.exact_accuracy,
            "validation_exact_accuracy": validation_metrics.exact_accuracy,
            "train_hamming_error": train_metrics.hamming_error,
            "validation_hamming_error": validation_metrics.hamming_error,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_result)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        elif epoch >= min_epochs:
            epochs_without_improvement += 1

        scheduler.step(validation_loss)

        stopping = should_stop_early(epoch, min_epochs, epochs_without_improvement, patience)
        if epoch == 1 or epoch % 10 == 0 or stopping:
            print(
                f"fold={fold_index} epoch={epoch:03d} "
                f"train_loss={train_loss:.4f} val_loss={validation_loss:.4f} "
                f"train_char={train_metrics.char_accuracy:.4f} "
                f"val_char={validation_metrics.char_accuracy:.4f} "
                f"train_exact={train_metrics.exact_accuracy:.4f} "
                f"val_exact={validation_metrics.exact_accuracy:.4f}",
                flush=True,
            )
        if stopping:
            break

    torch.save(model.state_dict(), final_checkpoint_path)
    final_result = history[-1]
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    train_loss_at_best, train_metrics_at_best = _evaluate(model, evaluation_train_loader, device)
    validation_loss_at_best, validation_metrics_at_best = _evaluate(
        model, validation_loader, device
    )
    return {
        "fold": fold_index,
        "train_size": len(train_indices),
        "validation_size": len(validation_indices),
        "best_epoch": best_epoch,
        "checkpoint": checkpoint_path.name,
        "final_checkpoint": final_checkpoint_path.name,
        "final": final_result,
        "train_at_best": {
            "loss": train_loss_at_best,
            **asdict(train_metrics_at_best),
        },
        "validation_at_best": {
            "loss": validation_loss_at_best,
            **asdict(validation_metrics_at_best),
        },
        "history": history,
    }


def _load_snapshot(
    labeled_dir: Path,
    snapshot_report: Path,
) -> tuple[LabeledCaptchaDataset, list[list[int]]]:
    reference = json.loads(snapshot_report.read_text(encoding="utf-8"))
    filenames = reference["dataset_snapshot"]
    dataset = LabeledCaptchaDataset(labeled_dir, filenames=filenames)
    assignments_path = snapshot_report.parent / "folds.json"
    assignments = json.loads(assignments_path.read_text(encoding="utf-8"))
    fold_count = int(reference["fold_count"])
    folds: list[list[int]] = [[] for _ in range(fold_count)]
    for sample_index, path in enumerate(dataset.paths):
        try:
            fold_index = int(assignments[path.name]) - 1
        except KeyError as error:
            raise ValueError(f"snapshot fold is missing for {path.name}") from error
        if not 0 <= fold_index < fold_count:
            raise ValueError(f"invalid snapshot fold for {path.name}: {fold_index + 1}")
        folds[fold_index].append(sample_index)
    if any(not fold for fold in folds):
        raise ValueError("snapshot contains an empty fold")
    return dataset, folds


def run_cross_validation(
    labeled_dir: Path,
    artifacts_dir: Path,
    fold_count: int = 5,
    epochs: int = 300,
    min_epochs: int = 100,
    patience: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    seed: int = DEFAULT_SEED,
    device_name: str | None = None,
    model_name: str = "position",
    snapshot_report: Path | None = None,
    augmentation_name: str = "none",
) -> Path:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"model_name must be one of {MODEL_NAMES}")
    if augmentation_name not in AUGMENTATION_NAMES:
        raise ValueError(f"augmentation_name must be one of {AUGMENTATION_NAMES}")
    if snapshot_report is None:
        dataset = LabeledCaptchaDataset(labeled_dir)
        labels = torch.stack([parse_label(path) for path in dataset.paths])
        folds = make_stratified_folds(labels, fold_count=fold_count, seed=seed)
    else:
        dataset, folds = _load_snapshot(labeled_dir, snapshot_report)
        if len(folds) != fold_count:
            raise ValueError(f"snapshot contains {len(folds)} folds but fold_count is {fold_count}")
    if len(dataset) < fold_count:
        raise ValueError(f"need at least {fold_count} labeled samples, found {len(dataset)}")
    if not 1 <= min_epochs <= epochs:
        raise ValueError("min_epochs must be between 1 and epochs")
    if patience < 1:
        raise ValueError("patience must be positive")

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_name = datetime.now().strftime(f"cv_{model_name}_{augmentation_name}_%Y%m%d_%H%M%S")
    output_dir = artifacts_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=False)

    fold_assignments = {
        dataset.paths[sample_index].name: fold_index
        for fold_index, fold in enumerate(folds, start=1)
        for sample_index in fold
    }
    (output_dir / "folds.json").write_text(
        json.dumps(fold_assignments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fold_results = []
    all_indices = set(range(len(dataset)))
    for fold_index, validation_indices in enumerate(folds, start=1):
        train_indices = sorted(all_indices - set(validation_indices))
        fold_results.append(
            _train_fold(
                dataset=dataset,
                train_indices=train_indices,
                validation_indices=validation_indices,
                output_dir=output_dir,
                fold_index=fold_index,
                device=device,
                epochs=epochs,
                min_epochs=min_epochs,
                patience=patience,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=seed,
                model_name=model_name,
                augmentation_name=augmentation_name,
            )
        )

    validation_exact_accuracies = np.array(
        [result["validation_at_best"]["exact_accuracy"] for result in fold_results],
        dtype=np.float64,
    )
    validation_char_accuracies = np.array(
        [result["validation_at_best"]["char_accuracy"] for result in fold_results],
        dtype=np.float64,
    )
    train_char_accuracies = np.array(
        [result["train_at_best"]["char_accuracy"] for result in fold_results],
        dtype=np.float64,
    )
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "dataset_snapshot": [path.name for path in dataset.paths],
        "sample_count": len(dataset),
        "fold_count": fold_count,
        "split_strategy": "position-wise approximate stratification",
        "group_isolation": False,
        "group_isolation_note": (
            "The current labeled snapshot belongs to one download batch, so strict Group K-Fold "
            "is not possible."
        ),
        "device": str(device),
        "model_name": model_name,
        "parameter_count": parameter_count(build_model(model_name)),
        "snapshot_report": str(snapshot_report) if snapshot_report else None,
        "config": {
            "epochs": epochs,
            "min_epochs": min_epochs,
            "patience": patience,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "augmentation": augmentation_name,
            "early_stopping_metric": "validation_loss",
            "checkpoint_metric": "validation_loss",
            "scheduler": {
                "name": "ReduceLROnPlateau",
                "factor": 0.3,
                "patience": 10,
                "min_lr": 1e-5,
            },
        },
        "folds": fold_results,
        "train_char_accuracy_mean_at_best": train_char_accuracies.mean().item(),
        "validation_char_accuracy_mean": validation_char_accuracies.mean().item(),
        "validation_char_accuracy_std": validation_char_accuracies.std(ddof=0).item(),
        "validation_exact_accuracy_mean": validation_exact_accuracies.mean().item(),
        "validation_exact_accuracy_std": validation_exact_accuracies.std(ddof=0).item(),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"done samples={len(dataset)} val_char={report['validation_char_accuracy_mean']:.4f} "
        f"val_exact={report['validation_exact_accuracy_mean']:.4f} report={report_path}",
        flush=True,
    )
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the captcha CNN with five-fold CV.")
    parser.add_argument("--labeled-dir", type=Path, default=Path("dataset/labeled"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--min-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--model", choices=MODEL_NAMES, default="position")
    parser.add_argument("--snapshot-report", type=Path)
    parser.add_argument("--augmentation", choices=AUGMENTATION_NAMES, default="none")
    args = parser.parse_args()
    run_cross_validation(
        labeled_dir=args.labeled_dir,
        artifacts_dir=args.artifacts_dir,
        fold_count=args.folds,
        epochs=args.epochs,
        min_epochs=args.min_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device_name=args.device,
        model_name=args.model,
        snapshot_report=args.snapshot_report,
        augmentation_name=args.augmentation,
    )


if __name__ == "__main__":
    main()
