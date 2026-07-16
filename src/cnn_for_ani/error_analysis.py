"""Out-of-fold error analysis for a completed cross-validation run."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from cnn_for_ani.dataset import LabeledCaptchaDataset
from cnn_for_ani.metrics import calculate_metrics
from cnn_for_ani.model import MODEL_NAMES, build_model


def confusion_matrix(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    class_count: int = 10,
) -> torch.Tensor:
    """Count rows as true classes and columns as predicted classes."""
    if targets.shape != predictions.shape:
        raise ValueError("targets and predictions must have the same shape")
    matrix = torch.zeros(class_count, class_count, dtype=torch.long)
    for target, prediction in zip(targets.flatten(), predictions.flatten(), strict=True):
        matrix[target.item(), prediction.item()] += 1
    return matrix


def analyze_errors(
    source_report: Path,
    labeled_dir: Path,
    model_name: str,
    error_sample_count: int = 100,
    batch_size: int = 32,
    seed: int = 20260715,
    device_name: str | None = None,
) -> Path:
    reference = json.loads(source_report.read_text(encoding="utf-8"))
    dataset = LabeledCaptchaDataset(labeled_dir, filenames=reference["dataset_snapshot"])
    assignments = json.loads((source_report.parent / "folds.json").read_text(encoding="utf-8"))
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    all_logits = torch.empty(len(dataset), 4, 10)
    all_targets = torch.empty(len(dataset), 4, dtype=torch.long)

    for fold_index in range(1, int(reference["fold_count"]) + 1):
        validation_indices = [
            index
            for index, path in enumerate(dataset.paths)
            if int(assignments[path.name]) == fold_index
        ]
        loader = DataLoader(
            Subset(dataset, validation_indices),
            batch_size=batch_size,
            shuffle=False,
        )
        model = build_model(model_name).to(device)
        model.load_state_dict(
            torch.load(
                source_report.parent / f"fold_{fold_index}.pt",
                map_location=device,
                weights_only=True,
            )
        )
        model.eval()
        fold_logits: list[torch.Tensor] = []
        fold_targets: list[torch.Tensor] = []
        with torch.no_grad():
            for images, targets in loader:
                fold_logits.append(model(images.to(device)).cpu())
                fold_targets.append(targets)
        all_logits[validation_indices] = torch.cat(fold_logits)
        all_targets[validation_indices] = torch.cat(fold_targets)

    probabilities = all_logits.softmax(dim=-1)
    confidences, predictions = probabilities.max(dim=-1)
    metrics = calculate_metrics(all_logits, all_targets)
    position_accuracies = predictions.eq(all_targets).float().mean(dim=0)
    aggregate_confusion = confusion_matrix(all_targets, predictions)
    position_confusions = [
        confusion_matrix(all_targets[:, position], predictions[:, position]).tolist()
        for position in range(4)
    ]
    digit_support = aggregate_confusion.sum(dim=1)
    digit_accuracies = aggregate_confusion.diag().float() / digit_support.clamp_min(1)

    incorrect_indices = predictions.ne(all_targets).any(dim=1).nonzero().flatten().tolist()
    random.Random(seed).shuffle(incorrect_indices)
    selected_indices = incorrect_indices[: min(error_sample_count, len(incorrect_indices))]
    output_dir = source_report.parent / "error_analysis"
    output_dir.mkdir(exist_ok=True)
    cases_dir = output_dir / "cases"
    cases_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "errors.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                "image",
                "source_filename",
                "true_label",
                "predicted_label",
                "confidence_1",
                "confidence_2",
                "confidence_3",
                "confidence_4",
            ),
        )
        writer.writeheader()
        for case_number, sample_index in enumerate(selected_indices, start=1):
            source_path = dataset.paths[sample_index]
            true_label = "".join(map(str, all_targets[sample_index].tolist()))
            predicted_label = "".join(map(str, predictions[sample_index].tolist()))
            copied_name = (
                f"{case_number:03d}_true_{true_label}_pred_{predicted_label}{source_path.suffix}"
            )
            shutil.copy2(source_path, cases_dir / copied_name)
            confidence_values = confidences[sample_index].tolist()
            writer.writerow(
                {
                    "image": f"cases/{copied_name}",
                    "source_filename": source_path.name,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "confidence_1": confidence_values[0],
                    "confidence_2": confidence_values[1],
                    "confidence_3": confidence_values[2],
                    "confidence_4": confidence_values[3],
                }
            )

    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "source_report": str(source_report),
        "model_name": model_name,
        "sample_count": len(dataset),
        "metrics": {
            "char_accuracy": metrics.char_accuracy,
            "exact_accuracy": metrics.exact_accuracy,
            "hamming_error": metrics.hamming_error,
        },
        "position_accuracies": position_accuracies.tolist(),
        "aggregate_confusion_matrix": aggregate_confusion.tolist(),
        "position_confusion_matrices": position_confusions,
        "digit_support": digit_support.tolist(),
        "digit_accuracies": digit_accuracies.tolist(),
        "incorrect_sample_count": len(incorrect_indices),
        "exported_error_count": len(selected_indices),
        "errors_csv": str(csv_path),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"oof_char={metrics.char_accuracy:.4f} oof_exact={metrics.exact_accuracy:.4f} "
        f"errors={len(incorrect_indices)} report={report_path}",
        flush=True,
    )
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze out-of-fold captcha errors.")
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--labeled-dir", type=Path, default=Path("dataset/labeled"))
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--error-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    analyze_errors(
        source_report=args.source_report,
        labeled_dir=args.labeled_dir,
        model_name=args.model,
        error_sample_count=args.error_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
