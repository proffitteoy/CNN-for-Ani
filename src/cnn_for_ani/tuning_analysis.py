"""Diagnostics used before changing model capacity or tuning data policy."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cnn_for_ani.dataset import LabeledCaptchaDataset, parse_label
from cnn_for_ani.error_analysis import confusion_matrix
from cnn_for_ani.metrics import calculate_metrics
from cnn_for_ani.model import build_model
from cnn_for_ani.prediction import calculate_top_k_exact_accuracy


def analyze_for_tuning(
    source_report: Path,
    labeled_dir: Path,
    batch_size: int = 32,
    error_sample_count: int = 100,
    seed: int = 20260715,
    device_name: str | None = None,
) -> Path:
    report = json.loads(source_report.read_text(encoding="utf-8"))
    split = json.loads((source_report.parent / report["split"]).read_text(encoding="utf-8"))
    train_names = split["train_samples"]
    holdout_names = split["holdout_samples"]
    holdout_dataset = LabeledCaptchaDataset(labeled_dir, filenames=holdout_names)
    loader = DataLoader(holdout_dataset, batch_size=batch_size, shuffle=False)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(report["model_name"]).to(device)
    model.load_state_dict(
        torch.load(
            source_report.parent / report["checkpoint"],
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    with torch.no_grad():
        for images, targets in loader:
            all_logits.append(model(images.to(device)).cpu())
            all_targets.append(targets)
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    probabilities = logits.softmax(dim=-1)
    confidences, predictions = probabilities.max(dim=-1)
    metrics = calculate_metrics(logits, targets)
    top_k = calculate_top_k_exact_accuracy(logits, targets, ks=(1, 2, 3, 5))
    position_accuracies = predictions.eq(targets).float().mean(dim=0)
    position_confusions = [
        confusion_matrix(targets[:, position], predictions[:, position]).tolist()
        for position in range(4)
    ]
    train_targets = torch.stack([parse_label(filename) for filename in train_names])
    position_support = {
        "train": [
            torch.bincount(train_targets[:, position], minlength=10).tolist()
            for position in range(4)
        ],
        "holdout": [
            torch.bincount(targets[:, position], minlength=10).tolist() for position in range(4)
        ],
    }
    second_confusion = torch.tensor(position_confusions[1])
    second_recall = second_confusion.diag().float() / second_confusion.sum(dim=1).clamp_min(1)

    second_error_indices = predictions[:, 1].ne(targets[:, 1]).nonzero().flatten().tolist()
    random.Random(seed).shuffle(second_error_indices)
    selected_indices = second_error_indices[: min(error_sample_count, len(second_error_indices))]
    output_dir = source_report.parent / "tuning_analysis"
    cases_dir = output_dir / "second_position_errors"
    cases_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "second_position_errors.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                "image",
                "source_filename",
                "true_label",
                "predicted_label",
                "second_true",
                "second_predicted",
                "second_confidence",
            ),
        )
        writer.writeheader()
        for case_number, sample_index in enumerate(selected_indices, start=1):
            source_path = holdout_dataset.paths[sample_index]
            true_label = "".join(map(str, targets[sample_index].tolist()))
            predicted_label = "".join(map(str, predictions[sample_index].tolist()))
            copied_name = (
                f"{case_number:03d}_true_{true_label}_pred_{predicted_label}{source_path.suffix}"
            )
            shutil.copy2(source_path, cases_dir / copied_name)
            writer.writerow(
                {
                    "image": f"second_position_errors/{copied_name}",
                    "source_filename": source_path.name,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "second_true": targets[sample_index, 1].item(),
                    "second_predicted": predictions[sample_index, 1].item(),
                    "second_confidence": confidences[sample_index, 1].item(),
                }
            )

    analysis = {
        "created_at": datetime.now().astimezone().isoformat(),
        "source_report": str(source_report),
        "sample_count": len(holdout_dataset),
        "metrics": {
            "char_accuracy": metrics.char_accuracy,
            "exact_accuracy": metrics.exact_accuracy,
            "hamming_error": metrics.hamming_error,
        },
        "top_k_exact_accuracy": {str(k): value for k, value in top_k.items()},
        "position_accuracies": position_accuracies.tolist(),
        "position_confusion_matrices": position_confusions,
        "position_digit_support": position_support,
        "second_position_digit_recall": second_recall.tolist(),
        "second_position_error_count": len(second_error_indices),
        "exported_second_position_errors": len(selected_indices),
        "second_position_errors_csv": str(csv_path),
    }
    analysis_path = output_dir / "report.json"
    analysis_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        " ".join(f"exact@{k}={value:.4f}" for k, value in top_k.items())
        + f" report={analysis_path}",
        flush=True,
    )
    return analysis_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a final model before tuning.")
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--labeled-dir", type=Path, default=Path("dataset/labeled"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--error-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    analyze_for_tuning(
        source_report=args.source_report,
        labeled_dir=args.labeled_dir,
        batch_size=args.batch_size,
        error_sample_count=args.error_samples,
        seed=args.seed,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
