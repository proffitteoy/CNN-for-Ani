"""Evaluate frozen cross-validation checkpoints on later labeled samples."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cnn_for_ani.dataset import LabeledCaptchaDataset
from cnn_for_ani.losses import mean_head_cross_entropy
from cnn_for_ani.metrics import calculate_metrics
from cnn_for_ani.model import MODEL_NAMES, build_model


def evaluate_temporal_holdout(
    source_report: Path,
    labeled_dir: Path,
    holdout_size: int,
    model_name: str,
    batch_size: int = 32,
    device_name: str | None = None,
) -> Path:
    reference = json.loads(source_report.read_text(encoding="utf-8"))
    snapshot_names = set(reference["dataset_snapshot"])
    candidates = sorted(
        (
            path
            for path in labeled_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
            and path.name not in snapshot_names
        ),
        key=lambda path: (path.stat().st_ctime_ns, path.name),
    )
    if len(candidates) < holdout_size:
        raise ValueError(f"need {holdout_size} post-snapshot samples, found {len(candidates)}")
    selected = candidates[:holdout_size]
    dataset = LabeledCaptchaDataset(labeled_dir, filenames=[path.name for path in selected])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))

    checkpoint_results: list[dict[str, object]] = []
    ensemble_logits: list[torch.Tensor] | None = None
    all_targets: list[torch.Tensor] = []
    for fold_index in range(1, int(reference["fold_count"]) + 1):
        checkpoint_path = source_report.parent / f"fold_{fold_index}.pt"
        model = build_model(model_name).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        model.eval()
        fold_logits: list[torch.Tensor] = []
        fold_targets: list[torch.Tensor] = []
        total_loss = 0.0
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(device)
                targets = targets.to(device)
                logits = model(images)
                total_loss += mean_head_cross_entropy(logits, targets).item() * len(images)
                fold_logits.append(logits.cpu())
                fold_targets.append(targets.cpu())
        logits = torch.cat(fold_logits)
        targets = torch.cat(fold_targets)
        metrics = calculate_metrics(logits, targets)
        checkpoint_results.append(
            {
                "fold": fold_index,
                "checkpoint": checkpoint_path.name,
                "loss": total_loss / len(dataset),
                **asdict(metrics),
            }
        )
        if ensemble_logits is None:
            ensemble_logits = [logits]
            all_targets = fold_targets
        else:
            ensemble_logits.append(logits)

    assert ensemble_logits is not None
    ensemble_metrics = calculate_metrics(
        torch.stack(ensemble_logits).mean(dim=0),
        torch.cat(all_targets),
    )
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "source_report": str(source_report),
        "source_snapshot_size": len(snapshot_names),
        "post_snapshot_candidates": len(candidates),
        "selection": "earliest creation time, then filename",
        "holdout_size": holdout_size,
        "holdout_samples": [path.name for path in selected],
        "model_name": model_name,
        "checkpoints": checkpoint_results,
        "ensemble": asdict(ensemble_metrics),
    }
    report_path = source_report.parent / f"temporal_holdout_{holdout_size}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"holdout={holdout_size} ensemble_char={ensemble_metrics.char_accuracy:.4f} "
        f"ensemble_exact={ensemble_metrics.exact_accuracy:.4f} report={report_path}",
        flush=True,
    )
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate later samples as a temporal holdout.")
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--labeled-dir", type=Path, default=Path("dataset/labeled"))
    parser.add_argument("--holdout-size", type=int, default=178)
    parser.add_argument("--model", choices=MODEL_NAMES, default="position")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    evaluate_temporal_holdout(
        source_report=args.source_report,
        labeled_dir=args.labeled_dir,
        holdout_size=args.holdout_size,
        model_name=args.model,
        batch_size=args.batch_size,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
