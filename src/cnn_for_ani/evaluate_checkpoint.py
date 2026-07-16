"""Evaluate a trained checkpoint on a named frozen split."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cnn_for_ani.dataset import LabeledCaptchaDataset
from cnn_for_ani.metrics import calculate_metrics
from cnn_for_ani.model import build_model
from cnn_for_ani.prediction import calculate_top_k_exact_accuracy


def evaluate_checkpoint(
    source_report: Path,
    split_path: Path,
    labeled_dir: Path,
    output_path: Path | None = None,
    batch_size: int = 32,
    device_name: str | None = None,
) -> Path:
    source = json.loads(source_report.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    dataset = LabeledCaptchaDataset(labeled_dir, filenames=split["holdout_samples"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(source["model_name"]).to(device)
    model.load_state_dict(
        torch.load(
            source_report.parent / source["checkpoint"],
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
    metrics = calculate_metrics(logits, targets)
    predictions = logits.argmax(dim=-1)
    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "source_report": str(source_report),
        "split": str(split_path),
        "model_name": source["model_name"],
        "sample_count": len(dataset),
        "metrics": asdict(metrics),
        "top_k_exact_accuracy": {
            str(k): value
            for k, value in calculate_top_k_exact_accuracy(logits, targets, ks=(1, 2, 3, 5)).items()
        },
        "position_accuracies": predictions.eq(targets).float().mean(dim=0).tolist(),
    }
    output_path = output_path or split_path.parent / f"baseline_{source['model_name']}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"model={source['model_name']} samples={len(dataset)} "
        f"char={metrics.char_accuracy:.4f} exact={metrics.exact_accuracy:.4f} "
        f"report={output_path}",
        flush=True,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on a frozen split.")
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--labeled-dir", type=Path, default=Path("dataset/labeled"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    evaluate_checkpoint(
        source_report=args.source_report,
        split_path=args.split,
        labeled_dir=args.labeled_dir,
        output_path=args.output,
        batch_size=args.batch_size,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
