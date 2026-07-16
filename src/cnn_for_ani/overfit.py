"""Real-data overfit diagnostic for the complete training path."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from cnn_for_ani.dataset import LabeledCaptchaDataset
from cnn_for_ani.losses import mean_head_cross_entropy
from cnn_for_ani.metrics import calculate_metrics
from cnn_for_ani.model import CaptchaCNN
from cnn_for_ani.training import DEFAULT_SEED


def run_overfit_check(
    labeled_dir: Path,
    artifacts_dir: Path,
    sample_count: int = 32,
    epochs: int = 1000,
    learning_rate: float = 3e-3,
    seed: int = DEFAULT_SEED,
    device_name: str | None = None,
) -> Path:
    dataset = LabeledCaptchaDataset(labeled_dir)
    if len(dataset) < sample_count:
        raise ValueError(f"need at least {sample_count} labeled samples, found {len(dataset)}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    selected_indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[
        :sample_count
    ].tolist()
    subset = Subset(dataset, selected_indices)
    loader = DataLoader(subset, batch_size=sample_count, shuffle=False)
    images, targets = next(iter(loader))
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    images = images.to(device)
    targets = targets.to(device)

    model = CaptchaCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history: list[dict[str, object]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        logits = model(images)
        loss = mean_head_cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(images)
            metrics = calculate_metrics(logits, targets)
        result = {"epoch": epoch, "loss": loss.item(), **asdict(metrics)}
        history.append(result)
        if epoch == 1 or epoch % 25 == 0 or metrics.exact_accuracy == 1.0:
            print(
                f"epoch={epoch:04d} loss={loss.item():.6f} "
                f"char={metrics.char_accuracy:.4f} exact={metrics.exact_accuracy:.4f} "
                f"hamming={metrics.hamming_error:.4f}",
                flush=True,
            )
        if metrics.exact_accuracy == 1.0:
            break

    run_name = datetime.now().strftime("overfit_%Y%m%d_%H%M%S")
    output_dir = artifacts_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_dir / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "dataset_size_at_start": len(dataset),
        "selected_samples": [dataset.paths[index].name for index in selected_indices],
        "sample_count": sample_count,
        "device": str(device),
        "config": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "seed": seed,
            "augmentation": False,
        },
        "reached_full_exact_accuracy": history[-1]["exact_accuracy"] == 1.0,
        "final": history[-1],
        "history": history,
        "checkpoint": checkpoint_path.name,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report={report_path}", flush=True)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit a fixed real-data sample.")
    parser.add_argument("--labeled-dir", type=Path, default=Path("dataset/labeled"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    run_overfit_check(
        labeled_dir=args.labeled_dir,
        artifacts_dir=args.artifacts_dir,
        sample_count=args.samples,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
