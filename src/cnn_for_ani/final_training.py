"""Train the selected model and evaluate one frozen temporal holdout."""

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
from torch.utils.data import DataLoader, Dataset

from cnn_for_ani.augmentation import light_augment_image
from cnn_for_ani.dataset import LabeledCaptchaDataset, parse_label
from cnn_for_ani.error_analysis import confusion_matrix
from cnn_for_ani.losses import mean_head_cross_entropy
from cnn_for_ani.metrics import CaptchaMetrics, calculate_metrics
from cnn_for_ani.model import MODEL_NAMES, build_model, parameter_count
from cnn_for_ani.preprocessing import preprocess_image


class LightTrainingDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = preprocess_image(light_augment_image(image))
        return tensor, parse_label(path)


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[float, CaptchaMetrics, torch.Tensor, torch.Tensor]:
    model.eval()
    total_loss = 0.0
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            total_loss += mean_head_cross_entropy(logits, targets).item() * len(images)
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    return total_loss / len(loader.dataset), calculate_metrics(logits, targets), logits, targets


def train_final_model(
    labeled_dir: Path,
    artifacts_dir: Path,
    model_name: str = "position",
    holdout_size: int = 250,
    epochs: int = 300,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    seed: int = 20260715,
    device_name: str | None = None,
    frozen_split_path: Path | None = None,
) -> Path:
    if frozen_split_path is None:
        paths = sorted(
            (
                path
                for path in labeled_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
            ),
            key=lambda path: (path.stat().st_ctime_ns, path.name),
        )
        if len(paths) <= holdout_size:
            raise ValueError("holdout_size must be smaller than the labeled dataset")
        train_paths = paths[:-holdout_size]
        holdout_paths = paths[-holdout_size:]
        selection = "newest creation time, then filename"
    else:
        frozen_split = json.loads(frozen_split_path.read_text(encoding="utf-8"))
        train_paths = [labeled_dir / filename for filename in frozen_split["train_samples"]]
        holdout_paths = [labeled_dir / filename for filename in frozen_split["holdout_samples"]]
        selection = frozen_split["selection"]
    run_name = datetime.now().strftime(f"final_{model_name}_%Y%m%d_%H%M%S")
    output_dir = artifacts_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    split = {
        "created_at": datetime.now().astimezone().isoformat(),
        "selection": selection,
        "source_split": str(frozen_split_path) if frozen_split_path else None,
        "train_samples": [path.name for path in train_paths],
        "holdout_samples": [path.name for path in holdout_paths],
    }
    saved_split_path = output_dir / "split.json"
    saved_split_path.write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(
        LightTrainingDataset(train_paths),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            loss = mean_head_cross_entropy(model(images), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(images)
        epoch_loss = total_loss / len(train_loader.dataset)
        history.append(
            {
                "epoch": epoch,
                "train_augmented_loss": epoch_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch={epoch:03d} train_augmented_loss={epoch_loss:.4f}", flush=True)

    checkpoint_path = output_dir / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)
    train_dataset = LabeledCaptchaDataset(
        labeled_dir, filenames=[path.name for path in train_paths]
    )
    holdout_dataset = LabeledCaptchaDataset(
        labeled_dir, filenames=[path.name for path in holdout_paths]
    )
    train_evaluation = _evaluate(
        model,
        DataLoader(train_dataset, batch_size=batch_size, shuffle=False),
        device,
    )
    holdout_evaluation = _evaluate(
        model,
        DataLoader(holdout_dataset, batch_size=batch_size, shuffle=False),
        device,
    )
    train_loss, train_metrics, _, _ = train_evaluation
    holdout_loss, holdout_metrics, holdout_logits, holdout_targets = holdout_evaluation
    holdout_predictions = holdout_logits.argmax(dim=-1)
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "model_name": model_name,
        "parameter_count": parameter_count(model),
        "device": str(device),
        "split": saved_split_path.name,
        "train_size": len(train_dataset),
        "holdout_size": len(holdout_dataset),
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": "AdamW",
            "augmentation": "light",
            "seed": seed,
        },
        "train": {"loss": train_loss, **asdict(train_metrics)},
        "holdout": {
            "loss": holdout_loss,
            **asdict(holdout_metrics),
            "position_accuracies": holdout_predictions.eq(holdout_targets)
            .float()
            .mean(dim=0)
            .tolist(),
            "confusion_matrix": confusion_matrix(holdout_targets, holdout_predictions).tolist(),
        },
        "history": history,
        "checkpoint": checkpoint_path.name,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"done train={len(train_dataset)} holdout={len(holdout_dataset)} "
        f"holdout_char={holdout_metrics.char_accuracy:.4f} "
        f"holdout_exact={holdout_metrics.exact_accuracy:.4f} report={report_path}",
        flush=True,
    )
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the selected final captcha model.")
    parser.add_argument("--labeled-dir", type=Path, default=Path("dataset/labeled"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--model", choices=MODEL_NAMES, default="position")
    parser.add_argument("--holdout-size", type=int, default=250)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--split", type=Path)
    args = parser.parse_args()
    train_final_model(
        labeled_dir=args.labeled_dir,
        artifacts_dir=args.artifacts_dir,
        model_name=args.model,
        holdout_size=args.holdout_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device_name=args.device,
        frozen_split_path=args.split,
    )


if __name__ == "__main__":
    main()
