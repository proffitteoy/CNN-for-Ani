"""Train and export the frozen captcha-v1.0 three-seed ensemble."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader

from cnn_for_ani.dataset import LabeledCaptchaDataset
from cnn_for_ani.final_training import LightTrainingDataset
from cnn_for_ani.losses import mean_head_cross_entropy
from cnn_for_ani.metrics import calculate_metrics
from cnn_for_ani.model import CaptchaEnsemble, build_model, parameter_count

ENSEMBLE_SEEDS = (42, 3407, 20260716)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    with torch.no_grad():
        for images, targets in loader:
            all_logits.append(model(images.to(device)).cpu())
            all_targets.append(targets)
    return torch.cat(all_logits), torch.cat(all_targets)


def export_ensemble_onnx(
    ensemble: CaptchaEnsemble,
    output_path: Path,
    sample_images: torch.Tensor,
) -> dict[str, object]:
    ensemble = ensemble.cpu().eval()
    example = torch.zeros(1, 1, 32, 96, dtype=torch.float32)
    torch.onnx.export(
        ensemble,
        example,
        output_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    onnx_model = onnx.load(output_path)
    output_dimensions = onnx_model.graph.output[0].type.tensor_type.shape.dim
    for dimension, value in zip(output_dimensions[1:], (4, 10), strict=True):
        dimension.ClearField("dim_param")
        dimension.dim_value = value
    onnx.save(onnx_model, output_path)
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        torch_logits = ensemble(sample_images).numpy()
    onnx_logits = session.run(["logits"], {"input": sample_images.numpy()})[0]
    absolute_error = np.abs(torch_logits - onnx_logits)
    max_absolute_error = float(absolute_error.max())
    if max_absolute_error >= 1e-4:
        raise RuntimeError(f"ensemble ONNX logits mismatch: {max_absolute_error}")
    return {
        "onnx_path": output_path.name,
        "onnx_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "onnx_size_bytes": output_path.stat().st_size,
        "opset_version": 17,
        "input_contract": ["batch", 1, 32, 96],
        "output_contract": ["batch", 4, 10],
        "verification_batch_size": len(sample_images),
        "max_absolute_error": max_absolute_error,
        "mean_absolute_error": float(absolute_error.mean()),
        "onnx_checker_passed": True,
        "onnxruntime_provider": "CPUExecutionProvider",
    }


def reexport_ensemble(source_report: Path, labeled_dir: Path) -> Path:
    report = json.loads(source_report.read_text(encoding="utf-8"))
    models = []
    for member in report["members"]:
        model = build_model(report["model_name"])
        model.load_state_dict(
            torch.load(
                source_report.parent / member["checkpoint"],
                map_location="cpu",
                weights_only=True,
            )
        )
        models.append(model)
    ensemble = CaptchaEnsemble(models).eval()
    manifest = json.loads((source_report.parent / report["manifest"]).read_text(encoding="utf-8"))
    sample_dataset = LabeledCaptchaDataset(
        labeled_dir,
        filenames=manifest["dataset_snapshot"][:8],
    )
    sample_images, _ = next(iter(DataLoader(sample_dataset, batch_size=8, shuffle=False)))
    report["onnx"] = export_ensemble_onnx(
        ensemble,
        source_report.parent / "captcha.onnx",
        sample_images,
    )
    source_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"reexported={source_report.parent / 'captcha.onnx'} "
        f"onnx_error={report['onnx']['max_absolute_error']:.8g}",
        flush=True,
    )
    return source_report.parent / "captcha.onnx"


def train_ensemble(
    labeled_dir: Path,
    artifacts_dir: Path,
    output_name: str = "captcha-v1.0",
    model_name: str = "wide_position",
    reference_epochs: int = 300,
    reference_sample_count: int = 2000,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    device_name: str | None = None,
) -> Path:
    dataset = LabeledCaptchaDataset(labeled_dir)
    sample_count = len(dataset)
    epochs = round(reference_epochs * reference_sample_count / sample_count)
    if epochs < 1:
        raise ValueError("adjusted epoch count must be positive")
    output_dir = artifacts_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "version": output_name,
        "dataset_snapshot": [path.name for path in dataset.paths],
        "sample_count": sample_count,
        "reference_epochs": reference_epochs,
        "reference_sample_count": reference_sample_count,
        "adjusted_epochs": epochs,
        "reference_optimizer_updates": reference_epochs
        * math.ceil(reference_sample_count / batch_size),
        "final_optimizer_updates": epochs * math.ceil(sample_count / batch_size),
        "seeds": list(ENSEMBLE_SEEDS),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    models: list[torch.nn.Module] = []
    member_results: list[dict[str, object]] = []
    evaluation_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for seed in ENSEMBLE_SEEDS:
        _set_seed(seed)
        model = build_model(model_name).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        train_loader = DataLoader(
            LightTrainingDataset(dataset.paths),
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
                logits = model(images)
                loss = mean_head_cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(images)
            epoch_loss = total_loss / sample_count
            history.append({"epoch": epoch, "train_augmented_loss": epoch_loss})
            if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                print(
                    f"seed={seed} epoch={epoch:03d}/{epochs} loss={epoch_loss:.4f}",
                    flush=True,
                )
        checkpoint_path = output_dir / f"model_seed{seed}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        logits, targets = _evaluate(model, evaluation_loader, device)
        member_results.append(
            {
                "seed": seed,
                "checkpoint": checkpoint_path.name,
                "train_metrics": asdict(calculate_metrics(logits, targets)),
                "final_augmented_loss": history[-1]["train_augmented_loss"],
                "history": history,
            }
        )
        models.append(model.cpu())

    ensemble = CaptchaEnsemble(models).eval()
    ensemble_path = output_dir / "ensemble.pt"
    torch.save(ensemble.state_dict(), ensemble_path)
    ensemble_logits, ensemble_targets = _evaluate(ensemble, evaluation_loader, torch.device("cpu"))
    ensemble_metrics = calculate_metrics(ensemble_logits, ensemble_targets)
    sample_images, _ = next(iter(evaluation_loader))
    export_result = export_ensemble_onnx(
        ensemble,
        output_dir / "captcha.onnx",
        sample_images[:8],
    )
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "version": output_name,
        "model_name": model_name,
        "member_parameter_count": parameter_count(models[0]),
        "ensemble_parameter_count": parameter_count(ensemble),
        "sample_count": sample_count,
        "epochs": epochs,
        "config": {
            "optimizer": "AdamW",
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "scheduler": None,
            "dropout": None,
            "augmentation": "light",
            "normalization": "float32 [0, 1]",
        },
        "members": member_results,
        "ensemble_train_metrics": asdict(ensemble_metrics),
        "ensemble_checkpoint": ensemble_path.name,
        "manifest": manifest_path.name,
        "onnx": export_result,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"done version={output_name} samples={sample_count} epochs={epochs} "
        f"train_exact={ensemble_metrics.exact_accuracy:.4f} "
        f"onnx_error={export_result['max_absolute_error']:.8g} report={report_path}",
        flush=True,
    )
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the captcha-v1.0 ensemble.")
    parser.add_argument("--labeled-dir", type=Path, default=Path("dataset/labeled"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output-name", default="captcha-v1.0")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--reexport-report", type=Path)
    args = parser.parse_args()
    if args.reexport_report is not None:
        reexport_ensemble(args.reexport_report, args.labeled_dir)
        return
    train_ensemble(
        labeled_dir=args.labeled_dir,
        artifacts_dir=args.artifacts_dir,
        output_name=args.output_name,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
