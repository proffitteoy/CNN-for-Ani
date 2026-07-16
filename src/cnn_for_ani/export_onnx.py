"""Export a trained captcha checkpoint to ONNX and verify numerical parity."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image

from cnn_for_ani.model import build_model
from cnn_for_ani.preprocessing import preprocess_image


def _load_images(labeled_dir: Path, filenames: list[str]) -> torch.Tensor:
    tensors = []
    for filename in filenames:
        with Image.open(labeled_dir / filename) as image:
            tensors.append(preprocess_image(image))
    return torch.stack(tensors)


def _compare_logits(
    model: torch.nn.Module,
    session: ort.InferenceSession,
    images: torch.Tensor,
) -> dict[str, object]:
    with torch.no_grad():
        torch_logits = model(images).cpu().numpy()
    onnx_logits = session.run(["logits"], {"images": images.cpu().numpy()})[0]
    absolute_error = np.abs(torch_logits - onnx_logits)
    return {
        "input_shape": list(images.shape),
        "output_shape": list(torch_logits.shape),
        "max_absolute_error": float(absolute_error.max()),
        "mean_absolute_error": float(absolute_error.mean()),
    }


def export_onnx(
    source_report: Path,
    labeled_dir: Path,
    output_path: Path | None = None,
    opset_version: int = 17,
) -> Path:
    report = json.loads(source_report.read_text(encoding="utf-8"))
    split_path = source_report.parent / report["split"]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    checkpoint_path = source_report.parent / report["checkpoint"]
    output_path = output_path or source_report.parent / "captcha.onnx"

    model = build_model(report["model_name"])
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.eval()
    example = torch.zeros(1, 1, 32, 96, dtype=torch.float32)
    torch.onnx.export(
        model,
        example,
        output_path,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset_version,
        dynamo=False,
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    holdout_names = split["holdout_samples"]
    single_images = _load_images(labeled_dir, holdout_names[:1])
    batch_images = _load_images(labeled_dir, holdout_names[:8])
    single_comparison = _compare_logits(model, session, single_images)
    batch_comparison = _compare_logits(model, session, batch_images)
    max_absolute_error = max(
        single_comparison["max_absolute_error"],
        batch_comparison["max_absolute_error"],
    )
    if max_absolute_error >= 1e-4:
        raise RuntimeError(f"ONNX logits mismatch: max absolute error {max_absolute_error}")

    export_report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "source_report": str(source_report),
        "checkpoint": str(checkpoint_path),
        "onnx_path": str(output_path),
        "onnx_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "opset_version": opset_version,
        "input_contract": ["batch", 1, 32, 96],
        "output_contract": ["batch", 4, 10],
        "single_sample": {
            "filename": holdout_names[0],
            **single_comparison,
        },
        "batch_sample": {
            "filenames": holdout_names[:8],
            **batch_comparison,
        },
        "maximum_verified_absolute_error": max_absolute_error,
        "onnx_checker_passed": True,
        "onnxruntime_provider": "CPUExecutionProvider",
        "versions": {
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
    }
    export_report_path = output_path.with_suffix(".export.json")
    export_report_path.write_text(
        json.dumps(export_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"onnx={output_path} max_abs_error={max_absolute_error:.8g} report={export_report_path}",
        flush=True,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and verify the final captcha ONNX model.")
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--labeled-dir", type=Path, default=Path("dataset/labeled"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    export_onnx(
        source_report=args.source_report,
        labeled_dir=args.labeled_dir,
        output_path=args.output,
        opset_version=args.opset,
    )


if __name__ == "__main__":
    main()
