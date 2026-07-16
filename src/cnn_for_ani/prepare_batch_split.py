"""Freeze a complete labeled download batch as an untouched tuning holdout."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from cnn_for_ani.dataset import parse_sample_id

_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def prepare_batch_split(
    labeled_dir: Path,
    raw_dir: Path,
    holdout_batch: str,
    artifacts_dir: Path,
) -> Path:
    holdout_raw_dir = raw_dir / holdout_batch
    if not holdout_raw_dir.is_dir():
        raise FileNotFoundError(f"raw holdout batch does not exist: {holdout_raw_dir}")
    holdout_ids = {
        path.stem
        for path in holdout_raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    }
    labeled_paths = sorted(
        path
        for path in labeled_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )
    holdout_paths = [path for path in labeled_paths if parse_sample_id(path) in holdout_ids]
    train_paths = [path for path in labeled_paths if parse_sample_id(path) not in holdout_ids]
    if not holdout_paths:
        raise ValueError(f"no labeled samples found for holdout batch {holdout_batch}")

    run_name = datetime.now().strftime(f"split_{holdout_batch}_%Y%m%d_%H%M%S")
    output_dir = artifacts_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    split = {
        "created_at": datetime.now().astimezone().isoformat(),
        "selection": "complete raw download batch",
        "holdout_batch": holdout_batch,
        "train_samples": [path.name for path in train_paths],
        "holdout_samples": [path.name for path in holdout_paths],
    }
    split_path = output_dir / "split.json"
    split_path.write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"train={len(train_paths)} holdout={len(holdout_paths)} split={split_path}",
        flush=True,
    )
    return split_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze one raw batch as a labeled holdout.")
    parser.add_argument("--labeled-dir", type=Path, default=Path("dataset/labeled"))
    parser.add_argument("--raw-dir", type=Path, default=Path("dataset/raw"))
    parser.add_argument("--holdout-batch", required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    prepare_batch_split(
        labeled_dir=args.labeled_dir,
        raw_dir=args.raw_dir,
        holdout_batch=args.holdout_batch,
        artifacts_dir=args.artifacts_dir,
    )


if __name__ == "__main__":
    main()
