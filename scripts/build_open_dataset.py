"""Build the public labeled dataset from the append-only local source data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

from PIL import Image

FIELDS = (
    "filename",
    "label",
    "sample_id",
    "source",
    "batch",
    "purpose",
    "width",
    "height",
    "sha256",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    metadata = {row["id"]: row for row in rows}
    if len(metadata) != len(rows):
        raise ValueError("metadata contains duplicate sample IDs")
    return metadata


def build_dataset(labeled_dir: Path, metadata_path: Path, output_dir: Path) -> None:
    metadata = read_metadata(metadata_path)
    labeled_paths = sorted(path for path in labeled_dir.glob("*.png") if path.name != ".gitkeep")
    if len(labeled_paths) != len(metadata):
        raise ValueError(
            f"labeled/metadata count mismatch: {len(labeled_paths)} != {len(metadata)}"
        )

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []

    for source_path in labeled_paths:
        try:
            label, sample_id = source_path.stem.split("_", maxsplit=1)
        except ValueError as error:
            raise ValueError(f"invalid labeled filename: {source_path.name}") from error
        if len(label) != 4 or not label.isdigit():
            raise ValueError(f"invalid four-digit label: {source_path.name}")
        try:
            source_metadata = metadata[sample_id]
        except KeyError as error:
            raise ValueError(f"missing metadata for {source_path.name}") from error

        digest = file_sha256(source_path)
        if digest != source_metadata["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {source_path.name}")
        with Image.open(source_path) as image:
            expected_size = (int(source_metadata["width"]), int(source_metadata["height"]))
            if image.format != "PNG" or image.size != expected_size:
                raise ValueError(
                    f"image contract mismatch for {source_path.name}: "
                    f"format={image.format} size={image.size} expected={expected_size}"
                )
            image.verify()
        destination = images_dir / source_path.name
        if not destination.exists() or file_sha256(destination) != digest:
            shutil.copy2(source_path, destination)
        records.append(
            {
                "filename": source_path.name,
                "label": label,
                "sample_id": sample_id,
                "source": source_metadata["source"],
                "batch": source_metadata["batch"],
                "purpose": source_metadata["purpose"],
                "width": source_metadata["width"],
                "height": source_metadata["height"],
                "sha256": digest,
            }
        )

    expected_names = {record["filename"] for record in records}
    actual_names = {path.name for path in images_dir.glob("*.png")}
    if actual_names != expected_names:
        raise ValueError("output images contain stale or missing files")

    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    manifest_digest = file_sha256(output_dir / "manifest.csv")
    (output_dir / "SHA256SUMS").write_text(
        f"{manifest_digest}  manifest.csv\n",
        encoding="utf-8",
    )
    print(f"published {len(records)} samples to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled-dir", type=Path, default=Path("dataset/labeled"))
    parser.add_argument("--metadata", type=Path, default=Path("dataset/metadata.csv"))
    parser.add_argument("--output", type=Path, default=Path("dataset/captcha-v1"))
    args = parser.parse_args()
    build_dataset(args.labeled_dir, args.metadata, args.output)


if __name__ == "__main__":
    main()
