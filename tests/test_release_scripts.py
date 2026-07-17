import csv
import hashlib
from pathlib import Path

from PIL import Image

from scripts.build_open_dataset import build_dataset


def _write_png(path: Path, value: int) -> str:
    Image.new("L", (128, 40), color=value).save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_open_dataset_writes_verified_manifest(tmp_path: Path) -> None:
    labeled_dir = tmp_path / "labeled"
    labeled_dir.mkdir()
    first = labeled_dir / "1234_sample-a.png"
    second = labeled_dir / "5678_sample-b.png"
    first_hash = _write_png(first, 0)
    second_hash = _write_png(second, 255)
    metadata_path = tmp_path / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=(
                "id",
                "source",
                "width",
                "height",
                "batch",
                "purpose",
                "relative_path",
                "sha256",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": "sample-a",
                "source": "source-a",
                "width": 128,
                "height": 40,
                "batch": "batch-a",
                "purpose": "training_raw",
                "relative_path": "batch-a/sample-a.png",
                "sha256": first_hash,
            }
        )
        writer.writerow(
            {
                "id": "sample-b",
                "source": "source-b",
                "width": 128,
                "height": 40,
                "batch": "batch-b",
                "purpose": "preflight_test",
                "relative_path": "batch-b/sample-b.png",
                "sha256": second_hash,
            }
        )

    output_dir = tmp_path / "public"
    build_dataset(labeled_dir, metadata_path, output_dir)

    records = list(csv.DictReader((output_dir / "manifest.csv").open(encoding="utf-8")))
    assert [record["label"] for record in records] == ["1234", "5678"]
    assert [record["source"] for record in records] == ["source-a", "source-b"]
    assert {path.name for path in (output_dir / "images").iterdir()} == {
        first.name,
        second.name,
    }
