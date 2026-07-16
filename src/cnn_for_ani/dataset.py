"""验证码标签命名与样本读取契约。"""

import re
from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from cnn_for_ani.preprocessing import preprocess_image

_LABELED_FILENAME = re.compile(r"^(?P<label>\d{4})_(?P<sample_id>[^.]+)\.[^.]+$")
_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def parse_label(path: str | Path) -> torch.Tensor:
    """从 ``<四位数字>_<样本ID>.<扩展名>`` 中解析四个分类目标。"""
    filename = Path(path).name
    match = _LABELED_FILENAME.fullmatch(filename)
    if match is None:
        raise ValueError(
            f"labeled filename must match '<four digits>_<sample id>.<extension>', got {filename!r}"
        )
    return torch.tensor([int(digit) for digit in match.group("label")], dtype=torch.long)


def parse_sample_id(path: str | Path) -> str:
    """Return the sample identifier from a labeled filename."""
    filename = Path(path).name
    match = _LABELED_FILENAME.fullmatch(filename)
    if match is None:
        raise ValueError(
            f"labeled filename must match '<four digits>_<sample id>.<extension>', got {filename!r}"
        )
    return match.group("sample_id")


class LabeledCaptchaDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """按文件名读取已经人工确认标签的验证码数据。"""

    def __init__(self, root: str | Path, filenames: Sequence[str] | None = None) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"labeled dataset directory does not exist: {self.root}")
        if filenames is None:
            self.paths = sorted(
                path
                for path in self.root.iterdir()
                if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
            )
        else:
            if len(set(filenames)) != len(filenames):
                raise ValueError("snapshot filenames must be unique")
            self.paths = [self.root / filename for filename in filenames]
            missing = [path.name for path in self.paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"snapshot files are missing: {missing[:3]}")
        for path in self.paths:
            parse_label(path)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[index]
        label = parse_label(path)
        with Image.open(path) as image:
            tensor = preprocess_image(image)
        return tensor, label
