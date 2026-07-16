"""Python 与未来 Kotlin 实现共同遵守的确定性图像预处理。"""

import numpy as np
import torch
from PIL import Image

from cnn_for_ani.model import INPUT_HEIGHT, INPUT_WIDTH


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """将任意 Pillow 图片转换为 ``1 x 32 x 96`` 的 float32 张量。

    resize 显式使用 nearest，以固定 Pillow 的实际行为；Kotlin 端必须使用相同算法。
    """
    grayscale = image.convert("L")
    resized = grayscale.resize(
        (INPUT_WIDTH, INPUT_HEIGHT),
        resample=Image.Resampling.NEAREST,
    )
    pixels = np.asarray(resized, dtype=np.float32) / np.float32(255.0)
    return torch.from_numpy(pixels.copy()).unsqueeze(0)
