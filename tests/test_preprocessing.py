import numpy as np
import torch
from PIL import Image

from cnn_for_ani.preprocessing import preprocess_image


def test_preprocess_image_matches_kotlin_friendly_contract() -> None:
    pixels = np.array([[0, 255], [64, 128]], dtype=np.uint8)
    image = Image.fromarray(pixels, mode="L")

    tensor = preprocess_image(image)

    assert tensor.shape == (1, 32, 96)
    assert tensor.dtype == torch.float32
    assert tensor.min().item() == 0.0
    assert tensor.max().item() == 1.0
