import random

import numpy as np
from PIL import Image

from cnn_for_ani.augmentation import final_augment_image, light_augment_image


def test_light_augmentation_preserves_image_contract() -> None:
    random.seed(123)
    np.random.seed(123)
    image = Image.new("RGB", (128, 40), color=(240, 240, 240))

    augmented = light_augment_image(image)

    assert augmented.mode == "L"
    assert augmented.size == image.size
    assert np.asarray(augmented).dtype == np.uint8


def test_final_augmentation_preserves_image_contract() -> None:
    random.seed(123)
    np.random.seed(123)
    image = Image.new("RGB", (128, 40), color=(240, 240, 240))

    augmented = final_augment_image(image)

    assert augmented.mode == "L"
    assert augmented.size == image.size
    assert np.asarray(augmented).dtype == np.uint8
