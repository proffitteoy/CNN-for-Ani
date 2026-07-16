"""Training-only augmentations that reflect observed captcha variation."""

from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def light_augment_image(image: Image.Image) -> Image.Image:
    """Apply small translation, photometric jitter, blur, and controlled noise."""
    grayscale = image.convert("L")
    pixels = np.asarray(grayscale, dtype=np.uint8)
    background = int(np.median(pixels))
    translated = Image.new("L", grayscale.size, color=background)
    translated.paste(grayscale, (random.randint(-4, 4), random.randint(-2, 2)))

    brightness = ImageEnhance.Brightness(translated).enhance(random.uniform(0.9, 1.1))
    contrasted = ImageEnhance.Contrast(brightness).enhance(random.uniform(0.9, 1.1))
    if random.random() < 0.25:
        contrasted = contrasted.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.5)))
    if random.random() < 0.5:
        array = np.asarray(contrasted, dtype=np.float32)
        noise = np.random.normal(0.0, random.uniform(1.0, 4.0), size=array.shape)
        array = np.clip(array + noise, 0, 255).astype(np.uint8)
        contrasted = Image.fromarray(array, mode="L")
    return contrasted


def final_augment_image(image: Image.Image) -> Image.Image:
    """Apply the bounded geometric and photometric augmentation from the final plan."""
    grayscale = image.convert("L")
    pixels = np.asarray(grayscale, dtype=np.uint8)
    background = int(np.median(pixels))
    width, height = grayscale.size
    center_x, center_y = width / 2.0, height / 2.0
    translate_x = random.uniform(-2.0, 2.0)
    translate_y = random.uniform(-1.0, 1.0)
    angle = math.radians(random.uniform(-2.0, 2.0))
    scale = random.uniform(0.95, 1.05)
    cosine = math.cos(angle) / scale
    sine = math.sin(angle) / scale
    affine = (
        cosine,
        sine,
        center_x - cosine * (center_x + translate_x) - sine * (center_y + translate_y),
        -sine,
        cosine,
        center_y + sine * (center_x + translate_x) - cosine * (center_y + translate_y),
    )
    transformed = grayscale.transform(
        grayscale.size,
        Image.Transform.AFFINE,
        affine,
        resample=Image.Resampling.BILINEAR,
        fillcolor=background,
    )
    transformed = ImageEnhance.Brightness(transformed).enhance(random.uniform(0.9, 1.1))
    transformed = ImageEnhance.Contrast(transformed).enhance(random.uniform(0.9, 1.1))
    if random.random() < 0.15:
        transformed = transformed.filter(ImageFilter.GaussianBlur(radius=0.5))
    if random.random() < 0.5:
        array = np.asarray(transformed, dtype=np.float32)
        sigma = random.uniform(0.0, 0.03) * 255.0
        noise = np.random.normal(0.0, sigma, size=array.shape)
        transformed = Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8), mode="L")
    return transformed
