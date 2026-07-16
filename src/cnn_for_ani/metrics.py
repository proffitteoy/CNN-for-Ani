"""实验规范要求的验证码分类指标。"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CaptchaMetrics:
    """一个批次或完整验证集的三项核心指标。"""

    char_accuracy: float
    exact_accuracy: float
    hamming_error: float


def calculate_metrics(logits: torch.Tensor, targets: torch.Tensor) -> CaptchaMetrics:
    """由 ``[B, 4, 10]`` logits 和 ``[B, 4]`` 标签计算指标。"""
    if logits.ndim != 3 or tuple(logits.shape[1:]) != (4, 10):
        raise ValueError(f"expected logits shape [B, 4, 10], got {tuple(logits.shape)}")
    if tuple(targets.shape) != tuple(logits.shape[:2]):
        raise ValueError(
            f"expected targets shape {tuple(logits.shape[:2])}, got {tuple(targets.shape)}"
        )

    predictions = logits.argmax(dim=-1)
    matches = predictions.eq(targets)
    return CaptchaMetrics(
        char_accuracy=matches.float().mean().item(),
        exact_accuracy=matches.all(dim=1).float().mean().item(),
        hamming_error=(~matches).sum(dim=1).float().mean().item(),
    )
