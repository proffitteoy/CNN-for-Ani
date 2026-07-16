"""与 multi-head classification 理论约束一致的训练损失。"""

import torch
from torch.nn import functional as F


def mean_head_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    position_weights: tuple[float, float, float, float] | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """计算四个数字位置交叉熵；可为最终方案设置位置权重与标签平滑。"""
    if logits.ndim != 3 or tuple(logits.shape[1:]) != (4, 10):
        raise ValueError(f"expected logits shape [B, 4, 10], got {tuple(logits.shape)}")
    if tuple(targets.shape) != tuple(logits.shape[:2]):
        raise ValueError(
            f"expected targets shape {tuple(logits.shape[:2])}, got {tuple(targets.shape)}"
        )

    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
    weights = position_weights or (1.0, 1.0, 1.0, 1.0)
    if len(weights) != 4 or any(weight <= 0.0 for weight in weights):
        raise ValueError("position_weights must contain four positive values")

    losses = [
        F.cross_entropy(
            logits[:, index, :],
            targets[:, index],
            label_smoothing=label_smoothing,
        )
        * weights[index]
        for index in range(4)
    ]
    return torch.stack(losses).sum() / sum(weights)
