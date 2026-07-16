"""验证码解码和保守置信度计算。"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CaptchaPrediction:
    digits: torch.Tensor
    confidence: torch.Tensor


@dataclass(frozen=True)
class TopKCaptchaPrediction:
    digits: torch.Tensor
    log_probabilities: torch.Tensor


def decode_logits(logits: torch.Tensor) -> CaptchaPrediction:
    """解码 logits，置信度取四个位置最大概率中的最小值。"""
    if logits.ndim != 3 or tuple(logits.shape[1:]) != (4, 10):
        raise ValueError(f"expected logits shape [B, 4, 10], got {tuple(logits.shape)}")

    probabilities = logits.softmax(dim=-1)
    per_position_confidence, digits = probabilities.max(dim=-1)
    confidence = per_position_confidence.min(dim=-1).values
    return CaptchaPrediction(digits=digits, confidence=confidence)


def decode_top_k(logits: torch.Tensor, k: int) -> TopKCaptchaPrediction:
    """Return the top-k distinct four-digit sequences ordered by joint probability."""
    if logits.ndim != 3 or tuple(logits.shape[1:]) != (4, 10):
        raise ValueError(f"expected logits shape [B, 4, 10], got {tuple(logits.shape)}")
    if not 1 <= k <= 10_000:
        raise ValueError("k must be between 1 and 10000")

    log_probabilities = logits.log_softmax(dim=-1)
    batch_size = len(logits)
    beam_scores = torch.zeros(batch_size, 1, device=logits.device, dtype=logits.dtype)
    beam_digits = torch.empty(batch_size, 1, 0, device=logits.device, dtype=torch.long)
    for position in range(4):
        candidate_scores = beam_scores.unsqueeze(-1) + log_probabilities[:, position].unsqueeze(1)
        flattened_scores = candidate_scores.flatten(start_dim=1)
        beam_size = min(k, flattened_scores.shape[1])
        beam_scores, flattened_indices = flattened_scores.topk(beam_size, dim=1)
        parent_indices = flattened_indices // 10
        next_digits = flattened_indices % 10
        parent_digits = torch.gather(
            beam_digits,
            1,
            parent_indices.unsqueeze(-1).expand(-1, -1, beam_digits.shape[-1]),
        )
        beam_digits = torch.cat((parent_digits, next_digits.unsqueeze(-1)), dim=-1)
    return TopKCaptchaPrediction(digits=beam_digits, log_probabilities=beam_scores)


def calculate_top_k_exact_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ks: tuple[int, ...] = (1, 2, 3, 5),
) -> dict[int, float]:
    """Calculate whether the true sequence appears in each requested top-k beam."""
    if tuple(targets.shape) != tuple(logits.shape[:2]):
        raise ValueError(
            f"expected targets shape {tuple(logits.shape[:2])}, got {tuple(targets.shape)}"
        )
    if not ks:
        raise ValueError("ks must not be empty")
    prediction = decode_top_k(logits, max(ks))
    sequence_matches = prediction.digits.eq(targets.unsqueeze(1)).all(dim=-1)
    return {k: sequence_matches[:, :k].any(dim=1).float().mean().item() for k in ks}
