"""Fixed-length captcha CNN architectures used by controlled ablations."""

from typing import Final

import torch
from torch import nn

INPUT_CHANNELS: Final = 1
INPUT_HEIGHT: Final = 32
INPUT_WIDTH: Final = 96
CAPTCHA_LENGTH: Final = 4
NUM_CLASSES: Final = 10


def _validate_images(images: torch.Tensor) -> None:
    expected_shape = (INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    if images.ndim != 4 or tuple(images.shape[1:]) != expected_shape:
        raise ValueError(f"expected input shape [B, 1, 32, 96], got {tuple(images.shape)}")


def _baseline_features() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
    )


class FlattenCaptchaCNN(nn.Module):
    """Original 34k baseline where every head sees the complete feature map."""

    def __init__(self) -> None:
        super().__init__()
        self.features = _baseline_features()
        self.classifier = nn.Linear(16 * 4 * 12, CAPTCHA_LENGTH * NUM_CLASSES)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        _validate_images(images)
        logits = self.classifier(self.features(images).flatten(start_dim=1))
        return logits.view(-1, CAPTCHA_LENGTH, NUM_CLASSES)


class PositionCaptchaCNN(nn.Module):
    """Small shared position head used by the first spatial ablation."""

    def __init__(self) -> None:
        super().__init__()
        self.features = _baseline_features()
        self.position_pool = nn.AdaptiveAvgPool2d((4, CAPTCHA_LENGTH))
        self.classifier = nn.Linear(16 * 4, NUM_CLASSES)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        _validate_images(images)
        features = self.features(images)
        position_features = self.position_pool(features).permute(0, 3, 1, 2).flatten(start_dim=2)
        return self.classifier(position_features)


class MediumCaptchaCNN(nn.Module):
    """A 28k global position model without hard digit slots."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.feature_pool = nn.AdaptiveAvgPool2d((2, 12))
        self.classifier = nn.Sequential(
            nn.Linear(24 * 2 * 12, 32),
            nn.ReLU(),
            nn.Linear(32, CAPTCHA_LENGTH * NUM_CLASSES),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        _validate_images(images)
        features = self.feature_pool(self.features(images)).flatten(start_dim=1)
        return self.classifier(features).view(-1, CAPTCHA_LENGTH, NUM_CLASSES)


class WidePositionCaptchaCNN(nn.Module):
    """A wider position-preserving CNN for the capacity tuning stage."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.position_pool = nn.AdaptiveAvgPool2d((4, CAPTCHA_LENGTH))
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        _validate_images(images)
        features = self.features(images)
        position_features = self.position_pool(features).permute(0, 3, 1, 2).flatten(start_dim=2)
        return self.classifier(position_features)


class DepthwiseResidualBlock(nn.Module):
    """Depthwise-separable residual block used by the final Position-DS model."""

    def __init__(
        self,
        channels: int,
        *,
        dilation: tuple[int, int] = (1, 1),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = dilation
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.depthwise_norm = nn.BatchNorm2d(channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.pointwise_norm = nn.BatchNorm2d(channels)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = features
        features = self.activation(self.depthwise_norm(self.depthwise(features)))
        features = self.dropout(self.pointwise_norm(self.pointwise(features)))
        return self.activation(features + residual)


def _downsample(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(),
    )


class PositionDSCaptchaCNN(nn.Module):
    """Final position-preserving depthwise-separable residual CNN."""

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
        )
        self.stage1 = nn.Sequential(
            DepthwiseResidualBlock(32, dropout=dropout),
            DepthwiseResidualBlock(32, dropout=dropout),
        )
        self.downsample1 = _downsample(32, 48)
        self.stage2 = nn.Sequential(
            DepthwiseResidualBlock(48, dropout=dropout),
            DepthwiseResidualBlock(48, dropout=dropout),
        )
        self.downsample2 = _downsample(48, 72)
        self.stage3 = nn.Sequential(
            DepthwiseResidualBlock(72, dropout=dropout),
            DepthwiseResidualBlock(72, dropout=dropout),
            DepthwiseResidualBlock(72, dilation=(1, 2), dropout=dropout),
        )
        self.position_pool = nn.AdaptiveAvgPool2d((1, CAPTCHA_LENGTH))
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(72, 64),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, NUM_CLASSES),
                )
                for _ in range(CAPTCHA_LENGTH)
            ]
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        _validate_images(images)
        features = self.stage1(self.stem(images))
        features = self.stage2(self.downsample1(features))
        features = self.stage3(self.downsample2(features))
        position_features = self.position_pool(features).squeeze(2).permute(0, 2, 1)
        return torch.stack(
            [head(position_features[:, index]) for index, head in enumerate(self.heads)],
            dim=1,
        )


class CaptchaEnsemble(nn.Module):
    """Average logits from identically shaped captcha models."""

    def __init__(self, models: list[nn.Module]) -> None:
        super().__init__()
        if not models:
            raise ValueError("ensemble must contain at least one model")
        self.models = nn.ModuleList(models)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = torch.stack([model(images) for model in self.models], dim=0).mean(dim=0)
        return logits.reshape(-1, CAPTCHA_LENGTH, NUM_CLASSES)


MODEL_NAMES = ("flatten", "medium", "position", "position_ds", "wide_position")


def build_model(model_name: str) -> nn.Module:
    models = {
        "flatten": FlattenCaptchaCNN,
        "medium": MediumCaptchaCNN,
        "position": PositionCaptchaCNN,
        "position_ds": PositionDSCaptchaCNN,
        "wide_position": WidePositionCaptchaCNN,
    }
    try:
        return models[model_name]()
    except KeyError as error:
        raise ValueError(f"unknown model {model_name!r}; expected one of {MODEL_NAMES}") from error


CaptchaCNN = PositionCaptchaCNN


def parameter_count(model: nn.Module) -> int:
    """返回所有可训练参数数量。"""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
