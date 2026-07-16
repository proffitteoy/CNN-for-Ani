"""不依赖真实数据的模型契约冒烟检查。"""

import json

import torch

from cnn_for_ani.model import PositionDSCaptchaCNN, parameter_count


def main() -> None:
    model = PositionDSCaptchaCNN().eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 1, 32, 96))
    print(
        json.dumps(
            {
                "input_shape": [1, 1, 32, 96],
                "output_shape": list(output.shape),
                "model": "position_ds",
                "trainable_parameters": parameter_count(model),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
