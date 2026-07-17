"""Run one public CNN for Ani ONNX model on a single image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnxruntime as ort
import torch
from PIL import Image

from cnn_for_ani.prediction import decode_logits
from cnn_for_ani.preprocessing import preprocess_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("image", type=Path)
    args = parser.parse_args()

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    with Image.open(args.image) as image:
        input_tensor = preprocess_image(image).unsqueeze(0)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_tensor.numpy()})[0]
    logits = torch.from_numpy(output)
    prediction = decode_logits(logits)
    digits = "".join(str(digit) for digit in prediction.digits[0].tolist())
    result = {
        "image": str(args.image),
        "prediction": digits,
        "confidence": prediction.confidence[0].item(),
        "logits_shape": list(logits.shape),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
