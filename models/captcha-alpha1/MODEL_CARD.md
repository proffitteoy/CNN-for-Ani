# captcha-alpha1 模型卡

## 状态与用途

公开研究预览版，可用于验证 ONNX 推理接口、Python/Kotlin 前向实现和下游集成。它对应内部运行名 `captcha-v1.0`，不是仍在训练的最终 Position-DS 模型。

推荐仅用于已获授权的兼容性研究，不建议把训练快照指标当作生产可用性证明。

## 架构与契约

- 三个 `wide_position` CNN，seed `42 / 3407 / 20260716`；
- 单成员 40,394 参数，集成 121,182 参数；
- 输入灰度 `float32 [0, 1]`，`[B, 1, 32, 96]`；
- 输出 logits `[B, 4, 10]`；
- 集成方法为三个成员 logits 的算术平均；
- 置信度为四个位置最大 softmax 概率的最小值。

## 训练与指标

- 训练快照：2,477 张人工标注样本；
- 242 epochs，AdamW，batch size 32，light augmentation；
- 训练快照 CharAcc `91.6532%`；
- 训练快照 ExactAcc `72.0630%`；
- 训练快照 HammingError `0.3339`。

这些是全量训练快照上的拟合指标，不是独立留出或盲测结果。

## 导出验证

- ONNX opset 17，大小 496,620 bytes；
- SHA-256 `832a11cd98f07511b57a2a4dae236f8cc29ad430962cf8fd509580ab7025ed0a`；
- ONNX checker 通过；
- 8 张真实图片上 PyTorch/ONNX Runtime 最大 logits 误差 `4.76837158203125e-06`。

原始训练报告和样本快照分别为 [`report.json`](report.json) 与 [`manifest.json`](manifest.json)。
可使用 [`SHA256SUMS`](SHA256SUMS) 校验三个发布文件的完整性。

## 限制

该模型未经过当前 481 张多来源留出集的最终评估；对新来源、不同验证码长度或明显不同的生成器不作保证。模型按根目录 Apache-2.0 许可发布。
