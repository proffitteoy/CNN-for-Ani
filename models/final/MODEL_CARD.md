# 最终 Position-DS 模型卡

> 状态：已发布。生产权重来自全部 3,203 张冻结样本；准确率只引用独立于生产重训的 481 张多来源 eval 报告。

## 文件

- [`captcha.onnx`](captcha.onnx)：最终三成员生产集成；
- [`captcha-final-eval.onnx`](captcha-final-eval.onnx)：产生公开留出指标的三成员 eval 集成；
- [`eval-report.json`](eval-report.json)：481 张留出集的总体、分来源和逐类完整指标；
- [`production-report.json`](production-report.json)：全量重训配置和成员信息；
- [`split.json`](split.json)：2,447 train / 275 validation / 481 test 冻结划分；
- [`manifest.json`](manifest.json)：3,203 张生产训练快照；
- [`release.json`](release.json)：面向工具读取的发布摘要；
- [`SHA256SUMS`](SHA256SUMS)：公开产物完整性校验。

## 架构与输入输出

每个成员是 93,904 参数的 `PositionDSCaptchaCNN`：32/48/72 通道 depthwise-separable residual stages、两次 stride-2 下采样、一个横向 dilation `(1, 2)` block、`AdaptiveAvgPool2d((1, 4))` 和四个独立 `72 -> 64 -> 10` 分类头。三个 seed `42 / 3407 / 20260716` 的 logits 做算术平均，集成参数共 281,712。

- 输入：灰度 `float32 [0, 1]`，`[B, 1, 32, 96]`；
- 输出：四位置十分类 logits，`[B, 4, 10]`；
- 置信度：四个位置最大 softmax 概率的最小值；
- ONNX opset：17。

## 多来源评估

eval 版本：`captcha-final-eval_20260717_090514`。

| 指标 | 总体（481） | 新优酷（161） | 次元城动画（160） | 饭团动漫（160） |
| --- | ---: | ---: | ---: | ---: |
| CharAcc | 98.80% | 98.45% | 99.22% | 98.75% |
| ExactAcc@1 | 96.05% | 96.27% | 96.88% | 95.00% |
| HammingError | 0.0478 | 0.0621 | 0.0313 | 0.0500 |

- ExactAcc@2/@3/@5：98.34% / 98.54% / 98.75%；
- 位置 1/2/3/4 准确率：98.75% / 98.34% / 99.38% / 98.75%；
- 发布门槛：60%，理想线：65%，实际 ExactAcc@1：96.05%，发布通过；
- 共 462/481 张四位完全正确，23/1,924 个字符错误。

测试集有 218 张出现在早期结构实验 manifest 中（次元城 159、饭团 59、新优酷 0）。这些样本没有进入本次 eval 模型训练，但方案选择已受其影响，因此本结果应称为“多来源留出评估”，不能称为完全盲测。

## 生产重训与导出

生产版本：`captcha-final-prod_20260717_124015`。三个成员使用 eval 选出的 EMA 最佳更新数：seed 42 为 6,750，seed 3407 和 20260716 均为 5,500；随后在全部 3,203 张冻结样本上重训并集成。生产报告故意不计算训练集“准确率”，可信准确率来源保持为 eval 报告。

- `captcha.onnx`：1,173,896 bytes；
- SHA-256：`97731e093e77c69a768de81ed9d565bb5f81c6bef88df261b6dd460bca2cfd9a`；
- ONNX checker：通过；
- 真实样本 batch 8 的 PyTorch/ONNX Runtime 最大 logits 误差：`1.5497207641601562e-06`；
- 推理 provider：`CPUExecutionProvider`。

`captcha-final-eval.onnx` 仅用于复核留出指标，其 SHA-256 为 `7236033844544992e438145592ec844059e6bde2a76268367e185d726ad39bd1`；实际部署使用全量重训的 `captcha.onnx`。

## 限制与适用范围

- 只适用于与当前三个来源相近的固定四位数字验证码，不是通用 OCR。
- 仍需新增至少 159 张次元城和 59 张饭团样本，才能形成完全未参与方案选择的多来源盲测。
- Python/ONNX 已完成 logits 对齐；纯 Kotlin 前向的 `<1e-4` logits 对齐仍需在 Animeko 工程内执行。
- 仅用于已获授权的兼容性研究和自动化，不得用于未授权访问控制绕过。

模型按仓库根目录 Apache-2.0 许可发布。
