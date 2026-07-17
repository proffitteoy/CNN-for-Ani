# CNN for Ani

[![CI](https://github.com/proffitteoy/CNN-for-Ani/actions/workflows/ci.yml/badge.svg)](https://github.com/proffitteoy/CNN-for-Ani/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/code%20%26%20models-Apache--2.0-blue.svg)](LICENSE)
[![Dataset: CC BY 4.0](https://img.shields.io/badge/dataset-CC%20BY%204.0-green.svg)](dataset/captcha-v1/LICENSE.md)

面向 Animeko 固定四位数字验证码的轻量 CNN 科研仓库。项目完整开放训练代码、人工标注样本、实验记录、alpha1 权重与最终模型的评估/生产流水线，目标是在 Python 与后续纯 Kotlin 实现之间复现相同的 40 个 logits。

> 发布状态（2026-07-17）：最终 Position-DS 三模型集成与 `captcha-alpha1` 均已开放。最终模型在 481 张多来源留出集上达到 96.05% ExactAcc@1；生产 ONNX 使用全部 3,203 张冻结样本重训。

## 开放资产

| 资产 | 状态 | 内容 | 可信度说明 |
| --- | --- | --- | --- |
| [`models/captcha-alpha1/`](models/captcha-alpha1/) | 可下载、可推理 | 三个 `wide_position` 模型的 logits 平均 ONNX、报告、训练 manifest | 91.65% CharAcc / 72.06% ExactAcc 是 2,477 张训练快照指标，不是独立测试结论 |
| [`models/final/`](models/final/) | 已发布 | 最终 Position-DS 生产 ONNX、eval/prod 报告、split、manifest 和哈希 | 481 张多来源留出 ExactAcc@1 96.05%；其中 218 张参与过早期方案选择 |
| [`models/experimental/`](models/experimental/) | 已开放 | 早期 Position 与 Wide Position 单模型 ONNX、split、报告和导出校验 | 仅用于复现实验演进，不建议部署 |
| [`dataset/captcha-v1/`](dataset/captcha-v1/) | 已开放 | 3,203 张人工标注 PNG、来源/批次/哈希 manifest | 三个来源各 1,000 张正式样本，另含 203 张 preflight 样本 |
| [`experiments/`](experiments/) | 已开放 | 结构消融、跨批次留出、alpha1 与最终结果 | 明确区分 CV、holdout、训练快照和生产重训 |

## 快速开始

要求 Python 3.12 和 [uv](https://docs.astral.sh/uv/)。

```powershell
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
uv sync --python 3.12
uv run python -m cnn_for_ani
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

使用最终 ONNX 识别一张样本：

```powershell
uv run python examples/predict_onnx.py `
  models/final/captcha.onnx `
  dataset/captcha-v1/images/0000_7a831cfc-a80e-4b52-9a39-9603bcff967d-1784115459408-0838.png
```

输出包含四位预测、最小位置置信度和 `[1, 4, 10]` logits 形状。置信度是四个位置各自最大类别概率的最小值，不取平均值。

## 固定模型契约

- 任务：固定长度四位数字 multi-head classification，不切字、不使用 CTC/CRNN/Transformer。
- 输入：灰度图，resize 到 `96 x 32`，转换为 `float32 [0, 1]`，张量形状 `[B, 1, 32, 96]`；不使用 ImageNet normalization。
- 输出：四个位置、每位置十类的 logits，形状 `[B, 4, 10]`。
- 损失：四位置交叉熵；最终方案的位置权重为 `(1.0, 1.15, 1.15, 1.0)`，label smoothing 为 `0.02`。
- 指标：CharAcc、ExactAcc、HammingError；模型选择和发布以 ExactAcc 为主。
- 对齐要求：同一真实输入的 Python/ONNX/Kotlin 40 个 logits 最大绝对误差 `< 1e-4`，只比较 argmax 不算通过。

## 最终模型：Position-DS ensemble

最终模型不是当前 alpha1 的改名版本，而是一套已冻结结构和训练规则的新模型。每个成员为 93,904 参数的 `PositionDSCaptchaCNN`，三个种子模型的 logits 做算术平均，集成共 281,712 个可训练参数。

单成员结构：

1. `1 -> 32` 的 `3 x 3` stem，BatchNorm + SiLU，保持 `32 x 96`。
2. 两个 32 通道 depthwise-separable residual block。
3. stride-2 下采样到 48 通道，再接两个 residual block。
4. stride-2 下采样到 72 通道，再接三个 residual block；最后一个使用横向 dilation `(1, 2)`。
5. `AdaptiveAvgPool2d((1, 4))` 保留四个横向位置。
6. 四个独立的 `72 -> 64 -> 10` 分类头，输出 `[B, 4, 10]`。

训练规则已经冻结：

| 配置 | 值 |
| --- | ---: |
| seeds | `42, 3407, 20260716` |
| optimizer | AdamW |
| updates | 最多 25,000；至少 15,000 |
| batch size | 32 |
| learning rate | `1.5e-3 -> 1e-5`，500 updates warmup + cosine decay |
| weight decay | `1e-4` |
| EMA | `0.999` |
| gradient clipping | `1.0` |
| hard replay | 17,500 updates 后，困难样本权重 `2.0` |
| checkpoint | 每 250 updates 保存完整可精确恢复状态 |

评估阶段固定 481 张多来源留出集：新优酷 161、次元城动画 160、饭团动漫 160；其余 2,447 张训练、275 张验证。该留出集有 218 张曾出现在早期实验 manifest 中，因此属于“多来源留出评估”，不是完全盲测。报告必须同时给出总体和分来源指标。只有 eval 集成 ExactAcc@1 达到 60% 才能进入全量 3,203 张生产重训；理想线为 65%。

最终 eval 已通过发布线：

| 指标 | 总体（481） | 新优酷（161） | 次元城动画（160） | 饭团动漫（160） |
| --- | ---: | ---: | ---: | ---: |
| CharAcc | 98.80% | 98.45% | 99.22% | 98.75% |
| ExactAcc@1 | 96.05% | 96.27% | 96.88% | 95.00% |
| HammingError | 0.0478 | 0.0621 | 0.0313 | 0.0500 |

总体 ExactAcc@2/@3/@5 分别为 98.34% / 98.54% / 98.75%，四个位置准确率分别为 98.75% / 98.34% / 99.38% / 98.75%。最终生产模型按 eval 选出的成员更新数 `6750 / 5500 / 5500` 在全部 3,203 张样本上重训；生产训练不生成用于宣传的训练集准确率，可信指标始终来自 eval 报告。

公开的 `captcha.onnx` 大小为 1,173,896 bytes，SHA-256 为 `97731e093e77c69a768de81ed9d565bb5f81c6bef88df261b6dd460bca2cfd9a`。ONNX checker 通过，PyTorch/ONNX Runtime 最大 logits 误差为 `1.5497208e-06`。完整指标与文件说明见 [`models/final/MODEL_CARD.md`](models/final/MODEL_CARD.md)。

## alpha1

`captcha-alpha1` 是最终 Position-DS 方案冻结前的可运行研究版本，对应内部实验名 `captcha-v1.0`：

- 三个 `wide_position` 成员，seed 为 `42 / 3407 / 20260716`；
- 单成员 40,394 参数，集成 121,182 参数；
- 在启动时冻结的 2,477 张完整训练快照上训练 242 epochs；
- 训练快照 CharAcc 91.65%、ExactAcc 72.06%、HammingError 0.3339；
- ONNX 大小 496,620 bytes，opset 17；
- PyTorch/ONNX Runtime 最大 logits 误差 `4.7683716e-06`；
- 输入 `[B, 1, 32, 96]`，输出 `[B, 4, 10]`。

这些准确率只说明训练快照拟合情况，不代表跨来源泛化能力；新部署应优先使用最终 Position-DS，并继续在自己的独立数据上验证。

## 开放数据集

[`dataset/captcha-v1/manifest.csv`](dataset/captcha-v1/manifest.csv) 为每张图片记录：人工标签、样本 ID、来源、批次、用途、尺寸和 SHA-256。图片保持采集时的 `128 x 40` PNG 原始字节；训练时才执行确定性灰度 resize。

| 来源 | 正式训练采集 | preflight | 合计 |
| --- | ---: | ---: | ---: |
| 次元城动画 | 1,000 | 1 | 1,001 |
| 饭团动漫 | 1,000 | 101 | 1,101 |
| 新优酷 | 1,000 | 101 | 1,101 |
| 合计 | 3,000 | 203 | 3,203 |

数据全部经过人工标注；没有使用模型预测自动回写标签。数据卡、重建命令、已知偏差和许可边界见 [`dataset/captcha-v1/DATASET_CARD.md`](dataset/captcha-v1/DATASET_CARD.md)。

## 复现实验

人工标注：

```powershell
uv run python -m cnn_for_ani.labeling
```

32 张真实样本过拟合检查：

```powershell
uv run python -m cnn_for_ani.overfit
```

五折结构实验与历史 alpha1：

```powershell
uv run python -m cnn_for_ani.training --model position --augmentation light
uv run python -m cnn_for_ani.train_ensemble
```

最终 Position-DS 评估、续跑与生产重训：

```powershell
uv run python -m cnn_for_ani.final_pipeline --phase eval
uv run python -m cnn_for_ani.final_pipeline --resume artifacts/captcha-final-eval_...
uv run python -m cnn_for_ani.final_pipeline --phase prod `
  --eval-report artifacts/captcha-final-eval_.../report.json
```

长训练启动时会冻结文件列表、split、配置和随机状态；续跑会校验这些状态，不能偷偷吸收新样本或修改超参数。完整历史指标见 [`experiments/README.md`](experiments/README.md)，执行依据见 [`docs/实现过程.md`](docs/实现过程.md) 与 [`docs/最终方案.md`](docs/最终方案.md)。

## 仓库结构

```text
src/cnn_for_ani/          模型、预处理、训练、评估与导出
tests/                    模型/数据/指标/断点恢复契约测试
models/                   公开的 alpha1、最终模型状态与实验 ONNX
dataset/captcha-v1/       公开标注样本、manifest、数据卡与数据许可
experiments/              可审计的实验阶段指标
examples/                 最小推理示例
scripts/                  开放数据集重建与校验工具
artifacts/                本地训练运行目录，默认不提交
docs/                     实验规范与理论依据
```

## 限制与负责任使用

- 当前任务只覆盖三个来源、固定四位数字和相近的图像生成分布；不要外推到通用 OCR。
- 481 张最终留出集中有 218 张参与过早期方案选择，完全盲测仍需新增至少 159 张次元城和 59 张饭团样本。
- 本项目仅用于已获授权的兼容性研究、可复现实验和无障碍自动化。不得用于绕过未授权服务的访问控制或违反来源站点条款。
- 数据来源名称只用于科研溯源，不表示来源站点认可本项目。

## 参与和许可

提交变更前请阅读 [`CONTRIBUTING.md`](CONTRIBUTING.md)，安全问题见 [`SECURITY.md`](SECURITY.md)。引用方式见 [`CITATION.cff`](CITATION.cff)。

代码和模型权重使用 [Apache License 2.0](LICENSE)；开放样本集使用 [CC BY 4.0](dataset/captcha-v1/LICENSE.md)。
