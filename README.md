# CNN for Ani

面向 Animeko 固定四位数字验证码的轻量 CNN 实验工程。Python/PyTorch 用于数据验证、训练和参考前向，最终目标是导出固定二进制权重，并在 Kotlin 中复现相同 logits。

## 文档来源

- [`docs/实现过程.md`](docs/实现过程.md)：四阶段实验的主执行规范。
- [`docs/识别模型.md`](docs/识别模型.md)：multi-head CNN、损失、指标、增强与置信度的理论基础。
- [`docs/最终方案.md`](docs/最终方案.md)：前几轮实验完成后的最终 Position-DS 训练与验收规范。
- [`docs/冷启动结论.md`](docs/冷启动结论.md)：本次工程化取舍和当前阻塞。

## 当前状态

已完成冷启动基线，并已实现最终模型训练流水线：

- 固化 `1 x 32 x 96 -> 3 层 CNN -> 横向 4 位置池化 -> 4 x 10 logits` 模型契约；
- 固化灰度、resize、`float32 / 255` 预处理；
- 固化标签命名、三项核心指标和最小位置置信度；
- 提供可运行的冒烟检查、单元测试和静态检查命令。
- 新增 93,904 参数 Position-DS CNN、四个独立分类头、EMA、固定更新数调度和困难样本回放；
- 新增 eval/prod 两阶段入口，冻结测试集指标与全量生产重训严格分开。

仓库现有 3203 张真实验证码已全部标注。最终流水线按三个来源近似等额冻结 481 张测试集：新优酷 161 张、次元城动画 160 张、饭团动漫 160 张；其余数据分为 2447 训练和 275 验证。生产阶段在测试完成后再合并全部 3203 张。最终真实准确率必须来自 `captcha-final-eval_*`，不能使用全量生产训练指标替代。

## 环境与启动

前置要求：

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

```powershell
uv sync --python 3.12
uv run python -m cnn_for_ani
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## 人工标注

递归读取 `dataset/raw/` 中的原始图片，人工输入四位数字后复制到 `dataset/labeled/`：

```powershell
uv run python -m cnn_for_ani.labeling
```

标注窗口中输入正好四位数字并按 Enter，会保存为：

```text
dataset/labeled/<四位数字>_<原始文件名不含扩展名>.<扩展名>
```

已存在于 `dataset/labeled/` 的样本 ID 会自动跳过；`dataset/raw/` 原图不会被修改。

若 Windows 上用户级 uv 缓存不可写，可临时把缓存放到项目目录：

```powershell
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
uv sync --python 3.12
```

## 五折训练

训练会在启动时固定当前 `dataset/labeled/` 文件列表，因此可以与后续人工标注并行进行：

```powershell
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
uv run python -m cnn_for_ani.training
```

结构消融使用 AdamW、`lr=1e-3`、`batch_size=32`、最多 300 epoch，不使用数据增强，并用 `ReduceLROnPlateau(factor=0.3, patience=10, min_lr=1e-5)` 调整学习率。前 100 epoch 禁止 early stopping，之后按验证损失使用 patience 50；ExactAcc 只作为业务指标，不参与早停或选模。每次运行输出到独立的 `artifacts/cv_<模型>_<时间>/`，包含五折划分、验证损失最佳检查点、最终检查点，以及逐 epoch 的 Train/Val loss、CharAcc、ExactAcc、HammingError 和学习率。

若当前标签只来自一个下载批次，严格 Group K-Fold 无法执行，训练入口会使用固定随机种子按四个位置的数字边际分布做近似均衡划分，并在报告中记录 `group_isolation=false`。这类结果可能受同批次相似样本泄漏影响。

### 746 张固定快照消融

2026-07-16 使用完全相同的 746 张快照、folds、AdamW、ReduceLROnPlateau 和停止规则完成结构消融：

| 模型 | 参数量 | 轻量增强 | CV CharAcc | CV ExactAcc |
| --- | ---: | --- | ---: | ---: |
| flatten 全局头 | 34,328 | 否 | 14.68% | 0.00% |
| medium 全局头 | 27,728 | 否 | 20.96% | 0.27% |
| position 空间头 | 4,218 | 否 | 39.07% | 1.47% |
| position 空间头 | 4,218 | 是 | 47.15% | 5.50% |

按固定规则选择 4,218 参数的 position 空间头。轻量增强还把同一 178 张 temporal holdout 的五模型集成 CharAcc 从 41.43% 提升到 54.07%，ExactAcc 从 3.37% 提升到 9.55%。这些结果仍不足以支持模型投入使用。

增强组 OOF 错误分析显示四个位置准确率依次为 48.79%、45.84%、47.32%、46.65%；主要数字混淆包括 `2 -> 3`、`1 -> 7`、`6 -> 0`、`9 -> 0/5` 和 `8 -> 0/6`。完整混淆矩阵和 100 张错误案例位于对应训练目录的 `error_analysis/`。

### 最终时间留出结果

结构与增强固定后，从 1868 张标注中按创建时间冻结最新 250 张 `fantuan` 样本，其余 1618 张用于 position 空间头的 300 epoch 全量训练。最终结果为：

- Train CharAcc 65.73%，Train ExactAcc 20.33%；
- Final Holdout CharAcc 65.90%，Final Holdout ExactAcc 19.60%；
- Final Holdout 四位置准确率为 68.80%、58.40%、68.40%、68.00%。

第二位仍明显较弱，主要混淆包括 `1 -> 7`、`8 -> 6`、`3 -> 2`、`6 -> 8` 和 `5 -> 9`。最终 PyTorch checkpoint、不可变切分和报告位于 `artifacts/final_position_20260716_100236/`。

最终 `captcha.onnx` 已通过 ONNX checker，并在冻结 holdout 的单张和 batch 8 真实图片上完成 PyTorch/ONNX Runtime logits 对齐。最大绝对误差为 `3.5762787e-06`，满足 `<1e-4` 契约；输入为动态 batch 的 `[B, 1, 32, 96]`，输出为 `[B, 4, 10]`。

### 宽版 Position 调参结果

新增 `xinyouku` 批次后，将当时已标注的 184 张完整冻结为跨批次 holdout，并使用 `ciyuancheng + fantuan` 共 2000 张训练。40,394 参数的 `wide_position` 只扩大卷积通道和共享位置分类头，其他训练变量与轻量增强保持不变：

- 4,218 参数基线在 `xinyouku`：CharAcc 62.23%，ExactAcc 17.39%；
- 40,394 参数宽版模型：Train CharAcc 87.31%，Train ExactAcc 58.80%；
- 宽版模型跨批次 Holdout CharAcc 82.34%，ExactAcc@1 50.00%；
- ExactAcc@2/@3/@5 分别为 64.67%、70.65%、77.72%；
- 四位置准确率为 85.33%、77.72%、78.26%、88.04%。

模型容量是当前阶段的主要瓶颈之一。第二、三位置仍然较弱，第二位置主要混淆为 `7 -> 1`、`0 -> 6` 和 `8 -> 2`。宽版 ONNX 的 PyTorch/ONNX Runtime 最大 logits 误差为 `7.6293945e-06`，满足 `<1e-4` 契约。调参产物位于 `artifacts/final_wide_position_20260716_104107/`。

训练链路的真实数据过拟合诊断会固定抽取 32 张图片，不使用验证集和数据增强，并尝试训练到 Train ExactAcc 100%：

```powershell
uv run python -m cnn_for_ani.overfit
```

可复用历史报告中的样本快照和五折划分进行模型消融：

```powershell
uv run python -m cnn_for_ani.training --model flatten --snapshot-report artifacts/cv_.../report.json
```

历史五折检查点可直接评估快照之后新增的时间留出集：

```powershell
uv run python -m cnn_for_ani.temporal_holdout --source-report artifacts/cv_.../report.json --holdout-size 178
```

胜出模型可基于五折 OOF 预测生成位置准确率、混淆矩阵和错误案例：

```powershell
uv run python -m cnn_for_ani.error_analysis --source-report artifacts/cv_position_.../report.json --model position
```

轻量增强 A/B 复用相同快照和 folds，只对训练集应用小幅平移、亮度、对比度、模糊与噪声：

```powershell
uv run python -m cnn_for_ani.training --model position --augmentation light --snapshot-report artifacts/cv_.../report.json
```

结构与增强固定后，最终训练会冻结最新 250 张作为时间留出集，并用其余标注训练 position 模型：

```powershell
uv run python -m cnn_for_ani.final_training --model position --holdout-size 250
```

最终 checkpoint 可导出为 ONNX，并用冻结 holdout 的真实图片验证 PyTorch/ONNX Runtime logits：

```powershell
uv run python -m cnn_for_ani.export_onnx --source-report artifacts/final_position_.../report.json
```

调参前可计算整码 Top-k、按位置混淆矩阵，并导出第二位置错误案例：

```powershell
uv run python -m cnn_for_ani.tuning_analysis --source-report artifacts/final_position_.../report.json
```

新增下载批次可整体冻结为跨批次 holdout，再按该 split 训练宽版位置模型：

```powershell
uv run python -m cnn_for_ani.prepare_batch_split --holdout-batch batch_20260715_xinyouku
uv run python -m cnn_for_ani.final_training --model wide_position --split artifacts/split_.../split.json
```

旧版架构和超参数冻结后，可复现此前的宽版三种子训练：

```powershell
uv run python -m cnn_for_ani.train_ensemble
```

### captcha-v1.0

这是最终方案确定前的宽版实验版本。它在启动时冻结 2477 张完整标注快照。参考模型为 2000 张训练 300 epoch，共 18,900 次优化器更新；按样本数换算后最终训练 242 epoch，共 18,876 次更新。三个成员固定使用 seed 42、3407、20260716：

- 单模型参数量 40,394，三模型集成参数量 121,182；
- 完整训练快照上的集成 CharAcc 91.65%，ExactAcc 72.06%；
- 最终 `captcha.onnx` 大小 496,446 bytes；
- PyTorch/ONNX Runtime 最大 logits 误差 `4.7683716e-06`；
- ONNX 输入 `input: [B, 1, 32, 96]`，输出 `logits: [B, 4, 10]`。

由于全部 2477 张数据均按当时的收尾方案合并训练，91.65%/72.06% 是训练快照指标，不是独立测试集结论。部署产物、三个成员权重、集成权重、数据 manifest 和报告位于 `artifacts/captcha-v1.0/`。

### Position-DS 最终流水线

默认一次执行可信评估和全量生产重训：

```powershell
uv run python -m cnn_for_ani.final_pipeline --phase all
```

也可以拆开执行，先检查 eval 报告达到发布线，再启动生产训练：

```powershell
uv run python -m cnn_for_ani.final_pipeline --phase eval
uv run python -m cnn_for_ani.final_pipeline --phase prod --eval-report artifacts/captcha-final-eval_.../report.json
```

eval 默认使用 25,000 updates、500 updates warmup、cosine decay、EMA 0.999、seed 42/3407/20260716，并在 17,500 updates 后启用 2 倍困难样本回放。`report.json` 包含三个单模型和集成的总体及分来源指标，包括 ExactAcc@1/2/3/5、四位置准确率、数字 precision/recall、混淆矩阵及 ONNX logits 误差。生产报告不生成训练集“准确率”；可信指标始终引用 eval 报告。

为尽量保留测试独立性，切分器优先选择未出现在历史实验 manifest 中的样本。当前来源均衡测试仍有 218 张曾参与前轮结构实验（次元城动画 159、饭团动漫 59、新优酷 0）；这些样本不会进入最终 eval 模型训练，但最终报告必须保留这项方案选择污染说明。若要得到完全未见的多来源盲测，应额外采集并标注至少 159 张次元城动画和 59 张饭团动漫样本。

若 eval 集成 ExactAcc@1 未达到 60%，流水线默认阻止生产重训；仅在明确接受低于发布线的风险时，才可传入 `--allow-below-threshold`。

### 断点续训

eval 和 prod 默认每 250 updates 保存一次完整断点，包括原始模型、EMA、optimizer、当前 update、最佳指标、困难样本权重、采样器状态及 Python/NumPy/PyTorch 随机状态。意外中断后，对运行目录执行：

```powershell
uv run python -m cnn_for_ani.final_pipeline --resume artifacts/captcha-final-eval_...
```

生产阶段同样使用对应的 `captcha-final-prod_...` 目录。已完成的 seed 会直接加载并跳过，未完成的 seed 从 `model_seed<seed>.resume.pt` 继续；恢复时自动使用 `run_state.json` 中冻结的原始配置和 split，不能修改 batch size、更新数或其他训练参数。默认最多损失最近 249 updates；可在首次启动时用 `--checkpoint-interval` 调整保存间隔。

## 数据目录契约

```text
dataset/
├── raw/                         # 原始图片，只增不改
│   └── 000001.png
└── labeled/                     # 人工确认后的图片
    └── 5271_000001.png
```

采集阶段应另行保存 `metadata.csv`，至少包含样本 ID、来源、宽度、高度和下载批次。相同验证码可能重复出现，因此标签文件不能只命名为 `5271.png`。

真实数据在纳入实验前至少检查：是否始终为四位数字、尺寸与生成器是否固定、是否有粘连/旋转/干扰线，以及四个位置的数字边际分布。

## 实验阶段

1. 在 Animeko 验证码图片进入 UI 前旁路保存原始字节，并按时间批次采集数据。
2. 人工标注为 `<label>_<id>`，检查分布，按下载批次优先使用 5 折 Group K-Fold。
3. 用 AdamW、`lr=1e-3`、`batch_size=64`、最多 50 epoch、early stopping patience 8 训练基线，主要比较 ExactAcc。
4. 根据交叉验证最佳 epoch 的中位数用全量数据重训，导出固定顺序的 `captcha.weights`，最后执行 Python/Kotlin logits 对齐测试。

只有真实验证集的四位完全准确率才能支持模型可用性结论；目标是稳定达到 98%，理想值约 99%。

## 代码结构

```text
src/cnn_for_ani/
├── model.py          # 文档指定的轻量 CNN
├── preprocessing.py  # Kotlin 必须复现的确定性预处理
├── dataset.py        # 标签解析与样本读取契约
├── labeling.py       # raw 到 labeled 的人工标注工具
├── training.py       # 可复现五折划分、训练、早停与报告
├── overfit.py        # 32 张真实样本过拟合诊断
├── losses.py         # 四个位置交叉熵的平均值
├── metrics.py        # CharAcc / ExactAcc / HammingError
└── prediction.py     # 解码与最小位置置信度
tests/                # 当前 Python 契约测试
```
