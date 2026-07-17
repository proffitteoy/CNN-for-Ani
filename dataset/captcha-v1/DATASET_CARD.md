# Captcha v1 数据卡

## 概述

Captcha v1 是 CNN for Ani 的固定四位数字验证码研究数据集，共 3,203 张 `128 x 40` PNG。图片来自三个公开可访问的验证码响应，由人工逐张确认四位标签；没有把模型预测写回为人工标签。

## 组成

| 来源 | 批次 | purpose | 数量 |
| --- | --- | --- | ---: |
| 次元城动画 | `batch_20260715_ciyuancheng` | `training_raw` | 1,000 |
| 饭团动漫 | `batch_20260715_fantuan` | `training_raw` | 1,000 |
| 新优酷 | `batch_20260715_xinyouku` | `training_raw` | 1,000 |
| 次元城动画 | `test_20260715_preflight` | `preflight_test` | 1 |
| 饭团动漫 | `test_20260715_preflight` | `preflight_test` | 101 |
| 新优酷 | `test_20260715_preflight` | `preflight_test` | 101 |

`images/` 保持采集到的原始 PNG 字节。文件名为 `<label>_<sample_id>.png`；`manifest.csv` 提供 label、来源、批次、用途、尺寸和单文件 SHA-256。

## 推荐划分

不要随机拆分高度相似的同批次样本后声称跨来源泛化。仓库最终流水线冻结 481 张来源均衡留出集（161/160/160），其余拆为 2,447 训练和 275 验证，并在报告中保留分来源指标。

该 481 张留出集中有 218 张出现在早期结构实验的 manifest 中，因此不是完全盲测。完全未见的多来源评估仍需新增至少 159 张次元城和 59 张饭团样本。

## 重建与完整性

从本地 append-only 数据重建公开目录：

```powershell
uv run python scripts/build_open_dataset.py
```

脚本要求 3,203 个 metadata ID 与 3,203 个标注文件一一对应，并逐文件校验 SHA-256。`SHA256SUMS` 记录 manifest 自身的哈希；每张图片的哈希位于 manifest。

## 预处理

训练/推理时转换为灰度、resize 到 `96 x 32`，再转为 `float32 [0, 1]` 的 `[1, 32, 96]` 张量。不使用 ImageNet normalization，不改写公开原图。

## 已知偏差与限制

- 只包含三个来源、固定四位数字和 2026-07-15 的采集分布。
- 同一批次内生成风格高度相似，随机 image-level split 可能高估泛化。
- preflight 数据参与过早期探索，不能作为最终盲测。
- 标签虽经人工确认，但尚未进行双人独立复核；发现错标请按 sample ID 提交 issue。

## 隐私、来源与使用边界

数据不包含账户、Cookie、令牌、用户输入或个人身份信息。来源名称只用于科研溯源，不表示相关站点认可本项目。使用者仍需遵守所在地法律和来源站点条款；禁止用于未获授权的访问控制绕过。

数据集按 [`LICENSE.md`](LICENSE.md) 的 CC BY 4.0 条款发布。许可只覆盖发布者有权许可的内容，不授予任何第三方商标或服务名称权利。
