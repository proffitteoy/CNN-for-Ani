# 实验阶段模型

本目录保存对最终架构选择有直接证据价值的两代单模型产物，而不是所有临时 checkpoint。

| 目录 | 模型 | 数据划分 | CharAcc | ExactAcc | 说明 |
| --- | --- | --- | ---: | ---: | --- |
| [`position/`](position/) | Position CNN，4,218 参数 | 1,618 train / 250 时间留出 | 65.90% | 19.60% | 证明横向位置头优于全局头 |
| [`wide-position/`](wide-position/) | Wide Position CNN，40,394 参数 | 2,000 train / 184 新优酷跨批次留出 | 82.34% | 50.00% | 证明容量是主要瓶颈，并成为 alpha1 成员架构 |

每个目录包含 ONNX、原始报告、split 和导出校验。路径中的 `final_*` 是当时实验脚本的命名，不表示现在的最终发布模型。

这些模型仅用于复现研究演进；部署时优先使用 `models/captcha-alpha1/`，最终 Position-DS 发布后再迁移到 `models/final/`。
