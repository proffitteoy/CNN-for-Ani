# 实验记录

[`results.csv`](results.csv) 是面向论文复核和模型卡引用的精简指标索引；原始开放报告位于 `models/`，本地完整 epoch 历史仍由被忽略的 `artifacts/` 保存。

## 结论链

1. 32 张真实样本过拟合达到 100% Train ExactAcc，确认数据解析、损失和反向传播链路可学习。
2. 746 张同快照消融中，全局 flatten/medium 头明显弱于保留横向位置的 Position 头；light augmentation 将 Position CV ExactAcc 从 1.47% 提升到 5.50%。
3. 250 张时间留出上，4,218 参数 Position 达到 19.60% ExactAcc；扩大到 40,394 参数 Wide Position 后，在 184 张新优酷跨批次留出上达到 50.00%。
4. 三个 Wide Position 成员组成 alpha1。其 72.06% ExactAcc 来自 2,477 张训练快照，只能证明拟合能力。
5. 最终阶段改用 93,904 参数 Position-DS、来源均衡 481 张留出和分来源报告。训练尚未完成，因此表中不填写最终准确率。

## 指标解释

- `5-fold CV`：结构比较信号；早期单批次条件下可能存在相似样本泄漏。
- `temporal/cross-batch holdout`：比随机 CV 更接近分布迁移，但仍只覆盖有限来源。
- `training snapshot`：不能用于部署结论。
- `multi-source holdout`：当前最终发布依据，但其中 218 张曾参与早期方案选择，不是完全盲测。

所有百分比均来自当前仓库报告或冻结 README 记录，没有用 smoke run、合成数据或训练中间日志替代最终结果。
