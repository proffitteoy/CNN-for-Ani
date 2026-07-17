# 最终 Position-DS 模型卡

> 状态：训练和多来源评估中，尚未发布权重。此文件是冻结的发布契约，不是完成声明。

## 预定发布内容

最终目录完成时应包含：

- `captcha.onnx`：在全部 3,203 张冻结快照上生产重训的三成员集成；
- `eval-report.json`：481 张多来源留出集的总体与分来源指标；
- `production-report.json`：全量重训配置和成员信息，不伪造生产训练集“准确率”；
- `split.json`：2,447/275/481 的冻结划分；
- `manifest.json`：全量生产训练快照；
- `release.json`：发布版本、门槛、指标摘要和 SHA-256。

## 架构

每个成员是 93,904 参数的 `PositionDSCaptchaCNN`：32/48/72 通道 depthwise-separable residual stages、两次 stride-2 下采样、一个横向 dilation `(1, 2)` block、`AdaptiveAvgPool2d((1, 4))` 和四个独立 `72 -> 64 -> 10` 分类头。三个 seed `42 / 3407 / 20260716` 的 logits 平均后输出 `[B, 4, 10]`，集成参数共 281,712。

完整训练超参数和数据划分见根目录 README。

## 发布门槛

- eval 集成 ExactAcc@1 必须 `>= 60%`，理想值 `>= 65%`；
- 必须报告 CharAcc、ExactAcc@1/2/3/5、HammingError、逐位置准确率、数字 precision/recall、混淆矩阵和分来源指标；
- ONNX checker 通过，真实图片 PyTorch/ONNX logits 最大绝对误差 `< 1e-4`；
- eval 报告必须保留“测试集中 218 张参与过早期方案选择”的限制；
- 生产报告的准确率来源必须指向 eval 报告。

最终训练完成后，由仓库维护者上传上述文件并填写真实评估指标。在文件真实生成并复核前，README 不填写最终准确率或声称模型已发布。
