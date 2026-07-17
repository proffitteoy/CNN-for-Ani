# 参与贡献

感谢参与 CNN for Ani。请保持固定四位数字、`[B, 1, 32, 96] -> [B, 4, 10]` 的当前实验边界；改变任务定义、预处理或权重格式前请先发 issue 讨论。

## 本地检查

```powershell
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
uv sync --python 3.12
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

模型、预处理、标签解析、数据划分或导出格式的修改必须补充相应测试。实验结果必须说明数据来源、样本数、split、随机种子和指标性质；不得把训练快照或 smoke 结果表述为独立准确率。

## 数据贡献

- `dataset/raw/` 只增不改，不提交模型猜测标签。
- 标注文件命名为 `<四位数字>_<样本ID>.<扩展名>`。
- 新批次需要提供来源、批次、用途、尺寸和 SHA-256 元数据。
- 提交数据即表示贡献者有权按数据集许可发布该内容，并已去除账户、Cookie、令牌和个人信息。

提交 pull request 时请写明修改内容、验证命令、真实数据验证范围和剩余风险。
