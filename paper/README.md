# 课程论文

主文件为 `main.tex`。`references.bib` 作为参考文献源数据保留；正文引用与文末参考文献已经完整写入 `main.tex`，日常编译无需 BibTeX 或 Biber。

当前版本采用简化研究论文版式：

- 无独立封面；
- 无姓名、学号、课程、学院、日期栏；
- 仅保留中文摘要与关键词；
- 目录仅列一级章节；
- 正文页码置于页脚中央；
- 图表采用常规浮动，并限制在对应章节内；
- 参考文献单独起页；
- 数字引用编号在 `main.tex` 中预注册，单次 XeLaTeX 编译也不会产生 `[?]`。

## 最省事的编译方式

Windows 下直接双击：

```text
compile.bat
```

脚本优先调用 `latexmk`；系统没有 `latexmk` 时，会自动连续运行 XeLaTeX 三次。

## 手动编译

直接运行 XeLaTeX 即可：

```powershell
xelatex -interaction=nonstopmode -file-line-error main.tex
```

引用编号在第一遍即可正常显示。目录、图表编号与交叉引用仍建议再运行一次 XeLaTeX，以刷新页码和位置。

如果目录里残留旧版辅助文件，可先双击 `clean.bat`，再运行 `compile.bat`。
