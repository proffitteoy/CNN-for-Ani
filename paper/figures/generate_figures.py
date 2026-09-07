"""Generate all manuscript figures from repository data with Matplotlib only."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = Path(__file__).resolve().parent
MANIFEST = ROOT / "dataset" / "captcha-v1" / "manifest.csv"
IMAGE_DIR = ROOT / "dataset" / "captcha-v1" / "images"
RESULTS = ROOT / "experiments" / "results.csv"
EVAL_REPORT = ROOT / "models" / "final" / "eval-report.json"

FONT_PATH = Path("C:/Windows/Fonts/msyh.ttc")
if not FONT_PATH.exists():
    raise FileNotFoundError("Microsoft YaHei font is required at C:/Windows/Fonts/msyh.ttc")
font_manager.fontManager.addfont(FONT_PATH)
FONT_NAME = font_manager.FontProperties(fname=FONT_PATH).get_name()

BLUE = "#315B7D"
BLUE_MID = "#6689A6"
BLUE_LIGHT = "#BCD0DE"
ROSE = "#C78686"
ROSE_LIGHT = "#E7C4C4"
GOLD = "#D6A64B"
TEAL = "#5F9792"
INK = "#263238"
MID = "#69767D"
LIGHT = "#EEF2F4"
WHITE = "#FFFFFF"

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_NAME, "Microsoft YaHei", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
    }
)


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save editable vectors and a print-resolution PNG preview."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{name}.svg", bbox_inches="tight", facecolor=WHITE)
    fig.savefig(FIGURE_DIR / f"{name}.pdf", bbox_inches="tight", facecolor=WHITE)
    fig.savefig(FIGURE_DIR / f"{name}.png", dpi=360, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.08, 1.04, label, transform=ax.transAxes, fontsize=9, fontweight="bold")


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = LIGHT,
    edgecolor: str = BLUE,
    fontsize: float = 8,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=1,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
    )


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], **kwargs) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1,
            color=kwargs.get("color", MID),
            connectionstyle=kwargs.get("connectionstyle", "arc3"),
        )
    )


def figure_paradigms() -> None:
    """Compare input segmentation with feature-space positional modelling."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45))
    titles = ["显式字符分割", "通用序列识别", "本文：位置保持整图预测"]
    for ax, title in zip(axes, titles, strict=True):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title, pad=8, fontweight="bold")

    ax = axes[0]
    rounded_box(ax, (0.25, 0.76), 0.5, 0.14, "完整验证码", facecolor=BLUE_LIGHT)
    rounded_box(ax, (0.14, 0.48), 0.72, 0.14, "输入空间切割", facecolor=ROSE_LIGHT, edgecolor=ROSE)
    arrow(ax, (0.5, 0.76), (0.5, 0.63))
    for i in range(4):
        x = 0.06 + i * 0.235
        rounded_box(
            ax,
            (x, 0.18),
            0.19,
            0.13,
            f"字符图 {i + 1}",
            facecolor=WHITE,
            edgecolor=ROSE,
            fontsize=6.6,
        )
        arrow(ax, (0.5, 0.48), (x + 0.095, 0.32))
    ax.text(0.5, 0.05, "边界错误会先于分类器发生", ha="center", color=ROSE, fontsize=7)

    ax = axes[1]
    rounded_box(ax, (0.25, 0.76), 0.5, 0.14, "完整验证码", facecolor=BLUE_LIGHT)
    rounded_box(ax, (0.16, 0.48), 0.68, 0.14, "CNN 特征序列", facecolor=LIGHT)
    rounded_box(
        ax, (0.18, 0.20), 0.64, 0.14, "RNN / CTC / Decoder", facecolor=WHITE, edgecolor=TEAL
    )
    arrow(ax, (0.5, 0.76), (0.5, 0.63))
    arrow(ax, (0.5, 0.48), (0.5, 0.35))
    ax.text(0.5, 0.05, "面向变长与未知对齐", ha="center", color=TEAL, fontsize=7)

    ax = axes[2]
    rounded_box(ax, (0.25, 0.76), 0.5, 0.14, "完整验证码", facecolor=BLUE_LIGHT)
    rounded_box(ax, (0.18, 0.48), 0.64, 0.14, "共享 CNN 特征", facecolor=LIGHT)
    arrow(ax, (0.5, 0.76), (0.5, 0.63))
    for i in range(4):
        x = 0.06 + i * 0.235
        rounded_box(
            ax,
            (x, 0.18),
            0.19,
            0.13,
            f"槽位 {i + 1}\n0--9",
            facecolor=WHITE,
            edgecolor=BLUE,
            fontsize=6.6,
        )
        arrow(ax, (0.5, 0.48), (x + 0.095, 0.32))
    ax.text(
        0.5,
        0.05,
        "特征有位置，输入没有切字",
        ha="center",
        color=BLUE,
        fontsize=7,
        fontweight="bold",
    )

    fig.subplots_adjust(wspace=0.14)
    save_figure(fig, "fig1_paradigms")


def load_manifest() -> list[dict[str, str]]:
    with MANIFEST.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def figure_dataset() -> None:
    """Show traceable raw examples and source composition."""
    rows = load_manifest()
    source_order = ["次元城动画", "饭团动漫", "新优酷"]
    selected: dict[str, list[dict[str, str]]] = {source: [] for source in source_order}
    for row in rows:
        source = row["source"]
        if source in selected and row["purpose"] == "training_raw" and len(selected[source]) < 3:
            selected[source].append(row)

    fig = plt.figure(figsize=(7.2, 4.25))
    grid = fig.add_gridspec(2, 3, height_ratios=[2.35, 1], hspace=0.42, wspace=0.16)
    image_axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    count_ax = fig.add_subplot(grid[1, :])

    for ax, source in zip(image_axes, source_order, strict=True):
        ax.axis("off")
        ax.set_title(source, fontsize=8, fontweight="bold", pad=5)
        for index, row in enumerate(selected[source]):
            image = np.asarray(Image.open(IMAGE_DIR / row["filename"]).convert("L"))
            inset = ax.inset_axes([0.04, 0.69 - index * 0.31, 0.92, 0.24])
            inset.imshow(
                image, cmap="gray", vmin=0, vmax=255, interpolation="nearest", aspect="auto"
            )
            inset.set_xticks([])
            inset.set_yticks([])
            for spine in inset.spines.values():
                spine.set_color("#B4BDC2")
                spine.set_linewidth(0.6)
            inset.text(
                0.99,
                0.04,
                row["label"],
                transform=inset.transAxes,
                ha="right",
                va="bottom",
                fontsize=6.5,
                color=INK,
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": WHITE,
                    "edgecolor": "none",
                    "alpha": 0.82,
                },
            )

    panel_label(image_axes[0], "a")
    source_counts = {source: Counter() for source in source_order}
    for row in rows:
        if row["source"] in source_counts:
            key = "正式采集" if row["purpose"] == "training_raw" else "预检样本"
            source_counts[row["source"]][key] += 1
    y = np.arange(len(source_order))
    formal = np.array([source_counts[s]["正式采集"] for s in source_order])
    preflight = np.array([source_counts[s]["预检样本"] for s in source_order])
    count_ax.barh(y, formal, color=BLUE_MID, height=0.56, label="正式采集")
    count_ax.barh(y, preflight, left=formal, color=GOLD, height=0.56, label="预检样本")
    count_ax.set_yticks(y, source_order)
    count_ax.invert_yaxis()
    count_ax.set_xlabel("图像数量")
    count_ax.set_xlim(0, 1180)
    count_ax.legend(ncol=2, loc="lower right", bbox_to_anchor=(1.0, 1.02))
    count_ax.tick_params(axis="y", length=0)
    for index, total in enumerate(formal + preflight):
        count_ax.text(total + 14, index, f"{total:,}", va="center", fontsize=7, color=INK)
    panel_label(count_ax, "b")
    save_figure(fig, "fig2_dataset")


def figure_architecture() -> None:
    """Render the actual Position-DS stage sequence and independent heads."""
    fig, ax = plt.subplots(figsize=(7.2, 3.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    stages = [
        (0.02, 0.37, 0.115, "完整输入\n1×32×96", BLUE_LIGHT),
        (0.17, 0.37, 0.115, "Stem\n32×32×96", LIGHT),
        (0.32, 0.37, 0.12, "DS-Res ×2\n32×32×96", LIGHT),
        (0.48, 0.37, 0.12, "Down + Res ×2\n48×16×48", LIGHT),
        (0.64, 0.37, 0.12, "Down + Res ×3\n72×8×24", LIGHT),
        (0.80, 0.37, 0.13, "位置池化\n72×1×4", BLUE_LIGHT),
    ]
    for x, y, width, label, colour in stages:
        rounded_box(ax, (x, y), width, 0.22, label, facecolor=colour, fontsize=7.4)
    for left, right in zip(stages[:-1], stages[1:], strict=True):
        arrow(ax, (left[0] + left[2], 0.48), (right[0], 0.48))

    head_x = [0.55, 0.66, 0.77, 0.88]
    for index, x in enumerate(head_x):
        rounded_box(
            ax,
            (x, 0.07),
            0.09,
            0.15,
            f"Head {index + 1}\n72→64→10",
            facecolor=WHITE,
            edgecolor=ROSE if index in (1, 2) else BLUE,
            fontsize=6.6,
        )
        arrow(ax, (0.865, 0.37), (x + 0.045, 0.22), connectionstyle="arc3,rad=0.05")

    ax.text(0.46, 0.76, "共享整图特征提取", ha="center", fontsize=8, color=INK, fontweight="bold")
    ax.plot([0.16, 0.77], [0.71, 0.71], color=BLUE_MID, lw=1.2)
    ax.text(0.84, 0.76, "横向槽位", ha="center", fontsize=8, color=BLUE, fontweight="bold")
    ax.text(0.78, 0.01, "四个分类头独立，前面的感受野共享", ha="center", fontsize=7, color=MID)
    ax.text(0.64, 0.62, "末块横向 dilation (1, 2)", fontsize=6.5, color=TEAL)
    ax.text(0.02, 0.92, "单成员 93,904 参数；三种子模型在 logits 层求平均", fontsize=8, color=INK)
    save_figure(fig, "fig3_architecture")


def figure_evolution() -> None:
    """Plot point estimates while preserving evaluation-protocol boundaries."""
    with RESULTS.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    labels = [
        "A\nFlatten",
        "B\nMedium",
        "C\nPosition",
        "D\nPos.+Aug",
        "E\nPosition",
        "F\nWide Pos.",
        "G*\nAlpha1",
        "H\nPos.-DS",
    ]
    char = np.array([float(row["char_accuracy"]) * 100 for row in rows])
    exact = np.array([float(row["exact_accuracy"]) * 100 for row in rows])
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(7.2, 3.85))
    spans = [
        (-0.45, 3.45, "同快照 5-fold CV", "#F3F5F7"),
        (3.55, 5.45, "时间 / 跨批次留出", "#EAF1F5"),
        (5.55, 6.45, "训练快照", "#F8EFE4"),
        (6.55, 7.45, "多来源留出", "#EDF4EE"),
    ]
    for left, right, title, colour in spans:
        ax.axvspan(left, right, color=colour, zorder=0)
        ax.text((left + right) / 2, 102.5, title, ha="center", va="bottom", fontsize=6.5, color=MID)
    ax.plot(x, char, color=BLUE, marker="o", lw=1.8, ms=5, label="CharAcc")
    ax.plot(x, exact, color=ROSE, marker="s", lw=1.8, ms=4.5, label="ExactAcc")
    for xi, value in zip(x, exact, strict=True):
        ax.text(
            xi,
            value - 5 if value > 12 else value + 4,
            f"{value:.2f}",
            ha="center",
            fontsize=6.4,
            color=ROSE,
        )
    ax.set_xticks(x, labels)
    ax.set_ylabel("准确率（%）")
    ax.set_ylim(-4, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(ncol=2, loc="upper left", bbox_to_anchor=(0.0, 0.98))
    ax.tick_params(axis="x", length=0)
    ax.text(
        0.0,
        -0.22,
        "* G 为训练快照；各背景区的数据划分与训练预算不同，折线表示项目演化，不构成严格同条件排名。",
        transform=ax.transAxes,
        fontsize=6.6,
        color=MID,
    )
    save_figure(fig, "fig4_evolution")


def figure_final_evaluation() -> None:
    """Summarise source robustness, positional errors, and digit confusions."""
    report = json.loads(EVAL_REPORT.read_text(encoding="utf-8"))
    overall = report["ensemble_test"]
    by_source = report["ensemble_test_by_source"]
    sources = ["总体", "新优酷", "次元城", "饭团"]
    source_keys = ["新优酷", "次元城动画", "饭团动漫"]
    char = [overall["char_accuracy"] * 100]
    exact = [overall["exact_accuracy"] * 100]
    for key in source_keys:
        char.append(by_source[key]["char_accuracy"] * 100)
        exact.append(by_source[key]["exact_accuracy"] * 100)
    positions = np.array(overall["position_accuracies"]) * 100
    matrix = np.array(overall["confusion_matrix"], dtype=int)
    off_diagonal = []
    for true_digit in range(10):
        for pred_digit in range(10):
            if true_digit != pred_digit and matrix[true_digit, pred_digit] > 0:
                off_diagonal.append((matrix[true_digit, pred_digit], f"{true_digit}→{pred_digit}"))
    off_diagonal.sort(reverse=True)
    top_pairs = off_diagonal[:6]

    fig = plt.figure(figsize=(7.2, 6.25))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.25], hspace=0.48, wspace=0.34)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    x = np.arange(len(sources))
    width = 0.34
    bars1 = ax_a.bar(x - width / 2, char, width, color=BLUE, label="CharAcc")
    bars2 = ax_a.bar(x + width / 2, exact, width, color=ROSE, label="ExactAcc")
    ax_a.set_xticks(x, sources)
    ax_a.set_ylim(93, 100.2)
    ax_a.set_yticks([94, 96, 98, 100])
    ax_a.set_ylabel("准确率（%）")
    ax_a.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax_a.tick_params(axis="x", length=0)
    for bars in (bars1, bars2):
        for bar in bars:
            ax_a.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.12,
                f"{bar.get_height():.2f}",
                ha="center",
                fontsize=5.8,
            )
    panel_label(ax_a, "a")

    bars = ax_b.bar(np.arange(4), positions, color=[BLUE_MID, ROSE, BLUE, BLUE_MID], width=0.62)
    ax_b.set_xticks(np.arange(4), ["第1位", "第2位", "第3位", "第4位"])
    ax_b.set_ylim(97.5, 100.0)
    ax_b.set_yticks([98, 99, 100])
    ax_b.set_ylabel("位置准确率（%）")
    ax_b.tick_params(axis="x", length=0)
    for bar, value in zip(bars, positions, strict=True):
        ax_b.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.06,
            f"{value:.2f}",
            ha="center",
            fontsize=6.3,
        )
    panel_label(ax_b, "b")

    image = ax_c.imshow(matrix, cmap="Blues", aspect="equal")
    ax_c.set_xticks(range(10))
    ax_c.set_yticks(range(10))
    ax_c.set_xlabel("预测数字")
    ax_c.set_ylabel("真实数字")
    ax_c.tick_params(length=0)
    for i in range(10):
        for j in range(10):
            value = matrix[i, j]
            if value > 0:
                colour = WHITE if value > matrix.max() * 0.55 else INK
                ax_c.text(j, i, str(value), ha="center", va="center", fontsize=5.2, color=colour)
    colourbar = fig.colorbar(image, ax=ax_c, fraction=0.046, pad=0.04)
    colourbar.set_label("字符数", fontsize=7)
    colourbar.ax.tick_params(labelsize=6)
    panel_label(ax_c, "c")

    counts = [item[0] for item in top_pairs][::-1]
    pair_labels = [item[1] for item in top_pairs][::-1]
    y = np.arange(len(counts))
    ax_d.barh(y, counts, color=ROSE, height=0.56)
    ax_d.set_yticks(y, pair_labels)
    ax_d.set_xlabel("错误字符数")
    ax_d.set_xticks(range(0, max(counts) + 2, 2))
    for yi, value in zip(y, counts, strict=True):
        ax_d.text(value + 0.12, yi, str(value), va="center", fontsize=6.5)
    ax_d.set_title("23 个字符错误中的主要混淆", loc="left", fontsize=7.5, color=INK, pad=4)
    panel_label(ax_d, "d")

    save_figure(fig, "fig5_final_evaluation")


def main() -> None:
    figure_paradigms()
    figure_dataset()
    figure_architecture()
    figure_evolution()
    figure_final_evaluation()


if __name__ == "__main__":
    main()
