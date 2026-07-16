"""人工标注原始验证码图片的小工具。"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from tkinter import BOTH, LEFT, RIGHT, Button, Entry, Frame, Label, StringVar, Tk, messagebox

from PIL import Image, ImageTk

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
LABEL_PATTERN = re.compile(r"^\d{4}$")
LABELED_FILENAME_PATTERN = re.compile(r"^\d{4}_(?P<sample_id>.+)\.[^.]+$")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def iter_raw_images(raw_dir: Path) -> list[Path]:
    """递归列出待标注原图。"""
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"raw dataset directory does not exist: {raw_dir}")
    return sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def labeled_sample_ids(labeled_dir: Path) -> set[str]:
    """读取已经存在于 labeled 目录中的样本 ID。"""
    if not labeled_dir.exists():
        return set()
    sample_ids: set[str] = set()
    for path in labeled_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        match = LABELED_FILENAME_PATTERN.fullmatch(path.name)
        if match is not None:
            sample_ids.add(match.group("sample_id"))
    return sample_ids


def build_labeled_path(labeled_dir: Path, raw_path: Path, label: str) -> Path:
    """生成 ``<四位数字>_<样本ID>.<扩展名>`` 标注文件路径。"""
    if LABEL_PATTERN.fullmatch(label) is None:
        raise ValueError(f"label must be exactly four digits, got {label!r}")
    return labeled_dir / f"{label}_{raw_path.stem}{raw_path.suffix.lower()}"


def unlabeled_raw_images(raw_dir: Path, labeled_dir: Path) -> list[Path]:
    """过滤掉已经标注过的原图。"""
    completed_ids = labeled_sample_ids(labeled_dir)
    return [path for path in iter_raw_images(raw_dir) if path.stem not in completed_ids]


class LabelingApp:
    """逐张显示验证码，并把人工输入复制到 labeled 目录。"""

    def __init__(self, raw_dir: Path, labeled_dir: Path) -> None:
        self.raw_dir = raw_dir
        self.labeled_dir = labeled_dir
        self.images = unlabeled_raw_images(raw_dir, labeled_dir)
        self.index = 0
        self.saved_count = 0
        self.photo: ImageTk.PhotoImage | None = None

        self.root = Tk()
        self.root.title("CNN for Ani 验证码人工标注")
        self.root.geometry("900x520")
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

        self.status_var = StringVar()
        self.path_var = StringVar()
        self.label_var = StringVar()
        self.error_var = StringVar()

        Label(self.root, textvariable=self.status_var, font=("Microsoft YaHei UI", 12)).pack(pady=8)
        Label(self.root, textvariable=self.path_var, wraplength=860).pack(pady=4)
        self.image_label = Label(self.root)
        self.image_label.pack(expand=True, fill=BOTH, padx=12, pady=12)

        input_frame = Frame(self.root)
        input_frame.pack(pady=8)
        Label(input_frame, text="四位数字：").pack(side=LEFT)
        self.entry = Entry(
            input_frame, textvariable=self.label_var, width=10, font=("Consolas", 18)
        )
        self.entry.pack(side=LEFT, padx=8)
        self.entry.bind("<Return>", self.save_current)

        button_frame = Frame(self.root)
        button_frame.pack(pady=6)
        Button(button_frame, text="保存并下一张 (Enter)", command=self.save_current).pack(
            side=LEFT, padx=6
        )
        Button(button_frame, text="跳过", command=self.skip_current).pack(side=LEFT, padx=6)
        Button(button_frame, text="退出", command=self.quit).pack(side=RIGHT, padx=6)

        Label(self.root, textvariable=self.error_var, fg="red").pack(pady=4)
        self.show_current()

    def run(self) -> None:
        self.root.mainloop()

    def current_path(self) -> Path | None:
        if self.index >= len(self.images):
            return None
        return self.images[self.index]

    def show_current(self) -> None:
        path = self.current_path()
        self.label_var.set("")
        self.error_var.set("")
        if path is None:
            self.status_var.set(f"全部完成：本次新增 {self.saved_count} 张标注。")
            self.path_var.set("")
            self.image_label.configure(image="", text="没有待标注图片。")
            self.entry.configure(state="disabled")
            return

        total = len(self.images)
        relative_path = path.relative_to(self.raw_dir)
        self.status_var.set(f"待标注 {self.index + 1}/{total}，本次已保存 {self.saved_count} 张")
        self.path_var.set(str(relative_path))
        self.photo = ImageTk.PhotoImage(self.display_image(path))
        self.image_label.configure(image=self.photo, text="")
        self.entry.configure(state="normal")
        self.entry.focus_set()

    def display_image(self, path: Path) -> Image.Image:
        with Image.open(path) as image:
            display = image.convert("RGB")
        max_width = 840
        max_height = 320
        width, height = display.size
        scale = min(max_width / width, max_height / height)
        scale = max(1, min(8, int(scale)))
        if scale > 1:
            display = display.resize((width * scale, height * scale), Image.Resampling.NEAREST)
        return display

    def save_current(self, _event: object | None = None) -> None:
        path = self.current_path()
        if path is None:
            return
        label = self.label_var.get().strip()
        if LABEL_PATTERN.fullmatch(label) is None:
            self.error_var.set("请输入正好四位数字，例如 0834。")
            self.entry.focus_set()
            return

        self.labeled_dir.mkdir(parents=True, exist_ok=True)
        target = build_labeled_path(self.labeled_dir, path, label)
        if target.exists():
            self.error_var.set(f"目标文件已存在，未覆盖：{target.name}")
            self.entry.focus_set()
            return

        shutil.copy2(path, target)
        self.saved_count += 1
        self.index += 1
        self.show_current()

    def skip_current(self) -> None:
        self.index += 1
        self.show_current()

    def quit(self) -> None:
        if self.saved_count > 0:
            messagebox.showinfo("标注已暂停", f"本次新增 {self.saved_count} 张标注。")
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="逐张打开 raw 图片，并保存人工四位数字标注。")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "raw",
        help="原始验证码目录，默认 dataset/raw，支持递归批次目录。",
    )
    parser.add_argument(
        "--labeled-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "labeled",
        help="标注输出目录，默认 dataset/labeled。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = LabelingApp(args.raw_dir, args.labeled_dir)
    app.run()


if __name__ == "__main__":
    main()
