from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")


def _configure_chinese_font() -> None:
    for font_path in (FONT_REGULAR, FONT_BOLD):
        if not font_path.is_file():
            raise FileNotFoundError(f"缺少中文字体：{font_path}")
        font_manager.fontManager.addfont(font_path)
    family = font_manager.FontProperties(fname=FONT_REGULAR).get_name()
    mpl.rcParams["font.family"] = family
    mpl.rcParams["font.sans-serif"] = [family]
    mpl.rcParams["axes.unicode_minus"] = False


def _box(axis, x: float, y: float, width: float, text: str, color: str) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            0.68,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor=color,
            edgecolor="white",
        )
    )
    axis.text(x + width / 2, y + 0.34, text, ha="center", va="center", fontsize=9)


def _draw_rate_split(axis) -> None:
    axis.text(0.0, 2.45, "相机：固定 30 Hz", fontsize=13, weight="bold", color="#1565c0")
    axis.text(0.0, 1.12, "动作：sample_factor=2 → 60 Hz", fontsize=13, weight="bold", color="#2e7d32")
    for index in range(5):
        x = index * 2.2
        _box(axis, x, 1.68, 1.75, f"图像 I{index}\n只读取一次", "#bbdefb")
        for substep in range(2):
            _box(axis, x + substep * 0.92, 0.35, 0.82, f"动作\n{index * 2 + substep}", "#c8e6c9")
        axis.add_patch(
            FancyArrowPatch(
                (x + 0.88, 1.68),
                (x + 0.88, 1.05),
                arrowstyle="-[,widthB=1.45",
                mutation_scale=14,
                color="#607d8b",
            )
        )
    axis.text(
        11.3,
        1.22,
        "一张图像供两个动作周期使用\n图像没有插帧、复制读取或改帧率",
        fontsize=11,
        va="center",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#fff8e1", "edgecolor": "#ffb300"},
    )
    axis.set_xlim(-0.2, 15.2)
    axis.set_ylim(0.0, 2.9)
    axis.axis("off")
    axis.set_title("① 图像频率与动作消费频率彻底解耦", loc="left", fontsize=15, weight="bold")


def _draw_chunk_expansion(axis) -> None:
    for index in range(30):
        axis.add_patch(Rectangle((index, 1.4), 0.86, 0.62, facecolor="#90caf9", edgecolor="white"))
        if index % 5 == 0 or index == 29:
            axis.text(index + 0.43, 1.71, f"A{index}", ha="center", va="center", fontsize=7)
    for index in range(60):
        axis.add_patch(Rectangle((index / 2, 0.25), 0.43, 0.62, facecolor="#a5d6a7", edgecolor="white"))
        if index % 10 == 0 or index == 59:
            axis.text(index / 2 + 0.21, 0.56, f"A′{index}", ha="center", va="center", fontsize=6.5)
    axis.annotate(
        "线性插值，只增加动作采样点",
        xy=(15.0, 1.12),
        xytext=(20.0, 2.45),
        arrowprops={"arrowstyle": "->", "linewidth": 1.8, "color": "#6a1b9a"},
        fontsize=11,
        color="#6a1b9a",
        weight="bold",
    )
    axis.text(-0.8, 1.71, "ACT 原始块", ha="right", va="center", fontsize=11, weight="bold")
    axis.text(-0.8, 0.56, "处理后动作块", ha="right", va="center", fontsize=11, weight="bold")
    axis.text(
        31.0,
        1.15,
        "30步@30Hz = 1秒\n60步@60Hz = 仍然1秒",
        fontsize=11,
        va="center",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#e8f5e9", "edgecolor": "#43a047"},
    )
    axis.set_xlim(-4.5, 37.0)
    axis.set_ylim(-0.1, 2.85)
    axis.axis("off")
    axis.set_title("② 一个 30 步 ACT 块如何变成 60 个控制点", loc="left", fontsize=15, weight="bold")


def _draw_replan(axis) -> None:
    submit_step = 30
    install_step = 42
    for step in range(61):
        color = "#90caf9" if step < install_step else "#bbdefb"
        axis.add_patch(Rectangle((step, 1.45), 0.88, 0.55, facecolor=color, edgecolor="white"))
    for index in range(60):
        step = submit_step + index
        color = "#ef9a9a" if step < install_step else "#81c784"
        axis.add_patch(Rectangle((step, 0.35), 0.88, 0.55, facecolor=color, edgecolor="white"))
    axis.axvline(submit_step, color="#1565c0", linestyle="--", linewidth=1.8)
    axis.axvline(install_step, color="#212121", linestyle="--", linewidth=1.8)
    axis.text(submit_step + 0.4, 2.23, "step 30：第2次推理\n= 第15张图像 = 0.5秒", fontsize=10)
    axis.text(install_step + 0.4, 2.23, "约200ms后返回\nB0～B11过期", fontsize=10, color="#c62828")
    axis.text(20.0, 1.72, "旧块 A", fontsize=11, weight="bold")
    axis.text(20.0, 0.62, "新块 B", fontsize=11, weight="bold")
    axis.text(
        64.0,
        1.1,
        "对齐同一目标时刻：A42 ↔ B12\n随后6个60Hz动作步平滑交接\n权重：25%→40%→55%→67.5%→80%→80%",
        fontsize=10.5,
        va="center",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f3e5f5", "edgecolor": "#8e24aa"},
    )
    axis.set_xlim(0, 92)
    axis.set_ylim(0.0, 2.9)
    axis.set_yticks([])
    axis.set_xticks(np.arange(0, 91, 10))
    axis.set_xlabel("全局动作控制 step（60 Hz）")
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_title(
        "③ 两个30步原始块：2 Hz推理在sample_factor=2下如何衔接", loc="left", fontsize=15, weight="bold"
    )


def _draw_formula(axis) -> None:
    cards = (
        (0.0, "动作频率", "30 × sample_factor\nS=1 → 30 Hz\nS=2 → 60 Hz", "#e3f2fd"),
        (4.7, "推理间隔", "round(动作频率 ÷ 推理Hz)\nS=1, 2Hz → 15步\nS=2, 2Hz → 30步", "#e8f5e9"),
        (9.4, "动作块时长", "30×S 个动作点\n消费频率也是30×S\n所以始终覆盖约1秒", "#fff8e1"),
    )
    for x, title, body, color in cards:
        axis.add_patch(
            FancyBboxPatch(
                (x, 0.15),
                4.25,
                1.65,
                boxstyle="round,pad=0.08,rounding_size=0.12",
                facecolor=color,
                edgecolor="#607d8b",
            )
        )
        axis.text(x + 0.25, 1.45, title, fontsize=12, weight="bold")
        axis.text(x + 0.25, 1.15, body, fontsize=10, va="top", linespacing=1.45)
    axis.text(
        14.2,
        0.95,
        "关键：调整S时保持 BRIDGE_REPLAN_HZ 不变，\n程序会自动按新的动作频率换算控制步间隔。",
        fontsize=11,
        weight="bold",
        color="#1b5e20",
        va="center",
    )
    axis.set_xlim(-0.2, 19.3)
    axis.set_ylim(0.0, 2.15)
    axis.axis("off")
    axis.set_title("④ 配置关系", loc="left", fontsize=15, weight="bold")


def generate(output: Path, *, overwrite: bool = False) -> Path:
    output = output.resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"拒绝覆盖已有图片：{output}")
    _configure_chinese_font()
    figure = Figure(figsize=(19, 14), dpi=160, constrained_layout=True)
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(4, 1, height_ratios=(1.05, 1.05, 1.15, 0.9))
    _draw_rate_split(figure.add_subplot(grid[0]))
    _draw_chunk_expansion(figure.add_subplot(grid[1]))
    _draw_replan(figure.add_subplot(grid[2]))
    _draw_formula(figure.add_subplot(grid[3]))
    figure.suptitle("2 Hz Bridge 自适应 sample_factor：图像不变，动作加密", fontsize=20, weight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="生成2 Hz Bridge适配sample_factor的中文说明图")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("w1_simulation/artifacts/explanations/bridge_sample_factor_2_cn.png"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(f"图片已生成：{generate(args.output, overwrite=args.overwrite)}")


if __name__ == "__main__":
    main()
