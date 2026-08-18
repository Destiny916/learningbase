from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Rectangle

FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
BLOCK_LENGTH = 30
BLOCK_B_SUBMIT_STEP = 15
BLOCK_B_INSTALL_STEP = 22
EXPIRED_STEPS = BLOCK_B_INSTALL_STEP - BLOCK_B_SUBMIT_STEP


def _configure_chinese_font() -> None:
    for font_path in (FONT_REGULAR, FONT_BOLD):
        if not font_path.is_file():
            raise FileNotFoundError(f"缺少中文字体：{font_path}")
        font_manager.fontManager.addfont(font_path)
    family = font_manager.FontProperties(fname=FONT_REGULAR).get_name()
    mpl.rcParams["font.family"] = family
    mpl.rcParams["font.sans-serif"] = [family]
    mpl.rcParams["axes.unicode_minus"] = False


def _draw_cell(axis, x: float, y: float, label: str, color: str, *, text_color: str = "#263238") -> None:
    axis.add_patch(Rectangle((x, y), 0.9, 0.72, facecolor=color, edgecolor="white", linewidth=0.8))
    axis.text(x + 0.45, y + 0.36, label, ha="center", va="center", fontsize=7, color=text_color)


def _draw_two_chunks(axis) -> None:
    for index in range(BLOCK_LENGTH):
        _draw_cell(axis, index, 1.12, f"A{index}", "#90caf9")
        block_b_color = "#ef9a9a" if index < EXPIRED_STEPS else "#81c784"
        _draw_cell(axis, BLOCK_B_SUBMIT_STEP + index, 0.08, f"B{index}", block_b_color)

    axis.text(-1.0, 1.48, "动作块 A", ha="right", va="center", fontsize=12, weight="bold")
    axis.text(-1.0, 0.44, "动作块 B", ha="right", va="center", fontsize=12, weight="bold")
    axis.axvline(BLOCK_B_INSTALL_STEP, color="#212121", linestyle="--", linewidth=2)
    axis.text(
        BLOCK_B_INSTALL_STEP + 0.25,
        2.02,
        "B 在这里返回：全局 step=22",
        color="#212121",
        fontsize=11,
        weight="bold",
    )
    axis.annotate(
        "B 在 step=15 按 2 Hz 节奏，使用最新图像和关节状态发起推理\n约 200 ms 后返回，因此 B0～B6 已经过期",
        xy=(BLOCK_B_INSTALL_STEP - 0.3, 0.42),
        xytext=(25.0, -0.75),
        arrowprops={"arrowstyle": "->", "color": "#c62828", "linewidth": 1.8},
        color="#c62828",
        fontsize=10,
    )
    axis.text(
        BLOCK_B_SUBMIT_STEP + EXPIRED_STEPS / 2,
        -0.18,
        "已过期：丢弃",
        ha="center",
        va="top",
        fontsize=10,
        color="#c62828",
        weight="bold",
    )
    axis.set_xlim(-1.3, 45.4)
    axis.set_ylim(-1.15, 2.55)
    axis.set_yticks([])
    axis.set_xticks(np.arange(0, 46, 5))
    axis.set_xlabel("全局控制 step（30 Hz）")
    axis.set_title("① 两个完整的 30 步动作块，按全局时间摆放", loc="left", fontsize=14, weight="bold")
    for spine in axis.spines.values():
        spine.set_visible(False)


def _draw_old_table(axis) -> None:
    control_steps = np.arange(BLOCK_B_INSTALL_STEP, BLOCK_B_INSTALL_STEP + 6)
    row_labels = ("当前控制时刻", "旧式执行", "这个动作本来属于", "时间误差")
    values = (
        [str(step) for step in control_steps],
        [f"B{index}" for index in range(6)],
        [f"step {BLOCK_B_SUBMIT_STEP + index}" for index in range(6)],
        ["落后 7 帧"] * 6,
    )
    colors = ("#eceff1", "#ef9a9a", "#ffebee", "#ffcdd2")
    cell_width = 1.65
    row_height = 0.74
    for row, (label, row_values, color) in enumerate(zip(row_labels, values, colors, strict=True)):
        y = 2.25 - row * row_height
        axis.text(-0.25, y + 0.31, label, ha="right", va="center", fontsize=10, weight="bold")
        for column, value in enumerate(row_values):
            x = column * cell_width
            axis.add_patch(Rectangle((x, y), cell_width - 0.05, 0.62, facecolor=color, edgecolor="white"))
            axis.text(x + (cell_width - 0.05) / 2, y + 0.31, value, ha="center", va="center", fontsize=9)
    axis.text(
        10.1,
        1.45,
        "结果：机器人执行的是过去的计划，\n容易回弹、重复动作或突然改变方向。",
        fontsize=11,
        color="#b71c1c",
        va="center",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#ffebee", "edgecolor": "#ef9a9a"},
    )
    axis.set_xlim(-3.4, 15.6)
    axis.set_ylim(-0.45, 3.15)
    axis.axis("off")
    axis.set_title(
        "② 旧式处理：B 返回后从 B0 开始，整条轨迹落后 7 帧",
        loc="left",
        fontsize=14,
        weight="bold",
        color="#b71c1c",
    )


def _draw_new_table(axis) -> None:
    control_steps = np.arange(BLOCK_B_INSTALL_STEP, BLOCK_B_INSTALL_STEP + 6)
    block_a = [f"A{step}" for step in control_steps]
    block_b = [f"B{step - BLOCK_B_SUBMIT_STEP}" for step in control_steps]
    old_weights = ("75%", "45%", "20%", "20%", "20%", "20%")
    new_weights = ("25%", "55%", "80%", "80%", "80%", "80%")
    rows = (
        ("当前控制时刻", [str(step) for step in control_steps], "#eceff1"),
        ("A 的同一时刻动作", block_a, "#bbdefb"),
        ("B 的同一时刻动作", block_b, "#c8e6c9"),
        (
            "输出公式",
            [
                f"{a}×{wa}+{b}×{wb}"
                for a, wa, b, wb in zip(block_a, old_weights, block_b, new_weights, strict=True)
            ],
            "#fff9c4",
        ),
    )
    cell_width = 2.35
    row_height = 0.74
    for row, (label, row_values, color) in enumerate(rows):
        y = 2.25 - row * row_height
        axis.text(-0.25, y + 0.31, label, ha="right", va="center", fontsize=10, weight="bold")
        for column, value in enumerate(row_values):
            x = column * cell_width
            axis.add_patch(Rectangle((x, y), cell_width - 0.05, 0.62, facecolor=color, edgecolor="white"))
            axis.text(x + (cell_width - 0.05) / 2, y + 0.31, value, ha="center", va="center", fontsize=8.2)
    axis.set_xlim(-3.4, 14.3)
    axis.set_ylim(-0.45, 3.15)
    axis.axis("off")
    axis.set_title(
        "③ 2 Hz Bridge：先对齐同一个全局 step，再用 3 帧完成交接",
        loc="left",
        fontsize=14,
        weight="bold",
        color="#1b5e20",
    )


def _draw_balance(axis) -> None:
    cards = (
        (
            0.0,
            "精度",
            "B0～B6 已过期，直接丢弃\n"
            "只融合 A22 与 B7 这类“同一时刻”的动作\n"
            "最新计划最终占 80%，不会被旧计划淹没",
            "#e3f2fd",
            "#1565c0",
        ),
        (
            7.3,
            "平滑",
            "不是从 A 突然跳到 B\n新块权重按 25% → 55% → 80% 增长\n旧块同步按 75% → 45% → 20% 退出",
            "#e8f5e9",
            "#2e7d32",
        ),
    )
    for x, title, body, facecolor, edgecolor in cards:
        axis.add_patch(
            FancyBboxPatch(
                (x, 0.05),
                6.7,
                1.65,
                boxstyle="round,pad=0.08,rounding_size=0.12",
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=2,
            )
        )
        axis.text(x + 0.3, 1.36, title, fontsize=14, weight="bold", color=edgecolor)
        axis.text(x + 0.3, 1.08, body, fontsize=10.5, va="top", linespacing=1.55)
    axis.text(
        14.6,
        0.86,
        "一句话：\n先把两张路线图的时间刻度对齐，\n再让新路线图逐渐接管方向盘。",
        ha="center",
        va="center",
        fontsize=12,
        weight="bold",
        color="#4a148c",
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#f3e5f5", "edgecolor": "#8e24aa"},
    )
    axis.set_xlim(-0.2, 18.1)
    axis.set_ylim(-0.1, 1.95)
    axis.axis("off")
    axis.set_title("④ 为什么能兼顾精度和平滑？", loc="left", fontsize=14, weight="bold")


def generate(output: Path, *, overwrite: bool = False) -> Path:
    output = output.resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"拒绝覆盖已有图片：{output}")
    _configure_chinese_font()
    figure = Figure(figsize=(19, 14), dpi=160, constrained_layout=True)
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(4, 1, height_ratios=(1.55, 1.2, 1.2, 0.95))
    _draw_two_chunks(figure.add_subplot(grid[0]))
    _draw_old_table(figure.add_subplot(grid[1]))
    _draw_new_table(figure.add_subplot(grid[2]))
    _draw_balance(figure.add_subplot(grid[3]))
    figure.suptitle(
        "两个 30 步 ACT 动作块：旧式执行与 2 Hz Bridge 的区别",
        fontsize=20,
        weight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="生成两个 30 步 ACT 动作块的中文 Bridge 对比图")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("w1_simulation/artifacts/explanations/bridge_two_30_step_chunks_2hz_cn.png"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(f"图片已生成：{generate(args.output, overwrite=args.overwrite)}")


if __name__ == "__main__":
    main()
