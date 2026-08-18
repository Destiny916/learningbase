from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, Rectangle

DEFAULT_TRAJECTORY = Path(
    "w1_simulation/artifacts/inference/act_sim_trajectory_bridge_temporal_200ms_full_20260810.npz"
)
ACT_JOINT_NAMES = (
    "WAIST",
    "LEFT_J1",
    "LEFT_J2",
    "LEFT_J3",
    "LEFT_J4",
    "LEFT_J5",
    "LEFT_J6",
    "LEFT_J7",
    "NECK1",
    "NECK2",
    "RIGHT_J1",
    "RIGHT_J2",
    "RIGHT_J3",
    "RIGHT_J4",
    "RIGHT_J5",
    "RIGHT_J6",
    "RIGHT_J7",
    "LEFT_GRIPPER",
    "RIGHT_GRIPPER",
)


def _required_array(trajectory: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key not in trajectory:
        raise KeyError(f"Trajectory is missing required array: {key}")
    return np.asarray(trajectory[key])


def _joint_index(value: str | None) -> int | None:
    if value is None:
        return None
    normalized = value.upper()
    if normalized in ACT_JOINT_NAMES:
        return ACT_JOINT_NAMES.index(normalized)
    try:
        index = int(value)
    except ValueError as exc:
        raise ValueError(f"Unknown ACT joint: {value}") from exc
    if not 0 <= index < len(ACT_JOINT_NAMES):
        raise ValueError(f"ACT joint index must be in [0, 18], got {index}")
    return index


def _select_event(
    chunks: np.ndarray,
    submit_steps: np.ndarray,
    install_steps: np.ndarray,
    record_index: int | None,
    joint_index: int | None,
) -> tuple[int, int]:
    if record_index is not None:
        if not 0 < record_index < len(chunks):
            raise ValueError(f"record-index must be in [1, {len(chunks) - 1}]")
        lag = int(install_steps[record_index] - submit_steps[record_index])
        if not 0 < lag < chunks.shape[1]:
            raise ValueError(f"Selected record has no usable delayed prefix: lag={lag}")
        if joint_index is None:
            differences = np.abs(chunks[record_index, 0, :17] - chunks[record_index, lag, :17])
            joint_index = int(np.argmax(differences))
        return record_index, joint_index

    best_score = -np.inf
    best_event: tuple[int, int] | None = None
    joint_indices = range(17) if joint_index is None else (joint_index,)
    for candidate in range(1, len(chunks)):
        lag = int(install_steps[candidate] - submit_steps[candidate])
        if not 0 < lag < chunks.shape[1]:
            continue
        for index in joint_indices:
            score = abs(float(chunks[candidate, 0, index] - chunks[candidate, lag, index]))
            if score > best_score:
                best_score = score
                best_event = (candidate, index)
    if best_event is None:
        raise ValueError("Trajectory does not contain a delayed installed chunk")
    return best_event


def _draw_timeline(axis, lag: int, *, old: bool) -> None:
    visible = min(max(lag + 6, 12), 18)
    colors = ("#90caf9", "#ef9a9a") if old else ("#81c784", "#eeeeee")
    for index in range(visible):
        expired = index < lag
        color = colors[int(expired)]
        axis.add_patch(Rectangle((index, 0), 0.92, 0.75, facecolor=color, edgecolor="white"))
        axis.text(index + 0.46, 0.38, str(index), ha="center", va="center", fontsize=9)
    axis.axvline(lag - 0.04, color="#263238", linestyle="--", linewidth=1.5)
    if old:
        axis.add_patch(
            FancyArrowPatch(
                (lag + 0.15, 1.25),
                (0.45, 0.82),
                arrowstyle="-|>",
                mutation_scale=15,
                color="#c62828",
                linewidth=2,
            )
        )
        axis.text(lag + 0.2, 1.28, "arrival -> restart at chunk[0]", color="#c62828", fontsize=10)
        axis.set_title("OLD FIFO: replay the expired prefix", loc="left", weight="bold")
    else:
        axis.add_patch(
            FancyArrowPatch(
                (lag + 2.8, 1.25),
                (lag + 0.45, 0.82),
                arrowstyle="-|>",
                mutation_scale=15,
                color="#2e7d32",
                linewidth=2,
            )
        )
        axis.text(lag + 2.9, 1.28, f"arrival -> start at chunk[{lag}]", color="#2e7d32", fontsize=10)
        axis.text(lag / 2, -0.28, "discarded", color="#616161", ha="center", fontsize=9)
        axis.set_title("NEW BRIDGE: jump to the action for the current step", loc="left", weight="bold")
    axis.set_xlim(-0.2, visible + 0.3)
    axis.set_ylim(-0.45, 1.65)
    axis.set_yticks([])
    axis.set_xlabel("action index inside the 30-step ACT chunk")
    for spine in axis.spines.values():
        spine.set_visible(False)


def _record_weight(
    record_index: int,
    record_rows: np.ndarray,
    weights: np.ndarray,
    steps: np.ndarray,
) -> np.ndarray:
    values = np.zeros(len(steps), dtype=np.float32)
    for output_index, step in enumerate(steps):
        matches = np.flatnonzero(record_rows[step] == record_index)
        if len(matches):
            values[output_index] = float(weights[step, matches[0]])
    return values


def generate_explanation(
    trajectory_path: Path,
    output_path: Path,
    *,
    record_index: int | None = None,
    joint: str | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    trajectory_path = trajectory_path.resolve()
    output_path = output_path.resolve()
    if not trajectory_path.is_file():
        raise FileNotFoundError(trajectory_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing explanation: {output_path}")

    with np.load(trajectory_path, allow_pickle=False) as trajectory:
        chunks = _required_array(trajectory, "processed_candidate_chunks").astype(np.float32)
        submit_steps = _required_array(trajectory, "chunk_submit_step").astype(np.int32)
        install_steps = _required_array(trajectory, "chunk_install_step").astype(np.int32)
        emitted = _required_array(trajectory, "act_action").astype(np.float32)
        ensemble_records = _required_array(trajectory, "ensemble_record_indices").astype(np.int32)
        ensemble_weights = _required_array(trajectory, "ensemble_weights").astype(np.float32)

    if chunks.ndim != 3 or chunks.shape[1:] != (30, 19):
        raise ValueError(f"Expected ACT chunk history shaped (N, 30, 19), got {chunks.shape}")
    selected_record, selected_joint = _select_event(
        chunks,
        submit_steps,
        install_steps,
        record_index,
        _joint_index(joint),
    )
    submit_step = int(submit_steps[selected_record])
    install_step = int(install_steps[selected_record])
    lag = install_step - submit_step
    end_step = min(install_step + 6, len(emitted) - 1, submit_step + chunks.shape[1] - 1)
    display_steps = np.arange(max(0, install_step - 3), end_step + 1, dtype=np.int32)
    relative_steps = display_steps - install_step

    old_fifo = np.full(len(display_steps), np.nan, dtype=np.float32)
    aligned = np.full(len(display_steps), np.nan, dtype=np.float32)
    for output_index, step in enumerate(display_steps):
        old_index = int(step - install_step)
        aligned_index = int(step - submit_step)
        if 0 <= old_index < chunks.shape[1]:
            old_fifo[output_index] = chunks[selected_record, old_index, selected_joint]
        if 0 <= aligned_index < chunks.shape[1]:
            aligned[output_index] = chunks[selected_record, aligned_index, selected_joint]

    selected_weight = _record_weight(
        selected_record,
        ensemble_records,
        ensemble_weights,
        display_steps,
    )
    other_weight = np.clip(1.0 - selected_weight, 0.0, 1.0)

    figure = Figure(figsize=(15, 11), dpi=160, constrained_layout=True)
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(4, 1, height_ratios=(1.0, 1.0, 1.35, 1.65))
    old_axis = figure.add_subplot(grid[0])
    new_axis = figure.add_subplot(grid[1])
    weight_axis = figure.add_subplot(grid[2])
    action_axis = figure.add_subplot(grid[3])

    _draw_timeline(old_axis, lag, old=True)
    _draw_timeline(new_axis, lag, old=False)

    weight_axis.plot(relative_steps, selected_weight, "o-", color="#2e7d32", label="selected new chunk")
    weight_axis.plot(relative_steps, other_weight, "o-", color="#546e7a", label="all older chunks")
    weight_axis.axvline(0, color="#263238", linestyle="--", linewidth=1.2)
    weight_axis.set_ylim(-0.05, 1.05)
    weight_axis.set_ylabel("weight")
    weight_axis.set_xlabel("control step relative to chunk installation")
    weight_axis.set_title("Soft handoff: 0.25 -> 0.55 -> 0.80", loc="left", weight="bold")
    weight_axis.grid(alpha=0.25)
    weight_axis.legend(loc="best")

    action_axis.plot(
        relative_steps,
        emitted[display_steps, selected_joint],
        "o-",
        color="#1565c0",
        linewidth=2.4,
        label="actual temporal-ensemble output",
    )
    action_axis.plot(
        relative_steps,
        aligned,
        "s--",
        color="#2e7d32",
        linewidth=1.6,
        label=f"time-aligned chunk[{lag}+k]",
    )
    action_axis.plot(
        relative_steps,
        old_fifo,
        "x--",
        color="#c62828",
        linewidth=1.6,
        markersize=8,
        label="old FIFO restart chunk[k]",
    )
    action_axis.axvline(0, color="#263238", linestyle="--", linewidth=1.2)
    action_axis.set_xlabel("control step relative to chunk installation")
    action_axis.set_ylabel("ACT action")
    action_axis.set_title(
        f"Real trajectory example: {ACT_JOINT_NAMES[selected_joint]} | "
        f"submit={submit_step}, install={install_step}, expired={lag} steps",
        loc="left",
        weight="bold",
    )
    action_axis.grid(alpha=0.25)
    action_axis.legend(loc="best")

    figure.suptitle(
        "ACT bridge: restart stale actions vs align and blend the current-time actions",
        fontsize=16,
        weight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    return {
        "trajectory": str(trajectory_path),
        "output": str(output_path),
        "record_index": selected_record,
        "joint": ACT_JOINT_NAMES[selected_joint],
        "submit_step": submit_step,
        "install_step": install_step,
        "discarded_prefix_steps": lag,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize old FIFO and timestamp-aligned ACT bridge handoff"
    )
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("w1_simulation/artifacts/explanations/bridge_old_vs_new_200ms.png"),
    )
    parser.add_argument("--record-index", type=int)
    parser.add_argument("--joint", help="ACT joint name or index; omitted selects the clearest body joint")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result = generate_explanation(
        args.trajectory,
        args.output,
        record_index=args.record_index,
        joint=args.joint,
        overwrite=args.overwrite,
    )
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
