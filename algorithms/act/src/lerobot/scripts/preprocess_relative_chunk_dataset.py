#!/usr/bin/env python

"""Create PI05-ready relative-state, anchored-action LeRobot datasets."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from copy import deepcopy
from pathlib import Path

import datasets
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import get_feature_stats
from lerobot.datasets.feature_utils import get_hf_features_from_features
from lerobot.datasets.io_utils import write_stats, write_table_one_row_group_per_episode

GRIPPER_INDICES = (6, 13)
DEFAULT_REPO_ID = "local/0704_bread_grasp_only_songling_robot_relative_chunk20"
DEFAULT_CHUNK_SIZE = 20
OBJECT_CENTER_STATE_NAMES = [
    "bread_x_m",
    "bread_y_m",
    "bread_z_m",
    "bowl_x_m",
    "bowl_y_m",
    "bowl_z_m",
]

logger = logging.getLogger(__name__)


def make_state_with_object_centers_features(source_features: dict) -> dict:
    """Append bread/bowl metric centers to a standard dual-arm 14D state."""
    if "observation.state" not in source_features or "action" not in source_features:
        raise ValueError("Source features must contain observation.state and action")
    if tuple(source_features["observation.state"].get("shape", ())) != (14,):
        raise ValueError("observation.state must have shape [14]")

    output_features = deepcopy(source_features)
    output_features["observation.state"]["shape"] = [20]
    names = list(output_features["observation.state"].get("names") or [])
    if len(names) != 14:
        raise ValueError("observation.state must provide exactly 14 names")
    output_features["observation.state"]["names"] = names + OBJECT_CENTER_STATE_NAMES
    return output_features


def make_relative_output_features(source_features: dict, chunk_size: int) -> dict:
    """Return the source schema with a fixed-horizon action feature."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    required = {"observation.state", "action"}
    missing = required - source_features.keys()
    if missing:
        raise ValueError(f"Source features are missing: {sorted(missing)}")
    if tuple(source_features["observation.state"].get("shape", ())) != (14,):
        raise ValueError("observation.state must have shape [14]")
    if tuple(source_features["action"].get("shape", ())) != (14,):
        raise ValueError("action must have shape [14]")

    output_features = deepcopy(source_features)
    output_features["action"]["shape"] = [chunk_size, 14]
    return output_features


def make_relative_state_and_action_chunks(
    absolute_states: np.ndarray,
    episode_lengths: list[int],
    chunk_size: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Build relative states and future targets without crossing episode boundaries.

    Arm state dimensions are q_t - q_{t-1}, with the first arm state in each
    episode set to zero. Gripper dimensions remain absolute. Action row t
    contains q_{t+k} - q_t for arm dimensions and absolute q_{t+k} grippers,
    for k in [1, chunk_size]; endpoint values are repeated at episode tails.
    """
    states = np.asarray(absolute_states)
    if states.ndim != 2 or states.shape[1] != 14:
        raise ValueError(f"absolute_states must have shape [N, 14], got {states.shape}")
    if not np.issubdtype(states.dtype, np.floating):
        raise ValueError(f"absolute_states must be floating point, got {states.dtype}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if any(length <= 0 for length in episode_lengths):
        raise ValueError("episode_lengths must contain only positive lengths")
    if sum(episode_lengths) != len(states):
        raise ValueError(
            f"episode lengths total {sum(episode_lengths)} does not match {len(states)} state rows"
        )

    arm_mask = np.ones(states.shape[1], dtype=bool)
    arm_mask[list(GRIPPER_INDICES)] = False
    relative_states = states.copy()
    action_chunks = np.empty((len(states), chunk_size, states.shape[1]), dtype=states.dtype)

    episode_start = 0
    horizon_offsets = np.arange(1, chunk_size + 1)
    for episode_length in episode_lengths:
        episode_stop = episode_start + episode_length
        episode_states = states[episode_start:episode_stop]

        relative_states[episode_start, arm_mask] = 0.0
        if episode_length > 1:
            relative_states[episode_start + 1 : episode_stop, arm_mask] = (
                episode_states[1:, arm_mask] - episode_states[:-1, arm_mask]
            )

        local_indices = np.arange(episode_length)[:, None]
        future_indices = np.minimum(local_indices + horizon_offsets, episode_length - 1)
        future_states = episode_states[future_indices]
        episode_actions = future_states.copy()
        episode_actions[..., arm_mask] -= episode_states[:, None, arm_mask]
        action_chunks[episode_start:episode_stop] = episode_actions
        episode_start = episode_stop

    return relative_states, action_chunks


def make_relative_action_stats(action_chunks: np.ndarray) -> dict[str, np.ndarray]:
    """Compute one 14D statistic vector over every future target in every chunk."""
    chunks = np.asarray(action_chunks)
    if chunks.ndim != 3 or chunks.shape[-1] != 14:
        raise ValueError(f"action_chunks must have shape [N, chunk_size, 14], got {chunks.shape}")
    return get_feature_stats(chunks.reshape(-1, chunks.shape[-1]), axis=0, keepdims=False)


def make_preprocessing_summary(
    *,
    source: str | Path,
    output: str | Path,
    chunk_size: int,
    total_episodes: int,
    total_frames: int,
    inherited_preprocessing: dict | list,
) -> dict:
    """Describe the relative transformation applied to a derived dataset."""
    return {
        "source": str(source),
        "output": str(output),
        "mode": "relative_state_and_anchored_action_chunk",
        "chunk_size": chunk_size,
        "state": {
            "arm": "q_t - q_(t-1)",
            "first_frame_arm": "0",
            "gripper": "absolute q_t",
            "gripper_indices": list(GRIPPER_INDICES),
        },
        "action": {
            "arm": "q_(t+k) - q_t for k=1..chunk_size",
            "gripper": "absolute q_(t+k) for k=1..chunk_size",
            "tail": "repeat each episode's final q",
        },
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "inherited_preprocessing": inherited_preprocessing,
    }


def _load_json(path: Path) -> dict | list:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)
        file.write("\n")


def _features_for_hf(features: dict) -> dict:
    output = deepcopy(features)
    for feature in output.values():
        if "shape" in feature:
            feature["shape"] = tuple(feature["shape"])
    return output


def _read_source(source: Path) -> tuple[dict, pa.Table, list[dict], np.ndarray]:
    info_path = source / "meta/info.json"
    data_path = source / "data/chunk-000/file-000.parquet"
    episodes_path = source / "meta/episodes/chunk-000/file-000.parquet"
    missing = [str(path) for path in (info_path, data_path, episodes_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Source dataset is incomplete; missing: {missing}")

    info = _load_json(info_path)
    data_table = pq.read_table(data_path)
    if "observation.state" not in data_table.column_names or "action" not in data_table.column_names:
        raise ValueError("Source parquet must contain observation.state and action")
    states = np.asarray(data_table["observation.state"].to_pylist(), dtype=np.float32)
    episodes = pq.read_table(episodes_path).to_pylist()
    if len(states) != info["total_frames"]:
        raise ValueError("Source parquet row count does not match meta/info.json")
    if len(episodes) != info["total_episodes"]:
        raise ValueError("Source episode count does not match meta/info.json")
    return info, data_table, episodes, states


def _episode_lengths(episodes: list[dict], total_frames: int) -> list[int]:
    lengths = [int(episode["length"]) for episode in episodes]
    if sum(lengths) != total_frames:
        raise ValueError(f"Episode lengths total {sum(lengths)} does not match {total_frames} frames")

    expected_from_index = 0
    for episode in episodes:
        from_index = int(episode["dataset_from_index"])
        to_index = int(episode["dataset_to_index"])
        length = int(episode["length"])
        if from_index != expected_from_index or to_index != from_index + length:
            raise ValueError(f"Episode {episode['episode_index']} has non-contiguous dataset frame bounds")
        expected_from_index = to_index
    return lengths


def _stats_as_lists(stats: dict[str, np.ndarray]) -> dict[str, list]:
    return {key: value.tolist() for key, value in stats.items()}


def _rewrite_episode_stats(
    episodes: list[dict], relative_states: np.ndarray, action_chunks: np.ndarray
) -> list[dict]:
    rewritten = deepcopy(episodes)
    start = 0
    for episode in rewritten:
        length = int(episode["length"])
        stop = start + length
        state_stats = get_feature_stats(relative_states[start:stop], axis=0, keepdims=False)
        action_stats = make_relative_action_stats(action_chunks[start:stop])
        for stat_name, value in _stats_as_lists(state_stats).items():
            episode[f"stats/observation.state/{stat_name}"] = value
        for stat_name, value in _stats_as_lists(action_stats).items():
            episode[f"stats/action/{stat_name}"] = value
        start = stop
    return rewritten


def _rewrite_numeric_parquet(
    build_root: Path,
    data_table: pa.Table,
    features: dict,
    relative_states: np.ndarray,
    action_chunks: np.ndarray,
) -> None:
    columns = {name: data_table[name].to_pylist() for name in data_table.column_names}
    columns["observation.state"] = relative_states.tolist()
    columns["action"] = action_chunks.tolist()
    hf_dataset = datasets.Dataset.from_dict(
        columns,
        features=get_hf_features_from_features(_features_for_hf(features)),
        split="train",
    )
    output_table = hf_dataset.with_format("arrow")[:]
    data_path = build_root / "data/chunk-000/file-000.parquet"
    write_table_one_row_group_per_episode(output_table, data_path)


def _rewrite_alignment_manifests(
    build_root: Path,
    final_output: Path,
    repo_id: str,
    episodes: list[dict],
    relative_states: np.ndarray,
    action_chunks: np.ndarray,
) -> None:
    summary_path = build_root / "catchpi_conversion_summary.json"
    if not summary_path.is_file():
        return

    conversion_summary = _load_json(summary_path)
    summary_episodes = conversion_summary.get("episodes")
    if not isinstance(summary_episodes, list) or len(summary_episodes) != len(episodes):
        raise ValueError("catchpi_conversion_summary.json does not match episode metadata")

    start = 0
    for episode, summary_episode in zip(episodes, summary_episodes, strict=True):
        if int(summary_episode["lerobot_episode_index"]) != int(episode["episode_index"]):
            raise ValueError("catchpi conversion summary episode order does not match metadata")
        length = int(episode["length"])
        stop = start + length
        manifest_path = build_root / "catchpi_alignment_manifests" / Path(
            summary_episode["alignment_manifest"]
        ).name
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing alignment manifest: {manifest_path}")
        frames = _load_json(manifest_path)
        if not isinstance(frames, list) or len(frames) != length:
            raise ValueError(f"Alignment manifest frame count differs for episode {episode['episode_index']}")
        for frame, state, action in zip(
            frames, relative_states[start:stop], action_chunks[start:stop], strict=True
        ):
            frame["state"] = state.tolist()
            frame["action"] = action.tolist()
        _write_json(manifest_path, frames)
        summary_episode["alignment_manifest"] = str(
            final_output / "catchpi_alignment_manifests" / manifest_path.name
        )
        start = stop

    conversion_summary["repo_id"] = repo_id
    conversion_summary["output_dir"] = str(final_output)
    conversion_summary["action_chunk_size"] = action_chunks.shape[1]
    conversion_summary["state_representation"] = "arm=q_t-q_(t-1), gripper=q_t"
    conversion_summary["action_representation"] = "arm=q_(t+k)-q_t, gripper=q_(t+k), k=1..chunk_size"
    _write_json(summary_path, conversion_summary)


def _rewrite_metadata(
    build_root: Path,
    source: Path,
    final_output: Path,
    repo_id: str,
    info: dict,
    features: dict,
    episodes: list[dict],
    relative_states: np.ndarray,
    action_chunks: np.ndarray,
) -> None:
    output_info = deepcopy(info)
    output_info["features"] = features
    _write_json(build_root / "meta/info.json", output_info)

    global_state_stats = get_feature_stats(relative_states, axis=0, keepdims=False)
    global_action_stats = make_relative_action_stats(action_chunks)
    source_stats = _load_json(build_root / "meta/stats.json")
    source_stats["observation.state"] = _stats_as_lists(global_state_stats)
    source_stats["action"] = _stats_as_lists(global_action_stats)
    write_stats(source_stats, build_root)

    rewritten_episodes = _rewrite_episode_stats(episodes, relative_states, action_chunks)
    episode_path = build_root / "meta/episodes/chunk-000/file-000.parquet"
    pq.write_table(pa.Table.from_pylist(rewritten_episodes), episode_path, compression="snappy", use_dictionary=True)

    inherited_preprocessing = _load_json(build_root / "preprocessing_summary.json")
    preprocessing_summary = make_preprocessing_summary(
        source=source,
        output=final_output,
        chunk_size=action_chunks.shape[1],
        total_episodes=output_info["total_episodes"],
        total_frames=output_info["total_frames"],
        inherited_preprocessing=inherited_preprocessing,
    )
    _write_json(build_root / "preprocessing_summary.json", preprocessing_summary)
    _rewrite_alignment_manifests(
        build_root,
        final_output,
        repo_id,
        episodes,
        relative_states,
        action_chunks,
    )


def convert_dataset(
    source: Path,
    output: Path,
    repo_id: str = DEFAULT_REPO_ID,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """Create a derived dataset while preserving all source videos byte-for-byte."""
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source dataset directory does not exist: {source}")
    if source == output:
        raise ValueError("Source and output paths must differ")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    build_root = output.parent / f".{output.name}.building"
    if build_root.exists():
        raise FileExistsError(f"Stale build directory exists: {build_root}")

    info, data_table, episodes, absolute_states = _read_source(source)
    episode_lengths = _episode_lengths(episodes, len(absolute_states))
    features = make_relative_output_features(info["features"], chunk_size)
    relative_states, action_chunks = make_relative_state_and_action_chunks(
        absolute_states, episode_lengths, chunk_size
    )

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, build_root, copy_function=shutil.copy2)
        _rewrite_numeric_parquet(build_root, data_table, features, relative_states, action_chunks)
        _rewrite_metadata(
            build_root,
            source,
            output,
            repo_id,
            info,
            features,
            episodes,
            relative_states,
            action_chunks,
        )
        conversion_path = build_root / "relative_chunk_conversion_summary.json"
        _write_json(
            conversion_path,
            {
                "source": str(source),
                "output": str(output),
                "repo_id": repo_id,
                "chunk_size": chunk_size,
                "state_representation": "arm=q_t-q_(t-1), gripper=q_t; first-frame arms are zero",
                "action_representation": "arm=q_(t+k)-q_t, gripper=q_(t+k), k=1..chunk_size",
                "episode_tail": "future indices clamp to each episode's final frame",
                "total_episodes": len(episodes),
                "total_frames": len(absolute_states),
                "videos": "copied byte-for-byte without re-encoding",
            },
        )
        build_root.rename(output)
    except BaseException:
        shutil.rmtree(build_root, ignore_errors=True)
        raise

    logger.info("Created %s with %d episodes and %d frames", output, len(episodes), len(absolute_states))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    convert_dataset(args.source, args.output, args.repo_id, args.chunk_size)


if __name__ == "__main__":
    main()
