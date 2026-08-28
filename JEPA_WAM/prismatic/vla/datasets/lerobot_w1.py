"""LeRobot v3.0 adapter primitives for the 19D DexForce W1 JEPA-WAM recipe.

The public JEPA-WAM release is LIBERO/RLDS-specific.  This module keeps the
W1 data contract independent from PyTorch so it can be tested on a host that
does not have the training environment installed.  Parquet/video iteration is
deferred to :class:`LeRobotW1Dataset` in the training container.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

W1_ACTION_HORIZON = 20
W1_STATE_DIM = 19
W1_ACTION_DIM = 19
W1_IMAGE_KEYS = (
    "observation.images.cam_high_right",
    "observation.images.cam_hand_left",
    "observation.images.cam_hand_right",
)
W1_RELATIVE_JOINT_INDICES = np.asarray(
    [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16], dtype=np.int64
)
W1_ABSOLUTE_INDICES = np.asarray([0, 8, 9, 17, 18], dtype=np.int64)


@dataclass(frozen=True)
class W1DataContract:
    state_dim: int = W1_STATE_DIM
    action_dim: int = W1_ACTION_DIM
    action_horizon: int = W1_ACTION_HORIZON
    image_keys: tuple[str, ...] = W1_IMAGE_KEYS
    pair_target_offset: int = 31


def validate_w1_info(info: Mapping[str, Any], contract: W1DataContract | None = None) -> None:
    contract = contract or W1DataContract()
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"W1 adapter requires LeRobot v3.0, got {info.get('codebase_version')!r}")
    features = info.get("features")
    if not isinstance(features, Mapping):
        raise ValueError("LeRobot metadata is missing `features`.")

    for name, expected_dim in (
        ("observation.state", contract.state_dim),
        ("action", contract.action_dim),
    ):
        shape = tuple(features.get(name, {}).get("shape", ()))
        if shape != (expected_dim,):
            raise ValueError(f"{name} must have shape ({expected_dim},), got {shape}")

    for key in contract.image_keys:
        shape = tuple(features.get(key, {}).get("shape", ()))
        if shape != (3, 224, 224):
            raise ValueError(f"{key} must have shape (3, 224, 224), got {shape}")


def normalize_with_quantiles(
    values: np.ndarray,
    q01: np.ndarray,
    q99: np.ndarray,
) -> np.ndarray:
    """Map values to [-1, 1] using one *specific* q01/q99 pair.

    State and action callers must pass separate statistics.  Constant
    dimensions are mapped to zero instead of producing NaN; all other values
    are clipped to their quantile interval before scaling.
    """

    values = np.asarray(values, dtype=np.float32)
    q01 = np.asarray(q01, dtype=np.float32)
    q99 = np.asarray(q99, dtype=np.float32)
    if values.shape[-1] != q01.size or q01.shape != q99.shape:
        raise ValueError(
            f"Quantile shape mismatch: values={values.shape}, q01={q01.shape}, q99={q99.shape}"
        )
    if not np.isfinite(values).all() or not np.isfinite(q01).all() or not np.isfinite(q99).all():
        raise ValueError("Values and quantiles must be finite.")
    if np.any(q99 < q01):
        raise ValueError("Every q99 must be greater than or equal to q01.")

    clipped = np.clip(values, q01, q99)
    span = q99 - q01
    safe_span = np.where(span > 0, span, 1.0)
    normalized = 2.0 * (clipped - q01) / safe_span - 1.0
    return np.where(span > 0, normalized, 0.0).astype(np.float32, copy=False)


def relative_state_representation(states: np.ndarray) -> np.ndarray:
    """Keep waist/neck/grippers absolute; difference only arm joint positions."""

    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != W1_STATE_DIM:
        raise ValueError(f"states must have shape [T, {W1_STATE_DIM}], got {states.shape}")
    result = states.copy()
    result[0, W1_RELATIVE_JOINT_INDICES] = 0.0
    result[1:, W1_RELATIVE_JOINT_INDICES] = np.diff(
        states[:, W1_RELATIVE_JOINT_INDICES], axis=0
    )
    return result


def relative_action_representation(actions: np.ndarray, anchor_state: np.ndarray) -> np.ndarray:
    """Express arm-joint action targets relative to the current state anchor."""

    actions = np.asarray(actions, dtype=np.float32)
    anchor_state = np.asarray(anchor_state, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != W1_ACTION_DIM:
        raise ValueError(f"actions must have shape [T, {W1_ACTION_DIM}], got {actions.shape}")
    if anchor_state.shape != (W1_STATE_DIM,):
        raise ValueError(f"anchor_state must have shape ({W1_STATE_DIM},), got {anchor_state.shape}")
    result = actions.copy()
    result[:, W1_RELATIVE_JOINT_INDICES] -= anchor_state[W1_RELATIVE_JOINT_INDICES]
    return result


def build_action_chunk(actions: np.ndarray, start: int, horizon: int = W1_ACTION_HORIZON) -> tuple[np.ndarray, np.ndarray]:
    """Build a fixed chunk, padding only with the final action of this episode."""

    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] == 0:
        raise ValueError(f"actions must be a non-empty [T,D] array, got {actions.shape}")
    if not 0 <= start < actions.shape[0]:
        raise ValueError(f"start must be in [0, {actions.shape[0]}), got {start}")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    indices = np.minimum(np.arange(start, start + horizon), actions.shape[0] - 1)
    valid = np.arange(start, start + horizon) < actions.shape[0]
    return actions[indices].copy(), valid


def build_pair_indices(length: int, start: int, offset: int = 31) -> np.ndarray:
    if length <= 0 or not 0 <= start < length:
        raise ValueError(f"invalid episode length/start: length={length}, start={start}")
    if offset < 0:
        raise ValueError("offset must be non-negative")
    return np.minimum(np.array([start, start + offset], dtype=np.int64), length - 1)


class LeRobotW1Dataset:
    """Small, dependency-lazy reader for W1 LeRobot v3.0 parquet samples.

    Video decoding is intentionally left to the caller/training collator; this
    reader emits deterministic row indices and normalized state/action tensors,
    avoiding a second video implementation in the data-contract layer.
    """

    def __init__(
        self,
        root: str,
        state_q01: np.ndarray,
        state_q99: np.ndarray,
        action_q01: np.ndarray,
        action_q99: np.ndarray,
        contract: W1DataContract | None = None,
    ) -> None:
        self.root = __import__("pathlib").Path(root).expanduser().resolve()
        self.contract = contract or W1DataContract()
        import json

        with (self.root / "meta" / "info.json").open(encoding="utf-8") as handle:
            info = json.load(handle)
        validate_w1_info(info, self.contract)
        self.state_q01 = np.asarray(state_q01, dtype=np.float32)
        self.state_q99 = np.asarray(state_q99, dtype=np.float32)
        self.action_q01 = np.asarray(action_q01, dtype=np.float32)
        self.action_q99 = np.asarray(action_q99, dtype=np.float32)
        if self.state_q01.shape != (self.contract.state_dim,) or self.state_q99.shape != self.state_q01.shape:
            raise ValueError("state q01/q99 must both be 19D")
        if self.action_q01.shape != (self.contract.action_dim,) or self.action_q99.shape != self.action_q01.shape:
            raise ValueError("action q01/q99 must both be 19D")
        self._episodes = self._read_episodes()

    def _read_episodes(self) -> dict[int, dict[str, np.ndarray]]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - exercised in host setup
            raise RuntimeError("LeRobotW1Dataset requires pyarrow in the training environment") from exc

        grouped: dict[int, list[tuple[int, np.ndarray, np.ndarray, int, int]]] = {}
        for path in sorted((self.root / "data").glob("**/*.parquet")):
            table = pq.read_table(path, columns=["observation.state", "action", "episode_index", "frame_index", "index"])
            for state, action, episode, frame, global_index in zip(
                table["observation.state"].to_pylist(),
                table["action"].to_pylist(),
                table["episode_index"].to_pylist(),
                table["frame_index"].to_pylist(),
                table["index"].to_pylist(),
                strict=True,
            ):
                grouped.setdefault(int(episode), []).append(
                    (int(frame), np.asarray(state, dtype=np.float32), np.asarray(action, dtype=np.float32), int(global_index), int(episode))
                )

        episodes: dict[int, dict[str, np.ndarray]] = {}
        for episode, rows in grouped.items():
            rows.sort(key=lambda row: row[0])
            if [row[0] for row in rows] != list(range(len(rows))):
                raise ValueError(f"episode {episode} frame indices are not contiguous")
            episodes[episode] = {
                "state": np.stack([row[1] for row in rows]),
                "action": np.stack([row[2] for row in rows]),
                "global_index": np.asarray([row[3] for row in rows], dtype=np.int64),
            }
        if not episodes:
            raise ValueError(f"No parquet rows found under {self.root / 'data'}")
        return episodes

    def __len__(self) -> int:
        return sum(rows["state"].shape[0] for rows in self._episodes.values())

    def __iter__(self):
        for episode in sorted(self._episodes):
            rows = self._episodes[episode]
            state_rep = relative_state_representation(rows["state"])
            state = normalize_with_quantiles(state_rep, self.state_q01, self.state_q99)
            for frame in range(rows["state"].shape[0]):
                action_abs, valid = build_action_chunk(rows["action"], frame, self.contract.action_horizon)
                action_rep = relative_action_representation(action_abs, rows["state"][frame])
                yield {
                    "episode_index": episode,
                    "frame_index": frame,
                    "global_index": int(rows["global_index"][frame]),
                    "state": state[frame],
                    "action": normalize_with_quantiles(action_rep, self.action_q01, self.action_q99),
                    "action_valid_mask": valid,
                    "pair_frame_indices": build_pair_indices(
                        rows["state"].shape[0], frame, self.contract.pair_target_offset
                    ),
                    "image_keys": self.contract.image_keys,
                }
