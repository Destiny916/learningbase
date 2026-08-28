"""Pure configuration helpers for the PC1 XWiz inference manager."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .runtime import prepare_client_config


LEROBOT_CAMERA_FEATURES = (
    ("observation.images.cam_high_left", "head_left_camera"),
    ("observation.images.cam_high_right", "head_right_camera"),
    ("observation.images.cam_hand_left", "left_wrist_camera"),
    ("observation.images.cam_hand_right", "right_wrist_camera"),
)
LEROBOT_19D_GROUPS = {
    "WAIST",
    "LEFT_ARM",
    "HEAD",
    "RIGHT_ARM",
    "LEFT_EEFGRIPPER",
    "RIGHT_EEFGRIPPER",
}


def lerobot_model_meta(config: Mapping[str, object]) -> tuple[list[str], set[str]]:
    """Translate the verified W1 19D LeRobot contract for the XWiz model panel."""
    inputs = config.get("input_features", {})
    outputs = config.get("output_features", {})
    if not isinstance(inputs, Mapping) or not isinstance(outputs, Mapping):
        return [], set()

    state = inputs.get("observation.state", {})
    action = outputs.get("action", {})
    if not isinstance(state, Mapping) or not isinstance(action, Mapping):
        return [], set()
    if tuple(state.get("shape", ())) != (19,) or tuple(action.get("shape", ())) != (19,):
        return [], set()

    cameras = [
        xwiz_name
        for feature_name, xwiz_name in LEROBOT_CAMERA_FEATURES
        if feature_name in inputs
    ]
    return cameras, set(LEROBOT_19D_GROUPS)


def prepare_resolved_configs(
    client_config: Mapping[str, object],
    server_config: Mapping[str, object],
    mode: int,
) -> tuple[dict[str, object], dict[str, object]]:
    client = prepare_client_config(client_config, mode)
    server = dict(server_config)
    server["data_type"] = "simulation" if mode == 1 else "real"
    server["action_horizon"] = 100
    return client, server


def read_task_mode(config_base_path: str | Path, task_id: int) -> int:
    path = Path(config_base_path) / "tasks" / str(int(task_id)) / "task_config.json"
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    mode = payload.get("client_config", {}).get("mode")
    if mode not in (1, 2):
        raise ValueError(f"task {task_id} must declare client_config.mode as 1 or 2")
    return int(mode)


def deploy_selected(manager: Any, model_id: int, task_id: int) -> bool:
    """Resolve and immediately start the exact model/task selected in XWiz."""
    mode = read_task_mode(manager.config_registry.config_base_path, task_id)
    client, server = manager.config_registry.resolve_config(model_id, task_id, mode)
    if not manager.client_controller.setup_config(client, server):
        return False
    manager.current_config = (client, server)
    return True
