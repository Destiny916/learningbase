import json
import os
import numpy as np
from act_async_infer_distributed_demo.scripts.utils_distributed import (
    log_warning,
    log_error,
)
from act_async_infer_distributed_demo.scripts.w1_mapping import TrajectoryKeys

GRIPPER_PUBLISH_TIMES = 10
_SEARCH_PATH = os.environ.get(
    "W1_RECORDS_DIR",
    "/home/dexforce/w1/dexe_mobile_application/script/records",
)


class JointTrajectoryLoader:
    """加载 JSON 目标位姿 + 生成插值轨迹。"""

    @staticmethod
    def load_targets(json_path: str) -> dict:
        if not os.path.isfile(json_path):
            log_error(
                f"Home position file not found: {json_path}\n"
                f"Expected path: {json_path}\n"
                f"Search directory: {_SEARCH_PATH}"
            )
        with open(json_path, "r") as f:
            data = json.load(f)
        if "frames" not in data or len(data["frames"]) == 0:
            log_error(
                f"Invalid home position file: {json_path} — "
                f"expected 'frames' list with at least one entry"
            )
        return dict(data["frames"][0]["data"])

    @staticmethod
    def build_trajectory(current: np.ndarray, target: np.ndarray,
                         n_frames: int = 200) -> np.ndarray:
        return np.linspace(current, target, n_frames)

    @staticmethod
    def normalize_json_path(raw: str) -> str:
        if raw is None:
            log_warning("No home position path provided, skip move to home position")
            return raw
        if raw.endswith(".json"):
            return raw
        return raw + ".json"
