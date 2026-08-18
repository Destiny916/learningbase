from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from w1_simulation.robot.commands import W1ControlEndpoints
from w1_simulation.robot.joints import ACT_STATE_JOINTS, BODY_JOINTS, HAND_POSITION_JOINTS

PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_PROFILE_PATH = PROJECT_ROOT / "configs" / "w1_popcorn_v1.json"


def _object(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"W1 profile requires object field: {key}")
    return value


def _resolve_path(value: object, asset_root: Path) -> Path:
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else asset_root / path).resolve()


@dataclass(frozen=True)
class W1Profile:
    source: Path
    payload: dict[str, Any]
    asset_root: Path

    @classmethod
    def load(cls, path: Path, asset_root: Path | None = None) -> W1Profile:
        source = Path(path).expanduser().resolve()
        with source.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError("W1 profile root must be an object")
        selected_root = (
            Path(os.environ.get("W1_SIMULATION_ASSET_ROOT", asset_root or WORKSPACE_ROOT))
            .expanduser()
            .resolve()
        )
        profile = cls(source=source, payload=payload, asset_root=selected_root)
        profile.validate()
        return profile

    @property
    def name(self) -> str:
        return str(self.payload["name"])

    @property
    def paths(self) -> dict[str, Path]:
        return {
            key: _resolve_path(value, self.asset_root)
            for key, value in _object(self.payload, "paths").items()
        }

    @property
    def checkpoint(self) -> Path:
        return self.paths["checkpoint"]

    @property
    def origin(self) -> Path:
        return self.paths["origin"]

    @property
    def artifacts(self) -> Path:
        override = os.environ.get("W1_SIMULATION_ARTIFACT_ROOT")
        return Path(override).expanduser().resolve() if override else self.paths["artifacts"]

    @property
    def urdf(self) -> Path:
        return self.paths["urdf"]

    @property
    def policy_script(self) -> Path:
        return self.paths["policy_script"]

    @property
    def bridge_script(self) -> Path:
        return self.paths["bridge_script"]

    @property
    def whole_script(self) -> Path:
        return self.paths["whole_script"]

    @property
    def act(self) -> dict[str, Any]:
        return _object(self.payload, "act")

    @property
    def commands(self) -> dict[str, Any]:
        return _object(self.payload, "commands")

    @property
    def hands(self) -> dict[str, Any]:
        return _object(self.payload, "hands")

    @property
    def camera_sources(self) -> dict[str, str]:
        return {str(key): str(value) for key, value in _object(self.payload, "camera_sources").items()}

    @property
    def simulation(self) -> dict[str, Any]:
        return _object(self.payload, "simulation")

    @property
    def runtime(self) -> dict[str, Any]:
        return _object(self.payload, "runtime")

    @property
    def hashes(self) -> dict[str, str]:
        return {str(key): str(value) for key, value in _object(self.payload, "hashes").items()}

    @property
    def locked_joint_values(self) -> dict[str, float]:
        values = self.commands["locked_body_positions_rad"]
        if not isinstance(values, dict):
            raise ValueError("commands.locked_body_positions_rad must be an object")
        return {str(key): float(value) for key, value in values.items()}

    @property
    def body_command_names(self) -> tuple[str, ...]:
        return tuple(str(name) for name in self.commands["body_order"])

    @property
    def endpoints(self) -> W1ControlEndpoints:
        return W1ControlEndpoints(
            body=str(self.commands["body_topic"]),
            left_hand=str(self.commands["left_hand_topic"]),
            right_hand=str(self.commands["right_hand_topic"]),
        )

    def validate(self) -> None:
        if int(self.payload.get("schema_version", 0)) != 1 or not self.name:
            raise ValueError("Unsupported W1 profile schema")
        required_paths = {
            "checkpoint",
            "origin",
            "artifacts",
            "urdf",
            "policy_script",
            "bridge_script",
            "whole_script",
        }
        if set(self.paths) != required_paths:
            raise ValueError(f"W1 profile paths must be exactly {sorted(required_paths)}")
        if tuple(self.act.get("state_action_order", ())) != ACT_STATE_JOINTS:
            raise ValueError("W1 profile ACT state/action order does not match the W1 contract")
        body_command_names = self.body_command_names
        if not body_command_names or len(set(body_command_names)) != len(body_command_names):
            raise ValueError("W1 profile body command order must be a non-empty unique sequence")
        if any(name not in BODY_JOINTS for name in body_command_names):
            raise ValueError("W1 profile body command order must be a subset of the ACT body order")
        if tuple(self.commands.get("hand_order", ())) != HAND_POSITION_JOINTS:
            raise ValueError("W1 profile hand command order does not match the W1 contract")
        _ = self.endpoints
        if set(self.locked_joint_values) != {"ANKLE", "KNEE", "BUTTOCK"}:
            raise ValueError("W1 profile must define all three locked body positions")
        image_keys = tuple(self.act.get("image_keys", ()))
        if image_keys != tuple(self.camera_sources):
            raise ValueError("W1 profile camera sources must preserve ACT image key order")
        if int(self.act.get("chunk_size", 0)) < 1 or int(self.act.get("n_action_steps", 0)) < 1:
            raise ValueError("W1 profile ACT horizons must be positive")
        for side in ("left", "right"):
            endpoints = self.hands.get(side)
            if not isinstance(endpoints, dict) or set(endpoints) != {"0", "100"}:
                raise ValueError(f"W1 profile requires {side} hand endpoints 0 and 100")
            for value in endpoints.values():
                if not isinstance(value, list) or len(value) != 6:
                    raise ValueError(f"W1 profile {side} hand endpoints must be 6D")
                if any(not 0.0 <= float(item) <= 100.0 for item in value):
                    raise ValueError(f"W1 profile {side} hand endpoints must stay in [0, 100]")
        compatibility = _object(self.payload, "compatibility")
        training_hash = self.hashes.get("training_kinematics_urdf_sha256")
        runtime_hash = self.hashes.get("runtime_urdf_sha256")
        if training_hash != runtime_hash and not bool(
            compatibility.get("allow_training_runtime_urdf_hash_mismatch", False)
        ):
            raise ValueError(
                "Training and runtime URDF hashes differ without an explicit compatibility waiver"
            )

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "schema_version": int(self.payload["schema_version"]),
            "source": str(self.source),
            "asset_root": str(self.asset_root),
            "paths": {key: str(value) for key, value in self.paths.items()},
            "hashes": dict(self.hashes),
            "compatibility": dict(_object(self.payload, "compatibility")),
        }


DEFAULT_PROFILE = W1Profile.load(
    Path(os.environ.get("W1_SIMULATION_PROFILE", DEFAULT_PROFILE_PATH)).expanduser().resolve()
)
DEFAULT_CHECKPOINT = DEFAULT_PROFILE.checkpoint
DEFAULT_ORIGIN = DEFAULT_PROFILE.origin
DEFAULT_ARTIFACT_ROOT = DEFAULT_PROFILE.artifacts
DEFAULT_POLICY_SCRIPT = DEFAULT_PROFILE.policy_script
DEFAULT_BRIDGE_SCRIPT = DEFAULT_PROFILE.bridge_script
DEFAULT_WHOLE_SCRIPT = DEFAULT_PROFILE.whole_script
DEFAULT_CAMERA_SOURCES = DEFAULT_PROFILE.camera_sources
ACT_IMAGE_KEYS = tuple(DEFAULT_PROFILE.act["image_keys"])
