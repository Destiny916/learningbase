"""Run the PC2 XWiz client with simulation/real isolation and one-chunk gating."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import types

import numpy as np


W1_ACT_ROOT = Path(__file__).resolve().parents[1]
if str(W1_ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(W1_ACT_ROOT))

# The vendor imports ActionLiPo even though one-chunk mode never uses it.
_lipo_module_name = "act_async_infer_distributed_demo.scripts.action_lipo"
if _lipo_module_name not in sys.modules:
    _stub = types.ModuleType(_lipo_module_name)

    class _DisabledActionLiPo:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("ActionLiPo is disabled for single-chunk XWiz execution")

    _stub.ActionLiPo = _DisabledActionLiPo
    sys.modules[_lipo_module_name] = _stub

import rclpy
from end_effector_interfaces.msg import EEFeedback, EEJointControl, EEJointControlMode
import end_effector_interfaces.msg as ee_messages

if not hasattr(ee_messages, "EEJointFeedback"):
    ee_messages.EEJointFeedback = EEFeedback

from act_async_infer_distributed_demo.scripts.client.robot_client_with_kingfisher import (
    OptimizedRobotClient,
    RobotState,
)
from act_async_infer_distributed_demo.scripts.inference_config import (
    ClientConfig,
    RequestType,
    ResponseKey,
)
from act_async_infer_distributed_demo.scripts.utils_distributed import log_info

from .runtime import (
    LEFT_CLOSED,
    LEFT_OPEN,
    RIGHT_CLOSED,
    RIGHT_OPEN,
    RuntimeContractError,
    SingleChunkGate,
    action_to_commands,
    feedback_positions_by_name,
    gripper_scalars_from_feedback,
    prepare_client_config,
    validate_observation_buffers,
    validate_robot_health,
    validate_robot_ready,
    validate_timed_actions,
)


_vendor_setup = OptimizedRobotClient._cmd_setup_config
_vendor_get_obs = OptimizedRobotClient.get_real_obs
_vendor_get_actions = OptimizedRobotClient.get_actions
_vendor_exec_action = OptimizedRobotClient.exec_action
_vendor_joint_callback = OptimizedRobotClient.joint_state_callback


def _buffer_snapshot(client):
    return {
        "head_left": client.head_left_buf,
        "head_right": client.head_right_buf,
        "wrist_left": client.hand_left_buf,
        "wrist_right": client.hand_right_buf,
        "joint_state": client.joint_state_buf,
        "left_hand": client.hand_qpos6_left_buf,
        "right_hand": client.hand_qpos6_right_buf,
    }


def _joint_state_callback(self, message):
    try:
        self._latest_robot_state = json.loads(message.data)
    except (json.JSONDecodeError, TypeError):
        self._latest_robot_state = None
    _vendor_joint_callback(self, message)


def _hand_feedback_callback(side):
    def callback(self, message):
        try:
            positions = feedback_positions_by_name(message.joint_states)
        except RuntimeContractError as exc:
            self.get_logger().error(f"{side} Linker feedback rejected: {exc}")
            return
        buffer = self.hand_qpos6_left_buf if side == "left" else self.hand_qpos6_right_buf
        buffer.append((time.time(), np.asarray(positions, dtype=np.float32)))

    return callback


def _setup_single_chunk(self, client_config, server_config, _home_position):
    mode = int(client_config.get("mode", self.cfg.mode))
    prepared = prepare_client_config(client_config, mode)
    prepared["end_effector_type"] = "gripper"
    self._single_chunk_gate = SingleChunkGate(100)

    try:
        if mode == 2:
            if not getattr(self, "_latest_robot_state", None):
                raise RuntimeContractError("real deployment has no robot state feedback")
            validate_robot_ready(self._latest_robot_state, tolerance_rad=0.05)
            validate_observation_buffers(
                _buffer_snapshot(self),
                use_wrist_images=bool(prepared.get("use_hand_camera", True)),
            )
            log_info("Real preflight passed: ACT default pose, robot health and observations ready")
        else:
            log_info("Simulation deployment selected: real control topics remain unused")
    except RuntimeContractError as exc:
        self._set_state(RobotState.ERROR, f"real preflight rejected deployment: {exc}")
        return {ResponseKey.SUCCESS: False, ResponseKey.MESSAGE: str(exc)}

    return _vendor_setup(self, prepared, server_config, "")


def _get_model_observation(self):
    deadline = time.monotonic() + 5.0
    while self.running and time.monotonic() < deadline:
        try:
            validate_observation_buffers(
                _buffer_snapshot(self),
                use_wrist_images=bool(self.cfg.use_hand_camera),
            )
            break
        except RuntimeContractError:
            time.sleep(0.02)
    else:
        raise RuntimeContractError("observation buffers did not become ready within 5 seconds")

    if int(self.cfg.mode) == 1:
        left_scalar = right_scalar = 0.0
    else:
        left_positions = self.hand_qpos6_left_buf[-1][1]
        right_positions = self.hand_qpos6_right_buf[-1][1]
        left_scalar, right_scalar = gripper_scalars_from_feedback(
            left_positions, right_positions
        )
    timestamp = time.time()
    self.gripper_qpos_left_buf.append((timestamp, [left_scalar]))
    self.gripper_qpos_right_buf.append((timestamp, [right_scalar]))
    return _vendor_get_obs(self)


def _ee_message(client, values):
    message = EEJointControl()
    message.header.stamp = client._now()
    message.mode = EEJointControlMode.POSITION
    message.joint_names = [
        "T_CMC_YAW", "T_MCP", "IF_MCP_PITCH",
        "MF_MCP_PITCH", "RF_MCP_PITCH", "LF_MCP_PITCH",
    ]
    message.values = [float(value) for value in values]
    return message


def _publish_gripper_openness(self, side, position, clip=True):
    scalar = float(np.clip(position, 0.0, 100.0)) if clip else float(position)
    if int(self.cfg.mode) == 1:
        message = EEJointControl()
        message.header.stamp = self._now()
        message.mode = EEJointControlMode.POSITION
        message.joint_names = ["PIPER_LEFT" if side == "left" else "PIPER_RIGHT"]
        message.values = [scalar]
        publisher = self._pub_gripper_left if side == "left" else self._pub_gripper_right
    else:
        closed, opened = (
            (LEFT_CLOSED, LEFT_OPEN) if side == "left" else (RIGHT_CLOSED, RIGHT_OPEN)
        )
        from .runtime import hand_command_from_openness

        message = _ee_message(self, hand_command_from_openness(scalar, closed, opened))
        publisher = self._pub_hand_left if side == "left" else self._pub_hand_right
    publisher.publish(message)


def _get_validated_actions(self):
    timed_actions = _vendor_get_actions(self)
    if timed_actions is not None:
        validate_timed_actions(timed_actions)
        log_info("Validated one finite ACT action chunk with shape (100, 19)")
    return timed_actions


def _finish_single_chunk(self, mode_name):
    with self.socket_lock:
        try:
            self.network_client.send_request(RequestType.STOP)
        except Exception:
            pass
    self._shutdown_event.set()
    self.shutdown_event.set()
    self._inference_running = False
    self._set_state(RobotState.IDLE)
    log_info(f"Single {mode_name} action chunk completed; stopped after exactly 100 frames")


def _execute_guarded_action(self, timed_action):
    if int(self.cfg.mode) == 1:
        result = _vendor_exec_action(self, timed_action)
        if self._single_chunk_gate.mark_published():
            _finish_single_chunk(self, "simulation")
        return result

    state = getattr(self, "_latest_robot_state", None)
    if not state:
        raise RuntimeContractError("robot state feedback disappeared during execution")
    validate_robot_health(state, allowed_status=("Idle", "Running"))
    command = action_to_commands(timed_action.get_action())
    self.publish_joint_positions(command.body_names, command.body_positions, clip=False)
    self._pub_hand_left.publish(_ee_message(self, command.left_hand))
    self._pub_hand_right.publish(_ee_message(self, command.right_hand))

    self.current_step += 1
    complete = self._single_chunk_gate.mark_published()
    log_info(
        f"Executed guarded real action frame {self.current_step}/100 "
        f"left_open={timed_action.get_action()[-2]:.3f} "
        f"right_open={timed_action.get_action()[-1]:.3f}"
    )
    if complete:
        _finish_single_chunk(self, "real")
    return timed_action


def install_hooks():
    OptimizedRobotClient.joint_state_callback = _joint_state_callback
    OptimizedRobotClient.hand_qpos6_left_callback = _hand_feedback_callback("left")
    OptimizedRobotClient.hand_qpos6_right_callback = _hand_feedback_callback("right")
    OptimizedRobotClient._cmd_setup_config = _setup_single_chunk
    OptimizedRobotClient.get_real_obs = _get_model_observation
    OptimizedRobotClient.publish_gripper_position = _publish_gripper_openness
    OptimizedRobotClient.get_actions = _get_validated_actions
    OptimizedRobotClient.exec_action = _execute_guarded_action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    install_hooks()
    config = ClientConfig.from_json_file(args.config)
    config.service = True

    rclpy.init()
    client = OptimizedRobotClient(config)
    client._start_manager_listener()
    log_info(f"XWiz dual-mode PC2 client listening on 0.0.0.0:{config.manager_port}")
    try:
        while rclpy.ok():
            rclpy.spin_once(client, timeout_sec=0.1)
            time.sleep(0.01)
    finally:
        client.stop()


if __name__ == "__main__":
    main()
