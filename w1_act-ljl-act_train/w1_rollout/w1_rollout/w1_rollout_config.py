from dataclasses import dataclass
from transformers.configuration_utils import PretrainedConfig
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch


class W1RolloutConfig(PretrainedConfig):
    model_type = 'w1_rollout_config'

    def __init__(
            self,
            # === Core ===
            policy_path: str = '/path/to/pretrained_model',
            device: str = 'cuda',
            policy_hz: float = 15.0,
            tolerance_ms: float = 50.0,
            # === Remote server ===
            remote_server_host: str = '127.0.0.1',
            remote_server_port: int = 8899,
            remote_connect_timeout_s: float = 10.0,
            remote_horizon_n: int = 30,
            remote_replan_trigger: int = 20,
            # === Topics ===
            joint_topic: str = '/feedback/robot_server_state',
            publish_topic: str = '/w1/policy/desired_joint_positions',
            # === Images / state keys ===
            head_target_width: int = 640,
            head_target_height: int = 360,
            hand_target_width: int = 640,
            hand_target_height: int = 480,
            image_hand_left_key: str = 'observation.images.cam_hand_left',
            image_hand_right_key: str = 'observation.images.cam_hand_right',
            state_key: str = 'observation.state',
            image_keys: List[str] = None,
            # === Body names ===
            ordered_body_names: List[str] = None,
            selected_body_names: List[str] = None,
            drop_joint_names: List[str] = None,
            joint_names: List[str] = None,
            # === Hands ===
            hand_input_mode: str = 'none',
            hand_sides: List[str] = None,
            hand_sides_str: str = '',
            left_hand_scalar_name: str = 'LEFT_GRIPPER',
            right_hand_scalar_name: str = 'RIGHT_GRIPPER',
            left_hand_qpos6_names: List[str] = None,
            right_hand_qpos6_names: List[str] = None,
            # === Hand topics (input) ===
            left_hand_scalar_topic: str = '/hand/left_scalar',
            right_hand_scalar_topic: str = '/hand/right_scalar',
            left_hand_qpos6_topic: str = '/feedback_sim/hand/left',
            right_hand_qpos6_topic: str = '/feedback_sim/hand/right',
            publish_hand_joint_name: List[str] = None,
            # === Hand topics (output) ===
            set_left_hand_qpos6_topic: str = '/control/ee/left',
            set_right_hand_qpos6_topic: str = '/control/ee/right',
            # === Camera topics ===
            cam_hand_left_topic: str = '/camera/left/image_raw',
            cam_hand_right_topic: str = '/camera/right/image_raw',
            # === Hand interpolation ===
            hand_interp_start: str = 'normal2',
            hand_interp_end: str = 'pinch',
            # === Gripper ===
            gripper_binarize: bool = True,
            gripper_thr: float = 0.5,
            gripper_hysteresis: bool = True,
            gripper_thr_close: float = 0.55,
            gripper_thr_open: float = 0.45,
            freeze_after_release_s: float = 0.0,
            # === Hand gestures ===
            hand_gestures: Dict[str, List[float]] = None,
            **kwargs
    ):
        # === Default list values ===
        if image_keys is None:
            image_keys = [
                "observation.images.cam_high_left",
                "observation.images.cam_high_right",
                "observation.images.cam_hand_left",
                "observation.images.cam_hand_right",
            ]
        if ordered_body_names is None:
            ordered_body_names = [
                'ANKLE', 'KNEE', 'BUTTOCK', 'WAIST',
                'LEFT_J1', 'LEFT_J2', 'LEFT_J3', 'LEFT_J4', 'LEFT_J5', 'LEFT_J6', 'LEFT_J7',
                'NECK1', 'NECK2',
                'RIGHT_J1', 'RIGHT_J2', 'RIGHT_J3', 'RIGHT_J4', 'RIGHT_J5', 'RIGHT_J6', 'RIGHT_J7',
            ]
        if selected_body_names is None:
            selected_body_names = ordered_body_names.copy()
        if drop_joint_names is None:
            drop_joint_names = ['ANKLE', 'KNEE', 'BUTTOCK']
        if joint_names is None:
            joint_names = [
                "ANKLE", "KNEE", "BUTTOCK", "WAIST", "NECK1", "NECK2",
                "LEFT_J1", "LEFT_J2", "LEFT_J3", "LEFT_J4", "LEFT_J5", "LEFT_J6", "LEFT_J7",
                "RIGHT_J1", "RIGHT_J2", "RIGHT_J3", "RIGHT_J4", "RIGHT_J5", "RIGHT_J6", "RIGHT_J7",
            ]
        if hand_sides is None:
            hand_sides = ['left', 'right']
        if left_hand_qpos6_names is None:
            left_hand_qpos6_names = [
                'LEFT_HAND_THUMB1', 'LEFT_HAND_THUMB2', 'LEFT_HAND_INDEX',
                'LEFT_HAND_MIDDLE', 'LEFT_HAND_RING', 'LEFT_HAND_PINKY'
            ]
        if right_hand_qpos6_names is None:
            right_hand_qpos6_names = [
                'RIGHT_HAND_THUMB1', 'RIGHT_HAND_THUMB2', 'RIGHT_HAND_INDEX',
                'RIGHT_HAND_MIDDLE', 'RIGHT_HAND_RING', 'RIGHT_HAND_PINKY'
            ]
        if hand_gestures is None:
            hand_gestures = {
                "normal": [0.0, 70.0, 0.0, 0.0, 0.0, 0.0],
                "normal2": [0.0, 100.0, 0.0, 0.0, 0.0, 0.0],
                "cup": [0.0, 100.0, 35.0, 45.0, 47.0, 37.0],
                "pinch": [65.0, 100.0, 70.0, 75.0, 100.0, 100.0],
                "fist": [100.0, 30.0, 100.0, 100.0, 100.0, 100.0],
                "like": [0.0, 0.0, 100.0, 100.0, 100.0, 100.0],
                "heart": [0.0, 100.0, 60.0, 70.0, 60.0, 60.0],
                "bull": [90.0, 80.0, 0.0, 100.0, 100.0, 0.0],
                "gun": [0.0, 0.0, 0.0, 100.0, 100.0, 100.0],
                "six": [0.0, 0.0, 100.0, 100.0, 100.0, 0.0],
                "one": [100.0, 70.0, 0.0, 100.0, 100.0, 100.0],
                "salute": [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "ok": [60.0, 90.0, 60.0, 0.0, 0.0, 0.0],
            }
        if publish_hand_joint_name is None:
            publish_hand_joint_name = ["T_CMC_YAW", "T_MCP", "IF_MCP_PITCH", "MF_MCP_PITCH", "RF_MCP_PITCH", "LF_MCP_PITCH"]

        # Core
        self.policy_path = policy_path
        self.device = device
        self.policy_hz = policy_hz
        self.tolerance_ms = tolerance_ms

        # Remote server
        self.remote_server_host = remote_server_host
        self.remote_server_port = remote_server_port
        self.remote_connect_timeout_s = remote_connect_timeout_s
        self.remote_horizon_n = remote_horizon_n
        self.remote_replan_trigger = remote_replan_trigger

        # Topics
        self.joint_topic = joint_topic
        self.publish_topic = publish_topic

        # Images / state keys
        self.head_target_width = head_target_width
        self.head_target_height = head_target_height
        self.hand_target_width = hand_target_width
        self.hand_target_height = hand_target_height
        self.image_hand_left_key = image_hand_left_key
        self.image_hand_right_key = image_hand_right_key
        self.state_key = state_key
        self.image_keys = image_keys

        # Body names
        self.ordered_body_names = ordered_body_names
        self.selected_body_names = selected_body_names
        self.drop_joint_names = drop_joint_names
        self.joint_names = joint_names

        # Hands
        self.hand_input_mode = hand_input_mode
        self.hand_sides = hand_sides
        self.hand_sides_str = hand_sides_str
        self.left_hand_scalar_name = left_hand_scalar_name
        self.right_hand_scalar_name = right_hand_scalar_name
        self.left_hand_qpos6_names = left_hand_qpos6_names
        self.right_hand_qpos6_names = right_hand_qpos6_names
        self.publish_hand_joint_name = publish_hand_joint_name

        # Hand topics (input)
        self.left_hand_scalar_topic = left_hand_scalar_topic
        self.right_hand_scalar_topic = right_hand_scalar_topic
        self.left_hand_qpos6_topic = left_hand_qpos6_topic
        self.right_hand_qpos6_topic = right_hand_qpos6_topic

        # Hand topics (output)
        self.set_left_hand_qpos6_topic = set_left_hand_qpos6_topic
        self.set_right_hand_qpos6_topic = set_right_hand_qpos6_topic

        # Camera topics
        self.cam_hand_left_topic = cam_hand_left_topic
        self.cam_hand_right_topic = cam_hand_right_topic

        # Hand interpolation
        self.hand_interp_start = hand_interp_start
        self.hand_interp_end = hand_interp_end

        # Gripper
        self.gripper_binarize = gripper_binarize
        self.gripper_thr = gripper_thr
        self.gripper_hysteresis = gripper_hysteresis
        self.gripper_thr_close = gripper_thr_close
        self.gripper_thr_open = gripper_thr_open
        self.freeze_after_release_s = freeze_after_release_s

        # Hand gestures
        self.hand_gestures = hand_gestures

        # Go
        super().__init__(**kwargs)

    @property
    def body_canonical(self):
        return self.ordered_body_names

    @property
    def left_scalar_name(self):
        return self.left_hand_scalar_name

    @property
    def right_scalar_name(self):
        return self.right_hand_scalar_name

    @property
    def left_q6_names(self):
        return self.left_hand_qpos6_names

    @property
    def right_q6_names(self):
        return self.right_hand_qpos6_names


class W1RolloutGripperProcess(object):

    def __init__(self, config: W1RolloutConfig):
        self._config = config
        self._freeze_until = 0.0
        self._grip_bin = {'left': 0, 'right': 0}
        self._prev_grip_bin = {'left': 0, 'right': 0}

    def postprocess_gripper_scalar(self, s_raw: float, side: str, t_now_second: float) -> float:
        """Map raw gripper scalar to a stable command.
        - Optional binarization (0/1) with hysteresis
        - Optional freeze-after-release to avoid 'do it twice' loops
        """
        s_raw = float(np.clip(float(s_raw), 0.0, 1.0))
        if not self._config.gripper_binarize:
            return s_raw

        if not self._config.gripper_hysteresis:
            s_bin = 1.0 if s_raw >= self._config.gripper_thr else 0.0
        else:
            st = int(self._grip_bin.get(side, 0))
            if st == 0 and s_raw >= self._config.gripper_thr_close:
                st = 1
            elif st == 1 and s_raw <= self._config.gripper_thr_open:
                st = 0
            self._grip_bin[side] = st
            s_bin = float(st)

        prev = int(self._prev_grip_bin.get(side, 0))
        nowb = int(round(s_bin))
        if prev == 1 and nowb == 0 and self._config.freeze_after_release_s > 0.0:
            self._freeze_until = max(self._freeze_until, t_now_second + self._config.freeze_after_release_s)
        self._prev_grip_bin[side] = nowb
        return s_bin


@dataclass(frozen=True)
class PositionCommand(object):
    joint_names: List[str]
    joint_values: List[float]


@dataclass
class W1PositionCommand(object):
    robot_cmd: Optional[PositionCommand] = None
    left_hand_cmd: Optional[PositionCommand] = None
    right_hand_cmd: Optional[PositionCommand] = None


class W1RolloutRobotDoF(object):

    def __init__(self, config: W1RolloutConfig):
        self._config = config
        self._full_order: List[str] = self._build_full_order(config, config.selected_body_names)
        self._full_dim: int = len(self._full_order)
        self._body_order: List[str] = [n for n in self._full_order if n in config.body_canonical]

        # Index for scalar left/right
        self._idx_left_scalar: Optional[int] = None
        self._idx_right_scalar: Optional[int] = None
        for i, n in enumerate(self._full_order):
            if n == config.left_scalar_name:  self._idx_left_scalar = i
            if n == config.right_scalar_name: self._idx_right_scalar = i

        # Index for q6 dof
        self._slice_left_q6: Optional[Tuple[int, int]] = None
        self._slice_right_q6: Optional[Tuple[int, int]] = None
        if 'left' in config.hand_sides and config.hand_input_mode == 'qpos6':
            self._slice_left_q6 = self._find_block(config.left_q6_names)
        if 'right' in config.hand_sides and config.hand_input_mode == 'qpos6':
            self._slice_right_q6 = self._find_block(config.right_q6_names)

        # Body index map
        self._body_index_map: np.ndarray = self._build_body_index_map(config)
        assert self._body_index_map is not None

    @property
    def full_order(self):
        return self._full_order

    @property
    def full_dim(self):
        return self._full_dim

    @property
    def body_order(self):
        return self._body_order

    @property
    def idx_left_scalar(self):
        return self._idx_left_scalar

    @property
    def idx_right_scalar(self):
        return self._idx_right_scalar

    @property
    def slice_left_q6(self):
        return self._slice_left_q6

    @property
    def slice_right_q6(self):
        return self._slice_right_q6

    @property
    def body_index_map(self):
        return self._body_index_map

    def map_scalar_to_qpos6(self, s: float) -> np.ndarray:
        s = float(np.clip(s, 0.0, 1.0))
        start, end = self._config.hand_interp_start, self._config.hand_interp_end
        if start not in self._config.hand_gestures or end not in self._config.hand_gestures:
            raise ValueError(f"Hand gesture must be in {list(self._config.hand_gestures.keys())}")
        base = np.asarray(self._config.hand_gestures[start], dtype=np.float32)
        target = np.asarray(self._config.hand_gestures[end], dtype=np.float32)
        return base * (1.0 - s) + target * s

    def action_to_np(self, action) -> Optional[np.ndarray]:
        if isinstance(action, torch.Tensor):
            act = action.detach().cpu().numpy()
            if act.ndim == 2:
                act = act[0]
        else:
            act = np.asarray(action, dtype=np.float32)

        if act.shape[-1] != self.full_dim:
            raise ValueError(
                f'Policy dim mismatch: expected {self.full_dim}, got {act.shape}. '
                f'Order: {self.full_order}'
            )
            return None

        return act.astype(np.float32, copy=True)

    def make_action(self,
                    act_np: np.ndarray,
                    processor: W1RolloutGripperProcess,
                    t_now_second: float) -> W1PositionCommand:
        # Body
        position_cmd = W1PositionCommand()
        pub_names_body: List[str] = list()
        pub_pos_body: List[float] = list()
        for i, n in enumerate(self.full_order):
            if n in self._config.body_canonical and n not in self._config.drop_joint_names:
                pub_names_body.append(n)
                pub_pos_body.append(float(act_np[i]))
        if len(pub_names_body) > 0:
            position_cmd.robot_cmd = PositionCommand(
                joint_names=pub_names_body, joint_values=pub_pos_body)

        # Hand
        if self._config.hand_input_mode == 'scalar':
            if 'left' in self._config.hand_sides and self.idx_left_scalar is not None:
                s_raw = float(act_np[self.idx_left_scalar])
                s = processor.postprocess_gripper_scalar(s_raw, side='left', t_now_second=t_now_second)
                act_np[self.idx_left_scalar] = s
                q6 = self.map_scalar_to_qpos6(s)
                position_cmd.left_hand_cmd = PositionCommand(
                    joint_names=self._config.publish_hand_joint_name,
                    joint_values=[float(x) for x in q6])

            if 'right' in self._config.hand_sides and self.idx_right_scalar is not None:
                s_raw = float(act_np[self.idx_right_scalar])
                s = processor.postprocess_gripper_scalar(s_raw, side='right', t_now_second=t_now_second)
                act_np[self.idx_right_scalar] = s
                q6 = self.map_scalar_to_qpos6(s)
                position_cmd.right_hand_cmd = PositionCommand(
                    joint_names=self._config.publish_hand_joint_name,
                    joint_values=[float(x) for x in q6])

        elif self._config.hand_input_mode == 'qpos6':
            if 'left' in self._config.hand_sides and self.slice_left_q6 is not None:
                a, b = self.slice_left_q6
                v = act_np[a:b]
                position_cmd.left_hand_cmd = PositionCommand(
                    joint_names=self._config.publish_hand_joint_name,
                    joint_values=[float(x) for x in v])
            if 'right' in self._config.hand_sides and self.slice_right_q6 is not None:
                a, b = self.slice_right_q6
                v = act_np[a:b]
                position_cmd.right_hand_cmd = PositionCommand(
                    joint_names=self._config.publish_hand_joint_name,
                    joint_values=[float(x) for x in v])

        # Done
        return position_cmd

    @staticmethod
    def _build_full_order(config: W1RolloutConfig, selected_raw: List[str]) -> List[str]:
        mode = config.hand_input_mode
        sides = config.hand_sides
        out: List[str] = []
        body_set = set(config.body_canonical)

        for n in selected_raw:
            if n in body_set:
                out.append(n)
                continue

            if mode == 'none':
                continue

            if mode == 'scalar':
                if n == config.left_scalar_name and 'left' in sides:
                    out.append(config.left_scalar_name)
                    continue
                if n == config.right_scalar_name and 'right' in sides:
                    out.append(config.right_scalar_name)
                    continue
                continue

            if mode == 'qpos6':
                if n == config.left_scalar_name and 'left' in sides:
                    out.extend(config.left_q6_names)
                    continue
                if n == config.right_scalar_name and 'right' in sides:
                    out.extend(config.right_q6_names)
                    continue
                if n in config.left_q6_names and 'left' in sides:
                    out.append(n);
                    continue
                if n in config.right_q6_names and 'right' in sides:
                    out.append(n);
                    continue
                continue

        if mode == 'scalar':
            if 'left' in sides and config.left_scalar_name not in out:
                out.append(config.left_scalar_name)
            if 'right' in sides and config.right_scalar_name not in out:
                out.append(config.right_scalar_name)
        elif mode == 'qpos6':
            def ensure_block(block: List[str]):
                if not any(n in out for n in block):
                    out.extend(block)

            if 'left' in sides:
                ensure_block(config.left_q6_names)
            if 'right' in sides:
                ensure_block(config.right_q6_names)

        return out

    def _find_block(self, names: List[str]) -> Optional[Tuple[int, int]]:
        L = len(names)
        for i in range(0, len(self._full_order) - L + 1):
            if self._full_order[i:i + L] == names:
                return i, i + L
        return None

    def _build_body_index_map(self, config: W1RolloutConfig) -> Optional[np.ndarray]:
        name_to_idx = {n: i for i, n in enumerate(config.joint_names)}
        idxs, missing = [], []
        for n in self._body_order:
            if n in name_to_idx:
                idxs.append(name_to_idx[n])
            else:
                missing.append(n)

        # Make output
        if len(missing) > 0:
            return None
        return np.asarray(idxs, dtype=np.int64)


# sandbox below
def _sandbox_rollout_config():
    inst = W1RolloutConfig(policy_hz=100)
    inst.to_json_file('test.json')
    out = W1RolloutConfig.from_json_file('test.json')
    print("policy_hz:", out.policy_hz)
    print("device:", out.device)
    print("remote_server_host:", out.remote_server_host)
    print("image_keys:", out.image_keys)
    print("ordered_body_names:", out.ordered_body_names)
    print("hand_sides:", out.hand_sides)
    print("gripper_binarize:", out.gripper_binarize)

    dof = W1RolloutRobotDoF(config=out)
    print(dof.full_order)


if __name__ == '__main__':
    _sandbox_rollout_config()
