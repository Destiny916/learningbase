"""LeRobot v3 modality contract for the Joint Songling dual-arm dataset."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class JointSonglingDataConfig:
    """Three-view, dual-arm data with state/action layout left-then-right."""

    video_keys = ["video.top", "video.gripper_left", "video.gripper_right"]
    state_keys = [
        "state.left_joints",
        "state.left_gripper",
        "state.right_joints",
        "state.right_gripper",
    ]
    action_keys = [
        "action.left_joints",
        "action.left_gripper",
        "action.right_joints",
        "action.right_gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    state_indices = [-1, 0]
    action_indices = list(range(50))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        state_modes = {key: "q99" for key in self.state_keys}
        action_modes = {key: "q99" for key in self.action_keys}
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(apply_to=self.state_keys, normalization_modes=state_modes),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(apply_to=self.action_keys, normalization_modes=action_modes),
            ]
        )


class JointSonglingXYZ17DataConfig(JointSonglingDataConfig):
    """Joint Songling data with absolute bread xyz appended to state."""

    state_keys = JointSonglingDataConfig.state_keys + ["state.bread_position"]


class JointSonglingEndpoint20DataConfig(JointSonglingDataConfig):
    """730 endpoint dataset with the complete 20D observation state."""

    state_keys = JointSonglingDataConfig.state_keys + [
        "state.right_endpoint",
        "state.left_endpoint",
    ]


class JointSonglingSwapEndpoint20DataConfig(JointSonglingDataConfig):
    """0806 swap dataset: joints, endpoint xyz, then gripper for each arm."""

    state_keys = [
        "state.left_joints",
        "state.left_endpoint",
        "state.left_gripper",
        "state.right_joints",
        "state.right_endpoint",
        "state.right_gripper",
    ]


class JointSonglingSwapEndpoint20Temporal2DataConfig(JointSonglingSwapEndpoint20DataConfig):
    """0806 swap data with a previous/current three-camera visual context."""

    observation_indices = [-1, 0]


ROBOT_TYPE_CONFIG_MAP = {
    "joint_songling_dualarm14": JointSonglingDataConfig(),
    "joint_songling_xyz17": JointSonglingXYZ17DataConfig(),
    "joint_songling_endpoint20": JointSonglingEndpoint20DataConfig(),
    "joint_songling_0806swap_endpoint20": JointSonglingSwapEndpoint20DataConfig(),
    "joint_songling_0806swap_endpoint20_t2": JointSonglingSwapEndpoint20Temporal2DataConfig(),
}
ROBOT_TYPE_TO_EMBODIMENT_TAG = {}
DATASET_NAMED_MIXTURES = {
    "joint_songling_full99": [("", 1.0, "joint_songling_dualarm14")],
    "joint_songling_xyz17_full37": [("", 1.0, "joint_songling_xyz17")],
    "joint_songling_730_endpoint20": [("", 1.0, "joint_songling_endpoint20")],
    "joint_songling_0806swap_endpoint20": [("", 1.0, "joint_songling_0806swap_endpoint20")],
    "joint_songling_0806swap_endpoint20_t2": [("", 1.0, "joint_songling_0806swap_endpoint20_t2")],
}
