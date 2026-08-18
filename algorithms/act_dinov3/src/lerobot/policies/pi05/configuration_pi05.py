#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import torch

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from ..rtc.configuration_rtc import RTCConfig

DEFAULT_IMAGE_SIZE = 224
RELATIVE_JOINT_FEATURE_NAMES = [f"joint_{index}" for index in range(6)] + ["gripper"]
POSE10D_FEATURE_NAMES = [
    "x",
    "y",
    "z",
    "rot6d_0",
    "rot6d_1",
    "rot6d_2",
    "rot6d_3",
    "rot6d_4",
    "rot6d_5",
    "gripper",
]
DUAL_ARM_20D_STATE_FEATURE_NAMES = (
    [f"left_joint_{index}" for index in range(6)]
    + ["left_endpoint_x", "left_endpoint_y", "left_endpoint_z", "left_gripper"]
    + [f"right_joint_{index}" for index in range(6)]
    + ["right_endpoint_x", "right_endpoint_y", "right_endpoint_z", "right_gripper"]
)
DUAL_ARM_14D_ACTION_FEATURE_NAMES = (
    [f"left_joint_{index}" for index in range(6)]
    + ["left_gripper"]
    + [f"right_joint_{index}" for index in range(6)]
    + ["right_gripper"]
)


@PreTrainedConfig.register_subclass("pi05")
@dataclass
class PI05Config(PreTrainedConfig):
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"  # Options: "bfloat16", "float32"

    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict, in openpi called "action_horizon"
    n_action_steps: int = 50  # Number of action steps to execute

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters: see openpi `PI0Pytorch`
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # Relative actions: converts absolute actions to relative (relative to state).
    use_relative_actions: bool = False
    # Joint names to exclude from relative (kept absolute). Empty list = all dims relative.
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    # Apply Piper/Pika physical action limits after inference. Disable for policies whose
    # action space is already native to another environment, such as LIBERO's 7D delta action.
    apply_action_limits: bool = True
    # Populated at runtime from dataset metadata by make_policy.
    state_feature_names: list[str] | None = None
    action_feature_names: list[str] | None = None

    # Joint representation for Piper/Pika 14D training.
    joint_representation: str = "absolute"  # Options: "absolute", "relative"
    # Independent pose10d workflow; this does not reuse the 7D joint branch above.
    end_effector_pose_representation: str = "absolute"  # Options: "absolute", "relative"
    pose_state_stats_path: str | None = None
    pose_action_stats_path: str | None = None
    joint_limit_profile: str = "piper_pika_14d"
    joint_limit_path: str | None = None
    joint_gripper_indices: list[int] = field(default_factory=lambda: [6, 13])
    state_gripper_indices: list[int] | None = None
    state_absolute_indices: list[int] = field(default_factory=list)
    condition_on_state: bool = True
    relative_state_stats_path: str | None = None
    relative_action_stats_path: str | None = None
    # Opt-in train-split quantiles for isolated absolute 7D and pose10d training.
    absolute_state_stats_path: str | None = None
    absolute_action_stats_path: str | None = None
    clip_quantiles: bool = True
    state_noise_std_rad: float = 0.0
    state_position_noise_std_m: float = 0.0
    state_position_indices: list[int] = field(default_factory=list)
    gripper_noise_std_m: float = 0.0
    # The dataset already stores relative state and [chunk_size, action_dim] targets.
    precomputed_relative_chunk: bool = False

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None
    # Maximum clean action-prefix length sampled during training. Zero disables trained RTC.
    rtc_training_max_delay: int = 0

    image_resolution: tuple[int, int] = (
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )  # see openpi `preprocessing_pytorch.py`
    image_feature_order: list[str] | None = None

    # Add empty images. Used to add empty cameras when no image features are present.
    empty_cameras: int = 0

    tokenizer_max_length: int = 200  # see openpi `__post_init__`
    tokenizer_name: str = "google/paligemma-3b-pt-224"

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for state
            "ACTION": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for action
        }
    )

    # Training settings
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization
    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    device: str | None = None  # Device to use for the model (None = auto-detect)

    # Finetuning settings
    freeze_vision_encoder: bool = False  # Freeze only the vision encoder
    freeze_language_model: bool = True  # Freeze only the language backbone, keep vision/projector trainable
    train_expert_only: bool = False  # Freeze entire VLM, train only action expert and projections
    # Optional visual-only initialization from another PI05/VISTA checkpoint.
    # Loads vision_tower and, by default, multi_modal_projector.
    visual_pretrained_path: str | None = None
    visual_pretrained_include_projector: bool = True

    # Optimizer settings: see openpi `AdamW`
    optimizer_lr: float = 2.5e-5  # see openpi `CosineDecaySchedule: peak_lr`
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # Scheduler settings: see openpi `CosineDecaySchedule`
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    tokenizer_max_length: int = 200  # see openpi `__post_init__`

    def __post_init__(self):
        super().__post_init__()

        # Validate configuration
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )
        if not 0 <= self.rtc_training_max_delay < self.chunk_size:
            raise ValueError(
                "rtc_training_max_delay must satisfy "
                f"0 <= delay < chunk_size ({self.chunk_size}), got {self.rtc_training_max_delay}"
            )

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")
        if self.joint_representation not in ["absolute", "relative"]:
            raise ValueError(f"Invalid joint_representation: {self.joint_representation}")
        if self.end_effector_pose_representation not in ["absolute", "relative"]:
            raise ValueError("Invalid end_effector_pose_representation")
        if min(self.state_noise_std_rad, self.state_position_noise_std_m, self.gripper_noise_std_m) < 0:
            raise ValueError("state noise standard deviations must be non-negative")

        if self.precomputed_relative_chunk and self.joint_representation != "relative":
            raise ValueError("precomputed_relative_chunk requires joint_representation='relative'")

        if self.freeze_language_model and self.train_expert_only:
            raise ValueError("freeze_language_model and train_expert_only are mutually exclusive")

        self._relative_joint_stats = None
        self._absolute_action_stats = None
        self._absolute_state_relative_action_stats = None

    def _validate_relative_7d_contract(self) -> None:
        self._relative_joint_stats = None
        if self.joint_representation != "relative" or self.precomputed_relative_chunk:
            return
        state_feature = self.robot_state_feature
        action_feature = self.action_feature
        if state_feature is None or action_feature is None:
            return
        if len(state_feature.shape) != 1 or len(action_feature.shape) != 1:
            raise ValueError(
                "relative joint representation requires one-dimensional state and action features; "
                f"got state shape {state_feature.shape} and action shape {action_feature.shape}"
            )
        state_dimension = state_feature.shape[0]
        action_dimension = action_feature.shape[0]
        if (
            self.action_feature_names is None
            or len(self.action_feature_names) != action_dimension
            or len(set(self.action_feature_names)) != action_dimension
        ):
            raise ValueError(
                f"matching relative joint action feature names must contain {action_dimension} unique names"
            )
        state_names = self.state_feature_names or (
            DUAL_ARM_20D_STATE_FEATURE_NAMES if state_dimension == 20 else self.action_feature_names
        )
        if len(state_names) != state_dimension or len(set(state_names)) != state_dimension:
            raise ValueError(
                f"matching relative joint state/action feature names must contain {state_dimension} unique names"
            )
        state_grippers = self.state_gripper_indices or [
            index for index, name in enumerate(state_names) if "gripper" in name.lower()
        ]
        if not state_grippers and state_dimension == action_dimension:
            state_grippers = list(self.joint_gripper_indices)
        if not self.joint_gripper_indices or len(set(self.joint_gripper_indices)) != len(self.joint_gripper_indices):
            raise ValueError("relative joint action gripper indices must be a nonempty unique list")
        if any(index < 0 or index >= action_dimension for index in self.joint_gripper_indices):
            raise ValueError(f"relative joint action gripper indices must be within [0, {action_dimension})")
        if not state_grippers or len(set(state_grippers)) != len(state_grippers):
            raise ValueError("relative joint state gripper indices must be a nonempty unique list")
        if any(index < 0 or index >= state_dimension for index in state_grippers):
            raise ValueError(f"relative joint state gripper indices must be within [0, {state_dimension})")
        if not self.relative_state_stats_path or not self.relative_action_stats_path:
            raise ValueError("relative 7D state and action stats path values must both be provided")

        from lerobot.datasets.relative_joint_stats import load_relative_joint_stats_paths

        relative_joint_stats = load_relative_joint_stats_paths(
            self.relative_state_stats_path,
            self.relative_action_stats_path,
            expected_horizon=self.chunk_size,
            expected_feature_names=self.action_feature_names,
            expected_gripper_indices=self.joint_gripper_indices,
            expected_state_feature_names=state_names,
            expected_state_gripper_indices=state_grippers,
        )
        self._relative_joint_stats = relative_joint_stats

    def _validate_relative_end_effector_pose_contract(self) -> None:
        if self.end_effector_pose_representation != "relative":
            return
        state_feature = self.robot_state_feature
        action_feature = self.action_feature
        if state_feature is None or action_feature is None or state_feature.shape != (10,) or action_feature.shape != (10,):
            raise ValueError("relative end-effector pose representation requires 10D state and action features")
        if self.joint_gripper_indices != [9]:
            raise ValueError("relative end-effector pose joint_gripper_indices must be [9]")

    def _validate_absolute_stats_contract(self) -> None:
        self._absolute_action_stats = None
        if self._absolute_state_relative_action_stats is not None:
            return
        if not self.absolute_state_stats_path and not self.absolute_action_stats_path:
            return
        if not self.absolute_state_stats_path or not self.absolute_action_stats_path:
            raise ValueError("absolute state and action stats path values must both be provided")
        state_feature = self.robot_state_feature
        action_feature = self.action_feature
        if state_feature is None or action_feature is None or state_feature.shape != action_feature.shape:
            raise ValueError("absolute statistics require matching state and action feature dimensions")

        dimension = state_feature.shape[0]
        if dimension == 7:
            if self.joint_representation != "absolute" or self.joint_gripper_indices != [6]:
                raise ValueError("absolute 7D statistics require absolute joints and joint_gripper_indices=[6]")
            feature_names = RELATIVE_JOINT_FEATURE_NAMES
            scaled_indices = list(range(7))
        elif dimension == 10:
            if self.end_effector_pose_representation != "absolute" or self.joint_gripper_indices != [9]:
                raise ValueError("absolute pose10d statistics require absolute pose and joint_gripper_indices=[9]")
            feature_names = POSE10D_FEATURE_NAMES
            scaled_indices = [0, 1, 2, 9]
        else:
            raise ValueError("absolute statistics support only 7D joints or pose10d features")
        if self.action_feature_names is not None and self.action_feature_names != feature_names:
            raise ValueError(f"absolute statistics require action feature names {feature_names}")

        from lerobot.datasets.absolute_action_stats import load_absolute_action_stats_paths

        self._absolute_action_stats = load_absolute_action_stats_paths(
            self.absolute_state_stats_path,
            self.absolute_action_stats_path,
            expected_horizon=self.chunk_size,
            feature_names=feature_names,
            scaled_indices=scaled_indices,
        )

    def _validate_absolute_state_relative_action_contract(self) -> None:
        self._absolute_state_relative_action_stats = None
        if self.joint_representation != "absolute" or not self.use_relative_actions:
            return
        if not self.absolute_state_stats_path or not self.relative_action_stats_path:
            raise ValueError(
                "absolute-state relative-action training requires absolute_state_stats_path "
                "and relative_action_stats_path"
            )
        if self.absolute_action_stats_path or self.relative_state_stats_path:
            raise ValueError(
                "absolute-state relative-action training must not provide absolute action or relative state stats"
            )

        state_feature = self.robot_state_feature
        action_feature = self.action_feature
        if state_feature is None or action_feature is None or action_feature.shape != (14,):
            raise ValueError("absolute-state relative-action training requires a 14D action feature")
        if state_feature.shape not in {(14,), (20,)}:
            raise ValueError("absolute-state relative-action training requires a 14D or 20D state feature")
        if self.joint_gripper_indices != [6, 13]:
            raise ValueError("absolute-state relative-action training requires joint_gripper_indices=[6,13]")
        if self.action_feature_names is None or len(self.action_feature_names) != 14:
            raise ValueError("absolute-state relative-action training requires 14 action feature names")

        excluded_indices = [
            index
            for index, name in enumerate(self.action_feature_names)
            if any(token.lower() in name.lower() for token in self.relative_exclude_joints)
        ]
        if excluded_indices != self.joint_gripper_indices:
            raise ValueError("relative_exclude_joints must select exactly the configured gripper indices")

        from lerobot.datasets.relative_joint_stats import load_relative_joint_stats_paths

        absolute_state_path = Path(self.absolute_state_stats_path)
        try:
            state_payload = json.loads(absolute_state_path.read_text(encoding="utf-8"))
            state_q01 = torch.tensor(state_payload["q01"], dtype=torch.float32)
            state_q99 = torch.tensor(state_payload["q99"], dtype=torch.float32)
            state_count = int(state_payload["count"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid absolute state statistics: {absolute_state_path}: {error}") from error
        if state_q01.shape != state_feature.shape or state_q99.shape != state_feature.shape:
            raise ValueError("absolute state q01/q99 dimensions do not match observation.state")
        absolute_bundle = SimpleNamespace(
            state=SimpleNamespace(q01=state_q01, q99=state_q99, count=state_count)
        )
        relative_action_path = Path(self.relative_action_stats_path)
        state_names = self.state_feature_names or (
            DUAL_ARM_20D_STATE_FEATURE_NAMES if state_feature.shape == (20,) else self.action_feature_names
        )
        state_grippers = self.state_gripper_indices or [
            index for index, name in enumerate(state_names) if "gripper" in name.lower()
        ]
        relative_bundle = load_relative_joint_stats_paths(
            relative_action_path.with_name("relative_state_q01_q99.json"),
            relative_action_path,
            expected_horizon=self.chunk_size,
            expected_feature_names=self.action_feature_names,
            expected_gripper_indices=self.joint_gripper_indices,
            expected_state_feature_names=state_names,
            expected_state_gripper_indices=state_grippers,
        )
        if absolute_bundle.state.count != relative_bundle.state.count:
            raise ValueError("absolute state and relative action statistics must describe the same frame count")
        self._absolute_state_relative_action_stats = (absolute_bundle, relative_bundle)

    @property
    def relative_joint_stats(self):
        return self._relative_joint_stats

    @property
    def absolute_action_stats(self):
        if self._absolute_action_stats is None and self.absolute_state_stats_path:
            self._validate_absolute_stats_contract()
        return self._absolute_action_stats

    @property
    def absolute_state_relative_action_stats(self):
        return self._absolute_state_relative_action_stats

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        if self.image_feature_order is not None and self.image_features:
            image_keys = list(self.image_features)
            if len(self.image_feature_order) != len(set(self.image_feature_order)):
                raise ValueError(f"image_feature_order contains duplicate keys: {self.image_feature_order}")
            if set(self.image_feature_order) != set(image_keys):
                raise ValueError(
                    "image_feature_order must contain every real image feature exactly once; "
                    f"expected {image_keys}, got {self.image_feature_order}"
                )
            non_image_features = {
                key: feature for key, feature in self.input_features.items() if key not in image_keys
            }
            self.input_features = {
                **{key: self.input_features[key] for key in self.image_feature_order},
                **non_image_features,
            }

        for i in range(self.empty_cameras):
            key = OBS_IMAGES + f".empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),  # Use configured image resolution
            )
            self.input_features[key] = empty_camera

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features[OBS_STATE] = state_feature

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features[ACTION] = action_feature

        state_feature = self.robot_state_feature
        action_feature = self.action_feature
        if (
            self.joint_representation == "relative"
            and state_feature is not None
            and action_feature is not None
            and state_feature.shape == (20,)
            and action_feature.shape == (14,)
        ):
            if self.state_feature_names is None:
                self.state_feature_names = list(DUAL_ARM_20D_STATE_FEATURE_NAMES)
            if self.action_feature_names is None:
                self.action_feature_names = list(DUAL_ARM_14D_ACTION_FEATURE_NAMES)

        self._validate_relative_end_effector_pose_contract()
        self._validate_relative_7d_contract()
        self._validate_absolute_state_relative_action_contract()
        self._validate_absolute_stats_contract()

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list[int] | None:
        if self.precomputed_relative_chunk:
            return None
        if self.joint_representation == "relative" or self.end_effector_pose_representation == "relative":
            return [-1, 0]
        return None

    @property
    def action_delta_indices(self) -> list[int] | None:
        if self.precomputed_relative_chunk:
            return None
        # action[t] is the target q_{t+1}; selecting index 0 yields t+1.
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
