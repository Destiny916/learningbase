#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig
from lerobot.utils.constants import OBS_STATE


RELATIVE_JOINT_FEATURE_NAMES = [f"joint_{index}" for index in range(6)] + ["gripper"]


@PreTrainedConfig.register_subclass("act")
@dataclass
class ACTConfig(PreTrainedConfig):
    """Configuration class for the Action Chunking Transformers policy.

    Defaults are configured for training on bimanual Aloha tasks like "insertion" or "transfer".

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_features` and `output_features`.

    Notes on the inputs and outputs:
        - Either:
            - At least one key starting with "observation.image is required as an input.
              AND/OR
            - The key "observation.environment_state" is required as input.
        - If there are multiple keys beginning with "observation.images." they are treated as multiple camera
          views. Right now we only support all images having the same shape.
        - May optionally work without an "observation.state" key for the proprioceptive robot state.
        - "action" is required as an output key.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        chunk_size: The size of the action prediction "chunks" in units of environment steps.
        n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
            This should be no greater than the chunk size. For example, if the chunk size size 100, you may
            set this to 50. This would mean that the model predicts 100 steps worth of actions, runs 50 in the
            environment, and throws the other 50 out.
        input_features: A dictionary defining the PolicyFeature of the input data for the policy. The key represents
            the input data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        output_features: A dictionary defining the PolicyFeature of the output data for the policy. The key represents
            the output data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        normalization_mapping: A dictionary that maps from a str value of FeatureType (e.g., "STATE", "VISUAL") to
            a corresponding NormalizationMode (e.g., NormalizationMode.MIN_MAX)
        vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
        pretrained_backbone_weights: Pretrained weights from torchvision to initialize the backbone.
            `None` means no pretrained weights.
        replace_final_stride_with_dilation: Whether to replace the ResNet's final 2x2 stride with a dilated
            convolution.
        pre_norm: Whether to use "pre-norm" in the transformer blocks.
        dim_model: The transformer blocks' main hidden dimension.
        n_heads: The number of heads to use in the transformer blocks' multi-head attention.
        dim_feedforward: The dimension to expand the transformer's hidden dimension to in the feed-forward
            layers.
        feedforward_activation: The activation to use in the transformer block's feed-forward layers.
        n_encoder_layers: The number of transformer layers to use for the transformer encoder.
        n_decoder_layers: The number of transformer layers to use for the transformer decoder.
        use_vae: Whether to use a variational objective during training. This introduces another transformer
            which is used as the VAE's encoder (not to be confused with the transformer encoder - see
            documentation in the policy class).
        latent_dim: The VAE's latent dimension.
        n_vae_encoder_layers: The number of transformer layers to use for the VAE's encoder.
        temporal_ensemble_coeff: Coefficient for the exponential weighting scheme to apply for temporal
            ensembling. Defaults to None which means temporal ensembling is not used. `n_action_steps` must be
            1 when using this feature, as inference needs to happen at every step to form an ensemble. For
            more information on how ensembling works, please see `ACTTemporalEnsembler`.
        dropout: Dropout to use in the transformer layers (see code for details).
        kl_weight: The weight to use for the KL-divergence component of the loss if the variational objective
            is enabled. Loss is then calculated as: `reconstruction_loss + kl_weight * kld_loss`.
    """

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    joint_representation: str = "absolute"
    # Independent pose10d workflow; the existing joint_representation remains 7D-only.
    end_effector_pose_representation: str = "absolute"
    pose_state_stats_path: str | None = None
    pose_action_stats_path: str | None = None
    condition_on_state: bool = True
    gripper_indices: list[int] = field(default_factory=lambda: [6])
    state_feature_names: list[str] | None = None
    state_gripper_indices: list[int] | None = None
    state_absolute_indices: list[int] = field(default_factory=list)
    action_feature_names: list[str] | None = None
    relative_state_stats_path: str | None = None
    relative_action_stats_path: str | None = None
    clip_quantiles: bool = True
    state_noise_std_rad: float = 0.0
    state_position_noise_std_m: float = 0.0
    gripper_noise_std_m: float = 0.0
    state_position_indices: list[int] = field(default_factory=list)

    # Opt-in StereoPolicy-style external stereo pair. The default keeps the existing ACT visual path exactly.
    stereo_visual_mode: str = "standard_rgb"

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Architecture.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False
    # Transformer layers.
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    # Note: Although the original ACT implementation has 7 for `n_decoder_layers`, there is a bug in the code
    # that means only the first layer is used. Here we match the original implementation by setting this to 1.
    # See this issue https://github.com/tonyzhaozh/act/issues/25#issue-2258740521.
    n_decoder_layers: int = 1
    # VAE.
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # Inference.
    # Note: the value used in ACT when temporal ensembling is enabled is 0.01.
    temporal_ensemble_coeff: float | None = None

    # Training and loss computation.
    dropout: float = 0.1
    kl_weight: float = 10.0

    # Training preset
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        super().__post_init__()

        if self.joint_representation not in {"absolute", "relative"}:
            raise ValueError(f"Invalid joint_representation: {self.joint_representation}")
        if self.end_effector_pose_representation not in {"absolute", "relative"}:
            raise ValueError(
                "end_effector_pose_representation must be 'absolute' or 'relative'"
            )
        if min(self.state_noise_std_rad, self.state_position_noise_std_m, self.gripper_noise_std_m) < 0:
            raise ValueError("state noise standard deviations must be non-negative")
        if self.stereo_visual_mode not in {
            "standard_rgb",
            "stereo_top_rgb",
            "stereo_top_rgbd",
            "five_camera_rgbd",
        }:
            raise ValueError(f"Invalid stereo_visual_mode: {self.stereo_visual_mode}")
        if self.stereo_visual_mode != "standard_rgb":
            # DINOv2 receives original RGB [0, 1] and applies its own ImageNet transform.
            self.normalization_mapping = {**self.normalization_mapping, "VISUAL": NormalizationMode.IDENTITY}
        if self.joint_representation == "relative":
            self.normalization_mapping = {
                **self.normalization_mapping,
                "STATE": NormalizationMode.QUANTILES,
                "ACTION": NormalizationMode.QUANTILES,
            }

        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. This is "
                "because the policy needs to be queried every step to compute the ensembled action."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )

        self._relative_joint_stats = None

    def _validate_relative_7d_contract(self) -> None:
        self._relative_joint_stats = None
        if self.joint_representation != "relative":
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
        state_names = self.state_feature_names or self.action_feature_names
        if len(state_names) != state_dimension or len(set(state_names)) != state_dimension:
            raise ValueError(
                f"matching relative joint state/action feature names must contain {state_dimension} unique names"
            )
        state_grippers = self.state_gripper_indices or [
            index for index, name in enumerate(state_names) if "gripper" in name.lower()
        ]
        if not state_grippers and state_dimension == action_dimension:
            state_grippers = list(self.gripper_indices)
        if not self.gripper_indices or len(set(self.gripper_indices)) != len(self.gripper_indices):
            raise ValueError("relative joint action gripper indices must be a nonempty unique list")
        if any(index < 0 or index >= action_dimension for index in self.gripper_indices):
            raise ValueError(f"relative joint action gripper indices must be within [0, {action_dimension})")
        if not state_grippers or len(set(state_grippers)) != len(state_grippers):
            raise ValueError("matching relative joint state gripper indices must be a nonempty unique list")
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
            expected_gripper_indices=self.gripper_indices,
            expected_state_feature_names=state_names,
            expected_state_gripper_indices=state_grippers,
            expected_state_absolute_indices=self.state_absolute_indices,
        )
        self._relative_joint_stats = relative_joint_stats

    @property
    def relative_joint_stats(self):
        return self._relative_joint_stats

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        self._relative_joint_stats = None
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")
        if self.joint_representation == "relative" and OBS_STATE not in self.input_features:
            raise ValueError("relative joint representation requires observation.state for label conversion")
        if self.stereo_visual_mode != "standard_rgb":
            if self.stereo_visual_mode == "five_camera_rgbd":
                required = {
                    "observation.images.top",
                    "observation.images.gripper_left",
                    "observation.images.gripper_right",
                    "observation.images.gripper_left_depth",
                    "observation.images.gripper_right_depth",
                }
            else:
                required = {
                    "observation.images.top_left",
                    "observation.images.top_right",
                    "observation.images.gripper_left",
                    "observation.images.gripper_right",
                }
                if self.stereo_visual_mode == "stereo_top_rgbd":
                    required.update(
                        {
                            "observation.images.gripper_left_depth",
                            "observation.images.gripper_right_depth",
                        }
                    )
            observed = set(self.image_features)
            if observed != required:
                raise ValueError(
                    f"{self.stereo_visual_mode} requires exactly {sorted(required)}, got {sorted(observed)}"
                )
        self._validate_relative_7d_contract()

    @property
    def observation_delta_indices(self) -> list[int] | None:
        if self.joint_representation == "relative" or self.end_effector_pose_representation == "relative":
            return [-1, 0]
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
