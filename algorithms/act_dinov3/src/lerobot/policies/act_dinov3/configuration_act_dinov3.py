from dataclasses import dataclass
from typing import Any

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("act_dinov3")
@dataclass
class ACTDINOv3Config(ACTConfig):
    """ACT configuration using one shared DINOv3 image encoder."""

    pretrained_backbone_weights: str | None = None
    dinov3_pretrained_path: str = (
        "/data/wengyikun/models/turbovla_joint_songling/dinov3-vitl16-pretrain-lvd1689m"
    )
    dinov3_learning_rate: float = 1e-6
    dinov3_gradient_checkpointing: bool = True
    dinov3_autocast_dtype: str = "bfloat16"
    dinov3_apply_image_normalization: bool = True
    dinov3_num_register_tokens: int = 4
    dinov3_patch_size: int = 16
    dinov3_model_config: dict[str, Any] | None = None
    scheduler_warmup_steps: int = 25_000
    scheduler_decay_steps: int = 500_000
    scheduler_decay_lr: float = 1e-6
    # Training-only auxiliary EE/FK metadata kept in saved checkpoints.
    # Declaring these fields preserves config compatibility during inference;
    # they do not alter the 19D ACT action head.
    ee_pose_loss_weight: float = 0.0
    fk_loss_weight: float = 0.0
    kinematics_urdf_path: str | None = None
    kinematics_urdf_sha256: str | None = None
    ee_reference_link: str | None = None
    ee_left_link: str | None = None
    ee_right_link: str | None = None
    ee_position_scale_m: float = 0.1
    ee_rotation_loss_weight: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        # ACT's generic visual normalization uses dataset mean/std. DINOv3
        # needs ImageNet normalization instead, applied inside its backbone.
        self.normalization_mapping = {
            **self.normalization_mapping,
            "VISUAL": NormalizationMode.IDENTITY,
        }
        if not self.dinov3_pretrained_path:
            raise ValueError("DINOv3 initialization path must be provided")
        if self.dinov3_autocast_dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError(
                "dinov3_autocast_dtype must be one of 'bfloat16', 'float16', or 'float32'"
            )
        if self.dinov3_num_register_tokens < 0:
            raise ValueError("dinov3_num_register_tokens must be non-negative")
        if self.dinov3_patch_size <= 0:
            raise ValueError("dinov3_patch_size must be positive")
        if self.scheduler_warmup_steps < 0 or self.scheduler_decay_steps <= self.scheduler_warmup_steps:
            raise ValueError("scheduler_decay_steps must be greater than scheduler_warmup_steps >= 0")

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            decay_after_warmup=True,
        )
