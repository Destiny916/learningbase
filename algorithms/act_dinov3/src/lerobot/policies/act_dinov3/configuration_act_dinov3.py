from dataclasses import dataclass
from typing import Any

from lerobot.configs import PreTrainedConfig
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
    dinov3_num_register_tokens: int = 4
    dinov3_patch_size: int = 16
    dinov3_model_config: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
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
