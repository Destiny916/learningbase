from collections import deque

import torch
from torch import Tensor, nn

from lerobot.policies.act.modeling_act import ACT, ACTPolicy, ACTTemporalEnsembler
from lerobot.policies.pretrained import PreTrainedPolicy

from .configuration_act_dinov3 import ACTDINOv3Config
from .dinov3_backbone import DINOv3SpatialBackbone


class _ACTDINOv3FeatureMap(nn.Module):
    """Expose the spatial adapter through ACT's existing backbone contract."""

    def __init__(self, spatial: DINOv3SpatialBackbone, output_dtype: torch.dtype) -> None:
        super().__init__()
        self.spatial = spatial
        self.output_dtype = output_dtype

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        return {"feature_map": self.spatial(image).to(dtype=self.output_dtype)}


class ACTDINOv3(ACT):
    def __init__(self, config: ACTDINOv3Config, *, dinov3_model: nn.Module | None = None) -> None:
        super().__init__(config)

        if dinov3_model is not None:
            spatial = DINOv3SpatialBackbone(
                model=dinov3_model,
                hidden_size=int(dinov3_model.config.hidden_size),
                num_register_tokens=config.dinov3_num_register_tokens,
                patch_size=config.dinov3_patch_size,
                gradient_checkpointing=config.dinov3_gradient_checkpointing,
                autocast_dtype=config.dinov3_autocast_dtype,
                apply_image_normalization=config.dinov3_apply_image_normalization,
            )
        elif config.dinov3_model_config is not None:
            spatial = DINOv3SpatialBackbone.from_model_config(
                config.dinov3_model_config,
                num_register_tokens=config.dinov3_num_register_tokens,
                patch_size=config.dinov3_patch_size,
                gradient_checkpointing=config.dinov3_gradient_checkpointing,
                autocast_dtype=config.dinov3_autocast_dtype,
                apply_image_normalization=config.dinov3_apply_image_normalization,
            )
        else:
            spatial = DINOv3SpatialBackbone.from_pretrained(
                config.dinov3_pretrained_path,
                num_register_tokens=config.dinov3_num_register_tokens,
                patch_size=config.dinov3_patch_size,
                gradient_checkpointing=config.dinov3_gradient_checkpointing,
                autocast_dtype=config.dinov3_autocast_dtype,
                apply_image_normalization=config.dinov3_apply_image_normalization,
            )
        model_config = spatial.model.config
        config.dinov3_model_config = (
            model_config.to_dict() if hasattr(model_config, "to_dict") else vars(model_config).copy()
        )

        self.encoder_img_feat_input_proj = nn.Conv2d(
            spatial.hidden_size, config.dim_model, kernel_size=1
        )
        self.backbone = _ACTDINOv3FeatureMap(
            spatial, output_dtype=self.encoder_img_feat_input_proj.weight.dtype
        )


class ACTDINOv3Policy(ACTPolicy):
    config_class = ACTDINOv3Config
    name = "act_dinov3"

    def __init__(
        self,
        config: ACTDINOv3Config,
        *,
        dinov3_model: nn.Module | None = None,
        **kwargs,
    ) -> None:
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.model = ACTDINOv3(config, dinov3_model=dinov3_model)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(
                config.temporal_ensemble_coeff, config.chunk_size
            )
        else:
            self._action_queue = deque([], maxlen=config.n_action_steps)

    def get_optim_params(self) -> list[dict]:
        dinov3_prefix = "model.backbone.spatial.model."
        main_parameters = []
        dinov3_parameters = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith(dinov3_prefix):
                dinov3_parameters.append(parameter)
            else:
                main_parameters.append(parameter)

        return [
            {"params": main_parameters, "lr": self.config.optimizer_lr},
            {"params": dinov3_parameters, "lr": self.config.dinov3_learning_rate},
        ]
