from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import Tensor, nn


_AUTOCAST_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class DINOv3SpatialBackbone(nn.Module):
    """Convert shared DINOv3 patch tokens into ACT-compatible spatial maps."""

    def __init__(
        self,
        *,
        model: nn.Module,
        hidden_size: int,
        num_register_tokens: int,
        patch_size: int,
        gradient_checkpointing: bool,
        autocast_dtype: str,
    ) -> None:
        super().__init__()
        self.model = model
        self.hidden_size = hidden_size
        self.num_register_tokens = num_register_tokens
        self.patch_size = patch_size
        self.autocast_dtype = _AUTOCAST_DTYPES[autocast_dtype]
        model_config = getattr(model, "config", None)
        model_image_size = getattr(model_config, "image_size", 224)
        model_patch_size = getattr(model_config, "patch_size", patch_size)
        model_register_tokens = getattr(model_config, "num_register_tokens", num_register_tokens)
        if model_image_size != 224:
            raise ValueError(f"DINOv3 model image size must be 224, got {model_image_size}")
        if model_patch_size != patch_size:
            raise ValueError(
                f"DINOv3 model patch size {model_patch_size} does not match configured patch size {patch_size}"
            )
        if model_register_tokens != num_register_tokens:
            raise ValueError(
                "DINOv3 model register-token count "
                f"{model_register_tokens} does not match configured count {num_register_tokens}"
            )
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        *,
        num_register_tokens: int,
        patch_size: int,
        gradient_checkpointing: bool,
        autocast_dtype: str,
    ) -> "DINOv3SpatialBackbone":
        path = Path(pretrained_path)
        if not path.is_dir():
            raise ValueError(f"DINOv3 initialization path does not exist: {pretrained_path}")

        from transformers import AutoModel

        model = AutoModel.from_pretrained(path, local_files_only=True, torch_dtype=torch.float32)
        hidden_size = int(model.config.hidden_size)
        return cls(
            model=model,
            hidden_size=hidden_size,
            num_register_tokens=num_register_tokens,
            patch_size=patch_size,
            gradient_checkpointing=gradient_checkpointing,
            autocast_dtype=autocast_dtype,
        )

    @classmethod
    def from_model_config(
        cls,
        model_config: dict,
        *,
        num_register_tokens: int,
        patch_size: int,
        gradient_checkpointing: bool,
        autocast_dtype: str,
    ) -> "DINOv3SpatialBackbone":
        from transformers import AutoConfig, AutoModel

        model_type = model_config.get("model_type")
        if not model_type:
            raise ValueError("Saved DINOv3 model config is missing model_type")
        config_kwargs = {key: value for key, value in model_config.items() if key != "model_type"}
        config = AutoConfig.for_model(model_type, **config_kwargs)
        model = AutoModel.from_config(config)
        return cls(
            model=model,
            hidden_size=int(config.hidden_size),
            num_register_tokens=num_register_tokens,
            patch_size=patch_size,
            gradient_checkpointing=gradient_checkpointing,
            autocast_dtype=autocast_dtype,
        )

    def _autocast_context(self, image: Tensor):
        enabled = image.device.type == "cuda" and self.autocast_dtype != torch.float32
        if not enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)

    def forward(self, image: Tensor) -> Tensor:
        if image.ndim != 4 or tuple(image.shape[-2:]) != (224, 224):
            raise ValueError(f"DINOv3 expects BxCx224x224 images, got {tuple(image.shape)}")

        with self._autocast_context(image):
            hidden_state = self.model(pixel_values=image).last_hidden_state

        patch_tokens = hidden_state[:, 1 + self.num_register_tokens :]
        patches_per_side = 224 // self.patch_size
        expected_patch_tokens = patches_per_side**2
        if patch_tokens.shape[1] != expected_patch_tokens:
            raise ValueError(
                f"DINOv3 must return {expected_patch_tokens} patch tokens after special-token removal, "
                f"got {patch_tokens.shape[1]}"
            )
        if patch_tokens.shape[2] != self.hidden_size:
            raise ValueError(
                f"DINOv3 hidden size must be {self.hidden_size}, got {patch_tokens.shape[2]}"
            )

        return patch_tokens.transpose(1, 2).reshape(
            image.shape[0], self.hidden_size, patches_per_side, patches_per_side
        )

    def forward_cameras(self, images: Sequence[Tensor]) -> list[Tensor]:
        return [self(image) for image in images]
