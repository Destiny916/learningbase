from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE


class TinyDINOv3(nn.Module):
    def __init__(self, hidden_size: int = 8, output_dtype: torch.dtype | None = None):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.output_dtype = output_dtype
        self.proj = nn.Conv2d(3, hidden_size, kernel_size=16, stride=16, bias=False)
        self.gradient_checkpointing_enabled = False
        self.seen_camera_values: list[float] = []

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True

    def forward(self, pixel_values: torch.Tensor):
        self.seen_camera_values.append(pixel_values[0, 0, 0, 0].item())
        patches = self.proj(pixel_values).flatten(2).transpose(1, 2)
        special = torch.zeros(
            pixel_values.shape[0],
            5,
            self.config.hidden_size,
            device=pixel_values.device,
            dtype=patches.dtype,
        )
        hidden_state = torch.cat([special, patches], dim=1)
        if self.output_dtype is not None:
            hidden_state = hidden_state.to(self.output_dtype)
        return SimpleNamespace(last_hidden_state=hidden_state)


def _config(tmp_path):
    from lerobot.policies.act_dinov3.configuration_act_dinov3 import ACTDINOv3Config

    weights = tmp_path / "dinov3"
    weights.mkdir()
    return ACTDINOv3Config(
        dinov3_pretrained_path=str(weights),
        pretrained_backbone_weights=None,
        chunk_size=2,
        n_action_steps=2,
        dim_model=16,
        n_heads=4,
        dim_feedforward=32,
        n_encoder_layers=1,
        n_decoder_layers=1,
        latent_dim=4,
        n_vae_encoder_layers=1,
        input_features={
            "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "observation.images.gripper_left": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 224, 224)
            ),
            "observation.images.gripper_right": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 224, 224)
            ),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
    )


def test_policy_preserves_camera_order_and_backpropagates_finite_act_loss(tmp_path):
    from lerobot.policies.act_dinov3.modeling_act_dinov3 import ACTDINOv3Policy

    dino = TinyDINOv3()
    policy = ACTDINOv3Policy(_config(tmp_path), dinov3_model=dino)
    policy.train()
    batch = {
        "observation.images.top": torch.full((1, 3, 224, 224), 1.0),
        "observation.images.gripper_left": torch.full((1, 3, 224, 224), 2.0),
        "observation.images.gripper_right": torch.full((1, 3, 224, 224), 3.0),
        OBS_STATE: torch.randn(1, 4),
        ACTION: torch.randn(1, 2, 3),
        "action_is_pad": torch.zeros(1, 2, dtype=torch.bool),
    }

    loss, loss_dict = policy(batch)
    loss.backward()

    expected_camera_values = [(value - 0.485) / 0.229 for value in (1.0, 2.0, 3.0)]
    torch.testing.assert_close(
        torch.tensor(dino.seen_camera_values), torch.tensor(expected_camera_values)
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(torch.tensor(loss_dict["l1_loss"]))
    assert dino.proj.weight.grad is not None
    assert torch.isfinite(dino.proj.weight.grad).all()
    assert dino.proj.weight.grad.abs().sum() > 0


def test_policy_optimizer_groups_are_complete_disjoint_and_use_two_learning_rates(tmp_path):
    from lerobot.policies.act_dinov3.modeling_act_dinov3 import ACTDINOv3Policy

    policy = ACTDINOv3Policy(_config(tmp_path), dinov3_model=TinyDINOv3())

    groups = policy.get_optim_params()
    grouped = [parameter for group in groups for parameter in group["params"]]
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]

    assert len(groups) == 2
    assert groups[0]["lr"] == policy.config.optimizer_lr
    assert groups[1]["lr"] == policy.config.dinov3_learning_rate
    assert len(grouped) == len({id(parameter) for parameter in grouped})
    assert {id(parameter) for parameter in grouped} == {id(parameter) for parameter in trainable}


def test_policy_preset_builds_scheduler_without_losing_differential_learning_rates(tmp_path):
    from types import SimpleNamespace

    from lerobot.optim.factory import make_optimizer_and_scheduler
    from lerobot.policies.act_dinov3.modeling_act_dinov3 import ACTDINOv3Policy

    policy = ACTDINOv3Policy(_config(tmp_path), dinov3_model=TinyDINOv3())
    train_config = SimpleNamespace(
        use_policy_training_preset=True,
        optimizer=policy.config.get_optimizer_preset(),
        scheduler=policy.config.get_scheduler_preset(),
        steps=100,
    )

    optimizer, scheduler = make_optimizer_and_scheduler(train_config, policy)

    assert scheduler is not None
    assert [group["initial_lr"] for group in optimizer.param_groups] == [1e-5, 1e-6]
    assert optimizer.param_groups[0]["lr"] / optimizer.param_groups[1]["lr"] == pytest.approx(10)

    # The 500k schedule is auto-scaled to this 100-step unit test, so warmup is 5 steps.
    for _ in range(5):
        optimizer.step()
        scheduler.step()

    assert [group["lr"] for group in optimizer.param_groups] == [1e-5, 1e-6]


def test_policy_casts_dinov3_features_to_projection_dtype(tmp_path):
    from lerobot.policies.act_dinov3.modeling_act_dinov3 import ACTDINOv3Policy

    policy = ACTDINOv3Policy(
        _config(tmp_path), dinov3_model=TinyDINOv3(output_dtype=torch.bfloat16)
    )

    feature_map = policy.model.backbone(torch.ones(1, 3, 224, 224))["feature_map"]

    assert feature_map.dtype == policy.model.encoder_img_feat_input_proj.weight.dtype


def test_factory_returns_independent_act_dinov3_policy_class():
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act_dinov3.modeling_act_dinov3 import ACTDINOv3Policy
    from lerobot.policies.factory import get_policy_class

    assert get_policy_class("act") is ACTPolicy
    assert get_policy_class("act_dinov3") is ACTDINOv3Policy
