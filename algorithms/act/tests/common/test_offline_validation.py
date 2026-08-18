#!/usr/bin/env python

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import lerobot.common.offline_validation as offline_validation
from lerobot.common.offline_validation import (
    SampleMetricAccumulator,
    evaluate_offline,
    make_action_unnormalizer,
    make_inference_batch,
    make_pi05_validation_randomness,
    physical_action_mse_parts,
    reduce_metric_parts,
    restore_absolute_joint_actions,
    restore_absolute_pose_actions,
    stable_validation_seed,
)
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor.normalize_processor import UnnormalizerProcessorStep
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.scripts.convert_right_end_effector_pose_to_lerobot_v30 import (
    pose10d_from_end_pose,
    relative_pose10d,
)


class FakeAccelerator:
    def __init__(self):
        self.autocast_entries = 0
        self.autocast_active = False
        self.gathered: list[torch.Tensor] = []

    @contextmanager
    def autocast(self):
        self.autocast_entries += 1
        self.autocast_active = True
        try:
            yield
        finally:
            self.autocast_active = False

    def gather_for_metrics(self, tensor: torch.Tensor) -> torch.Tensor:
        self.gathered.append(tensor.detach().clone())
        return tensor

    def unwrap_model(self, policy):
        return policy

    def reduce(self, tensor: torch.Tensor, reduction: str):
        assert reduction == "sum"
        return tensor


class FakeFlowModel:
    @staticmethod
    def sample_time(batch_size: int, device: torch.device | str) -> torch.Tensor:
        return torch.rand(batch_size, device=device)


class FakePI05Policy(nn.Module):
    def __init__(self, predicted: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(
            type="pi05", chunk_size=3, max_action_dim=7, joint_gripper_indices=[6]
        )
        self.model = FakeFlowModel()
        self.predicted = predicted
        self.inference_batches: list[dict[str, torch.Tensor]] = []
        self.initial_noises: list[torch.Tensor] = []

    def forward(
        self,
        batch,
        reduction="mean",
        *,
        noise=None,
        time=None,
        return_action_chunk=False,
    ):
        if return_action_chunk:
            assert ACTION not in batch
            assert "action_is_pad" not in batch
            self.inference_batches.append(batch)
            self.initial_noises.append(noise.detach().clone())
            return self.predicted.to(self.weight.device)
        assert reduction == "none"
        assert not self.training
        assert not torch.is_grad_enabled()
        assert noise is not None
        assert time is not None
        assert ACTION in batch
        torch.rand(1)
        return torch.ones(batch[ACTION].shape[0]), {
            "loss_sum_per_sample": torch.tensor([14.0, 7.0]),
            "loss_count_per_sample": torch.tensor([14, 7]),
            "gripper_loss_sum_per_sample": torch.tensor([2.0, 1.0]),
            "gripper_loss_count_per_sample": torch.tensor([2, 1]),
        }

    def predict_action_chunk(self, batch, *, noise=None):
        raise AssertionError("validation prediction must pass through the wrapped policy forward")


class FakeACTPolicy(nn.Module):
    def __init__(self, predicted: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(type="act", gripper_indices=[6])
        self.predicted = predicted
        self.inference_batches: list[dict[str, torch.Tensor]] = []

    def forward(self, batch, reduction="mean", *, return_action_chunk=False):
        if return_action_chunk:
            assert ACTION not in batch
            assert "action_is_pad" not in batch
            self.inference_batches.append(batch)
            return self.predicted.to(self.weight.device)
        assert reduction == "none"
        assert not self.training
        assert not torch.is_grad_enabled()
        assert ACTION in batch
        assert torch.count_nonzero(batch[OBS_STATE]) == 0
        batch_size = batch[ACTION].shape[0]
        return torch.ones(batch_size), {
            "loss_sum_per_sample": torch.full((batch_size,), 14.0),
            "loss_count_per_sample": torch.full((batch_size,), 14),
            "gripper_loss_sum_per_sample": torch.full((batch_size,), 2.0),
            "gripper_loss_count_per_sample": torch.full((batch_size,), 2),
        }

    def predict_action_chunk(self, batch):
        raise AssertionError("validation prediction must pass through the wrapped policy forward")


def _linear_action_unnormalizer(action: torch.Tensor) -> torch.Tensor:
    q01 = torch.arange(7, dtype=action.dtype, device=action.device)
    half_range = torch.arange(1, 8, dtype=action.dtype, device=action.device)
    return q01 + (action + 1.0) * half_range


def test_stable_validation_seed_depends_on_identity_and_purpose():
    same = stable_validation_seed(42, 4, 31, "flow_noise")
    assert same == stable_validation_seed(42, 4, 31, "flow_noise")
    assert same != stable_validation_seed(42, 4, 32, "flow_noise")
    assert same != stable_validation_seed(42, 4, 31, "initial_noise")


def test_pi05_randomness_is_stable_across_batch_and_sample_order():
    policy = SimpleNamespace(
        config=SimpleNamespace(chunk_size=3, max_action_dim=7),
        model=FakeFlowModel(),
    )
    ids = [(0, 12), (4, 31)]
    together = make_pi05_validation_randomness(policy, ids, seed=42, device="cpu")
    separate = [
        make_pi05_validation_randomness(policy, [sample_id], seed=42, device="cpu")
        for sample_id in ids
    ]

    for index in range(3):
        torch.testing.assert_close(together[index], torch.cat([item[index] for item in separate]))

    reversed_values = make_pi05_validation_randomness(policy, ids[::-1], seed=42, device="cpu")
    for index in range(3):
        torch.testing.assert_close(reversed_values[index].flip(0), together[index])


def test_make_inference_batch_removes_all_action_labels_but_keeps_zero_state():
    zero_state = torch.zeros(2, 7)
    inference = make_inference_batch(
        {
            OBS_STATE: zero_state,
            ACTION: torch.ones(2, 3, 7),
            "action_is_pad": torch.zeros(2, 3, dtype=torch.bool),
            "observation.images.right": torch.zeros(2, 3, 8, 8),
        }
    )

    assert ACTION not in inference
    assert "action_is_pad" not in inference
    assert inference[OBS_STATE] is zero_state
    assert torch.count_nonzero(inference[OBS_STATE]) == 0


def test_physical_action_mse_parts_excludes_padding_and_uses_gripper_index_six():
    predicted = torch.zeros(2, 3, 7)
    target = torch.zeros_like(predicted)
    predicted[0, 0] = torch.arange(1, 8)
    predicted[0, 1] = 1000
    action_is_pad = torch.tensor([[False, True, False], [False, True, True]])

    parts = physical_action_mse_parts(predicted, target, action_is_pad, gripper_indices=[6])

    torch.testing.assert_close(parts["action_mse_sum_per_sample"], torch.tensor([140.0, 0.0]))
    torch.testing.assert_close(parts["action_mse_count_per_sample"], torch.tensor([14, 7]))
    torch.testing.assert_close(parts["gripper_mse_sum_per_sample"], torch.tensor([49.0, 0.0]))
    torch.testing.assert_close(parts["gripper_mse_count_per_sample"], torch.tensor([2, 1]))
    torch.testing.assert_close(
        parts["dimension_mse_sum_per_sample"],
        torch.tensor([[1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0], [0.0] * 7]),
    )
    torch.testing.assert_close(
        parts["dimension_mse_count_per_sample"],
        torch.tensor([[2] * 7, [1] * 7]),
    )


def test_restore_absolute_joint_actions_adds_current_anchor_only_to_arm():
    relative = torch.tensor(
        [[[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 0.08], [0.5, 0.0, -0.5, 1.0, 2.0, 3.0, 0.02]]]
    )
    paired_absolute_state = torch.tensor(
        [[[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 0.01], [11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 0.03]]]
    )

    absolute = restore_absolute_joint_actions(relative, paired_absolute_state, gripper_indices=[6])

    torch.testing.assert_close(
        absolute[..., :6],
        relative[..., :6] + paired_absolute_state[:, -1, None, :6],
    )
    torch.testing.assert_close(absolute[..., 6], relative[..., 6])


def test_restore_absolute_pose_actions_composes_se3_and_keeps_absolute_gripper():
    current = pose10d_from_end_pose(
        {"x": 0.1, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 1.57079632679}, 0.02
    )
    target = pose10d_from_end_pose(
        {"x": 0.1, "y": 0.1, "z": 0.0, "roll": 0.0, "pitch": 0.1, "yaw": 1.57079632679}, 0.08
    )
    relative = relative_pose10d(current, target)

    restored = restore_absolute_pose_actions(
        torch.tensor([[relative]]), torch.tensor([[current, current]])
    )

    torch.testing.assert_close(restored[0, 0], torch.from_numpy(target), atol=1e-6, rtol=0)


def test_reduce_metric_parts_uses_global_numerator_and_count_not_sample_means():
    accelerator = FakeAccelerator()
    metric = reduce_metric_parts(
        accelerator,
        [
            (torch.tensor([9.0, 1.0]), torch.tensor([1, 9])),
            (torch.tensor([6.0]), torch.tensor([3])),
        ],
    )

    assert metric == pytest.approx(16 / 13)
    assert len(accelerator.gathered) == 4


def test_sample_metric_accumulator_covers_every_microbatch_with_equal_sample_weight():
    accumulator = SampleMetricAccumulator("gripper_loss")
    accumulator.update(
        {
            "gripper_loss_sum_per_sample": torch.tensor([9.0, 1.0]),
            "gripper_loss_count_per_sample": torch.tensor([1, 9]),
            "gripper_loss_per_sample": torch.tensor([9.0, 1.0 / 9.0]),
        }
    )
    accumulator.update(
        {
            "gripper_loss_sum_per_sample": torch.tensor([6.0]),
            "gripper_loss_count_per_sample": torch.tensor([3]),
            "gripper_loss_per_sample": torch.tensor([2.0]),
        }
    )

    assert accumulator.compute_global(FakeAccelerator()) == pytest.approx((9 + 1 / 9 + 2) / 3)


def test_action_unnormalizer_does_not_run_later_absolute_action_steps():
    step = UnnormalizerProcessorStep(
        features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        norm_map={FeatureType.ACTION: NormalizationMode.QUANTILES},
        stats={ACTION: {"q01": torch.arange(7), "q99": torch.arange(7) + 10}},
    )

    class ForbiddenAbsoluteStep:
        def __call__(self, _transition):
            raise AssertionError("validation must not reconstruct absolute actions")

    unnormalize = make_action_unnormalizer(SimpleNamespace(steps=[step, ForbiddenAbsoluteStep()]))
    result = unnormalize(torch.zeros(1, 1, 7))

    torch.testing.assert_close(result, torch.arange(7, dtype=torch.float32).reshape(1, 1, 7) + 5)


def test_action_unnormalizer_rejects_non_quantile_action_stats():
    step = UnnormalizerProcessorStep(
        features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD},
        stats={ACTION: {"mean": torch.zeros(7), "std": torch.ones(7)}},
    )

    with pytest.raises(ValueError, match="q01/q99 QUANTILES"):
        make_action_unnormalizer(SimpleNamespace(steps=[step]))


def test_evaluate_offline_reports_physical_per_joint_and_gripper_rmse_metrics():
    predicted = torch.zeros(2, 3, 7)
    predicted[0, 0] = 1.0
    predicted[0, 1] = 100.0
    policy = FakePI05Policy(predicted)
    policy.train()
    accelerator = FakeAccelerator()
    batch = {
        ACTION: torch.zeros(2, 3, 7),
        "action_is_pad": torch.tensor([[False, True, False], [False, True, True]]),
        OBS_STATE: torch.zeros(2, 7),
        "episode_index": torch.tensor([0, 4]),
        "frame_index": torch.tensor([12, 31]),
    }
    parameter_before = policy.weight.detach().clone()
    cpu_rng_before = torch.random.get_rng_state().clone()

    metrics = evaluate_offline(
        policy,
        [batch],
        preprocessor=lambda item: item,
        accelerator=accelerator,
        action_unnormalizer=_linear_action_unnormalizer,
        seed=42,
    )

    assert set(metrics) == {
        "valid/loss",
        "valid/gripper_loss",
        "valid/action_mse",
        "valid/gripper_mse",
        "valid/joint_0_mse_rad2",
        "valid/joint_0_rmse_rad",
        "valid/joint_0_rmse_deg",
        "valid/joint_1_mse_rad2",
        "valid/joint_1_rmse_rad",
        "valid/joint_1_rmse_deg",
        "valid/joint_2_mse_rad2",
        "valid/joint_2_rmse_rad",
        "valid/joint_2_rmse_deg",
        "valid/joint_3_mse_rad2",
        "valid/joint_3_rmse_rad",
        "valid/joint_3_rmse_deg",
        "valid/joint_4_mse_rad2",
        "valid/joint_4_rmse_rad",
        "valid/joint_4_rmse_deg",
        "valid/joint_5_mse_rad2",
        "valid/joint_5_rmse_rad",
        "valid/joint_5_rmse_deg",
        "valid/gripper_rmse_m",
        "valid/gripper_rmse_mm",
    }
    assert metrics["valid/loss"] == pytest.approx(1.0)
    assert metrics["valid/gripper_loss"] == pytest.approx(1.0)
    assert metrics["valid/action_mse"] == pytest.approx(sum(i * i for i in range(1, 8)) / 21)
    assert metrics["valid/gripper_mse"] == pytest.approx(49 / 3)
    for index in range(6):
        expected_mse = (index + 1) ** 2 / 3
        assert metrics[f"valid/joint_{index}_mse_rad2"] == pytest.approx(expected_mse)
        assert metrics[f"valid/joint_{index}_rmse_rad"] == pytest.approx(expected_mse**0.5)
        assert metrics[f"valid/joint_{index}_rmse_deg"] == pytest.approx(expected_mse**0.5 * 180 / torch.pi)
    assert metrics["valid/gripper_rmse_m"] == pytest.approx((49 / 3) ** 0.5)
    assert metrics["valid/gripper_rmse_mm"] == pytest.approx((49 / 3) ** 0.5 * 1000)
    assert policy.training
    torch.testing.assert_close(policy.weight, parameter_before)
    assert torch.equal(torch.random.get_rng_state(), cpu_rng_before)
    assert accelerator.autocast_entries == 1
    assert ACTION not in policy.inference_batches[0]
    assert "action_is_pad" not in policy.inference_batches[0]
    assert policy.inference_batches[0][OBS_STATE].shape == (2, 7)
    assert policy.initial_noises[0].shape == (2, 3, 7)


def test_evaluate_offline_reports_individual_dual_gripper_metrics():
    predicted = torch.zeros(2, 3, 14)
    policy = FakePI05Policy(predicted)
    policy.config.max_action_dim = 14
    policy.config.joint_gripper_indices = [6, 13]
    policy.config.action_feature_names = [
        *[f"left_joint_{index}" for index in range(6)],
        "left_gripper",
        *[f"right_joint_{index}" for index in range(6)],
        "right_gripper",
    ]
    batch = {
        ACTION: torch.zeros(2, 3, 14),
        "action_is_pad": torch.zeros(2, 3, dtype=torch.bool),
        OBS_STATE: torch.zeros(2, 14),
        "episode_index": torch.tensor([0, 1]),
        "frame_index": torch.tensor([0, 1]),
    }

    metrics = evaluate_offline(
        policy,
        [batch],
        preprocessor=lambda item: item,
        accelerator=FakeAccelerator(),
        action_unnormalizer=lambda action: action,
        seed=42,
    )

    for gripper in ("left_gripper", "right_gripper"):
        assert metrics[f"valid/{gripper}_mse"] == pytest.approx(0.0)
        assert metrics[f"valid/{gripper}_rmse_m"] == pytest.approx(0.0)
        assert metrics[f"valid/{gripper}_rmse_mm"] == pytest.approx(0.0)


def test_evaluate_offline_absolute_joint_actions_do_not_add_the_current_state(monkeypatch):
    predicted = torch.full((1, 3, 7), 2.0)
    policy = FakePI05Policy(predicted)
    policy.config.joint_representation = "absolute"
    captured: dict[str, torch.Tensor] = {}
    real_mse_parts = physical_action_mse_parts

    def capture_mse_parts(predicted, target, action_is_pad, *, gripper_indices):
        captured["predicted"] = predicted.clone()
        captured["target"] = target.clone()
        return real_mse_parts(predicted, target, action_is_pad, gripper_indices=gripper_indices)

    monkeypatch.setattr(offline_validation, "physical_action_mse_parts", capture_mse_parts)
    target = torch.full((1, 3, 7), 1.0)
    batch = {
        ACTION: target,
        "action_is_pad": torch.zeros(1, 3, dtype=torch.bool),
        OBS_STATE: torch.full((1, 7), 100.0),
        "episode_index": torch.tensor([0]),
        "frame_index": torch.tensor([0]),
    }

    metrics = evaluate_offline(
        policy,
        [batch],
        preprocessor=lambda item: item,
        accelerator=FakeAccelerator(),
        action_unnormalizer=lambda action: action,
        seed=42,
    )

    torch.testing.assert_close(captured["predicted"], predicted)
    torch.testing.assert_close(captured["target"], target)
    assert metrics["valid/action_mse"] == pytest.approx(1.0)


def test_evaluate_offline_restores_state_when_validation_raises():
    class FailingPolicy(FakePI05Policy):
        def forward(self, *args, **kwargs):
            torch.rand(1)
            raise RuntimeError("validation failed")

    policy = FailingPolicy(torch.zeros(1, 3, 7))
    policy.train()
    rng_before = torch.random.get_rng_state().clone()
    batch = {
        ACTION: torch.zeros(1, 3, 7),
        "action_is_pad": torch.zeros(1, 3, dtype=torch.bool),
        OBS_STATE: torch.zeros(1, 7),
        "episode_index": torch.tensor([0]),
        "frame_index": torch.tensor([0]),
    }

    with pytest.raises(RuntimeError, match="validation failed"):
        evaluate_offline(
            policy,
            [batch],
            preprocessor=lambda item: item,
            accelerator=FakeAccelerator(),
            action_unnormalizer=lambda action: action,
            seed=42,
        )

    assert policy.training
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_evaluate_offline_act_uses_raw_anchor_before_image_only_zeroing(monkeypatch):
    policy = FakeACTPolicy(torch.ones(1, 3, 7))
    policy.train()
    raw_state = torch.tensor(
        [[[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 0.01], [11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 0.03]]]
    )
    batch = {
        ACTION: torch.zeros(1, 3, 7),
        "action_is_pad": torch.tensor([[False, False, True]]),
        OBS_STATE: raw_state,
        "episode_index": torch.tensor([0]),
        "frame_index": torch.tensor([1]),
    }
    captured: dict[str, torch.Tensor] = {}
    real_mse_parts = physical_action_mse_parts

    def capture_mse_parts(predicted, target, action_is_pad, *, gripper_indices):
        captured["predicted"] = predicted.clone()
        captured["target"] = target.clone()
        return real_mse_parts(
            predicted,
            target,
            action_is_pad,
            gripper_indices=gripper_indices,
        )

    monkeypatch.setattr(offline_validation, "physical_action_mse_parts", capture_mse_parts)

    def image_only_preprocessor(item):
        processed = dict(item)
        processed[OBS_STATE] = torch.zeros(item[OBS_STATE].shape[0], 7)
        return processed

    metrics = evaluate_offline(
        policy,
        [batch],
        preprocessor=image_only_preprocessor,
        accelerator=FakeAccelerator(),
        action_unnormalizer=lambda action: action,
        seed=42,
    )

    assert {key: metrics[key] for key in ("valid/loss", "valid/gripper_loss", "valid/action_mse", "valid/gripper_mse")} == {
        "valid/loss": pytest.approx(1.0),
        "valid/gripper_loss": pytest.approx(1.0),
        "valid/action_mse": pytest.approx(1.0),
        "valid/gripper_mse": pytest.approx(1.0),
    }
    for index in range(6):
        assert metrics[f"valid/joint_{index}_mse_rad2"] == pytest.approx(1.0)
        assert metrics[f"valid/joint_{index}_rmse_rad"] == pytest.approx(1.0)
        assert metrics[f"valid/joint_{index}_rmse_deg"] == pytest.approx(180 / torch.pi)
    assert metrics["valid/gripper_rmse_m"] == pytest.approx(1.0)
    assert metrics["valid/gripper_rmse_mm"] == pytest.approx(1000.0)
    assert ACTION not in policy.inference_batches[0]
    assert "action_is_pad" not in policy.inference_batches[0]
    assert policy.inference_batches[0][OBS_STATE].shape == (1, 7)
    assert torch.count_nonzero(policy.inference_batches[0][OBS_STATE]) == 0
    torch.testing.assert_close(
        captured["predicted"][..., :6],
        torch.ones(1, 3, 6) + raw_state[:, -1, None, :6],
    )
    torch.testing.assert_close(captured["predicted"][..., 6], torch.ones(1, 3))
    torch.testing.assert_close(
        captured["target"][..., :6],
        raw_state[:, -1, None, :6].expand(1, 3, 6),
    )
    torch.testing.assert_close(captured["target"][..., 6], torch.zeros(1, 3))
    assert policy.training


def test_evaluate_offline_gathers_metric_parts_before_requesting_next_batch():
    policy = FakePI05Policy(torch.zeros(2, 3, 7))
    accelerator = FakeAccelerator()
    batch = {
        ACTION: torch.zeros(2, 3, 7),
        "action_is_pad": torch.zeros(2, 3, dtype=torch.bool),
        OBS_STATE: torch.zeros(2, 7),
        "episode_index": torch.tensor([0, 0]),
        "frame_index": torch.tensor([0, 1]),
    }

    def batches():
        yield batch
        # Ten MSE/loss metrics, each with a numerator and denominator, must be gathered while
        # Accelerate still knows whether this is the dataloader's final padded batch.
        assert len(accelerator.gathered) == 20
        yield batch

    evaluate_offline(
        policy,
        batches(),
        preprocessor=lambda item: item,
        accelerator=accelerator,
        action_unnormalizer=lambda action: action,
        seed=42,
    )


def test_evaluate_offline_unnormalizes_fp32_outside_autocast():
    policy = FakePI05Policy(torch.zeros(1, 3, 7, dtype=torch.bfloat16))
    accelerator = FakeAccelerator()
    batch = {
        ACTION: torch.zeros(1, 3, 7, dtype=torch.bfloat16),
        "action_is_pad": torch.zeros(1, 3, dtype=torch.bool),
        OBS_STATE: torch.zeros(1, 7),
        "episode_index": torch.tensor([0]),
        "frame_index": torch.tensor([0]),
    }

    def checked_unnormalizer(action):
        assert not accelerator.autocast_active
        assert action.dtype is torch.float32
        return action

    evaluate_offline(
        policy,
        [batch],
        preprocessor=lambda item: item,
        accelerator=accelerator,
        action_unnormalizer=checked_unnormalizer,
        seed=42,
    )
