import torch

from starVLA.training.trainer_utils.trainer_tools import filter_compatible_checkpoint


def test_partial_checkpoint_load_skips_only_shape_mismatches():
    target = {
        "state_proj.net.0.weight": torch.zeros(17),
        "state_proj.net.1.weight": torch.zeros(256, 17),
        "action_head.weight": torch.zeros(14, 512),
    }
    checkpoint = {
        "state_proj.net.0.weight": torch.zeros(14),
        "state_proj.net.1.weight": torch.zeros(256, 14),
        "action_head.weight": torch.ones(14, 512),
        "unused.weight": torch.ones(3),
    }

    compatible, skipped = filter_compatible_checkpoint(checkpoint, target)

    assert list(compatible) == ["action_head.weight"]
    assert skipped == ["state_proj.net.0.weight", "state_proj.net.1.weight", "unused.weight"]


def test_legacy_turbovla_names_are_mapped_to_wrapper_state_dict():
    target = {
        "model.action_head.decoder.action_queries.weight": torch.zeros(2, 256),
        "model.action_head.state_projection.net.1.weight": torch.zeros(256, 17),
        "model.vision_projection.mlp.0.weight": torch.zeros(1024, 1024),
    }
    checkpoint = {
        "action_model.action_policy.action_queries.weight": torch.ones(2, 256),
        "action_model.state_proj.net.1.weight": torch.ones(256, 14),
        "vision_proj.mlp.0.weight": torch.ones(1024, 1024),
    }

    compatible, skipped = filter_compatible_checkpoint(checkpoint, target)

    assert list(compatible) == [
        "model.action_head.decoder.action_queries.weight",
        "model.vision_projection.mlp.0.weight",
    ]
    assert skipped == ["action_model.state_proj.net.1.weight"]
