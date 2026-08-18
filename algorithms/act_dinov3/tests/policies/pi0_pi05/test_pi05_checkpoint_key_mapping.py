#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: E402


def test_pi05_checkpoint_vision_model_keys_match_runtime_vision_tower_keys():
    checkpoint_key = (
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model."
        "embeddings.patch_embedding.weight"
    )
    runtime_key = (
        "paligemma_with_expert.paligemma.model.vision_tower.embeddings.patch_embedding.weight"
    )
    weight = torch.ones(1)

    fixed = PI05Policy._fix_pytorch_state_dict_keys(object(), {checkpoint_key: weight}, None)

    assert fixed == {runtime_key: weight}


def test_pi05_runtime_vision_tower_keys_are_not_changed():
    runtime_key = (
        "paligemma_with_expert.paligemma.model.vision_tower.embeddings.patch_embedding.weight"
    )
    weight = torch.ones(1)

    fixed = PI05Policy._fix_pytorch_state_dict_keys(object(), {runtime_key: weight}, None)

    assert fixed == {runtime_key: weight}


def test_pi05_pretrained_mapping_targets_nested_runtime_vision_tower_keys():
    class StateDictCarrier:
        def state_dict(self):
            return {
                "model.paligemma_with_expert.paligemma.model.vision_tower.vision_model."
                "embeddings.patch_embedding.weight": torch.ones(1)
            }

    source_key = (
        "model.paligemma_with_expert.paligemma.model.vision_tower."
        "embeddings.patch_embedding.weight"
    )
    expected_key = (
        "model.paligemma_with_expert.paligemma.model.vision_tower.vision_model."
        "embeddings.patch_embedding.weight"
    )

    prepared = PI05Policy._prepare_pretrained_state_dict(
        StateDictCarrier(), {source_key: torch.ones(1)}
    )

    assert prepared == {expected_key: torch.ones(1)}
