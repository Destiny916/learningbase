#!/usr/bin/env python

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.policies import pi_gemma


def _mask_inputs():
    return {
        "config": SimpleNamespace(),
        "inputs_embeds": torch.zeros(1, 2, 4),
        "attention_mask": torch.ones(1, 2, dtype=torch.bool),
        "cache_position": torch.arange(2),
        "past_key_values": SimpleNamespace(),
        "position_ids": torch.arange(2).unsqueeze(0),
    }


def test_causal_mask_compat_omits_removed_cache_position(monkeypatch: pytest.MonkeyPatch):
    received = {}

    def current_api(config, inputs_embeds, attention_mask, past_key_values, position_ids):
        received.update(locals())
        return "current-mask"

    monkeypatch.setattr(pi_gemma, "create_causal_mask", current_api)

    assert pi_gemma._create_causal_mask_compat(**_mask_inputs()) == "current-mask"
    assert "cache_position" not in received


def test_causal_mask_compat_preserves_legacy_cache_position(monkeypatch: pytest.MonkeyPatch):
    received = {}

    def legacy_api(
        config,
        inputs_embeds,
        attention_mask,
        cache_position,
        past_key_values,
        position_ids,
    ):
        received.update(locals())
        return "legacy-mask"

    monkeypatch.setattr(pi_gemma, "create_causal_mask", legacy_api)

    inputs = _mask_inputs()
    assert pi_gemma._create_causal_mask_compat(**inputs) == "legacy-mask"
    torch.testing.assert_close(received["cache_position"], inputs["cache_position"])
