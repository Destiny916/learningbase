import pytest
import torch

pytest.importorskip("transformers")

from lerobot.policies.pi05.modeling_pi05 import _assert_finite_for_debug, _register_finite_grad_debug_hook


def test_finite_debug_is_opt_in(monkeypatch):
    _assert_finite_for_debug("finite", torch.tensor([1.0]))
    _assert_finite_for_debug("nan_disabled", torch.tensor([float("nan")]))

    monkeypatch.setenv("LEROBOT_PI05_FINITE_DEBUG", "1")
    with pytest.raises(FloatingPointError, match="PI05 non-finite tensor: nan_enabled"):
        _assert_finite_for_debug("nan_enabled", torch.tensor([float("nan")]))


def test_finite_grad_debug_hook_reports_nonfinite_backward_gradient(monkeypatch):
    monkeypatch.setenv("LEROBOT_PI05_FINITE_DEBUG", "1")
    value = torch.tensor([1.0], requires_grad=True)
    _register_finite_grad_debug_hook("image_embedding_0", value)

    with pytest.raises(FloatingPointError, match="PI05 non-finite gradient: image_embedding_0"):
        (value * float("nan")).sum().backward()
