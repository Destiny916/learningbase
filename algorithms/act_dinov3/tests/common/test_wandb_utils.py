#!/usr/bin/env python

from types import SimpleNamespace

from lerobot.common.wandb_utils import WandBLogger


def test_log_dict_supports_valid_metric_namespace():
    logged = []
    logger = object.__new__(WandBLogger)
    logger._wandb = SimpleNamespace(log=lambda **kwargs: logged.append(kwargs))
    logger._wandb_custom_step_key = None

    logger.log_dict({"loss": 1.0, "action_mse": 2.0}, step=200, mode="valid")

    assert logged == [{"data": {"valid/loss": 1.0, "valid/action_mse": 2.0}, "step": 200}]
