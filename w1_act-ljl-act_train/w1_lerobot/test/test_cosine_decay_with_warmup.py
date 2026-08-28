import math

import torch

from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


def test_cosine_decay_starts_after_warmup_and_reaches_min_at_decay_step():
    peak_lr = 1e-5
    min_lr = 1e-6
    warmup_steps = 25_000
    decay_end_step = 500_000
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.tensor(0.0))], lr=peak_lr)
    config = CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=warmup_steps,
        num_decay_steps=decay_end_step,
        peak_lr=peak_lr,
        decay_lr=min_lr,
    )

    scheduler = config.build(optimizer, num_training_steps=decay_end_step)
    lr_factor = scheduler.lr_lambdas[0]

    assert math.isclose(lr_factor(warmup_steps), 1.0, rel_tol=0.0, abs_tol=1e-12)
    midpoint = warmup_steps + (decay_end_step - warmup_steps) // 2
    assert math.isclose(lr_factor(midpoint), 0.55, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(lr_factor(decay_end_step), 0.1, rel_tol=0.0, abs_tol=1e-12)
