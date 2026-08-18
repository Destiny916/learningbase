from starVLA.dataloader import is_main_process
from starVLA.training.train_starvla import (
    is_main_process as is_training_main_process,
    synchronize_processes,
)
from starVLA.training.trainer_utils.trainer_tools import is_main_process as is_trainer_main_process
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils
import torch


def test_single_process_dataloader_treats_an_uninitialized_process_group_as_main(monkeypatch):
    monkeypatch.setattr("starVLA.dataloader.dist.is_initialized", lambda: False)

    assert is_main_process() is True


def test_single_process_training_treats_an_uninitialized_process_group_as_main(monkeypatch):
    monkeypatch.setattr("starVLA.training.train_starvla.dist.is_initialized", lambda: False)

    assert is_training_main_process() is True


def test_single_process_training_does_not_call_distributed_barrier(monkeypatch):
    monkeypatch.setattr("starVLA.training.train_starvla.dist.is_initialized", lambda: False)
    monkeypatch.setattr(
        "starVLA.training.train_starvla.dist.barrier",
        lambda: (_ for _ in ()).throw(AssertionError("barrier must not be called")),
    )

    synchronize_processes()


def test_single_process_trainer_helpers_treat_an_uninitialized_process_group_as_main(monkeypatch):
    monkeypatch.setattr("starVLA.training.trainer_utils.trainer_tools.dist.is_initialized", lambda: False)

    assert is_trainer_main_process() is True


def test_print_trainable_parameters_does_not_require_a_process_group(monkeypatch):
    monkeypatch.setattr("starVLA.training.trainer_utils.trainer_tools.dist.is_initialized", lambda: False)

    TrainerUtils.print_trainable_parameters(torch.nn.Linear(2, 1))
