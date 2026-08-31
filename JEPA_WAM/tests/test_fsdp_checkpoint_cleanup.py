from unittest.mock import patch

from prismatic.training.strategies.fsdp import release_checkpoint_memory


def test_release_checkpoint_memory_clears_state_and_cuda_cache() -> None:
    full_state = {"weight": object()}
    split_state = {"llm": {"weight": object()}}

    with patch("prismatic.training.strategies.fsdp.gc.collect") as collect, patch(
        "prismatic.training.strategies.fsdp.torch.cuda.is_available", return_value=True
    ), patch("prismatic.training.strategies.fsdp.torch.cuda.empty_cache") as empty_cache:
        release_checkpoint_memory(full_state, split_state)

    assert full_state == {}
    assert split_state == {}
    collect.assert_called_once_with()
    empty_cache.assert_called_once_with()
