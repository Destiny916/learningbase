"""Build the fixed LIBERO RLDS dataset and collator."""

from pathlib import Path
from typing import Tuple, Type

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.constants import NUM_TOKENS
from prismatic.vla.datasets import RLDSDataset, VLABatchTransform
from prismatic.vla.datasets.lerobot_w1 import W1Collator, W1DataContract, W1LeRobotTorchDataset


def get_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    shuffle_buffer_size: int = 20_000,
    visual_token_pair_offset: int = 31,
    target_action_dim: int = 7,
    target_proprio_dim: int = 8,
) -> Tuple[Dataset, PaddedCollatorForActionPrediction]:
    batch_transform = VLABatchTransform(
        base_tokenizer=tokenizer,
        image_transform=image_transform,
        prompt_builder_fn=prompt_builder_fn,
        flow_gr00t_placeholder_tokens=NUM_TOKENS,
        visual_token_pair_offset=visual_token_pair_offset,
    )
    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length,
        tokenizer.pad_token_id,
        padding_side="right",
        target_action_dim=target_action_dim,
        target_proprio_dim=target_proprio_dim,
    )
    dataset = RLDSDataset(
        data_root_dir,
        data_mix,
        batch_transform,
        resize_resolution=default_image_resolution[1:],
        shuffle_buffer_size=shuffle_buffer_size,
        visual_token_pair_offset=visual_token_pair_offset,
    )
    return dataset, collator


def get_w1_dataset_and_collator(
    data_root_dir: Path,
    image_transform,
    tokenizer,
    prompt_builder_fn,
    state_q01_q99: Path,
    action_q01_q99: Path,
    action_token_id: int,
    placeholder_tokens: int,
    action_horizon: int = 20,
):
    import json

    def read_quantiles(path: Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload["q01"], payload["q99"]

    state_q01, state_q99 = read_quantiles(state_q01_q99)
    action_q01, action_q99 = read_quantiles(action_q01_q99)
    dataset = W1LeRobotTorchDataset(
        str(data_root_dir), state_q01, state_q99, action_q01, action_q99,
        contract=W1DataContract(action_horizon=action_horizon),
    )
    collator = W1Collator(
        tokenizer,
        image_transform,
        prompt_builder_fn,
        action_token_id=action_token_id,
        placeholder_tokens=placeholder_tokens,
    )
    return dataset, collator
