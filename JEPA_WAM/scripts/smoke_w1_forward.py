#!/usr/bin/env python3
"""Single-process W1 model forward/backward smoke test."""
from pathlib import Path
from types import SimpleNamespace
import json
import torch

from prismatic.training.train import build_vla_from_base_vlm
from prismatic.vla.materialize import get_w1_dataset_and_collator

base = Path(__import__('os').environ['W1_BASE_VLM'])
dataset_root = Path(__import__('os').environ['W1_DATASET'])
state_stats = Path(__import__('os').environ['W1_STATE_Q'])
action_stats = Path(__import__('os').environ['W1_ACTION_Q'])

vla = SimpleNamespace(
    vjepa_checkpoint_path=__import__('os').environ['W1_VJEPA'], d_action=19, d_proprio=19,
    action_horizon=20, fm_hidden_size=1024, fm_num_layers=16,
    fm_num_inference_timesteps=4, fm_num_timestep_buckets=1000,
    fm_noise_beta_alpha=1.5, fm_noise_beta_beta=1.0, fm_noise_s=0.999,
    fm_num_target_vision_tokens=32, fm_add_pos_embed=True, fm_max_seq_len=1024,
    fm_state_dropout=0.5, flow_gr00t_placeholder_tokens=64,
    lambda_visual_token_cosine=0.5, enable_mixed_precision_training=False,
)
cfg = SimpleNamespace(vla=vla, llm_checkpoint_path=Path(__import__('os').environ['W1_QWEN']))
torch.cuda.set_device(0)
model = build_vla_from_base_vlm(base, cfg, None).cuda()
model.train()
with open(state_stats) as f: sq = json.load(f)
with open(action_stats) as f: aq = json.load(f)
dataset, collator = get_w1_dataset_and_collator(
    dataset_root, model.vision_backbone.get_image_transform(),
    model.llm_backbone.get_tokenizer(), model.llm_backbone.prompt_builder_fn,
    state_stats, action_stats, 151386, 64, 20)
from torch.utils.data import DataLoader
batch = next(iter(DataLoader(dataset, batch_size=1, collate_fn=collator, num_workers=0)))
batch = {k: (v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
batch.pop('action_valid_mask', None)
print('SMOKE_BATCH', {k: tuple(v.shape) for k, v in batch.items() if isinstance(v, torch.Tensor)})
out = model(**batch)
loss = out['loss']
assert torch.isfinite(loss).item(), loss
loss.backward()
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
assert torch.isfinite(torch.as_tensor(grad_norm)).item(), grad_norm
print(f'SMOKE_TEST_OK loss={loss.item():.6f} grad_norm={float(grad_norm):.6f} action_shape={tuple(batch["actions"].shape)} state_shape={tuple(batch["proprio"].shape)}')
