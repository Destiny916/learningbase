#!/usr/bin/env python3
"""Single-process W1 model forward/backward smoke test."""
from pathlib import Path
from types import SimpleNamespace
import json
import torch

from prismatic.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform, get_vlm
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
with open(base / 'config.json') as f: model_cfg = json.load(f)['model']
ckpt = torch.load(base / 'checkpoints/latest-checkpoint.pt', map_location='cpu', weights_only=False)['model']
vision, _ = get_vision_backbone_and_transform(model_cfg['vision_backbone_id'], model_cfg['image_resize_strategy'], checkpoint_path=vla.vjepa_checkpoint_path)
llm, _ = get_llm_backbone_and_tokenizer(model_cfg['llm_backbone_id'], llm_max_length=model_cfg.get('llm_max_length', 32768), inference_mode=False, custom_hf_path=str(cfg.llm_checkpoint_path))
model = get_vlm(model_cfg['model_id'], model_cfg['arch_specifier'], vision, llm, enable_mixed_precision_training=False,
    d_action=19, d_proprio=19, action_horizon=20, fm_hidden_size=1024, fm_num_layers=16,
    fm_num_inference_timesteps=4, fm_num_timestep_buckets=1000, fm_noise_beta_alpha=1.5,
    fm_noise_beta_beta=1.0, fm_noise_s=0.999, fm_num_target_vision_tokens=32,
    fm_add_pos_embed=True, fm_max_seq_len=1024, fm_state_dropout=0.5,
    flow_gr00t_placeholder_tokens=64, lambda_visual_token_cosine=0.5, d_jepa=vision.embed_dim)
model.llm_backbone.load_state_dict(ckpt['llm_backbone'])
model.projector.load_state_dict(ckpt['projector'])
model = model.to(dtype=torch.float32).cuda()
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
