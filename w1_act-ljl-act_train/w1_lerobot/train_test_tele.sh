lerobot-train \
  --dataset.repo_id=/home/ubuntu/LeForge/Data/Preprocess_lerobot/hello_0317 \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30 \
  --policy.vision_encoder_type="resnet" \
  --policy.vision_normalize_in_processor=true \
  --policy.vision_normalize_in_model=false \
  --policy.vision_freeze=false \
  --policy.push_to_hub=false \
  --policy.repo_id="" \
  --policy.type=act \
  --num_workers=16 \
  --steps=2000000 \
  --output_dir=outputs/train/testing1 \
  --job_name=w1_hello_world \
  --policy.device=cuda \
  --wandb.enable=false \
  --log_freq=20 

## allowed_encoder_types = {"resnet", "siglip2", "dinov2", "dinov3"}
