# PI05 官方 LIBERO 闭环 Rollout 冒烟验证

日期：2026-07-17

评估器在远端以 `uid=1009 gid=1008` 运行，仅向 Docker 暴露主机 GPU 3。它加载：

```text
/data/wengyikun/pi05_official_libero_smoke/train_out/checkpoints/000001/pretrained_model
```

评估配置为 `libero_spatial` task 0、相对控制、一个环境和一个 episode。

`lerobot-pi05-libero-smoke:20260717-libero-rootfix-assets` 镜像提供 EGL；将
包内层 `libero/` 目录用于 task init states；并将包的 assets 目录映射到预下载的
`/opt/libero-assets`，从而避免运行时重新下载。

评估器确认完整闭环：

```text
Built vec env | suite=libero_spatial | task_id=0 | n_envs=1
Loaded state dict from model.safetensors
All keys loaded successfully!
Running rollout with at most 280 steps: 100%
End of eval
```

产物：

```text
/data/wengyikun/pi05_official_libero_smoke/eval_out/eval_info.json
/data/wengyikun/pi05_official_libero_smoke/eval_out/videos/libero_spatial_0/eval_episode_0.mp4
```

保存指标：共一个 episode，`avg_sum_reward=0.0`，`avg_max_reward=0.0`，
`pc_success=0.0`，`successes=[false]`，`eval_s=86.342936`。

这证明离线微调 checkpoint 重载、MuJoCo reset、图像/状态到相对动作推理、环境
step、指标持久化和视频输出能够完整联通。它不是任务能力基准：策略只在一个离线
episode 上进行了一个优化器更新，且由于默认 `log_freq=200` 大于一步，冒烟配置
没有输出数值训练 loss。
