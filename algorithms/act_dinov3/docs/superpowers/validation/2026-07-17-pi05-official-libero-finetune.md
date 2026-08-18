# PI05 官方 LIBERO 微调冒烟验证

日期：2026-07-17

## 验证范围

这是一次受限的官方 LeRobot PI05 微调冒烟验证。它证明官方 LIBERO 数据集
契约、`lerobot/pi05_libero` 初始化、一次优化器更新、checkpoint 序列化与
checkpoint 重载能够协同工作。它不是基准测试，也不是生产训练。

## 远端命令

命令在 `183.230.224.121:50210` 上以 `uid=1009 gid=1008` 运行，使用预检
通过的镜像 `lerobot-pi05-libero-smoke:20260717`
（`sha256:a3987c43e745e7d1a5356921733e2325a2c4a4c99d04114568e517e21ae23d7b`）。
容器仅暴露主机 GPU 3，容器内显示为 `cuda:0`。

```bash
cd /home/wengyikun/lerobot
LEROBOT_GPUS=device=3 \
  bash run_scripts/remote_pi05_libero_smoke_container.sh \
  bash run_scripts/launch_pi05_official_libero_smoke.sh
```

固定的冒烟实验契约：

- 数据集：`HuggingFaceVLA/libero`，episode `[0]`，共 214 帧。
- 基础策略：`lerobot/pi05_libero`。
- LIBERO 原生动作：7D 相对控制，`joint_gripper_indices=[6]`。
- PI05：BF16，`chunk_size=50`，`n_action_steps=10`。
- 训练：`steps=1`，`batch_size=1`，梯度累积为 `1`，单进程，有效 batch size 为 `1`。
- checkpoint：第 1 step 保存；不进行环境评估、不上传 Hub。

数据加载器解析了四个 Hugging Face 数据集文件；策略加载器报告从
`lerobot/pi05_libero` 的 `model.safetensors` 成功加载。训练日志不能区分这四个
数据文件是缓存命中还是网络传输；预检阶段已准备 LIBERO assets 与 PaliGemma
tokenizer 缓存。

## 参数冻结审计

训练日志确认了预期参数策略：

| 模块 | 总参数量 | 可训练参数量 |
| --- | ---: | ---: |
| PaliGemma 语言模型 | 3B | 0 |
| 视觉编码器 | 412M | 412M |
| 多模态投影层 | 2M | 2M |
| Action Expert 与其他动作模块 | 1B | 1B |

完整策略共有 `4,143,404,816` 参数，其中 `1,634,873,104` 可训练。因此语言
骨干已冻结，而视觉路径、投影层和动作路径均可训练。

## 训练与 Checkpoint 证据

一次优化器更新耗时 20.10 秒，训练器随后写出 checkpoint 并正常退出：

```text
Training: 100%|...| 1/1 [00:20..., 20.10s/step]
Checkpoint policy after step 1
End of training
```

本次没有数值训练 loss：启动脚本未覆盖 LeRobot 默认的 `log_freq=200`，而冒烟
训练总共只有一步。这不影响“完成优化器更新并保存 checkpoint”的验证，但不能把它
当作离线 loss 测量。

持久化 checkpoint：

```text
/data/wengyikun/pi05_official_libero_smoke/train_out/checkpoints/000001/
```

它包含 `pretrained_model/config.json`、9,354,045,072 字节的
`pretrained_model/model.safetensors`、两个 policy processor JSON 及其归一化
张量，以及 optimizer、scheduler、RNG、参数组和 `training_step.json` 状态。
整个冒烟输出占用 14GB。

## 重载探针

在同一镜像、同一当前用户容器契约和隔离的 GPU 3 中，使用 PI05 配置注册表重建
保存的模型，并以 `strict=True` 加载。探针输出：

```text
Loaded state dict from model.safetensors
All keys loaded successfully!
CHECKPOINT_LOAD_OK PI05Policy 4143404816 cuda:0
```

不能直接用 `PI05Config.from_pretrained(...)` 加载该配置文件，因为其中保留了顶层
`type` 判别字段。成功探针先导入 PI05，再用
`PreTrainedConfig.from_pretrained(...)` 分派到 `PI05Config`，这与策略加载路径
一致。

## 后续闭环验证

后续任务使用该 `000001` checkpoint，在 `libero_spatial` task 0 执行一回合闭环
rollout。其 reward/success 衡量策略与环境的交互，和本次离线优化验证是不同指标。
