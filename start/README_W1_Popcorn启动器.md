# W1 Popcorn PC1/PC2 启动器索引

这些文件是独立归档，默认不会覆盖 `/home/dexforce/w1/w1_act` 中已有的启动器。每个 `.sh` 文件顶部均标注运行主机、模型、动作块和执行策略。
完整使用说明见：
`../howtotrain/W1_Popcorn三模型_PC1_PC2推理部署总说明.md`。

| 模型 | PC2 服务 | PC1 客户端 | PC1 控制器 | 合同 |
|---|---|---|---|---|
| 220000（EE/FK loss） | `pc2_server_220000_ee_loss_relative.sh` | `pc1_client_220000_ee_loss_relative.sh` | `control_220000_ee_loss_relative.py` | DINOv3，19D，相对臂关节，16 步，q01/q99 |
| 180000（DINOv3） | `pc2_server_180000_dinov3_relative.sh` | `pc1_client_180000_dinov3_relative.sh` | `control_180000_dinov3.py` | DINOv3，19D，相对臂关节，16 步，q01/q99 |
| 500000（旧 ACT） | `pc2_server_500000_absolute.sh` | `pc1_client_500000_absolute.sh` | `control_500000_absolute.py` | 绝对位姿；当前 PC2 artifact 已核对 chunk 16，替换权重后仍以 `config.json` 为准 |
| 200000（ACT async100） | `pc2_server_200000_chunk100.sh` | `pc1_client_200000_async100.sh` | `control_200000_async100.py` | 100 策略点，200 控制点，异步重规划；PC1 manager 端口 8896 |
| 200000（ACT async100 无插值） | 同上 | `pc1_client_200000_nointerp_async.sh` | `control_200000_nointerp_async.py` | 100 策略点，100 控制点，剩余 15 点触发，重叠 15 点线性融合；PC1 manager 端口 8897 |

## 启动器类型速查

| 文件组 | 主机 | 类型/用途 | 动作执行逻辑 |
|---|---|---|---|
| `pc2_server_*.sh` | PC2 | GPU policy server | 只加载 checkpoint 并返回动作，不发布 ROS 或电机命令 |
| `pc1_client_180000*`, `pc1_client_220000*` | PC1 | DINOv3 relative 客户端 | 相邻真实反馈形成相对臂关节 state；动作反归一化、还原绝对目标并限位后发布 |
| `pc1_client_500000_absolute.sh` | PC1 | 旧 ACT absolute 客户端 | state/action 按绝对值处理，不做增量还原 |
| `pc1_client_200000_sync100.sh` | PC1 | 100 点同步客户端 | 当前 100 点执行完成后才请求下一块，不插值、不提前请求 |
| `pc1_client_200000_async100.sh` | PC1 | 100 点异步+插值客户端 | 100 策略点以 `sample_factor=2` 变为 200 控制点；剩 30 控制点请求下一块；身体最多 30 点 LIPO，手部标量不融合 |
| `pc1_client_200000_nointerp_async.sh` | PC1 | 100 点异步无插值客户端 | 100 策略点直接执行；剩 15 点请求下一块；身体最多 15 点 LIPO，手部标量不融合 |
| `pc1_client_200000_chunk100.sh`, `pc1_client_500000_chunk100.sh` | PC1 | 100 点同步块客户端 | 每块 100 点直接执行；不启用异步重规划和融合 |

`control_*.py` 仅负责 PC1 manager 的 `status/start/stop`，不是模型服务；`config_*.json` 与 `client_runtime_*.json` 是参数快照，实际值以启动器传入配置为准。

## 归档内文件

- `profiles/popcorn_w1.json`：当前 W1 的通用基线（IP、路径、ROS、话题、关节合同和已核对相机身份）。
- `profiles/new_w1_template.json`：新 W1 迁移模板；相机序列号和任何差异必须先填写并现场核验。
- `preflight_w1_profile.py`：只读 profile/checkpoint 合同检查，不连接控制端口、不发布动作。
- `config_*.json`：PC2/PC1 参数快照；`client_runtime_*.json` 是控制器可直接读取的同等配置。
- `client_service_160000.py`、`client_runtime_160000.py`、`runtime.py`、`runtime_w1_contract.py`：相对 19D 客户端及安全合同实现。文件名中的 `160000` 是协议兼容名称，不代表使用 160000 checkpoint。
- `set_hand_scalar.py`：仅发送左右开合度标量的辅助工具。
- `control_200000_async100.py`：200000 async100 专用 status/start/stop 控制器；`--skip-default-pose-check` 必须显式指定。
- `convert_relative_checkpoint_180000.py`：180000 checkpoint 转换/审计工具。

## 运行位置

shell 启动器应在对应主机执行：PC2 执行 `pc2_server_*.sh`，PC1 执行 `pc1_client_*.sh`。
`control_*.py` 通过 PC1 manager 端口执行 `status/start/stop`。先 dry-run 和门禁，再由操作者明确发送 `start`。

启动器保留当前基线作为默认值，同时支持环境变量覆盖，例如：

```bash
W1_ACT_ROOT=/opt/w1/w1_act CONFIG_PATH=/opt/w1/direct_runtime/client_runtime_180000.json \
  ./pc1_client_180000_dinov3_relative.sh
```
