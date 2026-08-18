# W1 Profile、标准命令与动态 LIPO 验收记录

日期：2026-08-16

## 版本范围

- 默认配置统一到 `configs/w1_popcorn_v1.json`。
- ACT 19维动作通过纯W1适配层生成身体20维、左手6维、右手6维位置命令。
- ROS运行端固定发布 `/control/joint_position`、`/control/hand/left`、`/control/hand/right`。
- raw与bridge动作处理器固定选择，删除外部 `MODULE:FACTORY` 插件入口。
- 删除旧顶层模块别名、blocking执行入口、重复bridge-core参数和独立手映射配置。
- `run.py`与`evaluation/verify.py`保留为薄入口，执行、记录、摘要和验证实现分层。
- ACT三路视觉输入继续来自origin数据集，未改为MuJoCo相机。

## 静态与单元验收

- Ruff check：通过。
- Ruff format check：通过，46个目标文件保持格式化。
- bash语法检查：通过。
- compileall：通过。
- pytest：216 passed。

## 完整raw验收

运行目录：

`artifacts/runs/20260816_164105_976422_scheme_b_profile_raw_full`

- 1391/1391帧完成。
- 14次推理，完整chunk均执行到 `action_index=99` 后重新从0开始。
- 身体、左手、右手标准命令shape分别为 `[1391,20]`、`[1391,6]`、`[1391,6]`。
- held command为0，target step error最大绝对值为0。
- 总分92.6952。
- `ACT_SIM_VERIFICATION_STATUS=passed`。

## 完整bridge验收

运行目录：

`artifacts/runs/20260816_164311_947696_scheme_b_profile_bridge_full`

- 1391个origin帧生成2784个60 Hz控制点。
- 每个100策略点chunk插值为200控制点。
- 阈值0.5对应剩余50策略点/100控制点触发重规划。
- 重规划提交步为0、100、200至2700，共28次推理。
- 等待新轨迹时继续执行旧轨迹，held command为0。
- 每次LIPO为5策略点/10控制点；27次切换共270个blend-active控制点。
- 身体、左手、右手标准命令shape分别为 `[2784,20]`、`[2784,6]`、`[2784,6]`。
- target step error最大绝对值为0。
- 总分93.1747。
- `ACT_SIM_VERIFICATION_STATUS=passed`。

两个完整运行目录均包含 `summary.json`、`trajectory.npz`、`verification.json`、
`recording.rrd`、TensorBoard event、Rerun日志和TensorBoard日志。

完成最终blocking表面清理后，raw与bridge又分别执行3帧严格smoke，运行目录为
`artifacts/runs/20260816_164958_198127_scheme_b_final_raw_smoke` 和
`artifacts/runs/20260816_165017_768495_scheme_b_final_bridge_smoke`，两者状态均为passed。

## 未执行项

当前环境缺少 `rclpy`、`joint_interfaces` 和 `end_effector_interfaces`，因此ROS发送端使用注入的
消息类型和publisher完成序列化单元测试，没有声明真机ROS图验证通过。
