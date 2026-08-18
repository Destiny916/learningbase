# Bridge Sample Factor 2 实时可视化修复记录

本版本修复 `sample_factor=2` 时原生 Rerun 可视化反向拖慢控制循环的问题。播放现在以源数据时间轴为准：
动作按 60 Hz 执行，图像按 30 Hz 更新，机器人显示对应图像时刻已经执行控制后的真实 MuJoCo 状态。

## 修复原理

- 60 Hz 控制、关节状态和动作完整保存在 trajectory NPZ 与 TensorBoard 中。
- Rerun 不再重复写入每个 60 Hz 控制步；仅在 30 Hz 源图像切换时记录一次。
- 每个 Rerun 机器人姿态取当步 `sim_qpos_after`，不是目标姿态或插值占位姿态。
- Rerun 增加 `sim_time` 时间线，时间戳为源图像相对起始时间，保证回放速度与实际录制时间一致。
- 验证器要求 `visualization_step = source_frame_index × sample_factor`，并检查机器人和各路图像行数一致。

## 完整回放对比

| 指标 | 修复前 | 修复后 |
| --- | ---: | ---: |
| 动作频率 | 34.585 Hz | 59.999 Hz |
| 图像频率 | 17.293 Hz | 29.9996 Hz |
| 1391 图像帧播放时长 | 80.438 s | 46.367 s |
| p95 控制周期 | 41.258 ms | 16.873 ms |
| deadline miss | 2412 / 2782 | 3 / 2782 |
| Rerun 机器人/每路图像行数 | 不一致 | 1391 / 1391 |

正式运行 `bridge_sf2_live_decoupled_full_20260810` 完成 2782 个动作步、1391 个图像帧和 93 次
ACT 推理；严格验证通过，关节限位违规为 0，跟踪误差为 0。身体 MAE 为 0.05889 rad，腰部幅度覆盖率为 1.02127。

## 验证结果

- `w1_sim/tests + w1_act_sim/tests`：136 passed，1项本地端口测试单独 passed。
- Ruff、py_compile、`bash -n`：passed。
- RRD 结构与时序由严格 verifier 验证通过。

## Source Hashes

```text
1affa5f8546768ca6d8d51f45432658306e9cbda327986f4a1265af60c59970a  w1_sim/telemetry.py
2beef1ab0b281ec05bc6ecbf4f8226a645ce73f6986d6cde17a5dedcda3afb74  w1_act_sim/run.py
9d191ed374bc539581448090dd97fdb579817451e2a817cccac8aa84cce867b7  w1_act_sim/verify.py
ea9722e8ab6f522468b8b562f2332f4508623c73e33b1632a54795d94ac5303a  w1_sim/tests/test_telemetry.py
2383f7c34bd548e7311bc86f7cff9a8d51b829fa763901bdb80f67d3ec03c8e8  w1_act_sim/tests/test_runner_integration.py
91aa271d55617d25d9569470b38ecc0eef2baa97d7de15388295a12f1257fea1  w1_act_sim/README.md
```

## Artifact Hashes

```text
bc1587fa6a86fd7af511065e75a4eb24da0382bc2791eb68bbd21710ddb9b16a  act_sim_summary_bridge_sf2_live_decoupled_full_20260810.json
2f948da9a415b89c6e450c29198609fdff8a5ef6197bfb77b42bbff29afe40e5  act_sim_trajectory_bridge_sf2_live_decoupled_full_20260810.npz
40255fd9c141e590e40a6435a9da5cf7851d5cdc4c2401be6f9ab603a5aebf20  act_sim_bridge_sf2_live_decoupled_full_20260810.rrd
4d866d59fde040eb3f3d8a2fc6dbc25721b2e62087a64b9fe801221515f4aef9  verification_bridge_sf2_live_decoupled_full_20260810.json
```
