# Bridge 自适应 Sample Factor 版本记录

本版本基于已冻结的 `bridge_2hz_20260810.md`。旧版保持 2 Hz 推理、30 Hz 图像和30 Hz动作；
本版将图像频率、动作消费频率和模型推理频率解耦。

## 调度合同

- 图像固定 30 Hz，每个源帧只读取并写入 Rerun 一次。
- ACT 原始输出始终为 `30×19`。
- bridge 处理后长度为 `30×sample_factor`，动作频率为 `30×sample_factor` Hz，覆盖时长仍约1秒。
- replan 间隔自动计算为 `round(动作频率 / 请求推理Hz)`；默认请求2 Hz。
- `sample_factor=2` 时：60 Hz动作、30步 replan 间隔、60步处理后 chunk。
- 3步交接权重按动作频率扩展；`sample_factor=2` 为
  `0.25, 0.40, 0.55, 0.675, 0.80, 0.80`，物理交接时间仍为100 ms。
- 调度不满足“间隔 + 预期推理延迟 + 交接 ≤ chunk horizon”时启动即失败。

## 完整回放证据

- Run：`bridge_sf2_2hz_full_20260810`；独立 verification：passed。
- 1391个30 Hz源图像、2782个60 Hz动作步；93次推理。
- 实际动作频率59.999650 Hz；p95周期16.891 ms；deadline miss=0。
- 推理提交步差全部为30，即实际2 Hz；held action=0；target step error=0。
- 关节反馈链 `qpos_before[t+1] == qpos_after[t]` 最大误差0。
- 每两个动作步的模型图像哈希一致，证明同一30 Hz图像被安全复用；RRD每路图像1391行、关节2782行。
- 关节限位违规0；waist幅度覆盖率1.02127。
- 新实现的 `sample_factor=1` / 2 Hz / 29帧回归也通过独立 verification，确认旧行为未被破坏。

与 `sample_factor=1` 的2 Hz基线相比，身体动作单步变化均值从0.02713降至0.01372，p95从
0.05074降至0.02597；换算为每秒变化后基本一致，说明动作被加密而非减速。chunk安装时的身体动作
跳变量p95从0.05695降至0.03426。

## 验证

- `w1_sim/tests + w1_act_sim/tests`：136 passed，另1个本地端口测试单独 passed。
- Ruff、py_compile、`bash -n`：passed。

## Source Hashes

```text
0d006a3f36f419d9bd5a9fe4e0be2b9fd35b4f37d7b0d8e7cb9e7020ffb34f78  w1_act_sim/bridge_controller.py
bd99a8d1c52fafd8a0c3e09691e4e15a3c2ef7f1320ab8fa800e8ed846f6c72c  w1_act_sim/action_processor.py
ec65a241aa123c78c0fdfecb6e5c4955063549ce01c40ff93bc52983e9a4abb3  w1_act_sim/deployment.py
d2d0beb18d9f8008212e3b2fa0fd2bbc228282ae24407b58e1f2578d018396b6  w1_act_sim/run.py
6ea23f45755bd9789425d58fef4fc7e5bb2adefd0bb53ac5c2bef88589609679  w1_act_sim/launch.py
b83ef9709256cff9618b8ad34857b2198f424931fe5dc436ab71e881a742f011  w1_act_sim/verify.py
57aee584a7e28f3d0f7c7855302b49bfdf4505f406a51b13e6081cf503622167  run_act_sim.sh
22d5542a11a7826d77fdd482706548e2d377888777eeb523f0f8c85a3e7fa6e3  w1_act_sim/README.md
163886bdc995dffd727b1ed9d5b821bad562f109d73bb475e485150858070b44  w1_act_sim/explain_sample_factor_cn.py
```

## Artifact Hashes

```text
1a5af9347b950de7ac533a169e8bc8f4f8bf0fe3cfe1921da50eb0a67ac6bce1  act_sim_summary_bridge_sf2_2hz_full_20260810.json
fa47e80a461b021c8d83256185c7a90b61b586af5b8cc704a904023f79af8450  act_sim_trajectory_bridge_sf2_2hz_full_20260810.npz
34a6e36f0a40dd6c15cacccb2433dbb451dbfb5f1f1608db9d5fe508ddaafa1a  verification_bridge_sf2_2hz_full_20260810.json
5eb93a1413995c9607ce7a677dc48895ff85547d9a062f57a5f43518f0c1cb8f  bridge_sample_factor_2_cn.png
```
