# ACT动作质量指标版本记录

本版本基于 `bridge_continuous_gripper_20260811.md`，增加与模型及控制路径隔离的在线动作质量评估器。

## 质量合同

- 综合分范围0～100，默认由姿态40%、双手末端30%、运动方向20%、运动幅度10%组成。
- 各项可由两个启动脚本中的 `QUALITY_*` 开关单独启停；所选权重自动归一化。
- `gain` 是综合分相对保持初始姿态基线的提升。
- 参考关节按控制时间戳插值，只进入评估器，不进入ACT输入、动作处理或MuJoCo反馈。
- 逐步指标同时写入终端、TensorBoard、trajectory NPZ和summary；verifier会独立重算。

## 验证结果

```text
w1_act_sim/tests: 128 passed
Ruff: passed
compileall: passed
bash -n: passed
quality_metrics_final_smoke_20260811: strict verification passed
```

真实3源帧、6控制步smoke最终输出：

```text
quality=91.2 gain=+11.3 pose=0.002 ee=0.5cm dir=60% amp=100%
```

## Source Hashes

```text
d5ef9c013070a28f17895546fd377a7011c735d42db81217b71228f60b1e3271  w1_act_sim/quality.py
f92b6b540dd4de0348ea7e5c586db375a4d42aaa47f9e3743abbb213d52ab93c  w1_act_sim/run.py
a5a4bf17d11d4f9677cfd666fc07cda7b067fd532b9147a8b9402f4243db9423  w1_act_sim/launch.py
868b14171ae6b203488e1af8574d1a5f1c6a78228d92589430b0db4e514125eb  w1_act_sim/verify.py
dfeacd3d2356d8cb1f03ca1e3b3111fc5575291397c0e0ea95f9466abbadaa15  w1_act_sim/run_act_sim.sh
3061a19d0e7b40c990100e2639b4079d9f698f354183c0256e34c532b57ffb05  w1_act_sim/run_act_sim_bridge.sh
cdbbba9ccfc2e2104b418580fb26c8f7fe7267fd4a554e9e14a61c9384ab051b  w1_act_sim/README.md
5263a579564ff1c0d81375309050c6eadadca05fd27d525d2729d90900c759c3  w1_act_sim/tests/test_quality.py
b0188a71a1ce69d8c2bbc183f58be6483cccc1cf5883b5d54d6962cca4816552  w1_act_sim/tests/test_runner_integration.py
c542af183ba9fb5729762e03c9e5bf1a58c6e9a75b54677e6401558470a43abd  w1_act_sim/tests/test_launch.py
c50687cad66b09839e37ee783f69cc75154aefec5f502446d37ac3e08e8d01f9  w1_act_sim/tests/test_shell_entrypoints.py
```

## Artifact Hashes

```text
3e5bf13c91b58a4de0009f7caf8aef5fbfbb631dd4c2e0e227e6bac5556c3047  act_sim_summary_quality_metrics_final_smoke_20260811.json
a5069061524aa9de0d6ce70f04831387cf42550364d17b7872acd3bf887d2406  act_sim_trajectory_quality_metrics_final_smoke_20260811.npz
f330c6148f7fa47c82dfd66fb580b202e204a1d2945e7df936d2724a3ae2409a  verification_quality_metrics_final_smoke_20260811.json
```
