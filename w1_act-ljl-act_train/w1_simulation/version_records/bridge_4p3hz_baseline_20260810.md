# Bridge 4.3 Hz Baseline Record

Recorded at `2026-08-10 14:28:47 CST` before introducing fixed 2 Hz replanning.

## Scheduling Contract

- Control and image rate: 30 Hz.
- ACT chunk: 30 steps × 19 dimensions.
- Minimum simulated inference latency: 200 ms.
- Replan trigger: submit immediately after the previous asynchronous request is installed.
- Observed cadence: approximately one request every 7 control steps, or 4.3 Hz.
- Temporal alignment: `action_index = control_step - submit_step`.
- Candidate limit: 3.
- Newest-chunk handoff weights: `0.25, 0.55, 0.80`.

## Full-run Evidence

- Run: `bridge_temporal_200ms_full_20260810`.
- Frames: 1391.
- Completed policy calls: 200.
- Rollout duration: 46.367191521 s.
- Effective control rate: 29.999660414 Hz.
- Mean policy latency: 200.703851045 ms.
- p95 cycle time: 33.256054 ms.
- Deadline misses: 0.
- Verification status: passed.

## Source Hashes

```text
ce73cd780b1d48cfa34398c3cd1919afe3713290415895179b25b3314d8c787e  w1_act_sim/bridge_controller.py
ff1e1a2f5ebc702419a69fec2583888846904248d74dba60ee37381363ffdc2f  w1_act_sim/run.py
c8af673411e3b1240bf1d7800f02f6a995a5d21bea026a428c586ee34cbac323  w1_act_sim/launch.py
5b207a4d1691709d56f7c95b7f4657cf2ba9f3716e524b62fb6db55776f0b9d8  w1_act_sim/verify.py
5175e1b89aec37a01e2845c59da5984f044ff6d15dd13557a967385da08431be  run_act_sim.sh
bf087fbf48ad6f8ce47f1a91e104cb8d1ada88afbf19facd043e1a34f3b18a11  w1_act_sim/README.md
```

## Artifact Hashes

```text
d736f1b53eafd44b822a96254a8cc7430c6c904d1a012ee3c1c2ffa07ac6599d  act_sim_summary_bridge_temporal_200ms_full_20260810.json
b03895b3fca8222717286ebf565e4f61733d562b0434604ead612740cc127f56  act_sim_trajectory_bridge_temporal_200ms_full_20260810.npz
3d132842d53e00b7497697be5dbc6587dbf684019eac968fe7a7a5aa6d677b79  verification_bridge_temporal_200ms_full_20260810.json
```

This record is descriptive and does not contain a Git commit or a source snapshot. The hashes and retained
artifacts provide the immutable comparison reference.
