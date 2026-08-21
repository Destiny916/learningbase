# W1 Local Inference and PC1 Executor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run ACT inference on the local RTX host while PC1 collects W1 observations and safely publishes either simulation or explicitly armed real-robot actions, without PC2 or XWiz.

**Architecture:** Reuse the tested `xwiz_real_runtime.client_service` on PC1 and the existing local `xwiz-act-server.service`. Add a ROS-independent local CLI that sends the existing length-prefixed JSON manager protocol to PC1 `:8890`, plus a PID-safe PC1 lifecycle script for the client and black wrist publishers.

**Tech Stack:** Python 3.10/3.12, pytest, TCP length-prefixed JSON, ROS 2 Humble/CycloneDDS on PC1, systemd user service on the local host, Bash.

---

### Task 1: Add the ROS-independent control CLI

**Files:**
- Create: `w1_act-ljl-act_train/xwiz_real_runtime/control_cli.py`
- Create: `tests/xwiz_real_runtime/test_control_cli.py`

- [ ] **Step 1: Write failing tests for simulation, real confirmation, stop and status**

Create a fake controller that records `setup_config`, `stop`, and `get_status` calls. Test that `start-sim` accepts only task mode 1, `start-real` rejects every confirmation except `EXECUTE_100_REAL_FRAMES`, both paths force `action_horizon=max_steps=100`, and stop/status forward exactly once.

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```bash
PYTHONPATH=w1_act-ljl-act_train pytest -q tests/xwiz_real_runtime/test_control_cli.py
```

Expected: collection fails because `xwiz_real_runtime.control_cli` does not exist.

- [ ] **Step 3: Implement the minimal CLI**

Implement these public functions and constant:

```python
REAL_CONFIRMATION = "EXECUTE_100_REAL_FRAMES"

def load_task(path: str | Path) -> dict[str, object]: ...
def start_task(controller, task: Mapping[str, object], expected_mode: int,
               confirmation: str = "") -> bool: ...
def stop_runtime(controller) -> bool: ...
def runtime_status(controller) -> Mapping[str, object]: ...
def main() -> int: ...
```

`start_task` must reject a task whose `client_config.mode` differs from `expected_mode`, require the exact confirmation for mode 2, call `prepare_resolved_configs`, and invoke `InferenceClientController.setup_config`. The argparse commands are `start-sim --task`, `start-real --task --confirm`, `status`, and `stop`; `--client-host` defaults to `192.168.20.20` and `--client-port` defaults to `8890`.

- [ ] **Step 4: Run the focused test and full runtime tests**

Run:

```bash
PYTHONPATH=w1_act-ljl-act_train pytest -q \
  tests/xwiz_real_runtime/test_control_cli.py \
  tests/xwiz_real_runtime/test_runtime.py \
  tests/xwiz_real_runtime/test_manager_runtime.py
```

Expected: all tests pass.

### Task 2: Add PC1 runtime lifecycle management

**Files:**
- Create: `w1_act-ljl-act_train/xwiz_real_runtime/start_pc1_runtime.sh`
- Modify: `w1_act-ljl-act_train/xwiz_real_runtime/client_service.py`
- Test: `tests/xwiz_real_runtime/test_pc1_runtime_script.py`

- [ ] **Step 1: Write a failing structural test**

Test that the script supports `start|stop|status`, exports `ROS_DOMAIN_ID=20` and `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`, sources both Humble and `/home/dexforce/w1/install/setup.bash`, starts `black_wrist_images` and `client_service`, stores separate PID files, and never uses `pkill` or a broad process pattern.

- [ ] **Step 2: Run the structural test and verify failure**

Run:

```bash
PYTHONPATH=w1_act-ljl-act_train pytest -q tests/xwiz_real_runtime/test_pc1_runtime_script.py
```

Expected: failure because the script does not exist.

- [ ] **Step 3: Implement the lifecycle script**

Use fixed paths under `/home/dexforce/w1/w1_act`, logs under `/home/dexforce/`, and PID files under `/home/dexforce/.cache/xwiz-real-runtime/`. `start` must refuse to replace a live PID, start the black publisher first, wait for it, then start `client_service --config .../client_runtime.json` and verify `:8890`. `stop` must signal only validated saved PIDs, wait with a bounded loop, and remove stale PID files. `status` must report both processes and the listener.

- [ ] **Step 4: Make client logging host-neutral**

Change the startup log from `XWiz dual-mode PC2 client` to `XWiz dual-mode W1 client`; do not alter action or safety behavior.

- [ ] **Step 5: Validate shell syntax and tests**

Run:

```bash
bash -n w1_act-ljl-act_train/xwiz_real_runtime/start_pc1_runtime.sh
PYTHONPATH=w1_act-ljl-act_train pytest -q tests/xwiz_real_runtime
```

Expected: shell syntax succeeds and all runtime tests pass.

### Task 3: Deploy without PC2 and validate observation transport

**Files:**
- Deploy: PC1 `/home/dexforce/w1/w1_act/xwiz_real_runtime/`
- Use: `w1_act-ljl-act_train/xwiz_act_server/start_local.sh`

- [ ] **Step 1: Record the safe baseline**

Verify PC1 Auto/Tele/ACT/Map are inactive, `act_ros2` is exited, and `/control/joint_position`, `/control/ee/left`, `/control/ee/right` carry zero messages over five seconds.

- [ ] **Step 2: Stop the PC2 black publisher and any PC2 inference client**

Stop only their recorded or exact PIDs. Verify PC2 no longer publishes either wrist topic and has no `:8890` listener.

- [ ] **Step 3: Deploy and start the PC1 idle runtime**

Copy the tested runtime directory to PC1, run `start_pc1_runtime.sh start`, and verify PC1 owns one publisher for each black wrist topic, both images are 640x360 `bgr8` with all pixels zero, and PC1 listens on `:8890`.

- [ ] **Step 4: Start the existing local model service**

Run `systemctl --user start xwiz-act-server.service`; require `active`, a `:8889` listener, and a clean model-load log before continuing.

### Task 4: Run a simulation-only end-to-end acceptance test

**Files:**
- Use: `w1_act-ljl-act_train/xwiz_real_runtime/task_1_simulation.json`

- [ ] **Step 1: Capture real-topic publisher ownership**

Record verbose publisher identities for the three real control topics before inference.

- [ ] **Step 2: Start one simulation chunk from the local CLI**

Run:

```bash
PYTHONPATH=w1_act-ljl-act_train python -m xwiz_real_runtime.control_cli \
  --client-host 192.168.20.20 start-sim \
  --task w1_act-ljl-act_train/xwiz_real_runtime/task_1_simulation.json
```

Expected: setup succeeds, the model logs one finite `100x19` inference, and PC1 publishes exactly 100 simulation frames before returning Idle.

- [ ] **Step 3: Prove real topics remained untouched**

Compare verbose publisher identities and listen for five seconds. The inference client must not appear as a real-topic publisher and no real control message may be produced by this acceptance test.

- [ ] **Step 4: Verify stop and status commands**

Run CLI `status`, then `stop`; require a successful response and no active inference thread while the idle PC1 listener remains available.

### Task 5: Validate the real path without moving the robot and document operation

**Files:**
- Modify: `howtotrain/XWiz仿真推理使用手册.md`
- Create: `howtotrain/本机ACT推理与PC1安全执行器.md`

- [ ] **Step 1: Test the local real confirmation gate**

Invoke `start-real` without `--confirm` and with a wrong value. Both commands must exit nonzero before any TCP setup request reaches PC1.

- [ ] **Step 2: Dry-run the PC1 preflight without deployment**

Read current robot state and evaluate `validate_robot_ready` against it. Report whether the robot is at ACT default pose, but do not send the valid real confirmation string and do not publish any motion.

- [ ] **Step 3: Write the Chinese operating guide**

Document architecture, start/stop/status commands, simulation command, real confirmation command, prerequisites, logs, the 100-frame boundary, recovery, and the statement that starting services alone never moves the robot.

- [ ] **Step 4: Run final verification**

Run all `tests/xwiz_act_server` and `tests/xwiz_real_runtime`, shell syntax checks, fresh process/port checks, camera validation, and five-second real-topic silence checks. Leave the robot with Auto/Tele/ACT/Map inactive and no active inference.
