# XWiz W1 ACT Simulation Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make XWiz's “仿真推理” button load `act_popcorn_45w`, infer from a real head image plus two black wrist images, and publish only simulation actions through the PC2 client.

**Architecture:** A ROS-independent Python service on local host `192.168.20.164:8889` implements the PC2 client's framed-pickle protocol and adapts legacy W1 observations to the LeRobot ACT 19D/three-camera contract. The existing PC2 safety wrapper remains the only ROS component and enforces `mode=1`, empty home position, and simulation-only publishers. Model runtime is isolated behind an interface so protocol and mapping tests run without CUDA.

**Tech Stack:** Python 3.12, NumPy, PyTorch CUDA, custom LeRobot ACT source, safetensors, pytest, TCP framed pickle, ROS 2 Humble on PC2.

---

## File map

- Create `w1_act-ljl-act_train/xwiz_act_server/protocol.py`: framed-pickle encode/decode and one-client request server.
- Create `w1_act-ljl-act_train/xwiz_act_server/contract.py`: legacy observation decoding, 19D state assembly, RGB image mapping, and action grouping.
- Create `w1_act-ljl-act_train/xwiz_act_server/model_runtime.py`: strict checkpoint/preprocessor/postprocessor loading and ACT chunk inference.
- Create `w1_act-ljl-act_train/xwiz_act_server/server.py`: XWiz/PC2 request state machine and CLI.
- Create `w1_act-ljl-act_train/xwiz_act_server/__init__.py`: package marker.
- Create `w1_act-ljl-act_train/xwiz_act_server/runtime-requirements.txt`: only extra dependencies missing from the shared CUDA venv.
- Create `w1_act-ljl-act_train/xwiz_act_server/start_local.sh`: reproducible local launcher with explicit source/dependency paths.
- Create `tests/xwiz_act_server/test_protocol.py`: framing tests.
- Create `tests/xwiz_act_server/test_contract.py`: input/output contract and safety validation tests.
- Create `tests/xwiz_act_server/test_server.py`: request lifecycle tests using a fake model runtime.
- Modify local XWiz task `~/.dexforce/XWiz/model_deployments/tasks/1/task_config.json`: set local server, 100-step horizon, black wrist cameras, and no home position.
- Deploy existing safety wrapper files to PC2 `/home/dexforce/w1/w1_act/xwiz_safe_runtime/`.

### Task 1: Protocol framing

**Files:**
- Create: `w1_act-ljl-act_train/xwiz_act_server/__init__.py`
- Create: `w1_act-ljl-act_train/xwiz_act_server/protocol.py`
- Test: `tests/xwiz_act_server/test_protocol.py`

- [x] **Step 1: Write failing framing tests**

```python
def test_round_trip_frame_over_socketpair():
    left, right = socket.socketpair()
    send_message(left, {"type": "STATUS", "request_id": 7})
    assert recv_message(right) == {"type": "STATUS", "request_id": 7}

def test_rejects_frame_larger_than_limit():
    left, right = socket.socketpair()
    left.sendall(struct.pack(">I", MAX_FRAME_BYTES + 1))
    with pytest.raises(ProtocolError, match="too large"):
        recv_message(right)
```

- [x] **Step 2: Run RED**

Run: `PYTHONPATH=w1_act-ljl-act_train pytest -q tests/xwiz_act_server/test_protocol.py`

Expected: collection fails because `xwiz_act_server.protocol` does not exist.

- [x] **Step 3: Implement minimal framing**

Implement `recv_exact()`, `recv_message()`, and `send_message()` using a four-byte big-endian length and `pickle.dumps/loads`. Set `MAX_FRAME_BYTES = 64 * 1024 * 1024`; raise `ProtocolError` for EOF, invalid types, and oversized frames.

- [x] **Step 4: Run GREEN**

Run: `PYTHONPATH=w1_act-ljl-act_train pytest -q tests/xwiz_act_server/test_protocol.py`

Expected: all protocol tests pass.

- [x] **Step 5: Commit**

```bash
git add w1_act-ljl-act_train/xwiz_act_server tests/xwiz_act_server/test_protocol.py
git commit -m "feat: add XWiz framed pickle protocol"
```

### Task 2: W1 observation and action contract

**Files:**
- Create: `w1_act-ljl-act_train/xwiz_act_server/contract.py`
- Test: `tests/xwiz_act_server/test_contract.py`

- [x] **Step 1: Write failing mapping tests**

Create a legacy observation containing:

```python
states = {
    "waistqpos": np.array([3], np.float32),
    "left_armqpos": np.arange(10, 17, dtype=np.float32),
    "headqpos": np.array([20, 21], np.float32),
    "right_armqpos": np.arange(30, 37, dtype=np.float32),
    "left_eefgripper": np.array([40], np.float32),
    "right_eefgripper": np.array([41], np.float32),
}
```

Assert `decode_observation()` returns state order `[3,10..16,20,21,30..36,40,41]`, maps `cam_high`/wrist keys to the three training keys, converts BGR red `[0,0,255]` to RGB `[255,0,0]`, and rejects missing keys or non-640×360 images. Assert `group_action_chunk(np.zeros((100,19)))` produces group widths `1,7,2,7,1,1` and rejects NaN or wrong shape.

- [x] **Step 2: Run RED**

Run: `PYTHONPATH=w1_act-ljl-act_train pytest -q tests/xwiz_act_server/test_contract.py`

Expected: import fails because `contract.py` is absent.

- [x] **Step 3: Implement minimal contract functions**

Define constants for legacy/training keys and implement:

```python
def decode_observation(request: dict) -> dict[str, np.ndarray]: ...
def assemble_state(states: dict) -> np.ndarray: ...
def decode_bgr_image(data: bytes, target_size: tuple[int, int]) -> np.ndarray: ...
def group_action_chunk(actions: np.ndarray) -> dict[str, np.ndarray]: ...
```

Require exactly `(100,19)` finite actions and return `waistqpos`, `left_armqpos`, `headqpos`, `right_armqpos`, `left_eefgripper`, `right_eefgripper`.

- [x] **Step 4: Run GREEN**

Run: `PYTHONPATH=w1_act-ljl-act_train pytest -q tests/xwiz_act_server/test_contract.py`

Expected: all contract tests pass.

- [x] **Step 5: Commit**

```bash
git add w1_act-ljl-act_train/xwiz_act_server/contract.py tests/xwiz_act_server/test_contract.py
git commit -m "feat: adapt W1 observations and ACT actions"
```

### Task 3: Request state machine

**Files:**
- Create: `w1_act-ljl-act_train/xwiz_act_server/server.py`
- Test: `tests/xwiz_act_server/test_server.py`

- [x] **Step 1: Write failing lifecycle tests**

Use `FakeRuntime.predict()` returning a finite `(100,19)` array. Verify:

```python
assert app.handle({"type": "STATUS"})["state"] == "idle"
assert app.handle({"type": "SETUP_CONFIG", "config": {"data_type": "simulation"}})["success"]
assert app.handle({"type": "STATUS"})["state"] == "running"
assert app.handle(observation_request)["status"] == "received"
reply = app.handle({"type": "get_actions"})
assert reply["status"] == "success"
assert reply["actions"]["qpos"]["left_armqpos"].shape == (100, 7)
```

Also assert setup rejects `data_type="real"`, `get_actions` before inference returns `status="pending"`, and `STOP` clears stored actions and returns idle.

- [x] **Step 2: Run RED**

Run: `PYTHONPATH=w1_act-ljl-act_train pytest -q tests/xwiz_act_server/test_server.py`

Expected: import fails because `server.py` is absent.

- [x] **Step 3: Implement minimal state machine and TCP loop**

Implement `XWizActServerApp.handle()` for `SETUP_CONFIG`, `STATUS`, `observation`, `get_actions`, `STOP`, and `SHUTDOWN`. Preserve request IDs in replies. Only infer when `start_infer` is true; store timestamp/timestep with the latest grouped action. Implement a sequential TCP accept loop using `protocol.recv_message/send_message`.

- [x] **Step 4: Run GREEN**

Run: `PYTHONPATH=w1_act-ljl-act_train pytest -q tests/xwiz_act_server/test_server.py`

Expected: all lifecycle tests pass.

- [x] **Step 5: Commit**

```bash
git add w1_act-ljl-act_train/xwiz_act_server/server.py tests/xwiz_act_server/test_server.py
git commit -m "feat: add simulation-only XWiz ACT server"
```

### Task 4: Strict LeRobot ACT runtime

**Files:**
- Create: `w1_act-ljl-act_train/xwiz_act_server/model_runtime.py`
- Create: `w1_act-ljl-act_train/xwiz_act_server/runtime-requirements.txt`
- Create: `w1_act-ljl-act_train/xwiz_act_server/start_local.sh`
- Test: `tests/xwiz_act_server/test_model_runtime.py`

- [x] **Step 1: Write failing runtime boundary tests**

Test pure validation helpers without importing torch: checkpoint must contain `config.json`, `model.safetensors`, `policy_preprocessor.json`, and `policy_postprocessor.json`; `validate_action_chunk()` accepts finite `(100,19)` and rejects other shapes/non-finite values.

- [x] **Step 2: Run RED**

Run: `PYTHONPATH=w1_act-ljl-act_train pytest -q tests/xwiz_act_server/test_model_runtime.py`

Expected: import fails because `model_runtime.py` is absent.

- [x] **Step 3: Implement runtime**

Use lazy imports inside `LeRobotActRuntime.__init__`. Load `TrainPipelineConfig.from_pretrained(..., local_files_only=True)`, set device, then `ACTPolicy.from_pretrained(..., config=policy_config, strict=True)`. Load both processor pipelines, move normalizer to CUDA and unnormalizer to CPU, call `predict_action_chunk`, postprocess `{ACTION: chunk}`, and validate exactly `(100,19)`.

The launcher must export, in this order:

```bash
PYTHONPATH="${RUNTIME_DEPS}:${W1_ROOT}/w1_lerobot/src:${W1_ROOT}"
exec /home/wengyikun/workplace/joint_songling/lerobot/.venv/bin/python \
  -m xwiz_act_server.server --host 0.0.0.0 --port 8889 \
  --policy-path /home/wengyikun/workplace/popcorn/act_popcorn_45w --device cuda
```

- [x] **Step 4: Run GREEN unit tests**

Run: `PYTHONPATH=w1_act-ljl-act_train pytest -q tests/xwiz_act_server`

Expected: all pure tests pass without CUDA model load.

- [x] **Step 5: Install isolated small dependencies and run real checkpoint smoke test**

Run:

```bash
/home/wengyikun/workplace/joint_songling/lerobot/.venv/bin/pip install --no-deps \
  --target /home/wengyikun/.local/share/popcorn-xwiz-act/runtime-deps \
  -r w1_act-ljl-act_train/xwiz_act_server/runtime-requirements.txt
PYTHONPATH=/home/wengyikun/.local/share/popcorn-xwiz-act/runtime-deps:$PWD/w1_act-ljl-act_train/w1_lerobot/src:$PWD/w1_act-ljl-act_train \
  /home/wengyikun/workplace/joint_songling/lerobot/.venv/bin/python \
  -m xwiz_act_server.model_runtime --smoke-test \
  --policy-path /home/wengyikun/workplace/popcorn/act_popcorn_45w --device cuda
```

Expected: strict checkpoint load succeeds and one synthetic three-camera/19D inference returns finite `(100,19)`.

- [x] **Step 6: Commit**

```bash
git add w1_act-ljl-act_train/xwiz_act_server tests/xwiz_act_server/test_model_runtime.py
git commit -m "feat: load W1 LeRobot ACT checkpoint"
```

### Task 5: Local server and PC2 integration

**Files:**
- Modify: `/home/wengyikun/.dexforce/XWiz/model_deployments/tasks/1/task_config.json`
- Deploy: `/home/dexforce/w1/w1_act/xwiz_safe_runtime/` on PC2

- [x] **Step 1: Back up and patch XWiz task config**

Create a timestamped backup beside `task_config.json`, then set:

```json
{
  "server_host": "192.168.20.164",
  "server_port": 8889,
  "action_horizon": 100,
  "chunk_size_threshold": 0.0,
  "home_position": "",
  "hand_target_size": [640, 360],
  "mode": 1
}
```

Set `server_config.data_type` to `simulation` and `save_input` to false.

- [x] **Step 2: Start server and verify model-ready status**

Run `start_local.sh` under `nohup`, write `/home/wengyikun/xwiz_act_server.log`, and wait conditionally until `ss -ltnp` shows `0.0.0.0:8889`. Fail if the model load log contains an exception.

- [x] **Step 3: Re-deploy and verify PC2 safety client**

Copy the existing safe wrapper, restart only `safe_client_service.py`, keep the black wrist publisher, and verify `mode=1`, empty home, 8890 listener, and PC1 manager connection. Do not start Auto/Tele/ACT services.

- [x] **Step 4: Protocol integration without XWiz**

Send PC2 manager a simulation `SETUP_CONFIG`, observe server `STATUS=running`, allow one observation to produce a finite `100×19` action, and immediately issue `STOP`. Verify only `/mj_sim/control/*` receives the client publisher and real control topics gain no new client publisher.

### Task 6: XWiz click-path acceptance

**Files:**
- Update: `/home/wengyikun/workplace/popcorn/howtotrain/XWiz_本机使用说明.md`

- [x] **Step 1: Record safety baseline**

Record current owners/publishers for `/control/joint_position`, `/control/ee/left`, `/control/ee/right`, PC1 mode services, and PC2 inference processes. Confirm PC2 safe wrapper and local server are the intended owners.

- [x] **Step 2: Invoke the same ROS service payload used by XWiz simulation button**

Call `/inference/start_inference` with the selected model/task in simulation mode. This validates the button backend without GUI automation. Watch PC1 manager, PC2 client, and local server logs until one inference succeeds.

- [x] **Step 3: Verify success and stop**

Require all of:

- XWiz/ROS response reports success rather than reset timeout.
- Server logs strict checkpoint loaded and finite `(100,19)` inference.
- PC2 logs three camera inputs, action response, and simulation publisher use.
- `/mj_sim/control/*` has the inference publisher.
- Real control topics have no new inference-client publisher and robot feedback shows no commanded movement attributable to this test.

Then call `/inference/stop_inference` and verify client state returns idle.

- [x] **Step 4: Document operation and migration boundary**

Document start/stop commands, logs, task fields, black wrist behavior, and the future PC2 migration checklist in Chinese.

- [x] **Step 5: Final verification and commit**

Run:

```bash
PYTHONPATH=w1_act-ljl-act_train pytest -q tests/xwiz_act_server
git diff --check
```

Commit only repository files from this feature:

```bash
git add w1_act-ljl-act_train/xwiz_act_server tests/xwiz_act_server \
  howtotrain/XWiz_本机使用说明.md docs/superpowers/plans/2026-08-21-xwiz-act-simulation-inference.md
git commit -m "docs: describe XWiz ACT simulation inference"
```
