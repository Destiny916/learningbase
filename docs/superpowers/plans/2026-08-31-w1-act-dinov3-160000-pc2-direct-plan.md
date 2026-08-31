# W1 ACT-DINOv3 160000 PC2 Direct Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an isolated PC1/PC2 direct runtime for checkpoint `160000` using adjacent real feedback, q01/q99 normalization, 16-frame absolute action chunks, and no interpolation.

**Architecture:** Add a protocol-v2 payload carrying consecutive feedback snapshots. A dedicated 160000 adapter on PC2 computes relative arm state, invokes the existing ACT-DINOv3 policy, inversely normalizes actions, and reconstructs absolute targets anchored to current feedback. Dedicated launchers/configs select only `160000_pc2`; existing 500000 paths remain unchanged.

**Tech Stack:** Python 3, NumPy, PyTorch/LeRobot ACT-DINOv3, pickle length-prefixed TCP protocol, pytest, shell/systemd deployment files.

---

### Task 1: Protocol-v2 adjacent feedback contract

**Files:**
- Create: `w1_act-ljl-act_train/xwiz_act_server/relative_contract_160000.py`
- Test: `tests/xwiz_act_server/test_relative_contract_160000.py`

- [ ] Write tests for 19D field order, consecutive sequence/timestamp validation, and relative indices `1..7,10..16` only.
- [ ] Run `pytest tests/xwiz_act_server/test_relative_contract_160000.py -q` and confirm failure because the module is absent.
- [ ] Implement pure validation/conversion helpers with finite-value checks and no model or network dependencies.
- [ ] Run the focused tests and confirm pass.
- [ ] Commit `test and implement 160000 adjacent feedback contract`.

### Task 2: 160000 model adapter

**Files:**
- Create: `w1_act-ljl-act_train/xwiz_act_server/act_dinov3_160000_adapter.py`
- Test: `tests/xwiz_act_server/test_act_dinov3_160000_adapter.py`

- [ ] Write tests using synthetic q01/q99 and a fake normalized policy output to prove state normalization, action inverse normalization, and current-state anchoring.
- [ ] Run the focused tests and confirm failure.
- [ ] Implement adapter functions that load only the 160000 stats files, call policy pre/post processors once, and return finite `(16,19)` absolute actions.
- [ ] Add gripper `<95 -> 0` hardware adapter behavior at the output boundary without changing quantiles.
- [ ] Run focused tests and confirm pass.
- [ ] Commit adapter changes.

### Task 3: Server protocol-v2 integration

**Files:**
- Create: `w1_act-ljl-act_train/xwiz_act_server/server_160000.py`
- Test: `tests/xwiz_act_server/test_server_160000.py`

- [ ] Write tests proving v2 rejects missing previous feedback, non-consecutive sequence/timestamps, malformed images/states, and does not invoke policy on rejection.
- [ ] Run focused tests and confirm failure.
- [ ] Implement isolated server app using the existing framed protocol and adapter; old server entrypoint is untouched.
- [ ] Ensure each accepted observation produces exactly one 16x19 action chunk and each chunk is consumed once.
- [ ] Run server-focused tests and all existing `tests/xwiz_act_server` tests.
- [ ] Commit server integration.

### Task 4: PC1 synchronous 16-frame client adapter

**Files:**
- Create: `w1_act-ljl-act_train/xwiz_real_runtime/client_runtime_160000.py`
- Create: `w1_act-ljl-act_train/direct_runtime/client_runtime_160000.json`
- Test: `tests/xwiz_real_runtime/test_client_runtime_160000.py`

- [ ] Write tests for feedback snapshot pairing, exactly-16-frame chunk consumption, no interpolation, and next-request gating on fresh feedback.
- [ ] Run focused tests and confirm failure.
- [ ] Implement the client-side snapshot cache and isolated runtime hooks; commands sent to PC1 remain absolute 19D actions returned by PC2.
- [ ] Configure 30 Hz, horizon 16, `sample_factor=1`, threshold 0, and explicit camera topics/conversion metadata.
- [ ] Run focused tests and existing runtime tests.
- [ ] Commit client changes.

### Task 5: Dedicated launchers and deployment packaging

**Files:**
- Create: `w1_act-ljl-act_train/direct_runtime/start_pc2_act_dinov3_160000.sh`
- Create: `w1_act-ljl-act_train/direct_runtime/start_pc1_act_dinov3_160000.sh`
- Create: `w1_act-ljl-act_train/direct_runtime/w1-act-server-dinov3-160000.service`
- Create: `w1_act-ljl-act_train/direct_runtime/w1-act-client-dinov3-160000.service`
- Modify: deployment manifest/docs only if required by launcher paths.

- [ ] Write shell/config validation tests that assert explicit `160000_pc2`, protocol v2, port 8889/8890, and no fallback to 500000.
- [ ] Run validation tests and confirm failure.
- [ ] Implement non-starting launch scripts with strict environment/config checks.
- [ ] Run shellcheck where available and validation tests.
- [ ] Commit launch packaging.

### Task 6: Offline and network dry-run deployment

**Files:**
- Modify: `scripts/dryrun_act_dinov3_160000_pc2.py` only if adapter contract needs coverage.
- Create: `scripts/dryrun_protocol_v2_160000.py`
- Test: `tests/test_dryrun_protocol_v2_160000.py`

- [ ] Add no-motion dry-run tests using two synthetic real feedback frames and one 16x19 action chunk.
- [ ] Run strict checkpoint load and offline dry-run with `/home/wengyikun/workplace/popcorn/outputs/160000_pc2`.
- [ ] Verify image contract, q01/q99 paths, relative/absolute reconstruction, body limits, and gripper threshold.
- [ ] Push only new 160000 files/configs to PC1 and PC2, preserving existing services.
- [ ] Verify new units inactive, ports closed, no ACT action publisher, and report exact evidence.
- [ ] Commit final verification artifacts if any.
