# PI05 Official LIBERO Finetune And Rollout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run and record a reproducible, official LeRobot PI05 fine-tuning smoke test on LIBERO simulation data, then evaluate the resulting checkpoint with a closed-loop LIBERO rollout.

**Architecture:** Use the official `HuggingFaceVLA/libero` LeRobot dataset and the PI05 LIBERO checkpoint contract documented in `docs/source/libero.mdx`. A committed launcher preserves the exact train/eval arguments while all weights, dataset caches, logs, videos, and metrics remain under `/data/wengyikun/pi05_official_libero_smoke/` on the remote host. The rollout loads the checkpoint saved by the smoke train and runs one selected official LIBERO task in MuJoCo.

**Tech Stack:** LeRobot PI05, Hugging Face Hub dataset/model cache, LIBERO, MuJoCo EGL, PyTorch CUDA, Docker.

---

## Acceptance Criteria

- The remote Docker image runs as the remote user's UID/GID, uses exactly one idle GPU, and reports CUDA availability.
- The official LIBERO dataset metadata and PI05-LIBERO checkpoint are loadable.
- The fine-tune command completes at least one optimizer step and writes a reloadable checkpoint under `/data/wengyikun/pi05_official_libero_smoke/train_out/`.
- `lerobot-eval` loads that checkpoint, resets a LIBERO environment, executes a policy rollout, and writes JSON metrics under `/data/wengyikun/pi05_official_libero_smoke/eval_out/`.
- The final report distinguishes offline train loss from closed-loop rollout success/reward. It does not claim a benchmark success rate from this short smoke run.

## Fixed Experiment Contract

- Dataset: `HuggingFaceVLA/libero`, with `dataset.episodes=[0]` to bound the offline smoke run. LIBERO's dataset CLI does not expose suite/task filtering, so this sample is not asserted to belong to rollout task `libero_spatial` id `0`.
- Policy source: `lerobot/pi05_libero` when it is available in the remote cache or Hugging Face download succeeds. This is the official checkpoint family used by the repository's LIBERO PI05 reproduction.
- Action semantics: LIBERO's native relative 7D action, not the project's Piper 7D joint or pose10D processor modes.
- Training: one GPU, BF16, language model frozen, vision encoder/projector/action expert trainable, `chunk_size=50`, `n_action_steps=10`, no Hub upload.
- Rollout: `env.type=libero`, `env.task=libero_spatial`, `env.task_ids=[0]`, relative control, one episode, one vectorized environment, MuJoCo EGL.
- Remote execution: Docker `--user 1009:1008`, persistent mounts for `/data/wengyikun` and the checked-out repository, no changes to ongoing real-data jobs.

### Task 1: Add Reproducible Official LIBERO Launchers

**Files:**
- Create: `run_scripts/launch_pi05_official_libero_smoke.sh`
- Create: `run_scripts/eval_pi05_official_libero_smoke.sh`
- Create: `tests/training/test_pi05_official_libero_launch_scripts.py`

- [x] **Step 1: Write failing launcher-contract tests**

Assert that the train launcher uses `HuggingFaceVLA/libero`, `policy.type=pi05`, `policy.pretrained_path=lerobot/pi05_libero`, `env.type=libero`, `env.task=libero_spatial`, `env.task_ids=[0]`, BF16, `chunk_size=50`, `n_action_steps=10`, and a `/data/wengyikun/pi05_official_libero_smoke/train_out` output root. Assert that the eval launcher targets the saved `pretrained_model`, the same LIBERO task, `env.control_mode=relative`, and one rollout episode.

- [x] **Step 2: Run the test and confirm failure**

Run: `PYTHONPATH=src .venv/bin/pytest tests/training/test_pi05_official_libero_launch_scripts.py -q`

Expected: failure because the two launchers do not yet exist.

- [x] **Step 3: Implement the two launchers**

The train launcher must reject an existing `OUTPUT_DIR`, default to `/data/wengyikun/pi05_official_libero_smoke/train_out`, set `MUJOCO_GL=egl`, and invoke `python -m lerobot.scripts.lerobot_train` with `dataset.episodes=[0]`. The eval launcher must resolve `CHECKPOINT_STEP` (default `000001`) to `checkpoints/${CHECKPOINT_STEP}/pretrained_model` and invoke `python -m lerobot.scripts.lerobot_eval` with the fixed rollout contract.

- [x] **Step 4: Run the targeted launcher tests**

Run: `PYTHONPATH=src .venv/bin/pytest tests/training/test_pi05_official_libero_launch_scripts.py -q`

Expected: all tests pass.

- [x] **Step 5: Commit and push Task 1**

Run: `git add run_scripts/launch_pi05_official_libero_smoke.sh run_scripts/eval_pi05_official_libero_smoke.sh tests/training/test_pi05_official_libero_launch_scripts.py && git commit -m "test: add official PI05 LIBERO smoke launchers" && git push chengdu main`

### Task 2: Verify Remote PI05/LIBERO Prerequisites

**Files:**
- Create: `docker/Dockerfile.pi05_libero_smoke`
- Create: `docker/entrypoints/pi05_libero_smoke_entrypoint.sh`
- Create: `run_scripts/remote_pi05_libero_smoke_container.sh`
- Create: `tests/training/test_pi05_official_libero_container.py`
- Create: `docs/superpowers/validation/2026-07-17-pi05-official-libero-preflight.md`

- [x] **Step 1: Inspect the remote host without changing any training job**

Collect hostname, Docker image inventory, Docker user mapping, GPU memory/utilization, presence of `lerobot-pi05-train:20260706`, Python imports for `torch`, `lerobot`, `libero`, and the local Hugging Face cache status for the LIBERO dataset and PI05-LIBERO checkpoint. The initial probe found CUDA and PI05 available, but `libero` missing from `lerobot-pi05-train:20260706`.

- [x] **Step 2: Diagnose and test the missing LIBERO dependency**

Use the official `docker/Dockerfile.benchmark.libero` as the reference. The base image's `hf-libero` install initially failed because `hf-egl-probe` first used an isolated environment without CMake, and then CMake 4 rejected the package's old minimum policy. Verify `CMAKE_POLICY_VERSION_MINIMUM=3.5` with `--no-build-isolation` builds and imports `egl_probe==1.0.2`.

- [x] **Step 3: Add a reproducible PI05-LIBERO image contract**

Write failing static tests, then add a derived image based on `lerobot-pi05-train:20260706`, pin `hf-libero==0.1.4`, set `CMAKE_POLICY_VERSION_MINIMUM=3.5`, use `--no-build-isolation`, pre-download `lerobot/libero-assets`, and create a noninteractive `~/.libero/config.yaml` at runtime. Add a single-GPU current-user Docker launcher and run the static tests plus Bash syntax checks.

- [x] **Step 4: Build and probe the derived image**

Build `lerobot-pi05-libero-smoke:20260717` in a separate `/data/wengyikun/pi05_official_libero_smoke/build_context` and then run it on the least-busy GPU with `--user 1009:1008`. Require output containing `CUDA_OK`, `LIBERO_IMPORT_OK`, `PI05_IMPORT_OK`, and `LIBERO_CONFIG_OK`.

- [x] **Step 5: Record the preflight evidence**

Write the chosen Docker image, image id, selected host GPU, imports, cache locations, root-cause diagnosis, and exact probe command into the validation document. Do not add generated model or dataset files to Git.

- [x] **Step 6: Commit and push Task 2 evidence**

Run: `git add docs/superpowers/validation/2026-07-17-pi05-official-libero-preflight.md && git commit -m "docs: record PI05 LIBERO smoke preflight" && git push chengdu main`

### Task 3: Run The Official PI05 LIBERO Fine-Tune Smoke

**Files:**
- Create: `docs/superpowers/validation/2026-07-17-pi05-official-libero-finetune.md`

- [ ] **Step 1: Start the bounded remote training job**

Run `run_scripts/launch_pi05_official_libero_smoke.sh` in the preflight-verified image, as UID/GID `1009:1008`, on the selected idle GPU. Use `STEPS=1`, `SAVE_FREQ=1`, `ENV_EVAL_FREQ=0`, `EVAL_STEPS=0`, and `WANDB_ENABLE=false` so the task proves one fine-tuning optimizer update and checkpoint save without competing with active jobs.

- [ ] **Step 2: Verify the saved checkpoint can be loaded**

Confirm the expected `checkpoints/000001/pretrained_model/` files include `config.json`, `model.safetensors` or model shard/index files, and a processor configuration. Run a no-grad Python load probe using `PreTrainedPolicy.from_pretrained` or the repository policy factory.

- [ ] **Step 3: Record training evidence**

Record the output directory, selected GPU, exact command, effective batch size, parameter-freeze audit, first train loss, checkpoint path, and loading result. Explicitly note any official checkpoint/dataset download that occurred.

- [ ] **Step 4: Commit and push Task 3**

Run: `git add docs/superpowers/validation/2026-07-17-pi05-official-libero-finetune.md && git commit -m "docs: record PI05 LIBERO finetune smoke" && git push chengdu main`

### Task 4: Run Closed-Loop LIBERO Rollout And Report

**Files:**
- Create: `docs/superpowers/validation/2026-07-17-pi05-official-libero-rollout.md`

- [ ] **Step 1: Execute one closed-loop rollout**

Run `run_scripts/eval_pi05_official_libero_smoke.sh` in the same image and single-GPU isolation, pointing at Task 3's `000001` checkpoint. Preserve the evaluator JSON and any rollout video under `/data/wengyikun/pi05_official_libero_smoke/eval_out/`.

- [ ] **Step 2: Verify rollout completion**

Require evaluator output with at least one completed episode and inspect the saved JSON for reward, success, and episode count. A low success value is valid for a one-step smoke fine-tune; a loader, environment, shape, action, or MuJoCo failure is not.

- [ ] **Step 3: Record the distinction between metrics**

Document that train loss validates offline flow-matching optimization, while the evaluator's reward/success validates closed-loop behavior. Include the exact values without comparing this smoke result to the production 7D/pose10D Piper experiments.

- [ ] **Step 4: Commit and push Task 4**

Run: `git add docs/superpowers/validation/2026-07-17-pi05-official-libero-rollout.md && git commit -m "docs: record PI05 LIBERO rollout smoke" && git push chengdu main`

## Out Of Scope

- Long 6,000-step / 8-H100 reproduction or benchmark-level success-rate claims.
- Converting LIBERO action/state data into Piper joints or pose10D.
- Altering the active 7D/10D real-data PI05 or ACT training jobs.
- Uploading models, datasets, checkpoints, or rollout videos to Hugging Face Hub.
