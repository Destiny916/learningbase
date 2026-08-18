# PI05 Absolute Pose Training Design

## Goal

Add isolated PI05 training configurations for the existing 0714 joint7d and
endpose10d split datasets. The new paths train absolute state and absolute
future action chunks without changing the verified relative-pose paths.

## Data Contract

### Joint7d

- Input state is the raw absolute `q_t` with six arm joints and one absolute
  gripper value.
- Target chunk is the raw absolute future targets
  `[q_(t+1), ..., q_(t+50)]`.
- State and action receive separate per-dimension training-split `q01/q99`
  statistics. Action statistics flatten every valid future offset from 1 to
  the configured horizon; they are not copied from state statistics.
- Physical validation compares de-normalized predicted and target absolute
  chunks directly. It records loss, gripper loss, total action MSE, gripper
  MSE/RMSE, and per-arm-joint MSE/RMSE in radian and degree units.

### Endpose10d

- Input state is the raw absolute `xyz + rot6d + gripper` pose.
- Target chunk is the raw absolute future pose target for every horizon step.
- State and action receive separate training-split `q01/q99` values. Only
  `xyz` and the absolute gripper are scaled; `rot6d` remains unscaled.
- Physical validation de-normalizes both chunks and compares them directly;
  it does not compose either target with the current pose.

## Isolation

The implementation introduces opt-in absolute-stat paths and two new launch
scripts. Existing relative processor branches, relative statistics, relative
launch scripts, and active output directories remain unchanged. New outputs
use names containing `absolute` and no existing checkpoint is resumed.

## Validation

Tests cover exact state/action quantile construction, processor selection and
round trips, and the direct absolute-action MSE path. Launch-script tests
verify both datasets, stats files, frozen language model, trainable visual
encoder/projector, chunk size, periodic offline validation, and distinct
output directories.
