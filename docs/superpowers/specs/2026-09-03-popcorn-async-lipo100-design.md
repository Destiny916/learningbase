# Popcorn ACT 100-point asynchronous LIPO adapter

## Scope

Add an isolated runtime path that adapts the `temp` repository's timed
trajectory/LIPO ideas to the verified Popcorn W1 PC2-direct contract. Existing
launchers and runtime configurations remain unchanged.

## Contract

- Policy output is finite `(100, 19)`.
- `sample_factor=2` expands each policy chunk to 200 control points.
- A replan is submitted when 30 control points remain (15 policy points).
- Absolute `control_step` aligns old and new trajectories.
- A new session clears all old trajectory, transition, and pending-result state.
- Delayed results skip expired prefixes; the active old trajectory continues.
- Body dimensions are linearly blended for at most 30 control points.
- Left/right hand scalar dimensions are excluded from blending and use the new
  chunk directly.
- Final body actions are clipped to the W1 joint limits; non-finite actions are
  rejected. Hand conversion uses the current Popcorn EE feedback/endpoints and
  control topics.
- Camera topics and preprocessing remain the existing Popcorn contract.

## Components

The adapter exposes a small pure state machine for chunk expansion, session
reset, result installation, delayed-result handling, and per-step command
selection. The PC1 launcher is a separate entry point and uses its own manager
port/configuration. The PC2 server and model checkpoint are selected by the
launcher configuration, without changing existing services.

## Failure handling

Pending or expired results are discarded with diagnostics. If a trajectory is
temporarily unavailable, the runtime holds the last finite command and never
publishes NaN/Inf or an unclipped body joint. A session change invalidates
results from the previous session.

## Verification

Unit tests cover interpolation shape, first-chunk/no-old-blend behavior,
delayed results, expired results, body-only blending, hand exclusion, session
reset, and body-limit clipping. No real-robot start is part of this change.
