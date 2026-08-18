# Relative 17D State / 14D Action Implementation Plan

**Goal:** Add an isolated relative-joint contract for datasets whose observation state has 17 dimensions while the action chunk has 14 dimensions.

**Architecture:** Keep the existing 7D and 14D same-width behavior unchanged. The new contract stores independent state/action names and quantiles, maps each action name to the matching state name, computes relative state deltas for non-gripper state fields, keeps grippers absolute, and reconstructs absolute actions online from the mapped state anchor. ACT and PI05 use the same processor and separate normalization/output directories.

**Validation:** Unit tests cover independent quantile dimensions, name mapping, offline relative actions, online absolute reconstruction, and configuration loading. Dataset preflight must verify LeRobot v3 metadata, 17D state, 14D action, image/depth keys, finite values, and action/state alignment before GPU6 is used.

---

1. Add failing tests for independent state/action statistics and 17D-to-14D mapping.
2. Implement backward-compatible statistics serialization and relative processor mapping.
3. Wire ACT and PI05 configuration/factory metadata, including state names and optional position noise.
4. Add a dedicated 17D/14D launcher with its own normalization and output roots.
5. Run focused tests and dataset preflight, then inspect GPU6 before starting the isolated job.
