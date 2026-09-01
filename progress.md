# Progress

- Started audit; no new model or real-robot motion started yet.
- Stopped PC2 500000 model service; PC1 status verified idle.
- First load failure: undeclared EE/FK config fields. Added inference-compatible dataclass fields.
- Second load failure: checkpoint referenced training-server q01/q99 paths absent locally. Downloaded the exact stats package to `/home/wengyikun/act_stats/...` after `/data/wengyikun` was found non-writable.
- Verified locally that exact source commit `c8c674b` with Transformers `5.12.1` strictly loads all 343,814,885 checkpoint parameters.
- Completed transfer to PC2: `/home/dexforce/workspace/act_dinov3_c8c674b`, `/home/dexforce/workspace/act_dinov3_runtime_deps`, and `/home/dexforce/workspace/outputs/160000_pc2`.
- Verified all 11 model/config/stats SHA256 values match between local and PC2; PC2 service is inactive, port 8889 is closed, and no GPU compute process is active.
- Reproduced and fixed isolated dependency import failures without changing system packages: Transformers now imports as 5.12.1 with Hub 1.20.1 and Tokenizers 0.22.2; added missing Gymnasium and Termcolor to the same overlay.
- Python compatibility investigation found source requires >=3.12 while PC2 CUDA Torch uses Python 3.10. Full source compiles under 3.10; dry-run uses process-local `typing_extensions` shims for only `Self` and `Unpack`, without editing exact source.
- Completed the isolated PC2 runtime dependency layer, including the dataset package required by LeRobot's eager imports. Removed only stale NumPy 2.2.6 overlay remnants after proving they confused Accelerate; final NumPy is 1.26.4, matching Jetson Torch's ABI.
- Added `scripts/dryrun_act_dinov3_160000_pc2.py`, copied a fresh read-only robot/camera snapshot to PC2, and ran strict end-to-end GPU inference twice. The second run exited 0 without warnings or tracebacks.
- Final verification: preprocessor q01/q99 match true; relative-to-absolute postprocessor match true; raw and final shape 16x19 finite; body-limit violations 0; PC1 state idle; PC2 service inactive; 8889 closed; no GPU compute process remains.
