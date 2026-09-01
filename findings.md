# Findings

Record only verified facts for the 160000 ACT-DINOv3 dry-run.

- Old 500000 PC2 service is stopped; PC1 remains idle and no real inference is active.
- Checkpoint is ACT-DINOv3, 16x19, with relative arm indices 1..7 and 10..16; waist, neck and grippers indices 0,8,9,17,18 remain absolute.
- Preprocessor performs online arm state delta and q01/q99 normalization; postprocessor unnormalizes then reconstructs absolute arm targets from the chunk anchor.
- Checkpoint config contains EE/FK training-only metadata not declared by the current ACTDINOv3Config; minimal compatibility fields were added.
- Exact q01/q99 stats were found on the training server. Manifest format is v4, horizon 16, source is the 0827 next-state dataset, and all absolute/gripper indices match the checkpoint.
- Strict checkpoint loading requires source commit `c8c674b` and Transformers `5.12.1`; current repository HEAD and PC2's original Transformers are not checkpoint-compatible.
- The exact deployment copy is `outputs/160000_pc2`, with the matching stats bundled under `relative_stats` and PC2 path references written into its config.
- The prior rsync session completed all three transfers: exact source, pinned runtime dependencies, and the 160000 deployment checkpoint/stats.
- PC2 is Jetson aarch64/L4T R36.4.3 with CUDA-enabled Torch 2.8 on Python 3.10; checkpoint source declares Python >=3.12, but all files compile under 3.10 and its only unavailable `typing` names are `Self` and `Unpack`, both available from `typing_extensions`.
- The initial runtime bundle contained only Transformers 5.12.1. PC2 needed isolated overlays for Hub 1.20.1, Tokenizers 0.22.2, Gymnasium 1.3.0, and Termcolor 3.2.0. System packages were not overwritten.
- PC2 exact source tree checksum over non-pycache `src` files matches local: `65a65e94d6a29a4b81051635cf83706cd4513d3d80935d103d55d43197f7b123`.
- PC2 strict GPU load passed with 343,814,885 parameters, `chunk_size=16`, and `n_action_steps=16`.
- Fresh read-only capture timestamp `1788165140950895092` was Idle. The 19D input grippers reconstructed from Linker feedback were left `98.9855`, right `94.1234`.
- Dry-run image binding and conversion were: head `/camera/right_eye_resize` 960x540 -> centered black-pad 960x960 -> 224x224; model left wrist <- physical left `/camera_r` 640x360 -> 360x360 -> 224x224; model right wrist <- physical right `/camera_l` 640x480 -> 480x480 -> 224x224.
- Full preprocessor/model/postprocessor dry-run passed: first online relative arm state is zero, absolute indices 0/8/9/17/18 stay absolute, normalized state matches clipped q01/q99, raw and postprocessed outputs are finite 16x19, and inverse q01/q99 plus arm-anchor reconstruction matches numerically.
- The produced chunk had zero body-limit violations before runtime clipping. Raw grippers were left `100.02..101.03`, right `96.37..99.20`; runtime clamps left to 100 and `<95 -> 0` does not close either hand for this chunk.
- Final artifacts are on PC2 at `/home/dexforce/workspace/dryruns/160000_current/output`; model service remains inactive and port 8889 has no listener.
