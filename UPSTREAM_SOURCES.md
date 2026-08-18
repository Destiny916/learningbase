# Imported source revisions

The algorithm directories are ordinary files tracked by the Popcorn repository. They are not Git submodules, so local changes can be committed directly to `chengdu/popcorn.git`.

| Directory | Upstream URL | Upstream branch | Imported commit |
| --- | --- | --- | --- |
| `algorithms/act` | `http://192.168.10.28:3000/chengdu/lerobot_joint.git` | `main` | `8296f39e0db0d3f6a2da4eaf68d51da0505fe4de` |
| `algorithms/act_dinov3` | `http://192.168.10.28:3000/chengdu/lerobot_joint.git` | `feat/act-dinov3` | `c26cd5eec99d5978fd9ee374e5950e182586843a` |
| `algorithms/turbovla` | `http://192.168.10.28:3000/chengdu/turboVLA.git` | `main` | `7a9b9c948d0de6298133ad5f7df58b6df0b7339f` |
| `algorithms/turbovla_patchvision` | `http://192.168.10.28:3000/chengdu/turboVLA.git` | `feature/turbovla-patchvision-t2-act` | `4d7a5935b40ffdb26e1a721f892cd83b215ed43a` |

Each tree was exported from its exact commit using `git archive`, which includes all files tracked by that upstream commit and excludes upstream `.git` metadata.

`w1_act-ljl-act_train` is a source snapshot from `/home/wengyikun/workplace/popcorn/w1_act-ljl-act_train` on 2026-08-18. Source files and URDF/visual assets are included. Checkpoints, logs, caches, build outputs, and model weights are excluded. Its historical `.gitmodules` refers to `dexe_interfaces`, but that directory was not present in the local source and is therefore not represented as vendored code.
