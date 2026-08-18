# Popcorn training algorithms

This repository is the initial, versioned workspace for future training on the Popcorn W1 dataset. Algorithm and W1 ACT source files are committed directly in this repository so later changes can be made and pushed without updating separate submodules. The current phase only synchronizes source code, records the data/model contract, and provides offline checks. It does **not** start training, download model weights, build Docker images, or control a robot.

## Checkout

```bash
git clone http://192.168.10.28:3000/chengdu/popcorn.git
cd popcorn
bash scripts/verify_stack.sh
```

## Included sources

| Path | Imported source | Purpose |
| --- | --- | --- |
| `algorithms/act` | LeRobot `main` | Standard LeRobot ACT source |
| `algorithms/act_dinov3` | LeRobot `feat/act-dinov3` | ACT with the shared DINOv3 visual backbone |
| `algorithms/turbovla` | TurboVLA `main` | TurboVLA source and Joint Songling recipes |
| `algorithms/turbovla_patchvision` | TurboVLA PatchVision branch | PatchVision T2 implementation |
| `w1_act-ljl-act_train` | Local W1 source snapshot | Existing W1 ACT, simulation, inference, and URDF source |
| `dexchain` | local audited snapshot | Docker environment helpers; no credentials |
| `howtotrain` | local audited snapshot | Existing training and deployment notes |

Exact upstream URLs, branches, and imported commits are recorded in `UPSTREAM_SOURCES.md`. The algorithm directories intentionally contain no nested Git metadata.

## Popcorn contract

The observed checkpoint metadata is copied to `checkpoint_metadata/act_popcorn_45w` without model or optimizer weights. See `docs/POPCORN_19D_CONTRACT.md` before adapting any algorithm. Joint Songling's existing 20D-state/14D-relative-action recipes are reference material and are not directly compatible with Popcorn's 19D absolute contract.

## DexChain

The initial phase performs static validation only:

```bash
bash dexchain/tests/test_docker_config.sh
docker compose --env-file dexchain/.env.example -f dexchain/compose.yaml config --quiet
```

These commands do not start or rebuild the existing DexChain container.
