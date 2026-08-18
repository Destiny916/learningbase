# Popcorn training algorithms

This repository is the initial, versioned workspace for future training on the Popcorn W1 dataset. The current phase only synchronizes source code, records the data/model contract, and provides offline checks. It does **not** start training, download model weights, build Docker images, or control a robot.

## Checkout

```bash
git clone --recurse-submodules http://192.168.10.28:3000/chengdu/popcorn.git
cd popcorn
bash scripts/verify_stack.sh
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

## Included sources

| Path | Source branch | Purpose |
| --- | --- | --- |
| `algorithms/act` | `chengdu/lerobot_joint.git`, `main` | Standard LeRobot ACT source |
| `algorithms/act_dinov3` | `chengdu/lerobot_joint.git`, `feat/act-dinov3` | ACT with the shared DINOv3 visual backbone |
| `algorithms/turbovla` | `chengdu/turboVLA.git`, `main` | TurboVLA source and Joint Songling recipes |
| `algorithms/turbovla_patchvision` | `chengdu/turboVLA.git`, `feature/turbovla-patchvision-t2-act` | Historical PatchVision T2 implementation |
| `dexchain` | local audited snapshot | Docker environment helpers; no credentials |
| `howtotrain` | local audited snapshot | Existing training and deployment notes |

The submodule commit recorded by this repository is authoritative. The branch names in `.gitmodules` describe the intended update source; running `git submodule update --remote` is an explicit maintenance operation, not part of verification.

## Popcorn contract

The observed checkpoint metadata is copied to `checkpoint_metadata/act_popcorn_45w` without model or optimizer weights. See `docs/POPCORN_19D_CONTRACT.md` before adapting any algorithm. Joint Songling's existing 20D-state/14D-relative-action recipes are reference material and are not directly compatible with Popcorn's 19D absolute contract.

## DexChain

The initial phase performs static validation only:

```bash
bash dexchain/tests/test_docker_config.sh
docker compose --env-file dexchain/.env.example -f dexchain/compose.yaml config --quiet
```

These commands do not start or rebuild the existing DexChain container.
