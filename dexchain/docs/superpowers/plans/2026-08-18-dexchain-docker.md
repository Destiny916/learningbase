# DexChain Docker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a repeatable package-based Docker environment for DexEChain and EmbodiChain with working single-GPU CDI access.

**Architecture:** The documented private DexSDK image is the base. A thin derived image installs pinned packages from internal PyPI, while Compose and shell wrappers supply host networking, X11/Vulkan devices, project mounts, and the verified CDI GPU device.

**Tech Stack:** Docker 29, Docker Compose v2, NVIDIA Container Toolkit/CDI, Bash, Python 3.10, internal PyPI.

---

### Task 1: Configuration contract test

**Files:**
- Create: `tests/test_docker_config.sh`

- [x] Write a shell test that fails until `Dockerfile`, `compose.yaml`, `.env.example`, and scripts contain the approved image, pinned package versions, CDI device, X11 mounts, and import checks.
- [x] Run `bash tests/test_docker_config.sh` and confirm it fails because the configuration files do not exist.

### Task 2: Image and Compose configuration

**Files:**
- Create: `Dockerfile`
- Create: `compose.yaml`
- Create: `.env.example`
- Create: `.dockerignore`

- [x] Add a thin Dockerfile based on `${BASE_IMAGE}` and install `embodichain==0.2.4` plus `dexechain==0.1.6` from both internal indexes.
- [x] Add a Compose service named `dexchain` with host network/IPC, CDI `nvidia.com/gpu=0`, `/dev/dri`, X11, Vulkan/NVIDIA read-only mounts, and the project mounted at `/workspace/dexchain`.
- [x] Add environment defaults for image name, container name, CDI device, package versions, display, and PyPI endpoints.
- [x] Re-run `bash tests/test_docker_config.sh`; Docker configuration assertions should pass and the test should next fail on the not-yet-created operational scripts.

### Task 3: Operational scripts

**Files:**
- Create: `scripts/pull_image.sh`
- Create: `scripts/start.sh`
- Create: `scripts/verify.sh`

- [x] Add strict Bash wrappers that validate prerequisites and use `.env` when present.
- [x] Make `pull_image.sh` pull the selected base image and report its immutable image ID.
- [x] Make `start.sh` validate the CDI GPU with a short container probe before `docker compose up -d --build`.
- [x] Make `verify.sh` run container-side GPU, Python, and package import/version checks.
- [x] Run `bash -n scripts/*.sh`; the static contract should next fail only on the not-yet-created `README.md`.

### Task 4: Host prerequisites and image pull

**Files:**
- Modify: `/etc/docker/daemon.json` through root systemd units only

- [x] Install Docker Compose v2 if `docker compose version` is unavailable.
- [x] Keep `192.168.3.13:5000` in insecure registries and NVIDIA runtime configuration.
- [x] Set Docker `max-concurrent-downloads` to `1` to avoid the three concurrent large-layer stalls observed against this registry.
- [x] Validate daemon JSON, restart Docker, and re-check insecure registry, Compose, and CDI GPU access.
- [x] Run `scripts/pull_image.sh`; the approved base image was pulled after correcting the `Meta`/`tunx` policy route.

### Task 5: Documentation and end-to-end verification

**Files:**
- Create: `README.md`

- [x] Document quick start, `.env` overrides, source-mode limitation, CDI rationale, X11 access, and troubleshooting.
- [x] Run `docker compose config`.
- [x] Run `docker compose build`, `scripts/start.sh`, and `scripts/verify.sh`.
- [x] Record final commands, route correction, image IDs, verification result, and disk risk in the task planning files.
