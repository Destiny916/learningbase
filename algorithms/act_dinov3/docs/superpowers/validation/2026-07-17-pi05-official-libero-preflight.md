# PI05 Official LIBERO Preflight

Date: 2026-07-17

## Runtime Contract

- Remote host: `183.230.224.121:50210`
- Runtime user: `uid=1009(wengyikun) gid=1008(wengyikun)`
- Image: `lerobot-pi05-libero-smoke:20260717`
- Image ID: `sha256:a3987c43e745e7d1a5356921733e2325a2c4a4c99d04114568e517e21ae23d7b`
- Selected host GPU: `3` (the container exposed it as its only `cuda:0` device)
- Persistent runtime home: `/data/wengyikun/pi05_official_libero_smoke/home`
- Remaining `/data` space at preflight: `401G`

## Image Setup

The base `lerobot-pi05-train:20260706` image did not contain LIBERO. The derived
image installs `hf-libero==0.1.4` with `--no-build-isolation` and
`CMAKE_POLICY_VERSION_MINIMUM=3.5`. This is required because the package's EGL
extension uses an old CMake minimum policy which CMake 4 otherwise rejects.

Direct Hugging Face access on this host was unavailable. The image therefore
uses `HF_ENDPOINT=https://hf-mirror.com` and pre-downloads
`lerobot/libero-assets` serially (`max_workers=1`) to avoid mirror `429`
responses. The complete asset download contains 586 files and is configured at
runtime as `/opt/libero-assets` through `$HOME/.libero/config.yaml`.

## Probe

Command run as the remote user:

```bash
cd /home/wengyikun/lerobot
LEROBOT_GPUS=device=3 bash run_scripts/remote_pi05_libero_smoke_container.sh \
  bash -lc 'python -c "...CUDA, LIBERO, asset configuration, and PI05 imports..."'
```

Observed result:

```text
CUDA_OK NVIDIA GeForce RTX 4090 D
LIBERO_IMPORT_OK /opt/conda/lib/python3.11/site-packages/libero/__init__.py
LIBERO_CONFIG_OK /opt/libero-assets
PI05_IMPORT_OK PI05Policy
```

The probe also asserted `torch.cuda.device_count() == 1`, proving the container
was isolated to the selected single GPU.
