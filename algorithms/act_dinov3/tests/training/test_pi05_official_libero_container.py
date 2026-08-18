from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_pi05_libero_smoke_image_installs_compatible_official_dependencies():
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile.pi05_libero_smoke").read_text()

    assert "FROM lerobot-pi05-train:20260706" in dockerfile
    assert "ENV CMAKE_POLICY_VERSION_MINIMUM=3.5" in dockerfile
    assert "ENV HF_ENDPOINT=https://hf-mirror.com" in dockerfile
    assert '"hf-libero==0.1.4"' in dockerfile
    assert "--no-build-isolation" in dockerfile
    assert "libegl1" in dockerfile
    assert "libopengl0" in dockerfile
    assert "repo_id='lerobot/libero-assets'" in dockerfile
    assert "max_workers=1" in dockerfile
    assert "os.path.join(os.path.dirname(spec.origin), 'libero')" in dockerfile
    assert "pi05_libero_smoke_entrypoint.sh" in dockerfile


def test_pi05_libero_smoke_entrypoint_configures_noninteractive_assets_home():
    entrypoint = (REPO_ROOT / "docker" / "entrypoints" / "pi05_libero_smoke_entrypoint.sh").read_text()

    assert entrypoint.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert 'mkdir -p "$HOME/.libero"' in entrypoint
    assert '"$HOME/.libero/config.yaml"' in entrypoint
    assert "LIBERO_ASSETS_DIR" in entrypoint
    assert "LIBERO_PACKAGE_DIR" in entrypoint
    assert 'exec "$@"' in entrypoint


def test_pi05_libero_smoke_container_launcher_uses_current_user_and_one_gpu():
    launcher = (REPO_ROOT / "run_scripts" / "remote_pi05_libero_smoke_container.sh").read_text()

    assert launcher.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert ': "${LEROBOT_GPUS:?Set LEROBOT_GPUS to device=<gpu-id>' in launcher
    assert '--gpus "${LEROBOT_GPUS}"' in launcher
    assert '--user "$(id -u):$(id -g)"' in launcher
    assert "--gpus all" not in launcher
    assert "lerobot-pi05-libero-smoke:20260717-libero-rootfix-assets" in launcher
    assert "MUJOCO_GL=egl" in launcher
    assert "HF_ENDPOINT=https://hf-mirror.com" in launcher
    assert "HF_HOME=/data/wengyikun/.cache/huggingface" in launcher
    assert 'HF_DATASETS_CACHE="$HF_HOME/datasets"' in launcher
    assert "/data/wengyikun/pi05_official_libero_smoke/home" in launcher
