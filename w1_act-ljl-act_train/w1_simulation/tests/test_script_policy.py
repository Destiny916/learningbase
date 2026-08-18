from __future__ import annotations

import textwrap
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import pytest
from w1_simulation.inference.subprocess import ScriptPolicyRuntime
from w1_simulation.w1_profile import ACT_IMAGE_KEYS

FAKE_SERVER = r"""
import argparse
import json
from multiprocessing import shared_memory
from multiprocessing.connection import Listener

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
config = json.loads(open(args.config, encoding="utf-8").read())
listener = Listener(("127.0.0.1", config["port"]), authkey=b"w1_simulation_secret")
conn = listener.accept()
obs = acts = None
try:
    while True:
        message = conn.recv()
        if message["cmd"] == "SHM_INIT":
            assert config["device"] == "cuda:7"
            assert config["models"]["simulation"].endswith("checkpoint")
            assert message["obs_name"] != "policy_obs"
            assert message["acts_name"] != "policy_acts"
            obs = shared_memory.SharedMemory(name=message["obs_name"])
            acts = shared_memory.SharedMemory(name=message["acts_name"])
            metadata = message
            conn.send("OK")
        elif message["cmd"] == "RESET":
            conn.send("OK")
        elif message["cmd"] == "INFER_CHUNK":
            slot = metadata["slot_size"]
            state_offset = metadata["num_slots"] * slot
            state = np.ndarray((19,), np.float32, buffer=obs.buf, offset=state_offset)
            first_pixels = [obs.buf[index * slot] for index in range(metadata["num_slots"])]
            horizon = metadata["horizon_N"]
            output = np.ndarray((horizon, 19), np.float32, buffer=acts.buf)
            output[:] = state
            output[:, 0] += np.arange(horizon, dtype=np.float32)
            output[:, 1:1 + len(first_pixels)] += np.asarray(first_pixels, dtype=np.float32)
            conn.send({"status": "OK", "n_steps": message["steps"]})
finally:
    if obs is not None:
        obs.close()
    if acts is not None:
        acts.close()
"""


@pytest.fixture
def fake_runtime(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").touch()
    script = tmp_path / "fake_policy_infer_act.py"
    script.write_text(textwrap.dedent(FAKE_SERVER), encoding="utf-8")
    runtime = ScriptPolicyRuntime(
        checkpoint=checkpoint,
        script=script,
        device="cuda:7",
        startup_timeout_s=5,
        inference_timeout_s=5,
    )
    yield runtime
    runtime.close()


def test_predict_chunk_uses_script_shared_memory_protocol(fake_runtime: ScriptPolicyRuntime) -> None:
    state = np.arange(19, dtype=np.float32)
    images = {
        key: np.full((360, 640, 3), index + 1, dtype=np.uint8) for index, key in enumerate(ACT_IMAGE_KEYS)
    }

    chunk, latency_ms = fake_runtime.predict_chunk(state, images)

    assert fake_runtime.backend_name == "script"
    assert fake_runtime.script_path.name == "fake_policy_infer_act.py"
    assert fake_runtime.server_pid is not None
    assert chunk.shape == (30, 19)
    np.testing.assert_array_equal(chunk[:, 0], np.arange(30, dtype=np.float32))
    np.testing.assert_array_equal(chunk[0, 1:4], state[1:4] + np.asarray([1, 2, 3]))
    assert latency_ms >= 0
    assert fake_runtime.last_latency_ms == latency_ms


def test_script_runtime_uses_configured_execution_horizon(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").touch()
    script = tmp_path / "fake_policy_infer_act.py"
    script.write_text(textwrap.dedent(FAKE_SERVER), encoding="utf-8")

    with ScriptPolicyRuntime(
        checkpoint=checkpoint,
        script=script,
        device="cuda:7",
        execution_horizon=12,
        startup_timeout_s=5,
        inference_timeout_s=5,
    ) as runtime:
        images = {key: np.zeros((360, 640, 3), dtype=np.uint8) for key in ACT_IMAGE_KEYS}
        chunk, _ = runtime.predict_chunk(np.zeros(19, dtype=np.float32), images)

    assert chunk.shape == (12, 19)
    np.testing.assert_array_equal(chunk[:, 0], np.arange(12, dtype=np.float32))


def test_rejects_invalid_observations_before_ipc(fake_runtime: ScriptPolicyRuntime) -> None:
    images = {key: np.zeros((360, 640, 3), dtype=np.uint8) for key in ACT_IMAGE_KEYS}
    with pytest.raises(ValueError, match="finite 19D"):
        fake_runtime.predict_chunk(np.zeros(18, dtype=np.float32), images)
    images[ACT_IMAGE_KEYS[0]] = np.zeros((3, 360, 640), dtype=np.uint8)
    with pytest.raises(ValueError, match="uint8 HWC"):
        fake_runtime.predict_chunk(np.zeros(19, dtype=np.float32), images)


def test_close_is_idempotent_and_unlinks_shared_memory(fake_runtime: ScriptPolicyRuntime) -> None:
    obs_name = fake_runtime._obs_name
    acts_name = fake_runtime._acts_name
    fake_runtime.close()
    fake_runtime.close()

    assert fake_runtime.server_pid is None
    for name in (obs_name, acts_name):
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=name)


@pytest.mark.parametrize("num_images", [2, 4])
def test_supports_two_and_four_image_slot_layouts(tmp_path: Path, num_images: int) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").touch()
    script = tmp_path / "fake_policy_infer_act.py"
    script.write_text(textwrap.dedent(FAKE_SERVER), encoding="utf-8")
    image_shapes = {f"observation.images.cam_{index}": (360, 640, 3) for index in range(num_images)}
    images = {
        key: np.full(shape, index + 1, dtype=np.uint8)
        for index, (key, shape) in enumerate(image_shapes.items())
    }

    with ScriptPolicyRuntime(
        checkpoint=checkpoint,
        script=script,
        device="cuda:7",
        image_shapes=image_shapes,
        startup_timeout_s=5,
        inference_timeout_s=5,
    ) as runtime:
        chunk, _ = runtime.predict_chunk(np.arange(19, dtype=np.float32), images)

    np.testing.assert_array_equal(
        chunk[0, 1 : num_images + 1],
        np.arange(1, num_images + 1, dtype=np.float32) * 2,
    )


def test_supports_different_shapes_for_each_image_input(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").touch()
    script = tmp_path / "fake_policy_infer_act.py"
    script.write_text(textwrap.dedent(FAKE_SERVER), encoding="utf-8")
    image_shapes = {
        "observation.images.cam_left": (360, 640, 3),
        "observation.images.cam_right": (180, 320, 3),
    }
    images = {key: np.zeros(shape, dtype=np.uint8) for key, shape in image_shapes.items()}

    with ScriptPolicyRuntime(
        checkpoint=checkpoint,
        script=script,
        device="cuda:7",
        image_shapes=image_shapes,
        startup_timeout_s=5,
        inference_timeout_s=5,
    ) as runtime:
        chunk, _ = runtime.predict_chunk(np.zeros(19, dtype=np.float32), images)

    assert chunk.shape == (30, 19)
