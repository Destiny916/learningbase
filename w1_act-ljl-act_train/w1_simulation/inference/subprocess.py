from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import suppress
from multiprocessing import shared_memory
from multiprocessing.connection import Client, Connection
from pathlib import Path

import numpy as np

from w1_simulation.robot.joints import ACT_STATE_JOINTS
from w1_simulation.w1_profile import ACT_IMAGE_KEYS


class ScriptPolicyRuntime:
    backend_name = "script"

    def __init__(
        self,
        checkpoint: Path,
        script: Path,
        device: str = "cuda:0",
        image_shapes: dict[str, tuple[int, int, int]] | None = None,
        execution_horizon: int = 30,
        startup_timeout_s: float = 180.0,
        inference_timeout_s: float = 60.0,
    ) -> None:
        self.checkpoint = Path(checkpoint).resolve()
        self.script_path = Path(script).resolve()
        self.device = device
        self.startup_timeout_s = startup_timeout_s
        self.inference_timeout_s = inference_timeout_s
        self.last_latency_ms = 0.0
        self._lock = threading.Lock()
        self._closed = False
        self._conn: Connection | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._obs_shm: shared_memory.SharedMemory | None = None
        self._acts_shm: shared_memory.SharedMemory | None = None
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._log_file = None

        if startup_timeout_s <= 0 or inference_timeout_s <= 0:
            raise ValueError("Policy script timeouts must be positive")
        if isinstance(execution_horizon, bool) or execution_horizon <= 0:
            raise ValueError("Policy script execution horizon must be a positive integer")
        self.execution_horizon = int(execution_horizon)
        if not self.script_path.is_file():
            raise FileNotFoundError(self.script_path)
        if not (self.checkpoint / "model.safetensors").is_file():
            raise FileNotFoundError(self.checkpoint / "model.safetensors")

        requested_shapes = (
            dict.fromkeys(ACT_IMAGE_KEYS, (360, 640, 3)) if image_shapes is None else image_shapes
        )
        self.image_shapes = {key: tuple(shape) for key, shape in requested_shapes.items()}
        self._validate_image_shapes(self.image_shapes)
        self._slot_size = max(int(np.prod(shape)) for shape in self.image_shapes.values())
        self._obs_size = len(self.image_shapes) * self._slot_size + len(ACT_STATE_JOINTS) * 4
        self._acts_size = self.execution_horizon * len(ACT_STATE_JOINTS) * 4
        suffix = f"{uuid.uuid4().hex}_{os.getpid()}"
        self._obs_name = f"w1_simulation_obs_{suffix}"
        self._acts_name = f"w1_simulation_acts_{suffix}"

        try:
            self._start_server()
        except BaseException:
            self.close()
            raise

    @property
    def server_pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    def _start_server(self) -> None:
        port = self._find_free_port()
        self._temp_dir = tempfile.TemporaryDirectory(prefix="w1_simulation_script_")
        temp_path = Path(self._temp_dir.name)
        config_path = temp_path / "server_config.json"
        config_path.write_text(
            json.dumps(
                {
                    "port": port,
                    "device": self.device,
                    "models": {"simulation": str(self.checkpoint)},
                }
            ),
            encoding="utf-8",
        )
        self._log_file = (temp_path / "server.log").open("w+b")
        self._process = subprocess.Popen(
            [sys.executable, str(self.script_path), "--config", str(config_path)],
            stdin=subprocess.DEVNULL,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            cwd=str(self.script_path.parent),
        )
        self._obs_shm = shared_memory.SharedMemory(name=self._obs_name, create=True, size=self._obs_size)
        self._acts_shm = shared_memory.SharedMemory(name=self._acts_name, create=True, size=self._acts_size)

        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            return_code = self._process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"Policy script exited during startup with code {return_code}: {self._server_log()}"
                )
            try:
                self._conn = Client(("127.0.0.1", port), authkey=b"w1_simulation_secret")
                break
            except (ConnectionRefusedError, ConnectionResetError, EOFError, OSError):
                time.sleep(0.05)
        else:
            raise TimeoutError(
                f"Policy script was not ready after {self.startup_timeout_s:.1f}s: {self._server_log()}"
            )

        response = self._request(
            {
                "cmd": "SHM_INIT",
                "obs_name": self._obs_name,
                "acts_name": self._acts_name,
                "obs_size": self._obs_size,
                "acts_size": self._acts_size,
                "num_slots": len(self.image_shapes),
                "slot_size": self._slot_size,
                "state_dim": len(ACT_STATE_JOINTS),
                "horizon_N": self.execution_horizon,
                "image_keys": list(self.image_shapes),
                "image_shapes": {key: list(shape) for key, shape in self.image_shapes.items()},
                "state_key": "observation.state",
            },
            timeout_s=self.startup_timeout_s,
        )
        if response != "OK":
            raise RuntimeError(f"Policy script rejected SHM_INIT: {response!r}")
        if self._request({"cmd": "RESET"}, timeout_s=self.startup_timeout_s) != "OK":
            raise RuntimeError("Policy script rejected RESET")

    @staticmethod
    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    @staticmethod
    def _validate_image_shapes(
        image_shapes: dict[str, tuple[int, ...]],
    ) -> None:
        if not image_shapes:
            raise ValueError("Policy script requires at least one RGB image")
        for key, shape in image_shapes.items():
            if not isinstance(key, str) or not key:
                raise ValueError("Policy script image keys must be non-empty strings")
            if len(shape) != 3 or any(not isinstance(size, int) or size <= 0 for size in shape):
                raise ValueError(f"Invalid HWC image shape for {key}: {shape}")
            if shape[2] != 3:
                raise ValueError(f"Policy script requires RGB images, got {key}: {shape}")

    def _request(self, message: dict[str, object], timeout_s: float) -> object:
        if self._conn is None:
            raise RuntimeError("Policy script connection is closed")
        self._conn.send(message)
        if not self._conn.poll(timeout_s):
            raise TimeoutError(f"Policy script did not answer {message['cmd']} within {timeout_s:.1f}s")
        return self._conn.recv()

    def _server_log(self) -> str:
        if self._log_file is None:
            return "no server log"
        self._log_file.flush()
        position = self._log_file.tell()
        self._log_file.seek(0)
        contents = self._log_file.read().decode("utf-8", errors="replace").strip()
        self._log_file.seek(position)
        return contents[-4000:] if contents else "empty server log"

    def predict_chunk(self, state: np.ndarray, images: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
        state_array = np.asarray(state, dtype=np.float32)
        if state_array.shape != (len(ACT_STATE_JOINTS),) or not np.isfinite(state_array).all():
            raise ValueError(f"Expected finite 19D ACT state, got {state_array.shape}")
        if set(images) != set(self.image_shapes):
            raise ValueError(f"ACT image keys mismatch: {set(images)}")
        image_arrays: list[np.ndarray] = []
        for key, expected_shape in self.image_shapes.items():
            image = np.asarray(images[key])
            if image.shape != expected_shape or image.dtype != np.uint8:
                raise ValueError(
                    f"Expected uint8 HWC image {expected_shape} for {key}, got {image.shape} {image.dtype}"
                )
            image_arrays.append(image)

        with self._lock:
            if self._closed or self._obs_shm is None or self._acts_shm is None:
                raise RuntimeError("Policy script runtime is closed")
            for index, (image, shape) in enumerate(
                zip(image_arrays, self.image_shapes.values(), strict=True)
            ):
                target = np.ndarray(
                    shape,
                    dtype=np.uint8,
                    buffer=self._obs_shm.buf,
                    offset=index * self._slot_size,
                )
                np.copyto(target, image)
            state_target = np.ndarray(
                (len(ACT_STATE_JOINTS),),
                dtype=np.float32,
                buffer=self._obs_shm.buf,
                offset=len(self.image_shapes) * self._slot_size,
            )
            np.copyto(state_target, state_array)

            if self._request({"cmd": "RESET"}, self.inference_timeout_s) != "OK":
                raise RuntimeError("Policy script rejected RESET")
            started = time.perf_counter()
            response = self._request(
                {"cmd": "INFER_CHUNK", "steps": self.execution_horizon},
                timeout_s=self.inference_timeout_s,
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            if not isinstance(response, dict) or response.get("status") != "OK":
                raise RuntimeError(f"Policy script inference failed: {response!r}")
            if response.get("n_steps") != self.execution_horizon:
                raise ValueError(
                    f"Policy script returned {response.get('n_steps')} actions, "
                    f"expected {self.execution_horizon}"
                )
            chunk = np.ndarray(
                (self.execution_horizon, len(ACT_STATE_JOINTS)),
                dtype=np.float32,
                buffer=self._acts_shm.buf,
            ).copy()

        expected_shape = (self.execution_horizon, len(ACT_STATE_JOINTS))
        if chunk.shape != expected_shape or not np.isfinite(chunk).all():
            raise ValueError(f"Policy script returned an invalid {expected_shape} action chunk")
        self.last_latency_ms = latency_ms
        return chunk, latency_ms

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._conn is not None:
                with suppress(OSError):
                    self._conn.close()
                self._conn = None
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=5)
            self._process = None
            for shm in (self._obs_shm, self._acts_shm):
                if shm is None:
                    continue
                shm.close()
                with suppress(FileNotFoundError):
                    shm.unlink()
            self._obs_shm = None
            self._acts_shm = None
            if self._log_file is not None:
                self._log_file.close()
                self._log_file = None
            if self._temp_dir is not None:
                self._temp_dir.cleanup()
                self._temp_dir = None

    def __enter__(self) -> ScriptPolicyRuntime:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()
