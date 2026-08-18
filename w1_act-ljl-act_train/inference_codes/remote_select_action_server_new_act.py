#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
import traceback
from typing import Dict, Any

import numpy as np
import torch

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION
from act_async_infer_distributed_demo.scripts.network_utils_act import (
    NetworkServer,
    log_info,
    log_error,
)


class RemoteSelectActionServerNewACT:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.host = args.host
        self.port = args.port
        self.device = torch.device(
            args.device
            if (args.device == "cuda" and torch.cuda.is_available())
            else "cpu"
        )

        self.shutdown_event = threading.Event()
        self.client_connected = False
        self.client_timeout = float(args.client_timeout)
        self.last_client_activity = time.time()

        self.model_lock = threading.Lock()
        self.policy = self._load_policy(args.policy_path)
        self.preprocessor, self.postprocessor = self._load_processors(args.policy_path)

        self.network_server = NetworkServer(self.host, self.port)
        self.network_server.set_disconnect_callback(self._on_client_disconnected)
        self.network_server.register_handler("select_action", self._handle_select_action)
        self.network_server.register_handler("reset_policy", self._handle_reset_policy)
        self.network_server.register_handler("ping", self._handle_ping)
        self.network_server.register_handler("select_action_chunk", self._handle_select_action_chunk)

    @property
    def running(self) -> bool:
        return not self.shutdown_event.is_set()

    def _load_policy(self, policy_path: str) -> ACTPolicy:
        log_info(f"Loading lerobot ACTPolicy from {policy_path} on {self.device} ...")
        policy = ACTPolicy.from_pretrained(policy_path, local_files_only=True)
        policy.to(self.device).eval()
        if hasattr(policy, "config") and hasattr(policy.config, "device"):
            policy.config.device = self.device.type
        policy.reset()
        log_info("ACTPolicy loaded and reset.")
        return policy

    def _load_processors(
        self, policy_path: str
    ) -> tuple[PolicyProcessorPipeline | None, PolicyProcessorPipeline | None]:
        try:
            preprocessor = PolicyProcessorPipeline.from_pretrained(
                policy_path,
                config_filename="policy_preprocessor.json",
                local_files_only=True,
            )
            postprocessor = PolicyProcessorPipeline.from_pretrained(
                policy_path,
                config_filename="policy_postprocessor.json",
                local_files_only=True,
            )
        except Exception as exc:
            log_error(f"Failed to load ACT preprocess/postprocess pipelines: {exc}")
            return None, None

        for step in getattr(preprocessor, "steps", []):
            if step.__class__.__name__ == "DeviceProcessorStep":
                step.device = self.device.type
            elif step.__class__.__name__ == "NormalizerProcessorStep" and hasattr(step, "to"):
                step.to(device=self.device.type)

        for step in getattr(postprocessor, "steps", []):
            if step.__class__.__name__ == "DeviceProcessorStep":
                step.device = "cpu"
            elif step.__class__.__name__ == "UnnormalizerProcessorStep" and hasattr(step, "to"):
                step.to(device="cpu")

        log_info("Loaded ACT policy_preprocessor.json / policy_postprocessor.json on server.")
        return preprocessor, postprocessor

    def _build_observation_np_from_request(self, batch_raw: Dict[str, Any]) -> Dict[str, np.ndarray]:
        obs_np: Dict[str, np.ndarray] = {}
        for key, value in batch_raw.items():
            if value is None or not key.startswith("observation."):
                continue
            arr = np.asarray(value)
            if arr.dtype == np.object_:
                continue
            if "image" in key:
                if np.issubdtype(arr.dtype, np.integer):
                    arr = arr.astype(np.uint8, copy=False)
                elif arr.dtype == np.float64:
                    arr = arr.astype(np.float32)
            elif arr.dtype == np.float64:
                arr = arr.astype(np.float32)
            obs_np[key] = arr
        return obs_np

    def _preprocess_raw_batch_on_server(self, batch_raw: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        obs_np = self._build_observation_np_from_request(batch_raw)
        if len(obs_np) == 0:
            raise ValueError("No observation.* fields found in request batch")

        obs_t = prepare_observation_for_inference(obs_np, self.device)
        if self.preprocessor is None:
            batch_t: Dict[str, Any] = {}
            for key, value in obs_t.items():
                if isinstance(value, torch.Tensor) and value.ndim in (1, 3):
                    batch_t[key] = value.unsqueeze(0)
                else:
                    batch_t[key] = value
        else:
            batch_t = self.preprocessor(obs_t)

        batch_torch: Dict[str, torch.Tensor] = {}
        for key, value in batch_t.items():
            if key == ACTION or value is None:
                continue
            if isinstance(value, torch.Tensor):
                batch_torch[key] = value.to(self.device)
                continue

            arr = np.asarray(value)
            is_numeric_or_bool = np.issubdtype(arr.dtype, np.number) or np.issubdtype(
                arr.dtype, np.bool_
            )
            if not is_numeric_or_bool:
                continue
            if arr.dtype == np.float64:
                arr = arr.astype(np.float32)
            batch_torch[key] = torch.from_numpy(arr).to(self.device)

        if len(batch_torch) == 0:
            raise ValueError("No valid numeric tensor in preprocessed batch")
        return batch_torch

    def _postprocess_action(self, action: Any) -> Any:
        if self.postprocessor is None:
            return action
        try:
            out = self.postprocessor({ACTION: action})
        except Exception:
            out = self.postprocessor({"action": action})
        if isinstance(out, dict):
            if ACTION in out:
                return out[ACTION]
            if "action" in out:
                return out["action"]
        return out
    
    def _to_numpy_action(self, action: Any) -> np.ndarray:
        if isinstance(action, torch.Tensor):
            arr = action.detach().cpu().numpy()
            return np.asarray(arr)
        return np.asarray(action)

    def _touch_activity(self):
        self.last_client_activity = time.time()

    def _print_client_request(self, request_name: str, request: dict) -> None:
        request_id = request.get("request_id", "unknown")
        print(f"[SERVER][req {request_id}] Received '{request_name}' request from client.", flush=True)

    def _handle_ping(self, request: dict) -> dict:
        self._touch_activity()
        self._print_client_request("ping", request)
        return {"status": "ok"}

    def _handle_reset_policy(self, request: dict) -> dict:
        self._touch_activity()
        self._print_client_request("reset_policy", request)
        try:
            with self.model_lock:
                self.policy.reset()
            return {"status": "ok"}
        except Exception as exc:
            log_error(f"reset_policy failed: {exc}")
            return {"status": "error", "message": str(exc)}

    def _handle_select_action(self, request: dict) -> dict:
        self._touch_activity()
        self._print_client_request("select_action", request)
        try:
            batch = request.get("batch")
            if not isinstance(batch, dict) or len(batch) == 0:
                return {"status": "error", "message": "Invalid or empty batch"}

            batch_torch = self._preprocess_raw_batch_on_server(batch)
            with self.model_lock:
                with torch.no_grad():
                    action = self.policy.select_action(batch_torch)
                    action = self._postprocess_action(action)
            action_np = self._to_numpy_action(action)
            return {"status": "success", "action": action_np}
        except Exception as exc:
            traceback.print_exc()
            log_error(f"select_action failed: {exc}")
            return {"status": "error", "message": str(exc)}

    def _handle_select_action_chunk(self, request: dict) -> dict:
        self._touch_activity()
        self._print_client_request("select_action_chunk", request)
        try:
            batch = request.get("batch")
            if not isinstance(batch, dict) or len(batch) == 0:
                return {"status": "error", "message": "Invalid or empty batch"}

            n_action_steps = request.get("n_action_steps", None)
            if n_action_steps is not None:
                try:
                    n_action_steps = int(n_action_steps)
                    if n_action_steps <= 0:
                        n_action_steps = None
                except Exception:
                    n_action_steps = None

            batch_torch = self._preprocess_raw_batch_on_server(batch)
            with self.model_lock:
                with torch.no_grad():
                    actions = self.policy.predict_action_chunk(batch_torch)
                    actions = self._postprocess_action(actions)

            actions_np = np.asarray(self._to_numpy_action(actions))
            if actions_np.ndim == 3:
                actions_np = actions_np[0]
            if n_action_steps is not None:
                actions_np = actions_np[:n_action_steps]
            return {"status": "success", "actions": actions_np}
        except Exception as exc:
            traceback.print_exc()
            log_error(f"select_action_chunk failed: {exc}")
            return {"status": "error", "message": str(exc)}

    def _on_client_disconnected(self):
        self.client_connected = False

    def _client_timeout_monitor(self, stop_event: threading.Event):
        while self.running and not stop_event.is_set():
            if self.client_connected:
                idle = time.time() - self.last_client_activity
                if idle > self.client_timeout:
                    log_info(
                        f"Client timeout after {idle:.1f}s, closing connection and resetting policy."
                    )
                    self.client_connected = False
                    self.network_server.close()
            time.sleep(1.0)

    def start(self):
        stop_event = threading.Event()
        while self.running:
            try:
                log_info(f"Waiting for client on {self.host}:{self.port} ...")
                if not self.network_server.start():
                    time.sleep(1.0)
                    continue

                self.client_connected = True
                self._touch_activity()

                monitor_thread = threading.Thread(
                    target=self._client_timeout_monitor,
                    args=(stop_event,),
                    daemon=True,
                )
                monitor_thread.start()

                self.network_server.handle_requests(
                    lambda: self.running and self.client_connected
                )

                stop_event.set()
                monitor_thread.join(timeout=1.0)
                stop_event.clear()

                self.network_server.close()
                with self.model_lock:
                    self.policy.reset()
                log_info("Client session closed. Policy reset.")

            except Exception as exc:
                traceback.print_exc()
                log_error(f"Server main loop error: {exc}")
                self.network_server.close()
                time.sleep(1.0)

    def stop(self):
        self.shutdown_event.set()
        self.client_connected = False
        self.network_server.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remote lerobot ACT select_action server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--policy_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--client_timeout", type=float, default=30.0)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    server = RemoteSelectActionServerNewACT(args)

    def _sig_handler(sig, frame):
        log_info("Received stop signal, shutting down remote select_action server...")
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    try:
        server.start()
    finally:
        server.stop()


if __name__ == "__main__":
    main()
