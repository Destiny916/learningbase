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

from act.modeling_act import ACTPolicy
from act_async_infer_distributed_demo.scripts.network_utils_act import (
    NetworkServer,
    log_info,
    log_error,
)


class RemoteSelectActionServer:
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
        log_info(f"Loading ACTPolicy from {policy_path} on {self.device} ...")
        policy = ACTPolicy.from_pretrained(policy_path, local_files_only=True)
        policy.to(self.device).eval()
        policy.reset()
        log_info("ACTPolicy loaded and reset.")
        return policy

    def _to_torch_batch(self, batch_np: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        batch_torch: Dict[str, torch.Tensor] = {}
        for key, value in batch_np.items():
            arr = np.asarray(value)
            if arr.dtype == np.float64:
                arr = arr.astype(np.float32)
            batch_torch[key] = torch.from_numpy(arr).to(self.device)
        return batch_torch

    def _to_numpy_action(self, action: Any) -> np.ndarray:
        if isinstance(action, torch.Tensor):
            arr = action.detach().cpu().numpy()
            if arr.ndim == 2:
                arr = arr[0]
            return arr.astype(np.float32, copy=False)
        arr = np.asarray(action, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[0]
        return arr

    def _touch_activity(self):
        self.last_client_activity = time.time()

    def _handle_ping(self, request: dict) -> dict:
        self._touch_activity()
        return {"status": "ok"}

    def _handle_reset_policy(self, request: dict) -> dict:
        self._touch_activity()
        try:
            with self.model_lock:
                self.policy.reset()
            return {"status": "ok"}
        except Exception as exc:
            log_error(f"reset_policy failed: {exc}")
            return {"status": "error", "message": str(exc)}

    def _handle_select_action(self, request: dict) -> dict:
        self._touch_activity()
        try:
            batch = request.get("batch")
            if not isinstance(batch, dict) or len(batch) == 0:
                return {"status": "error", "message": "Invalid or empty batch"}

            batch_torch = self._to_torch_batch(batch)
            with self.model_lock:
                with torch.no_grad():
                    action = self.policy.select_action(batch_torch)
            action_np = self._to_numpy_action(action)
            return {"status": "success", "action": action_np}
        except Exception as exc:
            traceback.print_exc()
            log_error(f"select_action failed: {exc}")
            return {"status": "error", "message": str(exc)}
    def _handle_select_action_chunk(self, request: dict) -> dict:
        self._touch_activity()
        try:
            batch = request.get("batch")
            if not isinstance(batch, dict) or len(batch) == 0:
                return {"status": "error", "message": "Invalid or empty batch"}

            batch_torch = self._to_torch_batch(batch)
            with self.model_lock:
                with torch.no_grad():
                    actions = self.policy.predict_action_chunk(batch_torch)
            # actions: (batch_size, n_action_steps, action_dim)
            actions_np = actions.detach().cpu().numpy()
            if actions_np.ndim == 3:
                actions_np = actions_np[0]  # → (n_action_steps, action_dim)
            return {"status": "success", "actions": actions_np.astype(np.float32)}
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
    parser = argparse.ArgumentParser(description="Remote ACT select_action server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--policy_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--client_timeout", type=float, default=30.0)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    server = RemoteSelectActionServer(args)

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
