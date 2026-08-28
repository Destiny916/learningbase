"""Run the PC1 XWiz manager with guarded single or continuous configs."""

from __future__ import annotations

import json
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor

from act_async_infer_distributed_demo.scripts.manager.config_registry import ConfigRegistry
from act_async_infer_distributed_demo.scripts.manager.inference_manager import InferenceManager
from act_async_infer_distributed_demo.scripts.utils_distributed import log_error, log_info, log_warning

from .manager_runtime import deploy_selected, lerobot_model_meta, prepare_resolved_configs


_vendor_resolve_config = ConfigRegistry.resolve_config
_vendor_resolve_state_meta = ConfigRegistry.resolve_state_meta


def resolve_guarded_config(self, model_id, task_id, mode):
    client, server = _vendor_resolve_config(self, model_id, task_id, mode)
    return prepare_resolved_configs(client, server, int(mode))


def resolve_guarded_state_meta(self, model_id):
    cameras, groups = _vendor_resolve_state_meta(self, model_id)
    if cameras or any(group.is_selected for group in groups):
        return cameras, groups

    with open(self.find_state_meta_path(model_id), encoding="utf-8") as stream:
        fallback_cameras, selected_groups = lerobot_model_meta(json.load(stream))
    for group in groups:
        group.is_selected = group.group_name in selected_groups
    return fallback_cameras, groups


def deploy_and_start(self, request, response):
    """Deploy the selected task and immediately start its configured mode."""
    with self.lock:
        try:
            if not deploy_selected(self, request.model, request.task):
                response.success = False
                response.message = "PC2 inference client rejected deployment"
                return response
            response.success = True
            response.message = "Configuration deployed and inference started"
            log_info(f"Deployment started: model={request.model}, task={request.task}")
        except Exception as exc:
            response.success = False
            response.message = f"Failed to deploy: {exc}"
            log_error(response.message)
        return response


def resilient_watchdog(self):
    try:
        alive = self._node_is_present(self._target_node)
    except Exception as exc:
        now = time.monotonic()
        if now - getattr(self, "_last_watchdog_graph_warning", 0.0) >= 60.0:
            self._last_watchdog_graph_warning = now
            log_warning(f"看门狗读取 ROS graph 失败，暂不改变状态: {exc}")
        return
    if not self._monitoring:
        if alive:
            self._monitoring = True
            self._node_alive = True
            log_info(f"看门狗: {self._target_node} 已上线, 进入持续监控")
    elif not alive and self._node_alive:
        self._node_alive = False
        log_warning(f"看门狗: {self._target_node} 已下线, 触发停止推理")
        self._trigger_stop_service()
    elif alive and not self._node_alive:
        self._node_alive = True
        log_info(f"看门狗: {self._target_node} 已重新上线")


def safe_destroy(self):
    shutdown = getattr(self.client_controller, "shutdown", None)
    if callable(shutdown):
        shutdown()
    self.destroy_node()


def main() -> None:
    ConfigRegistry.resolve_config = resolve_guarded_config
    ConfigRegistry.resolve_state_meta = resolve_guarded_state_meta
    InferenceManager._handle_deploy = deploy_and_start
    InferenceManager._watchdog_callback = resilient_watchdog
    InferenceManager.destroy = safe_destroy

    rclpy.init()
    args = type("Args", (), {"client_host": "192.168.20.21", "client_port": 8890})()
    node = InferenceManager(args)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy()
        executor.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
