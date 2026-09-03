"""ROS-independent ACT server compatible with the vendor PC2 inference client."""

from __future__ import annotations

import argparse
import logging
import socket
import os
from typing import Any, Protocol

from .contract import ContractError, decode_observation, group_action_chunk
from .protocol import ProtocolError, recv_message, send_message


LOGGER = logging.getLogger("xwiz_act_server")


class ActionRuntime(Protocol):
    def predict(self, observation: dict[str, Any]) -> Any: ...

    def reset(self) -> None: ...


class XWizActServerApp:
    def __init__(self, runtime: ActionRuntime):
        self.runtime = runtime
        self.state = "idle"
        self.error: str | None = None
        self.latest_actions: dict[str, Any] | None = None
        self.latest_timestamp = 0.0
        self.latest_timestep = 0
        self.shutdown_requested = False

    @staticmethod
    def _with_request_id(request: dict[str, Any], reply: dict[str, Any]) -> dict[str, Any]:
        if "request_id" in request:
            reply["request_id"] = request["request_id"]
        return reply

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        request_type = request.get("type")
        try:
            if request_type == "SETUP_CONFIG":
                reply = self._setup(request.get("config", {}))
            elif request_type == "STATUS":
                reply = {"success": True, "state": self.state, "error": self.error}
            elif request_type == "observation":
                reply = self._observation(request)
            elif request_type == "get_actions":
                reply = self._get_actions()
            elif request_type == "STOP":
                reply = self._stop()
            elif request_type == "SHUTDOWN":
                reply = self._stop()
                self.shutdown_requested = True
            else:
                reply = {"success": False, "error": f"unknown request type: {request_type}"}
        except (ContractError, ValueError, RuntimeError) as exc:
            self.state = "error"
            self.error = str(exc)
            LOGGER.exception("request %s failed", request_type)
            reply = {"success": False, "state": "error", "error": str(exc)}
        return self._with_request_id(request, reply)

    def _setup(self, config: Any) -> dict[str, Any]:
        if not isinstance(config, dict):
            raise ValueError("config must be a mapping")
        data_type = config.get("data_type")
        if data_type not in {"simulation", "real"}:
            raise ValueError(f"unsupported data_type: {data_type!r}")
        self.runtime.reset()
        self.latest_actions = None
        self.error = None
        self.state = "running"
        return {"success": True, "state": "running"}

    def _observation(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.state == "idle":
            # STOP can race with one already-queued observation from PC2.
            # Acknowledge it without inference so the client remains idle.
            return {"status": "received", "inferred": False}
        if self.state != "running":
            raise RuntimeError(f"server is not running: {self.state}")
        inferred = bool(request.get("start_infer", False))
        if inferred:
            observation = decode_observation(request)
            actions = self.runtime.predict(observation)
            self.latest_actions = group_action_chunk(actions)
            self.latest_timestamp = float(request.get("timestamp", 0.0))
            self.latest_timestep = int(request.get("timestep", 0))
            horizon = int(os.environ.get("XWIZ_ACTION_HORIZON", "16"))
            LOGGER.info(
                "inference completed timestep=%d action_shape=(%d,19)",
                self.latest_timestep, horizon,
            )
        return {"status": "received", "inferred": inferred}

    def _get_actions(self) -> dict[str, Any]:
        if self.latest_actions is None:
            return {"status": "pending"}
        wire_actions = {
            name: values.tolist() for name, values in self.latest_actions.items()
        }
        reply = {
            "status": "success",
            "actions": {"qpos": wire_actions},
            "timestamp": self.latest_timestamp,
            "timestep": self.latest_timestep,
        }
        # The vendor client polls get_actions continuously. Each inference
        # chunk must be consumed exactly once or it will aggregate the same
        # chunk repeatedly before executing any simulation frame.
        self.latest_actions = None
        return reply

    def _stop(self) -> dict[str, Any]:
        self.latest_actions = None
        self.state = "idle"
        self.error = None
        return {"success": True, "state": "idle"}


def serve(host: str, port: int, app: XWizActServerApp) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen(1)
        LOGGER.info("listening on %s:%d", host, port)
        while not app.shutdown_requested:
            connection, address = listener.accept()
            LOGGER.info("client connected from %s:%d", *address)
            with connection:
                while not app.shutdown_requested:
                    try:
                        request = recv_message(connection)
                        send_message(connection, app.handle(request))
                    except ProtocolError as exc:
                        LOGGER.info("client disconnected: %s", exc)
                        break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8889)
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from .model_runtime import LeRobotActRuntime

    runtime = LeRobotActRuntime(args.policy_path, args.device)
    serve(args.host, args.port, XWizActServerApp(runtime))


if __name__ == "__main__":
    main()
