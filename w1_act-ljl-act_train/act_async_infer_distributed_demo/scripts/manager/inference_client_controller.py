import socket
import threading
import json
from typing import Dict, Optional
from act_async_infer_distributed_demo.scripts.utils_distributed import log_info, log_error
from act_async_infer_distributed_demo.scripts.inference_config import RequestType, ResponseKey, ManagerKey  


class InferenceClientController:
    def __init__(self, control_host: str = "192.168.20.20", control_port: int = 8890):
        self.control_host = control_host
        self.control_port = control_port
        self.control_socket: Optional[socket.socket] = None
        self.lock = threading.RLock()

    def connect(self) -> bool:
        with self.lock:
            try:
                if self.control_socket:
                    self.control_socket.close()
                self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.control_socket.settimeout(None)
                self.control_socket.connect((self.control_host, self.control_port))
                log_info(f"Connected to client at {self.control_host}:{self.control_port}")
                return True
            except Exception as e:
                log_error(f"Failed to connect to client: {e}")
                self.control_socket = None
                return False

    def _send_command(self, command: str, payload: Dict) -> Dict:
        with self.lock:
            if not self.control_socket:
                if not self.connect():
                    return {ResponseKey.SUCCESS: False, ResponseKey.MESSAGE: "Failed to connect to client"}
            try:
                message = {"command": command, "payload": payload}
                data = json.dumps(message).encode("utf-8")
                self.control_socket.sendall(len(data).to_bytes(4, "big"))
                self.control_socket.sendall(data)

                size_bytes = self.control_socket.recv(4)
                if not size_bytes:
                    self.control_socket = None
                    return {ResponseKey.SUCCESS: False, ResponseKey.MESSAGE: "No response"}
                size = int.from_bytes(size_bytes, "big")
                response_data = self.control_socket.recv(size)
                return json.loads(response_data.decode("utf-8"))
            except Exception as e:
                import traceback
                log_error(f"Command '{command}' failed: {e}")
                self.control_socket = None
                return {ResponseKey.SUCCESS: False, ResponseKey.MESSAGE: str(e)}

    def setup_config(self, client_config: Dict, server_config: Dict) -> bool:
        payload = {
            "client_config": client_config,
            "server_config": server_config,
            "home_position": client_config.get("home_position", ""),
        }
        return self._send_command(RequestType.SETUP_CONFIG, payload).get(ResponseKey.SUCCESS, False)

    def stop(self) -> bool:
        return self._send_command(RequestType.STOP, {}).get(ResponseKey.SUCCESS, False)

    def get_status(self) -> Dict:
        return self._send_command(RequestType.STATUS, {})
