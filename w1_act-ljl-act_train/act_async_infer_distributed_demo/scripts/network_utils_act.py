import socket
import pickle
import struct
import threading
import time
from typing import Any, Optional
import numpy as np
import traceback
import cv2
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
)


# 添加日志记录
def log_info(msg):
    logging.info(f"[INFO] {msg}")


def log_error(msg):
    logging.error(f"[ERROR] {msg}")


def log_warning(msg):
    logging.warning(f"[WARNING] {msg}")


class NetworkClient:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        self.lock = threading.Lock()
        self._request_id = 0

    def connect(self, timeout=10):
        """连接到服务器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # 禁用Nagle算法
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            # 优化socket缓冲区
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 10485760)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 10485760)
            self.socket.settimeout(timeout)
            self.socket.connect((self.host, self.port))
            self.connected = True
            log_info(f"Connected to server at {self.host}:{self.port}")
            return True
        except Exception as e:
            log_error(f"连接失败: {e}")
            return False

    def send_request(self, request_type: str, data: dict = None) -> Optional[dict]:
        """发送请求并等待响应"""
        if not self.connected:
            return None

        with self.lock:
            try:
                request_id = self._request_id
                self._request_id += 1

                # 准备请求数据
                request = {"type": request_type, "request_id": request_id}
                if data:
                    request.update(data)

                # 序列化并发送
                serialized_data = pickle.dumps(request)
                self.socket.sendall(struct.pack(">I", len(serialized_data)))
                self.socket.sendall(serialized_data)
                obs_size_bytes = len(serialized_data)
                obs_size_mb = obs_size_bytes / (1024 * 1024)

                log_info(
                    f"[req {request_id}] Sent '{request_type}' "
                    f"({obs_size_mb:.4f} MB, {obs_size_bytes} bytes)"
                )
                # 接收响应
                raw_len = self._recv_all(4)
                if not raw_len:
                    log_warning(f"[req {request_id}] No response header for '{request_type}'")
                    self.connected = False
                    return None
                data_len = struct.unpack(">I", raw_len)[0]

                # 数据长度验证
                if data_len > 1024 * 1024:  # 1MB限制
                    log_error(f"[req {request_id}] Response data too large")
                    return None

                serialized_response = self._recv_all(data_len)
                if not serialized_response:
                    log_warning(f"[req {request_id}] Incomplete response body for '{request_type}'")
                    self.connected = False
                    return None

                response = pickle.loads(serialized_response)
                response_id = response.get("request_id", request_id) if isinstance(response, dict) else request_id
                log_info(f"[req {response_id}] Received response for '{request_type}'")
                return response

            except Exception as e:
                log_error(f"[req {request_id}] 网络通信错误 during '{request_type}': {e}")
                self.connected = False
                return None

    def _recv_all(self, length: int) -> Optional[bytes]:
        """接收指定长度的数据"""
        data = b""
        while len(data) < length:
            try:
                packet = self.socket.recv(length - len(data))
                if not packet:
                    return None
                data += packet
            except socket.timeout:
                return None
        return data

    def close(self):
        """关闭连接"""
        if self.socket:
            self.socket.close()
        self.connected = False


class NetworkServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.socket = None
        self.client_socket = None
        self.client_address = None
        self.connected = False
        self.request_handlers = {}
        self.lock = threading.Lock()

        self.disconnect_callback = None
        self.last_activity_time = None

    def register_handler(self, request_type: str, handler):
        """注册请求处理器"""
        self.request_handlers[request_type] = handler

    def set_disconnect_callback(self, callback):
        """设置客户端断开连接回调"""
        self.disconnect_callback = callback

    def start(self):
        """启动服务器并等待客户端连接"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            self.socket.settimeout(1.0)  # 设置超时以便检查关闭事件

            log_info(f"Server listening on {self.host}:{self.port}")

            # 等待客户端连接
            while True:
                try:
                    self.client_socket, self.client_address = self.socket.accept()
                    self.client_socket.settimeout(1.0)  # 设置接收超时
                    self.connected = True
                    # 记录连接时间
                    self.last_activity_time = time.time()
                    log_info(f"Client connected from: {self.client_address}")
                    return True
                except socket.timeout:
                    continue
                except Exception as e:
                    log_error(f"Error accepting connection: {e}")
                    return False

        except Exception as e:
            log_error(f"服务器启动失败: {e}")
            return False

    def handle_requests(self, running_callback):
        """处理客户端请求"""
        while running_callback():
            try:
                # 接收请求
                raw_len = self._recv_all(4)
                if not raw_len:
                    # 客户端断开连接
                    log_info("Client disconnected (no data received)")
                    if self.disconnect_callback:
                        self.disconnect_callback()
                    break
                data_len = struct.unpack(">I", raw_len)[0]

                serialized_data = self._recv_all(data_len)
                if not serialized_data:
                    # 客户端断开连接
                    log_info("Client disconnected (incomplete data)")
                    if self.disconnect_callback:
                        self.disconnect_callback()
                    break

                # 更新最后活动时间
                self.last_activity_time = time.time()

                request = pickle.loads(serialized_data)

                request_type = request.get("type")
                request_id = request.get("request_id", "unknown")

                if request_type in self.request_handlers:
                    response = self.request_handlers[request_type](request)
                    if response:
                        if isinstance(response, dict) and "request_id" not in response:
                            response["request_id"] = request_id
                        self._send_response(response)

            except socket.timeout:
                # 接收超时，继续等待
                continue
            except Exception as e:
                log_error(f"请求处理错误: {e}")
                traceback.print_exc()
                time.sleep(0.1)

    def _send_response(self, response: dict):
        """发送响应"""
        try:
            serialized_data = pickle.dumps(response)
            with self.lock:
                self.client_socket.sendall(struct.pack(">I", len(serialized_data)))
                self.client_socket.sendall(serialized_data)
        except Exception as e:
            log_error(f"发送响应失败: {e}")
            self.connected = False
            # 调用断开连接回调
            if self.disconnect_callback:
                self.disconnect_callback()

    def _recv_all(self, length: int) -> Optional[bytes]:
        """接收指定长度的数据"""
        data = b""
        while len(data) < length:
            try:
                packet = self.client_socket.recv(length - len(data))
                if not packet:
                    # 客户端关闭连接
                    return None
                data += packet
            except socket.timeout:
                # 重新引发超时异常，让上层处理
                raise
            except ConnectionResetError:
                return None
        return data

    def close(self):
        """关闭服务器"""
        with self.lock:
            if self.client_socket:
                try:
                    self.client_socket.close()
                except:
                    pass
                self.client_socket = None
            if self.socket:
                try:
                    self.socket.close()
                except:
                    pass
                self.socket = None
        self.connected = False
        self.client_address = None
        self.last_activity_time = None


def compress_image(image, quality=80):
    """压缩图像以减少网络传输"""
    if image is None:
        return None
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encoded_image = cv2.imencode(".jpg", image, encode_param)
    if result:
        return encoded_image
    return None


def decompress_image(compressed_data):
    """解压缩图像"""
    if compressed_data is None:
        return None
    return cv2.imdecode(compressed_data, cv2.IMREAD_COLOR)
