import socket
import pickle
import struct
import threading
import time
from typing import Callable, Optional
import traceback
from act_async_infer_distributed_demo.scripts.utils_distributed import (
    log_info,
    log_error,
    log_warning,
)
import json


class NetworkClient:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        self.lock = threading.Lock()

    def connect(self):
        """连接到服务器"""
        max_retries = 5  # 最大重连次数
        retry_delay = 1  # 重连延迟（秒）
        self.close()
        for attempt in range(max_retries):
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # 禁用Nagle算法
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.socket.setsockopt(socket.SOL_TCP, socket.TCP_QUICKACK, 1)

                # 优化socket缓冲区
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16777216)
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16777216)
                self.socket.settimeout(None)
                self.socket.connect((self.host, int(self.port)))
                self.connected = True
                log_info(f"Connected to server at {self.host}:{self.port}")
                return True
            except Exception as e:
                log_warning(f"连接失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                elif self.connected == False:
                    log_error("Connect server failed")
        
        return False


    def send_request(self, request_type: str, data: dict = None) -> Optional[dict]:
        """发送请求并等待响应"""
        if not self.connected:
            return None

        with self.lock:
            try:
                # 准备请求数据
                request = {"type": request_type}
                if data:
                    request.update(data)

                # 序列化并发送
                serialized_data = pickle.dumps(request)
                self.socket.sendall(struct.pack(">I", len(serialized_data)))
                self.socket.sendall(serialized_data)
                obs_size_bytes = len(serialized_data)

                # 接收响应
                raw_len = self._recv_all(4)
                if not raw_len:
                    self.connected = False
                    return None
                data_len = struct.unpack(">I", raw_len)[0]

                # 数据长度验证
                if data_len > 1024 * 1024:  # 1MB限制
                    log_error("Response data too large")
                    return None

                serialized_response = self._recv_all(data_len)

                if not serialized_response:
                    self.connected = False
                    return None

                return pickle.loads(serialized_response)

            except Exception as e:
                traceback.print_exc()
                log_warning(f"Network: {e}")
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
        self.setup_config_mode = False
        
    def register_handler(self, request_type: str, handler):
        """注册请求处理器"""
        self.request_handlers[request_type] = handler

    def set_disconnect_callback(self, callback):
        """设置客户端断开连接回调"""
        self.disconnect_callback = callback

    def set_setup_config_mode(self, enabled: bool):
        """设置配置更新模式"""
        self.setup_config_mode = enabled
        if enabled:
            self._close_listening_socket()
    
    def _close_listening_socket(self):
        """关闭监听套接字"""
        with self.lock:
            if self.socket:
                try:
                    self.socket.close()
                except:
                    pass
                self.socket = None

    def start(self):
        """启动服务器并等待客户端连接"""
        try:

            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # 设置SO_LINGER以便快速关闭
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, 
                                 struct.pack('ii', 1, 0))

            self.socket.bind((self.host, int(self.port)))
            self.socket.listen(1)
            self.socket.settimeout(None)  # 设置超时以便检查关闭事件

            log_info(f"Server listening on {self.host}:{self.port}")

            # 等待客户端连接
            while True:
                try:
                    self.client_socket, self.client_address = self.socket.accept()
                    self.client_socket.settimeout(None)  # 设置接收超时
                    self.client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    self.client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)
                    self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16777216)
                    self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16777216)
                    self.connected = True
                    # 记录连接时间
                    self.last_activity_time = time.time()
                    log_info(f"Client connected from: {self.client_address}")
                    # 如果处于配置更新模式，关闭监听套接字
                    if self.setup_config_mode:
                        self._close_listening_socket()
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

                if request_type in self.request_handlers:
                    response = self.request_handlers[request_type](request)
                    if response:
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
        # 重置配置更新模式（连接生命周期结束）
        self.setup_config_mode = False



class SimpleJsonTcpServer:
    """轻量级 JSON TCP 服务器：bind/listen/accept"""
    def __init__(self, host: str, port: int, timeout: float = 1.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None
        self._running = False
    
    @staticmethod
    def create_listen_socket(host: str, port: int, 
                              timeout: float = 1.0,
                              backlog: int = 1) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(backlog)
        sock.settimeout(timeout)
        return sock
    
    @staticmethod
    def recv_exact(sock: socket.socket, n: int, timeout: Optional[float] = None) -> Optional[bytes]:
        """从 socket 接收 n 字节，超时/断开返回 None。"""
        data = b""
        if timeout is not None:
            sock.settimeout(timeout)
        try:
            while len(data) < n:
                packet = sock.recv(n - len(data))
                if not packet:
                    return None
                data += packet
            return data
        except socket.timeout:
            return None
    
    @staticmethod
    def send_json(sock: socket.socket, obj: dict) -> None:
        """发送 JSON 对象（4字节长度前缀）。"""
        payload = json.dumps(obj).encode("utf-8")
        sock.sendall(len(payload).to_bytes(4, "big"))
        sock.sendall(payload)
    
    @staticmethod
    def recv_json(sock: socket.socket) -> Optional[dict]:
        """接收 JSON 对象，失败返回 None。"""
        raw = SimpleJsonTcpServer.recv_exact(sock, 4)
        if not raw:
            return None
        length = int.from_bytes(raw, "big")
        data = SimpleJsonTcpServer.recv_exact(sock, length)
        if not data:
            return None
        return json.loads(data.decode("utf-8"))
    
    def start_in_thread(self, handler: Callable[[dict], dict],
                        daemon: bool = True) -> threading.Thread:
        """创建监听 socket 并在 daemon 线程中运行 accept 循环。"""
        self._sock = self.create_listen_socket(self.host, self.port, self.timeout)
        self._running = True
        thread = threading.Thread(
            target=self._accept_loop, args=(handler,), daemon=daemon
        )
        thread.start()
        log_info(f"JSON server listening on {self.host}:{self.port}")
        return thread
    
    def _accept_loop(self, handler: Callable[[dict], dict]):
        while self._running:
            try:
                conn, addr = self._sock.accept()
                log_info(f"Manager connected from {addr}")
                self._handle_one(conn, handler)
            except socket.timeout:
                continue
            except OSError:
                break
    
    def _handle_one(self, conn: socket.socket, handler: Callable[[dict], dict]):
        try:
            while True:
                msg = self.recv_json(conn)
                if msg is None:
                    break
                resp = handler(msg)
                self.send_json(conn, resp)
        except Exception:
            traceback.print_exc()
        finally:
            conn.close()
    
    def close(self):
        """关闭监听 socket。"""
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

