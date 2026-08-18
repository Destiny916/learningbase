import socket
import threading
import time
import numpy as np
from collections import deque
import struct
from act_async_infer_distributed_demo.scripts.utils_distributed import (
    log_info,
    log_error,
    log_warning,
    log_section
)

class NetworkServer:
    def __init__(self, host='0.0.0.0', port=8888):
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = False
        
        # 统计信息
        self.bytes_received = 0
        self.packets_received = 0
        self.start_time = 0
        self.stats_lock = threading.Lock()
        
        # 延迟统计
        self.latencies = deque(maxlen=1000)
        self.receive_times = deque(maxlen=1000)
        
    def start(self):
        """启动服务器"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.server_socket.settimeout(1.0)
        
        log_info(f"服务器启动在 {self.host}:{self.port}")
        log_info("等待客户端连接...")
        
        self.running = True
        self.start_time = time.time()
        
        # 启动统计线程
        stats_thread = threading.Thread(target=self._stats_worker, daemon=True)
        stats_thread.start()
        
        # 接受客户端连接
        while self.running:
            try:
                client_socket, addr = self.server_socket.accept()
                log_info(f"客户端连接来自: {addr}")
                
                # 为每个客户端创建处理线程
                client_thread = threading.Thread(
                    target=self._handle_client, 
                    args=(client_socket, addr),
                    daemon=True
                )
                client_thread.start()
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    log_error(f"接受连接错误: {e}")
    
    def _handle_client(self, client_socket, addr):
        """处理客户端连接"""
        client_socket.settimeout(5.0)
        
        try:
            while self.running:
                # 接收数据包头（包含数据大小和时间戳）
                header = self._receive_all(client_socket, 16)  # 8字节时间戳 + 8字节数据大小
                if not header:
                    break
                
                # 解析包头
                timestamp = struct.unpack('d', header[:8])[0]  # 客户端发送时间戳
                data_size = struct.unpack('Q', header[8:])[0]  # 数据大小
                
                # 接收实际数据
                data = self._receive_all(client_socket, data_size)
                if not data:
                    break
                
                receive_time = time.time()
                
                # 更新统计
                with self.stats_lock:
                    self.bytes_received += data_size
                    self.packets_received += 1
                    self.receive_times.append(receive_time)
                    
                    # 计算单程延迟（服务器收到时间 - 客户端发送时间）
                    one_way_latency = (receive_time - timestamp) * 1000  # ms
                    self.latencies.append(one_way_latency)
                
                # 发送确认（包含接收时间戳）
                ack_data = struct.pack('d', receive_time)  # 服务器接收时间戳
                client_socket.sendall(ack_data)
                
                # 可选：验证数据完整性（如果是numpy数组）
                if len(data) == 0.3 * 1024 * 1024:  # 0.3MB
                    # 如果是随机数据，可以跳过验证
                    pass
                
        except socket.timeout:
            log_warning(f"客户端 {addr} 超时")
        except Exception as e:
            log_warning(f"处理客户端 {addr} 错误: {e}")
        finally:
            client_socket.close()
            log_warning(f"客户端 {addr} 断开连接")
    
    def _receive_all(self, sock, n):
        """接收指定数量的数据"""
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return bytes(data)
    
    def _stats_worker(self):
        """统计信息工作线程"""
        while self.running:
            time.sleep(5.0)  # 每5秒输出一次统计
            
            with self.stats_lock:
                if self.packets_received == 0:
                    continue
                
                elapsed = time.time() - self.start_time
                mb_received = self.bytes_received / (1024 * 1024)
                
                if elapsed > 0:
                    throughput = mb_received / elapsed  # MB/s
                    packet_rate = self.packets_received / elapsed  # packets/s
                    
                    # 延迟统计
                    if self.latencies:
                        avg_latency = np.mean(self.latencies)
                        min_latency = np.min(self.latencies)
                        max_latency = np.max(self.latencies)
                        std_latency = np.std(self.latencies)
                    else:
                        avg_latency = min_latency = max_latency = std_latency = 0
                    
                    log_section(symbol="*")
                    log_info("服务器统计信息:")
                    log_info(f"  运行时间: {elapsed:.1f}秒")
                    log_info(f"  接收数据量: {mb_received:.2f} MB")
                    log_info(f"  接收包数: {self.packets_received}")
                    log_info(f"  吞吐量: {throughput:.2f} MB/s ({throughput*8:.1f} Mbps)")
                    log_info(f"  包速率: {packet_rate:.1f} packets/s")
                    log_info(f"  单程延迟统计(ms):")
                    log_info(f"    平均: {avg_latency:.2f}")
                    log_info(f"    最小: {min_latency:.2f}")
                    log_info(f"    最大: {max_latency:.2f}")
                    log_info(f"    标准差: {std_latency:.2f}")
                    log_section(symbol="*")
    
    def stop(self):
        """停止服务器"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        log_info("服务器已停止")

if __name__ == "__main__":
    server = NetworkServer(host='0.0.0.0', port=8888)
    
    try:
        server.start()
    except KeyboardInterrupt:
        log_warning("\n正在停止服务器...")
        server.stop()
