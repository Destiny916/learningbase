import socket
import time
import numpy as np
import struct
import threading
from collections import deque
import argparse
from act_async_infer_distributed_demo.scripts.utils_distributed import (
    log_info,
    log_error,
    log_warning,
    log_section
)
class NetworkClient:
    def __init__(self, server_host='localhost', server_port=8888, data_size_mb=0.3, frequency_hz=10):
        self.server_host = server_host
        self.server_port = server_port
        self.data_size_mb = data_size_mb
        self.frequency_hz = frequency_hz
        self.interval = 1.0 / frequency_hz
        
        # 统计数据
        self.bytes_sent = 0
        self.packets_sent = 0
        self.packets_received = 0
        self.start_time = 0
        self.running = False
        
        # 延迟统计
        self.rtt_latencies = deque(maxlen=1000)  # 往返延迟
        self.one_way_latencies = deque(maxlen=1000)  # 单程延迟（服务器处理）
        self.jitters = deque(maxlen=1000)  # 延迟抖动
        self.last_rtt = 0
        
        # 生成测试数据（0.3MB的随机数据）
        self.test_data = self._generate_test_data()
        
        # 网络连接
        self.sock = None
        
    def _generate_test_data(self):
        """生成测试数据"""
        # 0.3MB = 0.3 * 1024 * 1024 = 314572.8 bytes，取整数
        data_size = int(self.data_size_mb * 1024 * 1024)
        # 生成随机字节数据
        return np.random.bytes(data_size)
    
    def connect(self):
        """连接到服务器"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(5.0)
        
        log_info(f"连接到服务器 {self.server_host}:{self.server_port}...")
        self.sock.connect((self.server_host, self.server_port))
        log_info("连接成功!")
        
        # 设置更短的超时用于接收ACK
        self.sock.settimeout(1.0)
    
    def start(self):
        """开始发送数据"""
        if not self.sock:
            self.connect()
        
        self.running = True
        self.start_time = time.time()
        
        # 启动统计线程
        stats_thread = threading.Thread(target=self._stats_worker, daemon=True)
        stats_thread.start()
        
        # 发送线程
        send_thread = threading.Thread(target=self._send_worker)
        send_thread.start()
        
        try:
            send_thread.join()
        except KeyboardInterrupt:
            log_warning("\n正在停止客户端...")
            self.stop()
    
    def _send_worker(self):
        """发送数据的工作线程"""
        next_send_time = time.time()
        
        while self.running:
            try:
                # 等待到下一个发送时间
                current_time = time.time()
                if current_time < next_send_time:
                    time.sleep(next_send_time - current_time)
                
                # 记录发送时间戳
                send_timestamp = time.time()
                
                # 创建数据包：时间戳(8字节) + 数据大小(8字节) + 实际数据
                header = struct.pack('dQ', send_timestamp, len(self.test_data))
                packet = header + self.test_data
                
                # 发送数据
                start_send = time.perf_counter()
                self.sock.sendall(packet)
                send_duration = (time.perf_counter() - start_send) * 1000  # ms
                
                # 接收ACK（服务器返回的时间戳）
                ack_data = self._receive_all(self.sock, 8)  # 8字节时间戳
                if not ack_data:
                    log_warning("接收ACK失败")
                    break
                
                # 记录接收时间
                receive_time = time.time()
                server_receive_time = struct.unpack('d', ack_data)[0]
                
                # 计算延迟
                rtt = (receive_time - send_timestamp) * 1000  # 往返延迟(ms)
                one_way_server = (server_receive_time - send_timestamp) * 1000  # 服务器单程延迟(ms)
                network_rtt = rtt - (receive_time - server_receive_time) * 1000  # 网络RTT估计
                
                # 更新统计
                with threading.Lock():
                    self.bytes_sent += len(self.test_data)
                    self.packets_sent += 1
                    self.packets_received += 1
                    
                    self.rtt_latencies.append(rtt)
                    self.one_way_latencies.append(one_way_server)
                    
                    # 计算抖动（当前延迟与平均延迟的差值）
                    if len(self.rtt_latencies) > 1:
                        jitter = abs(rtt - self.last_rtt)
                        self.jitters.append(jitter)
                    self.last_rtt = rtt
                
                # 输出单次测量结果
                log_info(f"包 #{self.packets_sent}: RTT={rtt:.2f}ms, "
                      f"服务器延迟={one_way_server:.2f}ms, "
                      f"发送耗时={send_duration:.2f}ms")
                
                # 计算下一个发送时间
                next_send_time = send_timestamp + self.interval
                
            except socket.timeout:
                log_warning("发送或接收超时")
                break
            except Exception as e:
                log_error(f"发送数据错误: {e}")
                break
    
    def _receive_all(self, sock, n):
        """接收指定数量的数据"""
        data = bytearray()
        while len(data) < n:
            try:
                packet = sock.recv(n - len(data))
                if not packet:
                    return None
                data.extend(packet)
            except socket.timeout:
                return None
        return bytes(data)
    
    def _stats_worker(self):
        """统计信息工作线程"""
        while self.running:
            time.sleep(5.0)  # 每5秒输出一次统计
            
            if self.packets_sent == 0:
                continue
            
            elapsed = time.time() - self.start_time
            mb_sent = self.bytes_sent / (1024 * 1024)
            
            if elapsed > 0:
                throughput = mb_sent / elapsed  # MB/s
                packet_rate = self.packets_sent / elapsed  # packets/s
                
                # 延迟统计
                if self.rtt_latencies:
                    avg_rtt = np.mean(self.rtt_latencies)
                    min_rtt = np.min(self.rtt_latencies)
                    max_rtt = np.max(self.rtt_latencies)
                    std_rtt = np.std(self.rtt_latencies)
                    packet_loss = (self.packets_sent - self.packets_received) / self.packets_sent * 100
                    
                    # 抖动统计
                    if self.jitters:
                        avg_jitter = np.mean(self.jitters)
                        max_jitter = np.max(self.jitters)
                    else:
                        avg_jitter = max_jitter = 0
                    
                    log_section(symbol="*")
                    log_info("客户端统计信息:")
                    log_info(f"  运行时间: {elapsed:.1f}秒")
                    log_info(f"  发送数据量: {mb_sent:.2f} MB")
                    log_info(f"  发送包数: {self.packets_sent}")
                    log_info(f"  接收包数: {self.packets_received}")
                    log_info(f"  丢包率: {packet_loss:.2f}%")
                    log_info(f"  吞吐量: {throughput:.2f} MB/s ({throughput*8:.1f} Mbps)")
                    log_info(f"  实际频率: {packet_rate:.1f} packets/s (目标: {self.frequency_hz} Hz)")
                    log_info(f"  往返延迟(RTT)统计(ms):")
                    log_info(f"    平均: {avg_rtt:.2f}")
                    log_info(f"    最小: {min_rtt:.2f}")
                    log_info(f"    最大: {max_rtt:.2f}")
                    log_info(f"    标准差: {std_rtt:.2f}")
                    log_info(f"  抖动统计(ms):")
                    log_info(f"    平均: {avg_jitter:.2f}")
                    log_info(f"    最大: {max_jitter:.2f}")
                    
                    # 延迟分布
                    if len(self.rtt_latencies) >= 10:
                        percentiles = np.percentile(self.rtt_latencies, [50, 90, 95, 99])
                        log_info(f"  延迟百分位数(ms):")
                        log_info(f"    50% (中位数): {percentiles[0]:.2f}")
                        log_info(f"    90%: {percentiles[1]:.2f}")
                        log_info(f"    95%: {percentiles[2]:.2f}")
                        log_info(f"    99%: {percentiles[3]:.2f}")
                    log_section(symbol="*")
    
    def stop(self):
        """停止客户端"""
        self.running = False
        if self.sock:
            self.sock.close()
        log_info("客户端已停止")

def main():
    parser = argparse.ArgumentParser(description='网络延迟测试客户端')
    parser.add_argument('--host', default='localhost', help='服务器地址')
    parser.add_argument('--port', type=int, default=8888, help='服务器端口')
    parser.add_argument('--size', type=float, default=0.3, help='数据大小(MB)')
    parser.add_argument('--freq', type=float, default=10.0, help='发送频率(Hz)')
    parser.add_argument('--duration', type=float, default=60.0, help='测试持续时间(秒)')
    
    args = parser.parse_args()
    
    client = NetworkClient(
        server_host=args.host,
        server_port=args.port,
        data_size_mb=args.size,
        frequency_hz=args.freq
    )
    
    # 设置测试持续时间
    if args.duration > 0:
        def timer():
            time.sleep(args.duration)
            client.stop()
        
        timer_thread = threading.Thread(target=timer, daemon=True)
        timer_thread.start()
    
    try:
        client.start()
    except KeyboardInterrupt:
        log_info("\n测试被用户中断")
    finally:
        client.stop()
        
        # 输出最终统计
        if client.packets_sent > 0:
            log_section(symbol="*")
            log_info("最终测试结果:")
            log_info(f"总发送包数: {client.packets_sent}")
            log_info(f"总接收包数: {client.packets_received}")
            if client.rtt_latencies:
                avg_rtt = np.mean(client.rtt_latencies)
                log_info(f"平均RTT延迟: {avg_rtt:.2f} ms")
                log_info(f"理论最大频率: {1000/avg_rtt:.1f} Hz (基于RTT)")
            log_section(symbol="*")

if __name__ == "__main__":
    main()
