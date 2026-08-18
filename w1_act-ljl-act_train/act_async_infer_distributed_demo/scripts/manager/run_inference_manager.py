#!/usr/bin/env python3
import rclpy
from rclpy.executors import MultiThreadedExecutor
import signal
import sys
from act_async_infer_distributed_demo.scripts.manager.inference_manager import InferenceManager
from act_async_infer_distributed_demo.scripts.utils_distributed import (
    log_info,
    log_warning
)
import argparse

def signal_handler(sig, frame):
    log_warning("\nReceived interrupt signal, shutting down...")
    rclpy.shutdown()
    sys.exit(0)

def main():
    rclpy.init()
    parser = argparse.ArgumentParser(
        description="DexforceVLA Inference manager connect to client."
    )
    parser.add_argument("--client_host", type=str, default="0.0.0.0", help="Client host")
    parser.add_argument("--client_port", type=int, default=8890, help="Client port")
    args = parser.parse_args()

    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 创建InferenceManager节点
    inference_manager = InferenceManager(args)
    
    # 使用多线程执行器
    executor = MultiThreadedExecutor()
    executor.add_node(inference_manager)
    
    log_info("InferenceManager started")
    log_info("Services available:")
    log_info("  /inference/start_inference")
    log_info("  /inference/stop_inference")
    log_info("  /inference/get_model_info")
    log_info("Publishing /inference/status")
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        log_warning("Shutting down...")
    finally:
        inference_manager.destroy()
        executor.shutdown()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
