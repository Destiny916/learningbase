import argparse
import rclpy
import time
from act_async_infer_distributed_demo.scripts.client.robot_client_with_kingfisher import (
    OptimizedRobotClient,
)
from act_async_infer_distributed_demo.scripts.inference_config import ClientConfig
from act_async_infer_distributed_demo.scripts.utils_distributed import (
    log_info,
    log_error,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot Client")
    parser.add_argument(
        "--config", type=str, default="",
        help="客户端 JSON 配置文件路径"
    )
    parser.add_argument(
        "--service", action="store_true",
        help="以服务模式运行，等待Manager下发"
    )
    parser.add_argument(
        "--voice_control", action="store_true",
        help="服务模式 + 语音控制"
    )
    args = parser.parse_args()

    client_config = ClientConfig.from_json_file(args.config)

    rclpy.init()

    client_config.service = args.service
    client = OptimizedRobotClient(client_config)
    try:
        if args.service or args.voice_control:
            log_info("Starting robot client in service mode...")
            client._start_manager_listener()
            while rclpy.ok():
                time.sleep(1.0)
        else:
            log_info("Starting robot client in direct mode...")
            client.start()
    except KeyboardInterrupt:
        log_error("Client interrupted")
    finally:
        client.stop()
        rclpy.shutdown()
