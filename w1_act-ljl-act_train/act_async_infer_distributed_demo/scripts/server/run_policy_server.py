import argparse
import signal
import sys
from act_async_infer_distributed_demo.scripts.server.policy_server_pure_inference import (
    PolicyServerPureInference,
)
from act_async_infer_distributed_demo.scripts.inference_config import ServerConfig
from act_async_infer_distributed_demo.scripts.utils_distributed import log_info
def signal_handler(sig, frame):
    log_info("\nReceived interrupt signal, shutting down server...")
    if "server" in globals():
        server.stop()
    sys.exit(0)
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    parser = argparse.ArgumentParser(description="DexforceVLA Policy Server")
    parser.add_argument("--config", type=str, required=True,
                        help="Server JSON 配置文件路径")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="服务器监听地址")
    parser.add_argument("--port", type=int, default=8889,
                        help="服务器监听端口")

    args = parser.parse_args()
    
    server_cfg = ServerConfig.from_json_file(args.config)
    
    server = PolicyServerPureInference(server_cfg, args.host, args.port)
    log_info(f"Starting DexforceVLA policy server on {args.host}:{args.port}")
    log_info("Server will automatically reset when client disconnects")
    log_info("Press Ctrl+C to stop the server")