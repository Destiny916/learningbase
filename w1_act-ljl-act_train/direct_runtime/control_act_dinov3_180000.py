"""Control only the isolated W1 ACT-DINOv3 180000 PC1 client."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from act_async_infer_distributed_demo.scripts.manager.inference_client_controller import (  # noqa: E402
    InferenceClientController,
)


CONFIG_PATH = Path(__file__).with_name("client_runtime_180000.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "start", "stop"))
    parser.add_argument("--host", default="192.168.20.20")
    parser.add_argument("--port", type=int, default=8891)
    args = parser.parse_args()

    controller = InferenceClientController(args.host, args.port)
    if args.command == "status":
        result = controller.get_status()
    elif args.command == "stop":
        result = controller.stop()
    else:
        client = json.loads(CONFIG_PATH.read_text())
        server = {
            "data_type": "real",
            "protocol_version": 2,
            "action_horizon": 16,
            "inverse_gripper": False,
            "is_gripper_bool": False,
            "save_input": False,
        }
        result = controller.setup_config(client, server)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result else 1


if __name__ == "__main__":
    raise SystemExit(main())
