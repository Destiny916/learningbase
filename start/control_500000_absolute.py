"""Control the PC1 W1 client without XWiz GUI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from act_async_infer_distributed_demo.scripts.manager.inference_client_controller import (  # noqa: E402
    InferenceClientController,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("status", "start", "stop"))
    parser.add_argument("--host", default="192.168.20.20")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--config", default=str(Path(__file__).with_name("client_runtime_pc2.json")))
    args = parser.parse_args()
    controller = InferenceClientController(args.host, args.port)
    if args.command == "status":
        result = controller.get_status()
    elif args.command == "stop":
        result = controller.stop()
    else:
        with open(args.config, encoding="utf-8") as stream:
            client = json.load(stream)
        client["mode"] = 2
        server = {"data_type": "real", "inverse_gripper": False, "is_gripper_bool": False, "save_input": False}
        result = controller.setup_config(client, server)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result else 1


if __name__ == "__main__":
    raise SystemExit(main())
