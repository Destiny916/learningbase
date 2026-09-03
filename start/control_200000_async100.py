"""Control the isolated ACT-200000 async100 PC1 client.

This controller only sends manager commands.  ``start`` still performs the
client's robot-health and observation checks; the pose check can be bypassed
only with the explicit command-line switch that was authorized for this W1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "w1_act-ljl-act_train"))

from act_async_infer_distributed_demo.scripts.manager.inference_client_controller import (  # noqa: E402
    InferenceClientController,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("status", "start", "stop"))
    parser.add_argument("--host", default="192.168.20.20")
    parser.add_argument("--port", type=int, default=8896)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("client_runtime_200000_async100.json")),
    )
    parser.add_argument("--skip-default-pose-check", action="store_true")
    args = parser.parse_args()

    controller = InferenceClientController(args.host, args.port)
    if args.command == "status":
        result = controller.get_status()
    elif args.command == "stop":
        result = controller.stop()
    else:
        client = json.loads(Path(args.config).read_text(encoding="utf-8"))
        client["mode"] = 2
        if args.skip_default_pose_check:
            client["skip_default_pose_check"] = True
        server = {
            "data_type": "real",
            "inverse_gripper": False,
            "is_gripper_bool": False,
            "save_input": False,
        }
        result = controller.setup_config(client, server)

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result else 1


if __name__ == "__main__":
    raise SystemExit(main())
