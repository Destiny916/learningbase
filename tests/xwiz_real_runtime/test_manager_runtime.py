import json

import xwiz_real_runtime.manager_runtime as manager_runtime
from xwiz_real_runtime.manager_runtime import deploy_selected, prepare_resolved_configs


class RecordingController:
    def __init__(self):
        self.calls = []

    def setup_config(self, client, server):
        self.calls.append((client, server))
        return True


class FakeRegistry:
    def __init__(self, base_path):
        self.config_base_path = base_path
        self.calls = []

    def resolve_config(self, model, task, mode):
        self.calls.append((model, task, mode))
        return {"mode": mode}, {"data_type": "real" if mode == 2 else "simulation"}


class FakeManager:
    def __init__(self, base_path):
        self.current_config = None
        self.config_registry = FakeRegistry(base_path)
        self.client_controller = RecordingController()


def test_deploy_uses_selected_task_mode_and_immediately_starts_pc2(tmp_path):
    task_dir = tmp_path / "tasks" / "7"
    task_dir.mkdir(parents=True)
    (task_dir / "task_config.json").write_text(json.dumps({"client_config": {"mode": 2}}))
    manager = FakeManager(tmp_path)

    assert deploy_selected(manager, model_id=3, task_id=7) is True
    assert manager.config_registry.calls == [(3, 7, 2)]
    assert manager.client_controller.calls == [({"mode": 2}, {"data_type": "real"})]
    assert manager.current_config == ({"mode": 2}, {"data_type": "real"})


def test_resolved_simulation_config_is_one_chunk_and_simulation_tagged():
    client, server = prepare_resolved_configs(
        {"max_steps": 600, "home_position": "unsafe"},
        {"data_type": "real"},
        mode=1,
    )

    assert client["mode"] == 1
    assert client["max_steps"] == 16
    assert client["home_position"] == ""
    assert server["data_type"] == "simulation"
    assert server["action_horizon"] == 16


def test_resolved_real_config_is_one_chunk_and_real_tagged():
    client, server = prepare_resolved_configs({}, {"data_type": "simulation"}, mode=2)

    assert client["mode"] == 2
    assert client["action_horizon"] == 16
    assert client["sample_factor"] == 1.0
    assert client["chunk_size_threshold"] == 0.0
    assert server["data_type"] == "real"
    assert server["action_horizon"] == 16


def test_resolved_continuous_real_config_preserves_continuous_execution():
    client, server = prepare_resolved_configs(
        {
            "execution_mode": "continuous",
            "max_steps": 16,
            "server_host": "192.168.20.21",
            "server_port": 8889,
        },
        {"data_type": "simulation"},
        mode=2,
    )

    assert client["execution_mode"] == "continuous"
    assert client["max_steps"] > 16
    assert client["server_host"] == "192.168.20.21"
    assert client["server_port"] == 8889
    assert server["data_type"] == "real"


def test_lerobot_act_metadata_maps_model_features_to_xwiz_names():
    cameras, groups = manager_runtime.lerobot_model_meta(
        {
            "input_features": {
                "observation.state": {"type": "STATE", "shape": [19]},
                "observation.images.cam_high_left": {
                    "type": "VISUAL",
                    "shape": [3, 360, 640],
                },
                "observation.images.cam_hand_left": {
                    "type": "VISUAL",
                    "shape": [3, 360, 640],
                },
                "observation.images.cam_hand_right": {
                    "type": "VISUAL",
                    "shape": [3, 360, 640],
                },
            },
            "output_features": {"action": {"type": "ACTION", "shape": [19]}},
        }
    )

    assert cameras == [
        "head_left_camera",
        "left_wrist_camera",
        "right_wrist_camera",
    ]
    assert groups == {
        "WAIST",
        "LEFT_ARM",
        "HEAD",
        "RIGHT_ARM",
        "LEFT_EEFGRIPPER",
        "RIGHT_EEFGRIPPER",
    }
