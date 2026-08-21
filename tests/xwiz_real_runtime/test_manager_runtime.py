import json

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
    assert client["max_steps"] == 100
    assert client["home_position"] == ""
    assert server["data_type"] == "simulation"
    assert server["action_horizon"] == 100


def test_resolved_real_config_is_one_chunk_and_real_tagged():
    client, server = prepare_resolved_configs({}, {"data_type": "simulation"}, mode=2)

    assert client["mode"] == 2
    assert client["action_horizon"] == 100
    assert client["sample_factor"] == 1.0
    assert client["chunk_size_threshold"] == 0.0
    assert server["data_type"] == "real"
    assert server["action_horizon"] == 100
