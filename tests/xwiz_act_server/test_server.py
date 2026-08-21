import numpy as np

from xwiz_act_server.server import XWizActServerApp

from test_contract import observation_request


class FakeRuntime:
    def __init__(self):
        self.calls = []
        self.reset_calls = 0

    def predict(self, observation):
        self.calls.append(observation)
        return np.arange(100 * 19, dtype=np.float32).reshape(100, 19)

    def reset(self):
        self.reset_calls += 1


def setup_request(data_type="simulation"):
    return {
        "type": "SETUP_CONFIG",
        "request_id": 1,
        "config": {"data_type": data_type},
    }


def test_simulation_lifecycle_produces_legacy_action_response():
    runtime = FakeRuntime()
    app = XWizActServerApp(runtime)

    assert app.handle({"type": "STATUS"})["state"] == "idle"
    setup = app.handle(setup_request())
    assert setup == {"success": True, "state": "running", "request_id": 1}
    assert runtime.reset_calls == 1

    request = observation_request()
    request.update({"type": "observation", "request_id": 2})
    assert app.handle(request) == {
        "status": "received",
        "inferred": True,
        "request_id": 2,
    }
    assert len(runtime.calls) == 1

    reply = app.handle({"type": "get_actions", "request_id": 3})
    assert reply["status"] == "success"
    assert reply["request_id"] == 3
    assert reply["timestamp"] == 12.5
    assert reply["timestep"] == 8
    assert reply["actions"]["qpos"]["left_armqpos"].shape == (100, 7)


def test_setup_rejects_real_mode():
    app = XWizActServerApp(FakeRuntime())
    reply = app.handle(setup_request(data_type="real"))
    assert reply["success"] is False
    assert reply["state"] == "error"
    assert "simulation" in reply["error"]


def test_get_actions_is_pending_before_inference():
    app = XWizActServerApp(FakeRuntime())
    app.handle(setup_request())
    reply = app.handle({"type": "get_actions"})
    assert reply["status"] == "pending"


def test_observation_without_start_flag_is_received_but_not_inferred():
    runtime = FakeRuntime()
    app = XWizActServerApp(runtime)
    app.handle(setup_request())
    request = observation_request()
    request.update({"type": "observation", "start_infer": False})
    assert app.handle(request)["inferred"] is False
    assert runtime.calls == []


def test_stop_clears_actions_and_returns_idle():
    app = XWizActServerApp(FakeRuntime())
    app.handle(setup_request())
    request = observation_request()
    request.update({"type": "observation"})
    app.handle(request)

    reply = app.handle({"type": "STOP"})
    assert reply == {"success": True, "state": "idle"}
    assert app.handle({"type": "get_actions"})["status"] == "pending"
