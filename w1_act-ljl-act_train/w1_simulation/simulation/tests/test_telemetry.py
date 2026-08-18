import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from w1_simulation.simulation.telemetry import (
    CAMERA_STREAMS,
    RerunTelemetry,
    build_act_blueprint,
    gpu_metrics,
    sha256_array,
    sha256_file,
    write_json,
)


def _layout_node(kind):
    return lambda *args, **kwargs: {"kind": kind, "args": args, "kwargs": kwargs}


def test_json_and_sha256_helpers_are_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "report.json"
    write_json(path, {"z": 1, "a": [2, 3]})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": [2, 3], "z": 1}
    assert path.read_text(encoding="utf-8").index('"a"') < path.read_text(encoding="utf-8").index('"z"')
    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    assert sha256_array(image) == hashlib.sha256(image.tobytes()).hexdigest()


def test_default_camera_layout_matches_act_observations() -> None:
    assert CAMERA_STREAMS == ("cam_high_left", "cam_hand_left", "cam_hand_right")


def test_act_blueprint_places_robot_left_and_stacks_camera_views_right() -> None:
    rrb = SimpleNamespace(
        Spatial3DView=_layout_node("Spatial3DView"),
        Spatial2DView=_layout_node("Spatial2DView"),
        Horizontal=_layout_node("Horizontal"),
        Vertical=_layout_node("Vertical"),
        TimePanel=_layout_node("TimePanel"),
        Blueprint=_layout_node("Blueprint"),
    )

    blueprint = build_act_blueprint(rrb, CAMERA_STREAMS)

    horizontal = blueprint["args"][0]
    time_panel = blueprint["args"][1]
    robot, video_column = horizontal["args"]
    assert blueprint["kind"] == "Blueprint"
    assert blueprint["kwargs"] == {
        "auto_layout": False,
        "auto_views": False,
        "collapse_panels": True,
    }
    assert time_panel["kind"] == "TimePanel"
    assert time_panel["kwargs"] == {"state": "collapsed", "timeline": "sim_time"}
    assert horizontal["kind"] == "Horizontal"
    assert horizontal["kwargs"]["column_shares"] == [2.0, 1.0]
    assert robot["kind"] == "Spatial3DView"
    assert robot["kwargs"]["origin"] == "world"
    assert video_column["kind"] == "Vertical"
    assert video_column["kwargs"]["row_shares"] == [1.0, 1.0, 1.0]
    assert [view["kwargs"]["name"] for view in video_column["args"]] == list(CAMERA_STREAMS)


def test_eye_and_both_blueprints_show_mujoco_camera_view() -> None:
    rrb = SimpleNamespace(
        Spatial3DView=_layout_node("Spatial3DView"),
        Spatial2DView=_layout_node("Spatial2DView"),
        Horizontal=_layout_node("Horizontal"),
        Vertical=_layout_node("Vertical"),
        TimePanel=_layout_node("TimePanel"),
        Blueprint=_layout_node("Blueprint"),
    )

    eye = build_act_blueprint(rrb, CAMERA_STREAMS, view_mode="eye")
    both = build_act_blueprint(rrb, CAMERA_STREAMS, view_mode="both")

    eye_layout = eye["args"][0]
    both_layout = both["args"][0]
    eye_view, eye_camera_column = eye_layout["args"]
    both_primary, both_camera_column = both_layout["args"]
    robot_view, both_eye_view = both_primary["args"]
    assert eye_layout["kind"] == "Horizontal"
    assert eye_view["kind"] == "Spatial2DView"
    assert eye_view["kwargs"]["origin"].endswith("/w1_eye_camera")
    assert eye_view["kwargs"]["contents"].endswith("/w1_eye_camera/rgb")
    assert eye_camera_column["kind"] == "Vertical"
    assert [view["kwargs"]["name"] for view in eye_camera_column["args"]] == list(CAMERA_STREAMS)
    assert both_layout["kind"] == "Horizontal"
    assert both_primary["kind"] == "Vertical"
    assert robot_view["kind"] == "Spatial3DView"
    assert both_eye_view["kind"] == "Spatial2DView"
    assert both_camera_column["kind"] == "Vertical"
    assert [view["kwargs"]["name"] for view in both_camera_column["args"]] == list(CAMERA_STREAMS)


def test_log_state_emits_qpos_target_action_metrics_and_arbitrary_images() -> None:
    events = []
    stream = SimpleNamespace(
        set_time=lambda timeline, **value: events.append(("time", timeline, value)),
        log=lambda entity, value, static=False: events.append((entity, value, static)),
    )

    def scalar(value):
        return "scalars", np.asarray(value).tolist()

    class FakeImage:
        def __init__(self, value, color_model):
            self.shape = np.asarray(value).shape
            self.color_model = color_model

        def compress(self, jpeg_quality):
            return ("jpeg", self.shape, self.color_model, jpeg_quality)

    image = FakeImage
    telemetry = RerunTelemetry.__new__(RerunTelemetry)
    telemetry.enabled = True
    telemetry.streams = [stream]
    telemetry.link_entities = []
    telemetry.active_qpos_ids = list(range(29))
    telemetry._rr = SimpleNamespace(
        Scalars=scalar,
        Image=image,
        TextDocument=lambda value: ("text", value),
    )
    telemetry.log_state(
        step=4,
        data=SimpleNamespace(qpos=np.arange(29, dtype=np.float64)),
        target=np.ones(29),
        action=np.full(29, 0.5),
        metrics={"latency_ms": 3.0},
        images={"external_debug_camera": np.zeros((2, 3, 3), dtype=np.uint8)},
        time_seconds=0.125,
    )
    entities = [event[0] for event in events]
    assert entities == [
        "time",
        "time",
        "joints/qpos",
        "joints/target",
        "joints/action",
        "metrics/latency_ms",
        "observation/external_debug_camera",
        "observation/external_debug_camera/sha256",
    ]
    assert events[:2] == [
        ("time", "step", {"sequence": 4}),
        ("time", "sim_time", {"duration": 0.125}),
    ]
    assert events[-2][1] == ("jpeg", (2, 3, 3), "RGB", 85)
    assert events[-1][1] == ("text", hashlib.sha256(bytes(18)).hexdigest())


def test_gpu_metrics_parses_the_first_gpu(monkeypatch) -> None:
    completed = SimpleNamespace(stdout="73, 2048, 61\n10, 512, 45\n")
    monkeypatch.setattr(
        "w1_simulation.simulation.telemetry.subprocess.run", lambda *args, **kwargs: completed
    )
    assert gpu_metrics() == {
        "gpu/utilization_percent": 73.0,
        "gpu/memory_used_mb": 2048.0,
        "gpu/temperature_c": 61.0,
    }


def test_gpu_metrics_is_empty_when_nvidia_smi_is_unavailable(monkeypatch) -> None:
    def unavailable(*args, **kwargs):
        raise OSError("not installed")

    monkeypatch.setattr("w1_simulation.simulation.telemetry.subprocess.run", unavailable)
    assert gpu_metrics() == {}
