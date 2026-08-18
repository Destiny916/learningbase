import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


VIDEO_PATH = (
    Path(__file__).parents[1]
    / "third_party/starvla_runtime/starVLA/dataloader/gr00t_lerobot/video.py"
)


def _load_video_module(monkeypatch):
    fake_av = ModuleType("av")
    fake_av.open = lambda path: None
    fake_cv2 = ModuleType("cv2")
    fake_torch = ModuleType("torch")
    fake_torchvision = ModuleType("torchvision")
    fake_torchvision.io = SimpleNamespace()
    fake_torchvision.set_video_backend = lambda backend: None
    for name, module in {
        "av": fake_av,
        "cv2": fake_cv2,
        "torch": fake_torch,
        "torchvision": fake_torchvision,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("video_under_test", VIDEO_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_open_pyav_video_limits_ffmpeg_to_one_thread(monkeypatch):
    video = _load_video_module(monkeypatch)
    codec_context = SimpleNamespace(thread_count=0)
    stream = SimpleNamespace(codec_context=codec_context)
    container = SimpleNamespace(
        streams=SimpleNamespace(video=[stream]),
        close=lambda: None,
    )
    monkeypatch.setattr(video.av, "open", lambda path: container)

    opened_container, opened_stream = video._open_pyav_single_thread("sample.mp4")

    assert opened_container is container
    assert opened_stream is stream
    assert codec_context.thread_count == 1


def test_all_direct_pyav_open_calls_are_centralized():
    source = VIDEO_PATH.read_text()

    assert source.count("av.open(") == 1
    assert source.count("_open_pyav_single_thread(video_path)") == 3
