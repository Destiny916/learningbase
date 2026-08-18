from pathlib import Path


CONFIG = Path("turbovla/models/configuration.py")
VISION_ENCODER = Path("turbovla/models/vision_encoder.py")
CORE = Path("turbovla/models/turbovla.py")
FRAMEWORK = Path("third_party/starvla_runtime/starVLA/model/framework/VLM4A/TurboVLA.py")


def test_vision_config_exposes_a_validated_temporal_window_size():
    source = CONFIG.read_text()

    assert "temporal_window_size: int = 1" in source
    assert "vision.temporal_window_size < 1" in source


def test_vision_path_preserves_time_view_and_patch_axes_until_flattening():
    source = VISION_ENCODER.read_text()
    core = CORE.read_text()

    assert "pixel_values must be [B,V,3,H,W] or [B,T,V,3,H,W]" in source
    assert "batch_size, time_steps, num_views = pixel_values.shape[:3]" in source
    assert "tokens.view(batch_size, time_steps, num_views" in source
    assert "self.time_embedding" in core
    assert "return self._position_visual_tokens(tokens).flatten(1, 3)" in core


def test_framework_requires_two_complete_time_groups_for_t2():
    source = FRAMEWORK.read_text()

    assert "self.temporal_window_size = int(fw.vision.get(\"temporal_window_size\", 1))" in source
    assert "def _as_temporal_view_list" in source
    assert "expected {self.temporal_window_size} image time steps" in source
    assert "len(examples), self.temporal_window_size, self.num_views, *pixel_values.shape[1:]" in source
