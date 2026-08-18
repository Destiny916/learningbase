from pathlib import Path


REGISTRY = Path("experiments/joint_songling/data_registry/data_config.py")
DATASET_LOADER = Path("third_party/starvla_runtime/starVLA/dataloader/gr00t_lerobot/datasets.py")


def test_patchvision_t2_recipe_requests_previous_and_current_frames():
    registry = REGISTRY.read_text()

    assert "class JointSonglingSwapEndpoint20Temporal2DataConfig" in registry
    assert "observation_indices = [-1, 0]" in registry
    assert '"video.top", "video.gripper_left", "video.gripper_right"' in registry
    assert "state_indices = [-1, 0]" in registry
    assert "action_indices = list(range(50))" in registry


def test_patchvision_t2_recipe_has_a_distinct_dataset_mixture():
    registry = REGISTRY.read_text()

    assert '"joint_songling_0806swap_endpoint20_t2": [("", 1.0, "joint_songling_0806swap_endpoint20_t2")]' in registry


def test_loader_packs_all_requested_timesteps_in_time_then_camera_order():
    loader = DATASET_LOADER.read_text()

    assert "num_image_steps = len(data[self.modality_keys[\"video\"][0]])" in loader
    assert "for time_index in range(num_image_steps):" in loader
    assert "time_views.append(image)" in loader
    assert '"image": step_images[0] if len(step_images) == 1 else step_images' in loader
