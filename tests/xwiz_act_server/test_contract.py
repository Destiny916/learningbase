import numpy as np
import pytest

from xwiz_act_server.contract import (
    ContractError,
    assemble_state,
    decode_observation,
    group_action_chunk,
)


def legacy_states():
    return {
        "waistqpos": np.array([3], np.float32),
        "left_armqpos": np.arange(10, 17, dtype=np.float32),
        "headqpos": np.array([20, 21], np.float32),
        "right_armqpos": np.arange(30, 37, dtype=np.float32),
        "left_eefgripper": np.array([40], np.float32),
        "right_eefgripper": np.array([41], np.float32),
    }


def image_bytes(bgr=(0, 0, 255)):
    image = np.empty((360, 640, 3), dtype=np.uint8)
    image[:] = bgr
    return image.tobytes()


def observation_request():
    return {
        "states": legacy_states(),
        "cam_high": image_bytes(),
        "cam_left_wrist": image_bytes((0, 0, 0)),
        "cam_right_wrist": image_bytes((0, 0, 0)),
        "head_target_size": [640, 360],
        "hand_target_size": [640, 360],
        "timestamp": 12.5,
        "timestep": 8,
        "start_infer": True,
    }


def test_assemble_state_uses_checkpoint_order():
    state = assemble_state(legacy_states())
    expected = np.array(
        [3, *range(10, 17), 20, 21, *range(30, 37), 40, 41],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(state, expected)


def test_decode_observation_maps_images_and_converts_bgr_to_rgb():
    observation = decode_observation(observation_request())
    assert set(observation) == {
        "observation.state",
        "observation.images.cam_high_left",
        "observation.images.cam_hand_left",
        "observation.images.cam_hand_right",
    }
    head = observation["observation.images.cam_high_left"]
    assert head.shape == (360, 640, 3)
    assert head.dtype == np.uint8
    np.testing.assert_array_equal(head[0, 0], [255, 0, 0])
    assert observation["observation.images.cam_hand_left"].max() == 0
    assert observation["observation.images.cam_hand_right"].max() == 0


def test_decode_observation_rejects_missing_state_group():
    request = observation_request()
    del request["states"]["headqpos"]
    with pytest.raises(ContractError, match="headqpos"):
        decode_observation(request)


def test_decode_observation_rejects_wrong_image_size():
    request = observation_request()
    request["cam_high"] = b"short"
    with pytest.raises(ContractError, match="cam_high"):
        decode_observation(request)


def test_group_action_chunk_uses_six_legacy_groups():
    actions = np.arange(16 * 19, dtype=np.float32).reshape(16, 19)
    grouped = group_action_chunk(actions)
    assert {key: value.shape for key, value in grouped.items()} == {
        "waistqpos": (16, 1),
        "left_armqpos": (16, 7),
        "headqpos": (16, 2),
        "right_armqpos": (16, 7),
        "left_eefgripper": (16, 1),
        "right_eefgripper": (16, 1),
    }
    np.testing.assert_array_equal(grouped["headqpos"], actions[:, 8:10])


@pytest.mark.parametrize(
    "actions, message",
    [
        (np.zeros((32, 19), dtype=np.float32), "16, 19"),
        (np.zeros((16, 18), dtype=np.float32), "16, 19"),
        (np.full((16, 19), np.nan, dtype=np.float32), "finite"),
    ],
)
def test_group_action_chunk_rejects_invalid_output(actions, message):
    with pytest.raises(ContractError, match=message):
        group_action_chunk(actions)
