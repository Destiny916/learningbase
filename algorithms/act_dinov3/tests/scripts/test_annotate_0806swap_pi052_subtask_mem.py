from lerobot.scripts.annotate_0806swap_pi052_subtask_mem import (
    MEMORY_AFTER_RIGHT_PICKUP,
    SUBTASK_PICKUP_RIGHT,
    SUBTASK_TRANSFER_LEFT_AND_PLACE,
    build_persistent_rows,
    parse_split_guide,
    _replace_language_columns,
    _rename_episode_metadata_columns,
    _add_language_features,
)


def test_persistent_rows_activate_second_subtask_and_memory_after_split() -> None:
    rows = build_persistent_rows(
        frame_indices=[0, 1, 2, 3, 4],
        timestamps=[0.0, 0.04, 0.08, 0.12, 0.16],
        split_after_frame=2,
    )

    assert rows == [
        {
            "role": "assistant",
            "content": SUBTASK_PICKUP_RIGHT,
            "style": "subtask",
            "timestamp": 0.0,
            "camera": None,
            "tool_calls": None,
        },
        {
            "role": "assistant",
            "content": SUBTASK_TRANSFER_LEFT_AND_PLACE,
            "style": "subtask",
            "timestamp": 0.12,
            "camera": None,
            "tool_calls": None,
        },
        {
            "role": "assistant",
            "content": MEMORY_AFTER_RIGHT_PICKUP,
            "style": "memory",
            "timestamp": 0.12,
            "camera": None,
            "tool_calls": None,
        },
    ]


def test_persistent_rows_reject_missing_boundary_frame() -> None:
    import pytest

    with pytest.raises(ValueError, match="split_after_frame=1"):
        build_persistent_rows(
            frame_indices=[0, 1, 3, 4],
            timestamps=[0.0, 0.04, 0.12, 0.16],
            split_after_frame=1,
        )


def test_parse_split_guide_accepts_the_full_width_colon_used_by_the_guide(tmp_path) -> None:
    guide = tmp_path / "guide.md"
    guide.write_text("- episode0：第 342 帧后分开（总帧数 887）\n", encoding="utf-8")

    splits = parse_split_guide(guide)

    assert splits[0].split_after_frame == 342
    assert splits[0].total_frames == 887


def test_language_columns_have_one_empty_event_list_per_input_frame() -> None:
    import pyarrow as pa

    table = pa.table({"episode_index": [0, 0], "timestamp": [0.0, 0.04]})
    persistent = [
        build_persistent_rows(
            frame_indices=[0, 1, 2], timestamps=[0.0, 0.04, 0.08], split_after_frame=1
        )
    ] * 2

    annotated = _replace_language_columns(table, persistent)

    assert annotated.num_rows == 2
    assert annotated["language_events"].to_pylist() == [[], []]


def test_episode_metadata_uses_the_same_gripper_video_keys_as_info_json() -> None:
    import pyarrow as pa

    table = pa.table(
        {
            "videos/observation.images.wrist_left/chunk_index": [0],
            "stats/observation.images.wrist_right/q99": [1.0],
        }
    )

    renamed = _rename_episode_metadata_columns(table)

    assert renamed.column_names == [
        "videos/observation.images.gripper_left/chunk_index",
        "stats/observation.images.gripper_right/q99",
    ]


def test_dataset_info_declares_the_persistent_language_columns(tmp_path) -> None:
    import json

    meta = tmp_path / "meta"
    meta.mkdir()
    info_path = meta / "info.json"
    info_path.write_text(json.dumps({"features": {"action": {"dtype": "float32"}}}), encoding="utf-8")

    _add_language_features(tmp_path)

    features = json.loads(info_path.read_text(encoding="utf-8"))["features"]
    assert features["language_persistent"]["dtype"] == "language"
    assert features["language_events"]["dtype"] == "language"
