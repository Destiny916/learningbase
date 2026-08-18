from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from w1_simulation.evaluation.verification import verify_run
from w1_simulation.execution.rollout import run_act_simulation
from w1_simulation.robot.commands import position_command_contract
from w1_simulation.robot.mapping import ActHandGestureConfig, ActJointMapper
from w1_simulation.simulation.simulator import W1Simulator
from w1_simulation.simulation.telemetry import sha256_file

pytestmark = pytest.mark.integration

CUSTOM_CAMERA_SOURCES = {
    "observation.images.cam_high_left": "head_right",
    "observation.images.cam_hand_left": "hand_right",
    "observation.images.cam_hand_right": "hand_left",
}


@pytest.fixture(scope="module")
def three_frame_run(tmp_path_factory, checkpoint_root, origin_root, cuda_device) -> tuple[Path, dict]:
    artifact_root = tmp_path_factory.mktemp("act-runner-artifacts")
    summary_path = run_act_simulation(
        checkpoint=checkpoint_root,
        origin_root=origin_root,
        artifact_root=artifact_root,
        run_name="pytest_three_frames",
        max_frames=3,
        device=cuda_device,
        realtime=False,
        control_mode="kinematic",
        action_pipeline="raw",
        camera_sources=CUSTOM_CAMERA_SOURCES,
        quality_metrics=("pose", "end_effector", "motion_direction", "amplitude"),
    )
    return summary_path, json.loads(summary_path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def bridge_three_frame_run(tmp_path_factory, checkpoint_root, origin_root, cuda_device) -> tuple[Path, dict]:
    artifact_root = tmp_path_factory.mktemp("act-bridge-runner-artifacts")
    summary_path = run_act_simulation(
        checkpoint=checkpoint_root,
        origin_root=origin_root,
        artifact_root=artifact_root,
        run_name="pytest_bridge_three_frames",
        max_frames=3,
        device=cuda_device,
        realtime=False,
        control_mode="kinematic",
        action_pipeline="bridge",
        image_replay_mode="state",
        camera_sources=CUSTOM_CAMERA_SOURCES,
    )
    return summary_path, json.loads(summary_path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def bridge_sample_factor_two_run(
    tmp_path_factory, checkpoint_root, origin_root, cuda_device
) -> tuple[Path, dict]:
    artifact_root = tmp_path_factory.mktemp("act-bridge-sample-factor-two")
    summary_path = run_act_simulation(
        checkpoint=checkpoint_root,
        origin_root=origin_root,
        artifact_root=artifact_root,
        run_name="pytest_bridge_sample_factor_two",
        max_frames=16,
        device=cuda_device,
        realtime=False,
        control_mode="kinematic",
        action_pipeline="bridge",
        image_replay_mode="state",
        bridge_sample_factor=2,
        bridge_policy_hz=20.0,
        camera_sources=CUSTOM_CAMERA_SOURCES,
    )
    return summary_path, json.loads(summary_path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def raw_replacement_run(tmp_path_factory, checkpoint_root, origin_root, cuda_device) -> tuple[Path, dict]:
    artifact_root = tmp_path_factory.mktemp("act-raw-replacement-artifacts")
    summary_path = run_act_simulation(
        checkpoint=checkpoint_root,
        origin_root=origin_root,
        artifact_root=artifact_root,
        run_name="pytest_raw_replacement",
        max_frames=15,
        device=cuda_device,
        realtime=True,
        control_mode="kinematic",
        action_pipeline="raw",
        replan_interval=10,
        camera_sources=CUSTOM_CAMERA_SOURCES,
    )
    return summary_path, json.loads(summary_path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def no_recording_run(tmp_path_factory, checkpoint_root, origin_root, cuda_device) -> tuple[Path, dict]:
    artifact_root = tmp_path_factory.mktemp("act-no-rerun-recording-artifacts")
    summary_path = run_act_simulation(
        checkpoint=checkpoint_root,
        origin_root=origin_root,
        artifact_root=artifact_root,
        run_name="pytest_no_rerun_recording",
        max_frames=1,
        device=cuda_device,
        realtime=False,
        control_mode="kinematic",
        action_pipeline="raw",
        camera_sources=CUSTOM_CAMERA_SOURCES,
        save_artifacts=False,
    )
    return summary_path, json.loads(summary_path.read_text(encoding="utf-8"))


def test_three_frame_runner_writes_complete_trajectory(three_frame_run) -> None:
    _, summary = three_frame_run
    trajectory = np.load(summary["trajectory"])

    assert summary["status"] == "completed"
    assert summary["frames"] == 3
    assert trajectory["act_state"].shape == (3, 19)
    assert trajectory["act_action"].shape == (3, 19)
    assert trajectory["body_position_command"].shape == (3, 17)
    assert trajectory["body_position_command_names"].shape == (17,)
    assert trajectory["left_hand_position_command"].shape == (3, 6)
    assert trajectory["right_hand_position_command"].shape == (3, 6)
    assert trajectory["source_frame_id"].shape == (3, 3)
    assert trajectory["runtime_cycle_ms"].shape == (3,)
    assert trajectory["runtime_deadline_misses"].shape == (3,)


def test_runner_records_standard_body_and_dexterous_hand_command_contract(three_frame_run) -> None:
    _, summary = three_frame_run

    assert summary["robot_command_contract"] == position_command_contract()
    with np.load(summary["trajectory"], allow_pickle=False) as trajectory:
        assert np.isfinite(trajectory["body_position_command"]).all()
        assert np.all(trajectory["left_hand_position_command"] >= 0.0)
        assert np.all(trajectory["left_hand_position_command"] <= 100.0)
        assert np.all(trajectory["right_hand_position_command"] >= 0.0)
        assert np.all(trajectory["right_hand_position_command"] <= 100.0)


def test_raw_runner_uses_exactly_one_control_tick_per_source_frame(three_frame_run) -> None:
    _, summary = three_frame_run
    with np.load(summary["trajectory"], allow_pickle=False) as trajectory:
        assert summary["frames"] == summary["source_frames"]
        np.testing.assert_array_equal(
            trajectory["clock_source_frame_index"],
            np.arange(summary["source_frames"], dtype=np.int32),
        )


def test_kinematic_control_places_qpos_at_target_after_every_step(three_frame_run) -> None:
    _, summary = three_frame_run
    with np.load(summary["trajectory"], allow_pickle=False) as trajectory:
        np.testing.assert_allclose(
            trajectory["sim_qpos_after"],
            trajectory["sim_target"],
            atol=1e-6,
            rtol=0.0,
        )


def test_policy_state_uses_current_simulator_feedback(three_frame_run) -> None:
    _, summary = three_frame_run
    simulator = W1Simulator(summary["runtime_model"])
    mapper = ActJointMapper(
        simulator.model,
        gestures=ActHandGestureConfig.from_dict(summary["hand_mapping"]),
    )
    try:
        with np.load(summary["trajectory"], allow_pickle=False) as trajectory:
            expected_states = np.asarray(
                [mapper.act_state_from_sim(qpos) for qpos in trajectory["sim_qpos_before"]]
            )
            np.testing.assert_allclose(
                trajectory["act_state"],
                expected_states,
                atol=1e-5,
                rtol=0.0,
            )
    finally:
        simulator.close()


def test_three_frame_runner_persists_complete_candidate_chunk_history(three_frame_run) -> None:
    _, summary = three_frame_run
    with np.load(summary["trajectory"], allow_pickle=False) as trajectory:
        assert trajectory["candidate_chunks"].shape == (1, 100, 19)
        np.testing.assert_array_equal(trajectory["chunk_submit_step"], np.asarray([0]))
        np.testing.assert_array_equal(trajectory["chunk_install_step"], np.asarray([0]))
        np.testing.assert_array_equal(trajectory["act_action"], trajectory["candidate_chunks"][0, :3])
        np.testing.assert_array_equal(
            trajectory["processed_candidate_chunks"], trajectory["candidate_chunks"]
        )


def test_three_frame_runner_writes_tensorboard_events(three_frame_run) -> None:
    _, summary = three_frame_run
    event_files = list(Path(summary["tensorboard"]).glob("events.out.tfevents.*"))

    assert len(event_files) == 1
    assert event_files[0].stat().st_size > 0


def test_three_frame_runner_writes_rerun_recording(three_frame_run) -> None:
    summary_path, summary = three_frame_run
    recording = Path(summary["rerun_recording"])

    assert summary_path.name == "summary.json"
    assert summary["artifacts_persistent"] is True
    assert Path(summary["run_directory"]) == summary_path.parent
    assert Path(summary["trajectory"]) == summary_path.parent / "trajectory.npz"
    assert recording.is_file()
    assert recording == summary_path.parent / "recording.rrd"
    assert Path(summary["tensorboard"]) == summary_path.parent / "tensorboard"
    assert Path(summary["runtime_model"]).is_relative_to(summary_path.parent / "generated")
    assert recording.stat().st_size > 0


def test_runner_does_not_write_rerun_recording_when_disabled(no_recording_run) -> None:
    summary_path, summary = no_recording_run

    assert summary["rerun_recording_enabled"] is False
    assert summary["artifacts_persistent"] is False
    assert summary["run_directory"] is None
    assert summary["rerun_recording"] is None
    assert summary["rerun_recording_sha256"] is None
    assert not (summary_path.parent / "recording.rrd").exists()


def test_three_frame_runner_keeps_reference_joint_data_out_of_policy(three_frame_run) -> None:
    _, summary = three_frame_run

    assert summary["origin_pose_reads"] == 1
    assert summary["origin_pose_use"] == "reset_and_quality_evaluation"
    assert summary["recorded_joint_data_used_after_reset"] is True
    assert summary["recorded_joint_data_use_after_reset"] == "quality_evaluation_only"
    assert summary["recorded_joint_data_used_by_policy"] is False


def test_three_frame_runner_exposes_selected_motion_quality_metrics(three_frame_run) -> None:
    _, summary = three_frame_run
    quality = summary["quality_evaluation"]

    assert quality["enabled"] is True
    assert quality["metrics"] == ["pose", "end_effector", "motion_direction", "amplitude"]
    assert quality["reference_use"] == "evaluation_only_not_policy_input"
    assert quality["execution"] == "asynchronous_during_rollout"
    assert 0.0 <= quality["final"]["score"] <= 100.0
    with np.load(summary["trajectory"], allow_pickle=False) as trajectory:
        for tag in quality["tensorboard_tags"]:
            name = tag.removeprefix("quality/")
            values = trajectory[f"quality_{name}"]
            assert values.shape == (3,)
            assert np.isfinite(values).all()

    accumulator = EventAccumulator(summary["tensorboard"])
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags()["scalars"])
    assert set(quality["tensorboard_tags"]) <= scalar_tags


def test_three_frame_runner_reports_process_and_summary_scores(three_frame_run) -> None:
    _, summary = three_frame_run
    score = summary["run_score"]

    assert score["execution"] == "offline_post_rollout"
    assert score["valid"] is True
    assert 0.0 <= score["average_score"] <= 100.0
    assert score["selected_components"] == [
        "motion_reproduction",
        "smoothness",
        "amplitude",
    ]
    assert score["comparable"] is False
    assert summary["timing"]["quality_wait_seconds"] >= 0.0
    assert summary["timing"]["scoring_seconds"] >= 0.0
    with np.load(summary["trajectory"], allow_pickle=False) as trajectory:
        assert trajectory["score_motion_reproduction"].shape == (3,)
        assert trajectory["score_smoothness"].shape == (3,)
        assert trajectory["score_amplitude"].shape == (3,)

    accumulator = EventAccumulator(summary["tensorboard"])
    accumulator.Reload()
    assert set(score["tensorboard_tags"]) <= set(accumulator.Tags()["scalars"])


def test_three_frame_runner_records_kinematic_control_mode(three_frame_run) -> None:
    _, summary = three_frame_run

    assert summary["control_mode"] == "kinematic"


def test_three_frame_runner_records_dynamic_camera_and_policy_contract(three_frame_run) -> None:
    _, summary = three_frame_run

    assert summary["camera_sources"] == CUSTOM_CAMERA_SOURCES
    assert summary["policy_contract"]["images"] == list(CUSTOM_CAMERA_SOURCES)
    assert summary["policy_contract"]["state_dim"] == 19
    assert summary["policy_contract"]["chunk_shape"] == [100, 19]


def test_verifier_replays_dynamic_camera_sources(three_frame_run) -> None:
    summary_path, summary = three_frame_run

    report_path = verify_run(summary_path.parent)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["status"] == "passed"
    assert report["frames"] == 3


def test_verifier_rejects_tampered_run_average_score(three_frame_run, tmp_path: Path) -> None:
    _, original = three_frame_run
    summary = json.loads(json.dumps(original))
    summary["run_name"] = "pytest_tampered_run_score"
    summary["run_score"]["average_score"] += 1.0
    tampered_directory = tmp_path / "tampered_run_score"
    tampered_directory.mkdir()
    tampered_path = tampered_directory / "summary.json"
    tampered_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(AssertionError, match="average score"):
        verify_run(tampered_directory)


def test_raw_runner_replaces_active_chunk_and_restarts_at_index_zero(raw_replacement_run) -> None:
    summary_path, summary = raw_replacement_run
    with np.load(summary["trajectory"], allow_pickle=False) as trajectory:
        assert summary["inference_schedule"] == "synchronous_latest"
        assert summary["action_scheduler"] == {
            "mode": "receding_horizon_replace",
            "replan_interval": 10,
            "replan_frequency_hz": 3.0,
            "requested_replan_frequency_hz": None,
            "replan_interval_override": 10,
            "replan_interval_source": "cli",
            "low_watermark": None,
            "asynchronous": False,
            "minimum_inference_ms": None,
            "max_candidates": None,
            "handoff_weights": None,
            "handoff_duration_ms": None,
        }
        np.testing.assert_array_equal(trajectory["chunk_submit_step"], [0, 10])
        install_step = int(trajectory["chunk_install_step"][1])
        assert install_step == 10
        np.testing.assert_array_equal(trajectory["action_index"], [*range(10), *range(5)])
        assert trajectory["processor_record_index"][install_step] == 1
        assert trajectory["action_index"][install_step] == 0
        np.testing.assert_array_equal(
            trajectory["act_action"][install_step],
            trajectory["raw_candidate_chunks"][1, 0],
        )
        assert not np.any(trajectory["processor_record_index"][install_step:] == 0)
        np.testing.assert_array_equal(
            trajectory["replan_seed_state"][1],
            trajectory["act_state"][10],
        )
        np.testing.assert_array_equal(
            trajectory["replan_image_sha256"][1],
            trajectory["model_input_sha256"][10],
        )

    report_path = verify_run(summary_path.parent)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["inference_schedule"] == "synchronous_latest"


def test_bridge_runner_preserves_thirty_hz_images_and_action_chunk_length(bridge_three_frame_run) -> None:
    _, summary = bridge_three_frame_run
    with np.load(summary["trajectory"], allow_pickle=False) as trajectory:
        assert summary["action_pipeline"] == "bridge"
        assert summary["inference_schedule"] == "async_remaining_ratio_absolute_step_lipo"
        assert summary["action_scheduler"] == {
            "mode": "remaining_ratio_absolute_step_lipo",
            "replan_threshold": 0.5,
            "trigger_policy_points": 50,
            "trigger_control_points": 100,
            "lipo_blend_policy_points": 5,
            "lipo_blend_control_points": 10,
            "policy_hz": 20.0,
            "control_hz": 40.0,
            "asynchronous": True,
            "simulated_inference_ms": 200.0,
            "inference_budget_ms": 300.0,
            "inference_budget_policy_points": 6,
            "replan_margin_policy_points": 2,
            "required_policy_points": 13,
            "available_policy_points": 37,
            "max_in_flight": 1,
            "max_candidates": 1,
            "body_dimensions": 17,
            "gripper_blended": False,
        }
        assert summary["source_frames"] == 3
        assert summary["frames"] == 4
        assert summary["control_hz"] == 40.0
        assert summary["source_image_hz"] == 30.0
        assert summary["image_replay"]["mode"] == "state"
        assert summary["image_replay"]["reference_joint_use"] == "frame_selection_only_not_policy_state"
        assert summary["recorded_joint_data_used_by_policy"] is False
        assert summary["recorded_joint_data_used_for_image_selection"] is True
        assert summary["control_ticks_per_source_frame"] is None
        assert summary["replan_interval"] is None
        assert summary["replan_frequency_hz"] is None
        np.testing.assert_array_equal(trajectory["clock_source_frame_index"], [0, 0, 1, 2])
        assert trajectory["raw_candidate_chunks"].shape[1:] == (100, 19)
        assert trajectory["interpolated_chunks"].shape[1:] == (200, 19)
        assert trajectory["candidate_count"].shape == (4,)
        assert trajectory["observation_age_steps"].shape == (4,)
        assert trajectory["target_step_error"].shape == (4,)
        assert trajectory["discarded_prefix_steps"].shape == (4,)
        assert trajectory["blend_active"].shape == (4,)
        assert trajectory["blend_alpha"].shape == (4,)
        assert trajectory["clock_source_frame_index"].shape == (4,)
        assert trajectory["image_match_distance"].shape == (4,)
        assert trajectory["image_match_frozen"].shape == (4,)
        assert trajectory["image_selection_updated"].shape == (4,)
        assert np.all(np.diff(trajectory["source_frame_index"]) >= 0)
        assert np.all(np.diff(trajectory["source_frame_index"]) <= 2)
        assert np.all(trajectory["target_step_error"] == 0)
        assert np.all(trajectory["candidate_count"] == 1)
        np.testing.assert_array_equal(
            trajectory["act_action"][:, 17:], trajectory["processed_queue_action"][:, 17:]
        )
        np.testing.assert_array_equal(
            trajectory["replan_image_sha256"][0], trajectory["model_input_sha256"][0]
        )
        for record_index, submit_step in enumerate(trajectory["chunk_submit_step"]):
            np.testing.assert_allclose(
                trajectory["replan_seed_state"][record_index],
                trajectory["act_state"][submit_step],
                atol=0.0,
                rtol=0.0,
            )


def test_bridge_runner_verifier_replays_every_processing_stage(bridge_three_frame_run) -> None:
    summary_path, summary = bridge_three_frame_run

    report_path = verify_run(summary_path.parent)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["status"] == "passed"
    assert report["action_pipeline"] == "bridge"
    assert report["frames"] == 4


def test_sample_factor_two_keeps_images_at_30hz_and_consumes_actions_at_40hz(
    bridge_sample_factor_two_run,
) -> None:
    summary_path, summary = bridge_sample_factor_two_run
    with np.load(summary["trajectory"], allow_pickle=False) as trajectory:
        assert summary["source_frames"] == 16
        assert summary["frames"] == 21
        assert summary["source_image_hz"] == 30.0
        assert summary["control_hz"] == 40.0
        assert summary["action_sample_factor"] == 2
        assert summary["control_ticks_per_source_frame"] is None
        assert summary["visualization_hz"] == 30.0
        assert summary["visualization_frames"] == 16
        assert summary["visualization_state_source"] == "mujoco_qpos_after_control_step"
        assert summary["visualization_timebase"] == "control_clock_relative_seconds"
        assert summary["policy_contract"]["chunk_shape"] == [100, 19]
        assert summary["policy_contract"]["processed_chunk_shape"] == [200, 19]
        assert summary["action_scheduler"]["trigger_control_points"] == 100
        assert summary["action_scheduler"]["lipo_blend_control_points"] == 10
        assert summary["action_scheduler"]["policy_hz"] == 20.0
        assert np.all(np.diff(trajectory["source_frame_index"]) >= 0)
        assert trajectory["raw_candidate_chunks"].shape[1:] == (100, 19)
        assert trajectory["processed_candidate_chunks"].shape[1:] == (200, 19)
        np.testing.assert_array_equal(trajectory["chunk_submit_step"], [0])

    report_path = verify_run(summary_path.parent)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["rerun_view_mode"] == "eye"
    assert report["eye_camera"]["enabled"] is True
    with np.load(summary["trajectory"], allow_pickle=False) as trajectory:
        assert report["eye_camera"]["frames_rendered"] == len(trajectory["eye_camera_step"])
    assert report["rrd_rows"]["joints/qpos"] == summary["rerun_state_frames"]
    assert all(
        rows == summary["visualization_frames"]
        for entity, rows in report["rrd_rows"].items()
        if entity.startswith("observation/")
    )


def test_bridge_verifier_rejects_negative_scheduler_latency(bridge_three_frame_run, tmp_path: Path) -> None:
    _, original = bridge_three_frame_run
    summary = json.loads(json.dumps(original))
    summary["run_name"] = "pytest_bridge_invalid_scheduler_latency"
    summary["action_scheduler"]["simulated_inference_ms"] = -1.0
    tampered_directory = tmp_path / "invalid_scheduler_latency"
    tampered_directory.mkdir()
    tampered_path = tampered_directory / "summary.json"
    tampered_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(AssertionError, match="timing, trigger, blend"):
        verify_run(tampered_directory)


def test_bridge_verifier_rejects_inconsistent_threshold_contract(
    bridge_three_frame_run, tmp_path: Path
) -> None:
    _, original = bridge_three_frame_run
    summary = json.loads(json.dumps(original))
    summary["run_name"] = "pytest_bridge_invalid_trigger"
    summary["action_scheduler"]["trigger_control_points"] = 28
    tampered_directory = tmp_path / "invalid_trigger"
    tampered_directory.mkdir()
    tampered_path = tampered_directory / "summary.json"
    tampered_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(AssertionError, match="timing, trigger, blend"):
        verify_run(tampered_directory)


def test_bridge_verifier_rejects_measured_latency_below_200_ms(
    bridge_three_frame_run, tmp_path: Path
) -> None:
    _, original = bridge_three_frame_run
    summary = json.loads(json.dumps(original))
    summary["run_name"] = "pytest_bridge_invalid_measured_latency"
    original_trajectory = Path(summary["trajectory"])
    tampered_directory = tmp_path / "invalid_measured_latency"
    tampered_directory.mkdir()
    tampered_trajectory = tampered_directory / "trajectory.npz"
    with np.load(original_trajectory, allow_pickle=False) as trajectory:
        payload = {key: trajectory[key] for key in trajectory.files}
    payload["chunk_latency_ms"] = payload["chunk_latency_ms"].copy()
    payload["chunk_latency_ms"][0] = 199.0
    np.savez_compressed(tampered_trajectory, **payload)
    summary["trajectory"] = str(tampered_trajectory)
    summary["trajectory_sha256"] = sha256_file(tampered_trajectory)
    tampered_path = tampered_directory / "summary.json"
    tampered_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(AssertionError, match="violated the configured minimum"):
        verify_run(tampered_directory)


def test_raw_and_bridge_modes_start_from_the_same_act_chunk(three_frame_run, bridge_three_frame_run) -> None:
    _, raw_summary = three_frame_run
    _, bridge_summary = bridge_three_frame_run

    with (
        np.load(raw_summary["trajectory"], allow_pickle=False) as raw_trajectory,
        np.load(bridge_summary["trajectory"], allow_pickle=False) as bridge_trajectory,
    ):
        np.testing.assert_array_equal(
            raw_trajectory["raw_candidate_chunks"][0],
            bridge_trajectory["raw_candidate_chunks"][0],
        )
        np.testing.assert_array_equal(
            raw_trajectory["model_input_sha256"][0],
            bridge_trajectory["model_input_sha256"][0],
        )
        np.testing.assert_array_equal(raw_trajectory["act_state"][0], bridge_trajectory["act_state"][0])
