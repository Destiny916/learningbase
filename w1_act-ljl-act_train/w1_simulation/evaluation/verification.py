from __future__ import annotations

import argparse
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from w1_simulation.artifacts import sha256_file, simulation_artifact_paths
from w1_simulation.control.processing import build_action_processor, validate_trace
from w1_simulation.evaluation.behavior import KINEMATIC_BEHAVIOR_LIMITS
from w1_simulation.evaluation.behavior import behavior_metrics as _behavior_metrics
from w1_simulation.evaluation.behavior import enforce_kinematic_behavior as _enforce_kinematic_behavior
from w1_simulation.evaluation.behavior import reference_states as _reference_states
from w1_simulation.evaluation.quality import (
    MotionQualityEvaluator,
    load_reference_states,
    validate_quality_metrics,
)
from w1_simulation.evaluation.scoring import compute_run_score
from w1_simulation.evaluation.verification_report import save_verification_report
from w1_simulation.inference.direct import inspect_checkpoint_contract
from w1_simulation.replay.origin import OriginReplay, StateAlignedFrameSelector
from w1_simulation.robot.commands import W1ControlEndpoints, position_command_contract
from w1_simulation.robot.mapping import ActHandGestureConfig, ActJointMapper
from w1_simulation.simulation.camera import RERUN_VIEW_MODES, EyeCameraSchedule
from w1_simulation.simulation.config import BODY_JOINTS, SELF_COLLISION_EXCLUDES, SimulationConfig
from w1_simulation.simulation.simulator import W1Simulator

REQUIRED_TENSORBOARD_TAGS = {
    "control/joint_limit_violations",
    "control/tracking_rmse",
    "inference/e2e_ms",
    "input/camera_skew_ms",
    "input/state_aligned_frame_index",
    "input/state_match_distance",
    "input/state_match_frozen",
    "input/state_selection_updated",
    "runtime/cycle_ms",
    "runtime/deadline_misses",
    "scheduler/action_index",
    "scheduler/candidate_count",
    "scheduler/discarded_prefix_steps",
    "scheduler/replan_count",
    "scheduler/replan_interval",
    "scheduler/low_watermark",
    "scheduler/queue_size",
    "scheduler/held_last_command",
    "scheduler/observation_age_steps",
    "scheduler/target_step_error",
    "system/gpu/memory_used_mb",
}


def _verify_action_pipeline(
    trajectory: np.lib.npyio.NpzFile,
    summary: dict[str, object],
    gestures: ActHandGestureConfig,
) -> None:
    action_pipeline = str(summary["action_pipeline"])
    processor_entry = summary["action_processor"]
    processor = build_action_processor(action_pipeline, int(summary["action_sample_factor"]))
    if processor.name != processor_entry["name"]:
        raise AssertionError("Action processor name differs from the recorded manifest")
    processor.reset()
    processor_source = Path(str(processor_entry["source"]))
    if sha256_file(processor_source) != processor_entry["source_sha256"]:
        raise AssertionError("Action processor source hash mismatch")

    raw_chunks = np.asarray(trajectory["raw_candidate_chunks"], dtype=np.float32)
    processed_chunks = np.asarray(trajectory["processed_candidate_chunks"], dtype=np.float32)
    record_count = len(raw_chunks)
    prediction_shape = tuple(summary["policy_contract"]["prediction_chunk_shape"])
    execution_shape = tuple(summary["policy_contract"]["execution_chunk_shape"])
    if (
        prediction_shape[1:] != (19,)
        or execution_shape[1:] != (19,)
        or prediction_shape[0] < execution_shape[0]
        or summary["policy_contract"].get("chunk_shape") != list(execution_shape)
        or int(summary["policy_contract"].get("prediction_horizon", 0)) != prediction_shape[0]
        or int(summary["policy_contract"].get("execution_horizon", 0)) != execution_shape[0]
    ):
        raise AssertionError("ACT prediction/execution horizon contract is invalid")
    if raw_chunks.shape != (record_count, *execution_shape):
        raise AssertionError("Raw ACT chunk history has an invalid shape")
    processed_shape = tuple(summary["policy_contract"]["processed_chunk_shape"])
    if processed_chunks.shape != (record_count, *processed_shape):
        raise AssertionError("Processed action chunk history has an invalid shape")
    stage_names = tuple(summary["action_processor_stages"])
    for stage_name in stage_names:
        expected_stage_shape = raw_chunks.shape if stage_name == "raw" else processed_chunks.shape
        if trajectory[f"processor_stage_{stage_name}"].shape != expected_stage_shape:
            raise AssertionError(f"Action processor stage has invalid shape: {stage_name}")
    if trajectory["candidate_chunks"].shape != raw_chunks.shape:
        raise AssertionError("Raw candidate chunk alias has an invalid shape")
    np.testing.assert_array_equal(trajectory["candidate_chunks"], raw_chunks)
    if action_pipeline == "raw":
        np.testing.assert_array_equal(processed_chunks, raw_chunks)
    elif stage_names != ("raw", "interpolated", "processed"):
        raise AssertionError("LIPO bridge must expose only raw/interpolated/processed stages")

    submit_steps = np.asarray(trajectory["chunk_submit_step"], dtype=np.int32)
    install_steps = np.asarray(trajectory["chunk_install_step"], dtype=np.int32)
    chunk_latencies = np.asarray(trajectory["chunk_latency_ms"], dtype=np.float32)
    seed_states = np.asarray(trajectory["replan_seed_state"], dtype=np.float32)
    image_hashes = np.asarray(trajectory["replan_image_sha256"])
    if (
        submit_steps.shape != (record_count,)
        or install_steps.shape != (record_count,)
        or chunk_latencies.shape != (record_count,)
    ):
        raise AssertionError("Action inference record step arrays have invalid shapes")
    if not np.isfinite(chunk_latencies).all():
        raise AssertionError("Action inference latency history contains non-finite values")
    if seed_states.shape != (record_count, 19):
        raise AssertionError("Action seed-state history has an invalid shape")
    if image_hashes.shape != (record_count, len(summary["policy_contract"]["images"])):
        raise AssertionError("Action inference image-hash history has an invalid shape")
    if action_pipeline == "raw":
        replan_interval = int(summary["action_scheduler"]["replan_interval"])
        if submit_steps[0] != 0 or np.any(submit_steps[1:] % replan_interval != 0):
            raise AssertionError("Raw ACT replans outside the configured fixed interval")
        np.testing.assert_array_equal(install_steps, submit_steps)
    else:
        expected_minimum_ms = float(summary["action_scheduler"]["simulated_inference_ms"])
        if np.any(chunk_latencies < expected_minimum_ms):
            minimum_latency = float(chunk_latencies.min(initial=np.inf))
            raise AssertionError(
                f"Bridge inference latency violated the configured minimum: minimum={minimum_latency:.3f} ms"
            )
        if submit_steps[0] != 0 or install_steps[0] != 0:
            raise AssertionError("LIPO bootstrap inference must be installed synchronously at step zero")
        if np.any(submit_steps < 0) or np.any(install_steps[1:] < -1):
            raise AssertionError("LIPO inference record contains an invalid lifecycle step")
        completed = install_steps >= 0
        if np.any(install_steps[completed] < submit_steps[completed]):
            raise AssertionError("LIPO inference was installed before it was submitted")

    installs: dict[int, list[int]] = defaultdict(list)
    submits: dict[int, list[int]] = defaultdict(list)
    for record_index, (submit_step, install_step) in enumerate(zip(submit_steps, install_steps, strict=True)):
        submits[int(submit_step)].append(record_index)
        installs[int(install_step)].append(record_index)

    plan_queue: deque[np.ndarray] = deque()
    provenance: deque[tuple[int, int]] = deque()

    def install_record(record_index: int) -> None:
        previous_action = plan_queue[-1] if plan_queue else None
        if action_pipeline == "bridge":
            previous_action = None
        trace = validate_trace(
            processor.process_chunk(raw_chunks[record_index], previous_action),
            raw_chunks[record_index],
        )
        if action_pipeline == "raw":
            plan_queue.clear()
            provenance.clear()
        np.testing.assert_array_equal(trace.processed, processed_chunks[record_index])
        if tuple(trace.stages) != stage_names:
            raise AssertionError("Action processor stage schema differs from the recorded run")
        for stage_name, actual in trace.stages.items():
            np.testing.assert_allclose(
                actual,
                trajectory[f"processor_stage_{stage_name}"][record_index],
                atol=0.0,
                rtol=0.0,
            )
        if action_pipeline == "raw":
            for index, action in enumerate(trace.processed):
                plan_queue.append(action.copy())
                provenance.append((record_index, index))

    states = np.asarray(trajectory["act_state"], dtype=np.float32)
    feedback_states = np.asarray(trajectory["feedback_act_state"], dtype=np.float32)
    queue_actions = np.asarray(trajectory["processed_queue_action"], dtype=np.float32)
    effective_actions = np.asarray(trajectory["act_action"], dtype=np.float32)
    held = np.asarray(trajectory["held_last_command"], dtype=np.bool_)
    record_indices = np.asarray(trajectory["processor_record_index"], dtype=np.int32)
    action_indices = np.asarray(trajectory["action_index"], dtype=np.int32)
    queue_sizes = np.asarray(trajectory["action_queue_size"], dtype=np.int32)
    input_hashes = np.asarray(trajectory["model_input_sha256"])
    policy_inputs = np.asarray(trajectory["policy_input"], dtype=np.bool_)
    candidate_counts = np.asarray(trajectory["candidate_count"], dtype=np.int32)
    observation_age_steps = np.asarray(trajectory["observation_age_steps"], dtype=np.int32)
    target_step_errors = np.asarray(trajectory["target_step_error"], dtype=np.int32)
    discarded_prefix_steps = np.asarray(trajectory["discarded_prefix_steps"], dtype=np.int32)
    blend_active = np.asarray(trajectory["blend_active"], dtype=np.bool_)
    blend_alpha = np.asarray(trajectory["blend_alpha"], dtype=np.float32)
    old_record_indices = np.asarray(trajectory["lipo_old_record_index"], dtype=np.int32)
    new_record_indices = np.asarray(trajectory["lipo_new_record_index"], dtype=np.int32)
    expected_policy_steps = sorted(step for step in submits if 0 <= step < len(states))
    np.testing.assert_array_equal(np.flatnonzero(policy_inputs), expected_policy_steps)

    if action_pipeline == "bridge":
        scheduler = summary["action_scheduler"]
        trigger_control_points = int(scheduler["trigger_control_points"])
        blend_control_points = int(scheduler["lipo_blend_control_points"])
        body_dimensions = int(scheduler["body_dimensions"])
    else:
        trigger_control_points = blend_control_points = body_dimensions = 0

    if feedback_states.shape != states.shape:
        raise AssertionError("ACT feedback-state trajectory has an invalid shape")
    if action_pipeline == "bridge":
        np.testing.assert_allclose(states[0], feedback_states[0], atol=0.0, rtol=0.0)
        if len(states) > 1:
            np.testing.assert_allclose(states[1:], effective_actions[:-1], atol=0.0, rtol=0.0)

    last_command: np.ndarray | None = None

    for step in range(len(states)):
        for record_index in installs.get(step, []):
            install_record(record_index)

        if action_pipeline == "raw":
            expected_queue_action = plan_queue.popleft() if plan_queue else None
            expected_record_index = expected_action_index = -1
        else:
            expected_record_index = int(record_indices[step])
            expected_action_index = int(action_indices[step])
            expected_queue_action = None
            if expected_record_index >= 0:
                if expected_record_index >= record_count:
                    raise AssertionError(f"LIPO record index is out of range at step {step}")
                expected_action_index = step - int(submit_steps[expected_record_index])
                if action_indices[step] != expected_action_index:
                    raise AssertionError(f"LIPO action index is not absolute-step aligned at step {step}")
                if not 0 <= expected_action_index < processed_chunks.shape[1]:
                    raise AssertionError(f"LIPO action index is outside its trajectory at step {step}")
                expected_queue_action = processed_chunks[expected_record_index, expected_action_index].copy()
                if blend_active[step]:
                    new_record_index = int(new_record_indices[step])
                    old_record_index = int(old_record_indices[step])
                    if new_record_index != expected_record_index:
                        raise AssertionError(f"LIPO transition new-record provenance mismatch at step {step}")
                    install_step = int(install_steps[new_record_index])
                    if install_step < 0 or step < install_step:
                        raise AssertionError(
                            f"LIPO transition precedes trajectory installation at step {step}"
                        )
                    new_end = int(submit_steps[new_record_index]) + processed_chunks.shape[1] - 1
                    if old_record_index >= 0:
                        old_action_index = step - int(submit_steps[old_record_index])
                        if not 0 <= old_action_index < processed_chunks.shape[1]:
                            raise AssertionError(f"LIPO old trajectory is unavailable at step {step}")
                        old_action = processed_chunks[old_record_index, old_action_index]
                        overlap_end = min(
                            new_end,
                            int(submit_steps[old_record_index]) + processed_chunks.shape[1] - 1,
                        )
                    else:
                        if install_step <= 0:
                            raise AssertionError(
                                f"LIPO hold transition has no previous command at step {step}"
                            )
                        old_action = queue_actions[install_step - 1]
                        overlap_end = new_end
                    blend_length = min(blend_control_points, overlap_end - install_step + 1)
                    if blend_length <= 0 or step >= install_step + blend_length:
                        raise AssertionError(f"LIPO transition exceeds its overlap at step {step}")
                    expected_alpha = float(step - install_step + 1) / float(blend_length)
                    if not np.isclose(blend_alpha[step], expected_alpha, atol=1e-7, rtol=0.0):
                        raise AssertionError(f"LIPO blend alpha mismatch at step {step}")
                    expected_queue_action[:body_dimensions] = (
                        old_action[:body_dimensions] * (1.0 - expected_alpha)
                        + expected_queue_action[:body_dimensions] * expected_alpha
                    )
                    np.testing.assert_array_equal(
                        expected_queue_action[body_dimensions:],
                        processed_chunks[new_record_index, expected_action_index, body_dimensions:],
                    )
                elif blend_alpha[step] not in {0.0, 1.0}:
                    raise AssertionError(f"Non-transition LIPO alpha is invalid at step {step}")
        if expected_queue_action is None:
            if not held[step] or last_command is None:
                raise AssertionError(f"Action queue hold state is invalid at step {step}")
            expected_queue_action = last_command.copy()
            expected_record_index, expected_action_index = -1, -1
        else:
            if held[step]:
                raise AssertionError(f"Action queue provenance is invalid at step {step}")
            if action_pipeline == "raw":
                if not provenance:
                    raise AssertionError(f"Action queue provenance is invalid at step {step}")
                expected_record_index, expected_action_index = provenance.popleft()
            last_command = expected_queue_action.copy()
        np.testing.assert_allclose(expected_queue_action, queue_actions[step], atol=0.0, rtol=0.0)
        if record_indices[step] != expected_record_index or action_indices[step] != expected_action_index:
            raise AssertionError(f"Bridge emission provenance mismatch at step {step}")

        expected_effective = processor.process_action(expected_queue_action)
        np.testing.assert_allclose(expected_effective, effective_actions[step], atol=0.0, rtol=0.0)
        expected_queue_size = (
            len(plan_queue)
            if action_pipeline == "raw"
            else sum(
                int(submit_step <= step and (install_step < 0 or install_step > step))
                for submit_step, install_step in zip(submit_steps, install_steps, strict=True)
            )
        )
        if queue_sizes[step] != expected_queue_size:
            raise AssertionError(f"Action queue size mismatch at step {step}")
        if action_pipeline == "bridge" and expected_queue_size > 1:
            raise AssertionError(f"LIPO has more than one pending inference at step {step}")
        expected_candidate_count = int(expected_record_index >= 0)
        if candidate_counts[step] != expected_candidate_count:
            raise AssertionError(f"LIPO candidate count mismatch at step {step}")
        if expected_record_index >= 0:
            expected_age = step - int(submit_steps[expected_record_index])
            expected_error = step - (int(submit_steps[expected_record_index]) + expected_action_index)
            expected_discarded = max(
                int(install_steps[expected_record_index] - submit_steps[expected_record_index]), 0
            )
            if expected_action_index < expected_discarded:
                raise AssertionError(f"Expired ACT prefix executed at step {step}")
        else:
            expected_age = expected_error = expected_discarded = -1
        if observation_age_steps[step] != expected_age:
            raise AssertionError(f"Observation age mismatch at step {step}")
        if target_step_errors[step] != expected_error or expected_error not in {-1, 0}:
            raise AssertionError(f"Action target step mismatch at step {step}")
        if discarded_prefix_steps[step] != expected_discarded:
            raise AssertionError(f"Discarded prefix mismatch at step {step}")

        for record_index in submits.get(step, []):
            expected_seed = states[step]
            np.testing.assert_allclose(seed_states[record_index], expected_seed, atol=0.0, rtol=0.0)
            np.testing.assert_array_equal(image_hashes[record_index], input_hashes[step])
            if action_pipeline == "bridge" and record_index > 0:
                planning_record = int(new_record_indices[step])
                if planning_record < 0:
                    planning_record = int(record_indices[step])
                if planning_record >= 0:
                    remaining = int(submit_steps[planning_record]) + processed_chunks.shape[1] - step
                    if remaining > trigger_control_points:
                        raise AssertionError(
                            f"LIPO replan triggered before the remaining-point threshold at step {step}"
                        )

    for record_index in installs.get(-1, []):
        install_record(record_index)


def _rrd_rows(recording: Path, entity: str) -> int:
    completed = subprocess.run(
        ["rerun", "rrd", "print", "-vv", "--entity", entity, str(recording)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return sum(int(value) for value in re.findall(r"with (\d+) rows", completed.stdout))


def verify_run(run_directory: Path) -> Path:
    paths = simulation_artifact_paths(run_directory)
    summary_path = paths["summary"]
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    run_name = str(summary["run_name"])
    if summary.get("status") != "completed":
        raise AssertionError(f"ACT simulation is not complete: {summary.get('status')}")
    command_contract = summary.get("robot_command_contract")
    if not isinstance(command_contract, dict):
        raise AssertionError("W1 body and dexterous-hand command contract is missing or invalid")
    try:
        body_command_names = tuple(str(name) for name in command_contract["body"]["joint_names"])
        command_endpoints = W1ControlEndpoints(
            body=str(command_contract["body"]["topic"]),
            left_hand=str(command_contract["left_hand"]["topic"]),
            right_hand=str(command_contract["right_hand"]["topic"]),
        )
    except (KeyError, TypeError) as exc:
        raise AssertionError("W1 body and dexterous-hand command contract is missing or invalid") from exc
    if command_contract != position_command_contract(command_endpoints, body_command_names):
        raise AssertionError("W1 body and dexterous-hand command contract is missing or invalid")
    quality_enabled = bool(summary.get("quality_evaluation", {}).get("enabled", False))
    score_report = summary.get("run_score")
    if not isinstance(score_report, dict):
        raise AssertionError("Run score summary is missing")
    score_requested = score_report.get("requested")
    if not isinstance(score_requested, dict):
        raise AssertionError("Run score requested-component contract is missing")
    image_replay = summary.get("image_replay")
    if not isinstance(image_replay, dict) or image_replay.get("mode") not in {"time", "state"}:
        raise AssertionError("Image replay contract is missing or invalid")
    state_aligned_images = image_replay["mode"] == "state"
    reference_used = quality_enabled or bool(score_requested.get("smoothness", False))
    recorded_joint_data_used = reference_used or state_aligned_images
    if summary.get("origin_pose_reads") != 1:
        raise AssertionError("Origin reset pose must be read exactly once")
    if summary.get("recorded_joint_data_used_by_policy", False) is not False:
        raise AssertionError("Recorded joint data entered the policy or control path")
    if summary.get("recorded_joint_data_used_after_reset") is not recorded_joint_data_used:
        raise AssertionError("Recorded joint-data use differs from evaluation and frame-selection contracts")
    expected_use = (
        "quality_evaluation_and_frame_selection_only"
        if reference_used and state_aligned_images
        else "frame_selection_only"
        if state_aligned_images
        else "quality_evaluation_only"
        if reference_used
        else "none"
    )
    if summary.get("recorded_joint_data_use_after_reset", "none") != expected_use:
        raise AssertionError("Recorded joint-data use after reset has an invalid purpose")
    if summary.get("recorded_joint_data_used_for_image_selection") is not state_aligned_images:
        raise AssertionError("Recorded joint-data frame-selection use is inconsistent")
    control_mode = summary.get("control_mode")
    if control_mode not in {"kinematic", "dynamic"}:
        raise AssertionError(f"Unknown ACT simulation control mode: {control_mode}")
    action_pipeline = summary.get("action_pipeline")
    if action_pipeline not in {"raw", "bridge"}:
        raise AssertionError(f"Unknown action pipeline: {action_pipeline}")
    expected_state_source = (
        "session_initial_unpublished_and_last_published_selected_command"
        if action_pipeline == "bridge"
        else "mujoco_controlled_qpos"
    )
    if summary.get("state_source_first_inference") != "mujoco_controlled_qpos":
        raise AssertionError("First ACT inference state is not sourced from MuJoCo feedback")
    if summary.get("state_source_after_reset") != expected_state_source:
        raise AssertionError("ACT state source differs from the selected action pipeline")
    scheduler = summary.get("action_scheduler")
    if not isinstance(scheduler, dict):
        raise AssertionError("Action scheduler manifest is missing")
    sample_factor = int(summary.get("action_sample_factor", 0))
    policy_contract = summary.get("policy_contract")
    if not isinstance(policy_contract, dict):
        raise AssertionError("ACT policy contract is missing")
    prediction_shape = tuple(policy_contract.get("prediction_chunk_shape", ()))
    execution_shape = tuple(policy_contract.get("execution_chunk_shape", ()))
    prediction_horizon = int(policy_contract.get("prediction_horizon", 0))
    checkpoint_execution_horizon = int(policy_contract.get("checkpoint_execution_horizon", 0))
    execution_horizon = int(policy_contract.get("execution_horizon", 0))
    execution_horizon_source = policy_contract.get("execution_horizon_source")
    checkpoint_contract = inspect_checkpoint_contract(Path(str(summary["checkpoint"])))
    if (
        len(prediction_shape) != 2
        or len(execution_shape) != 2
        or prediction_shape[1] != 19
        or execution_shape[1] != 19
        or prediction_shape[0] < execution_shape[0]
        or prediction_horizon != prediction_shape[0]
        or execution_horizon != execution_shape[0]
        or prediction_horizon != checkpoint_contract.prediction_horizon
        or checkpoint_execution_horizon != checkpoint_contract.execution_horizon
        or execution_horizon_source not in {"checkpoint_config", "runtime_override"}
        or (
            execution_horizon_source == "checkpoint_config"
            and execution_horizon != checkpoint_execution_horizon
        )
    ):
        raise AssertionError("ACT checkpoint horizon contract is missing or invalid")
    source_image_hz = float(summary.get("source_image_hz", 0.0))
    control_hz = float(summary.get("control_hz", 0.0))
    if sample_factor < 1 or (action_pipeline == "raw" and sample_factor != 1):
        raise AssertionError("Action sample factor is invalid for the selected pipeline")
    if source_image_hz != 30.0:
        raise AssertionError("Source image rate must remain 30 Hz")
    if action_pipeline == "raw":
        if (
            control_hz != source_image_hz
            or summary.get("control_ticks_per_source_frame") != 1
            or int(summary.get("frames", -1)) != int(summary.get("source_frames", -2))
        ):
            raise AssertionError("Raw ACT image and control timebases are inconsistent")
    elif (
        float(scheduler.get("policy_hz", 0.0)) <= 0.0
        or not np.isclose(
            control_hz,
            float(scheduler["policy_hz"]) * sample_factor,
        )
        or summary.get("control_ticks_per_source_frame") is not None
        or not np.isclose(
            float(summary.get("control_to_image_rate_ratio", 0.0)),
            control_hz / source_image_hz,
        )
    ):
        raise AssertionError("LIPO bridge policy/control/image timebases are inconsistent")
    expected_visualization_timebase = (
        "control_clock_relative_seconds" if state_aligned_images else "origin_timestamp_relative_seconds"
    )
    rerun_view_mode = summary.get("rerun_view_mode")
    rerun_layout = summary.get("rerun_layout")
    eye_camera = summary.get("eye_camera")
    if (
        rerun_view_mode not in RERUN_VIEW_MODES
        or not isinstance(rerun_layout, dict)
        or not isinstance(eye_camera, dict)
    ):
        raise AssertionError("Rerun view, layout, and MuJoCo eye-camera contracts are missing")
    eye_camera_enabled = bool(eye_camera.get("enabled", False))
    if eye_camera_enabled != (rerun_view_mode != "standard"):
        raise AssertionError("Rerun view mode and MuJoCo eye-camera enablement differ")
    if (
        float(summary.get("visualization_hz", 0.0)) != source_image_hz
        or not 0 < int(summary.get("visualization_frames", -1)) <= int(summary.get("source_frames", -2))
        or not int(summary.get("visualization_frames", -1))
        <= int(summary.get("rerun_state_frames", -2))
        <= int(summary.get("frames", -3))
        or summary.get("visualization_state_source") != "mujoco_qpos_after_control_step"
        or summary.get("visualization_timebase") != expected_visualization_timebase
    ):
        raise AssertionError("Visualization does not use the source-rate actual simulator state contract")
    if action_pipeline == "raw":
        if (
            summary.get("inference_schedule") != "synchronous_latest"
            or scheduler.get("mode") != "receding_horizon_replace"
            or not isinstance(scheduler.get("replan_interval"), int)
            or int(scheduler["replan_interval"]) <= 0
            or float(scheduler.get("replan_frequency_hz", 0.0))
            != float(summary["control_hz"]) / int(scheduler["replan_interval"])
            or scheduler.get("low_watermark") is not None
            or scheduler.get("asynchronous") is not False
            or scheduler.get("replan_interval_source")
            not in {
                "cli",
                "checkpoint_execution_horizon",
                "runtime_execution_horizon",
            }
            or (
                scheduler.get("replan_interval_source") != "cli"
                and int(scheduler["replan_interval"]) != execution_shape[0]
            )
            or (
                scheduler.get("replan_interval_source") == "checkpoint_execution_horizon"
                and execution_horizon_source != "checkpoint_config"
            )
            or (
                scheduler.get("replan_interval_source") == "runtime_execution_horizon"
                and execution_horizon_source != "runtime_override"
            )
        ):
            raise AssertionError("Raw ACT does not use the synchronous replacement schedule")
    else:
        processed_shape = summary["policy_contract"].get("processed_chunk_shape")
        replan_threshold = float(scheduler.get("replan_threshold", np.nan))
        trigger_policy_points = int(scheduler.get("trigger_policy_points", 0))
        trigger_control_points = int(scheduler.get("trigger_control_points", 0))
        blend_policy_points = int(scheduler.get("lipo_blend_policy_points", 0))
        blend_control_points = int(scheduler.get("lipo_blend_control_points", 0))
        policy_hz = float(scheduler.get("policy_hz", 0.0))
        simulated_inference_ms = float(scheduler.get("simulated_inference_ms", np.nan))
        inference_budget_ms = float(scheduler.get("inference_budget_ms", np.nan))
        inference_budget_policy_points = int(scheduler.get("inference_budget_policy_points", -1))
        replan_margin_policy_points = int(scheduler.get("replan_margin_policy_points", -1))
        required_policy_points = int(scheduler.get("required_policy_points", -1))
        available_policy_points = int(scheduler.get("available_policy_points", -1))
        expected_trigger_policy_points = (
            int(np.ceil(execution_shape[0] * replan_threshold)) if np.isfinite(replan_threshold) else -1
        )
        expected_inference_budget_policy_points = (
            int(np.ceil(inference_budget_ms / 1000.0 * policy_hz))
            if np.isfinite(inference_budget_ms) and np.isfinite(policy_hz)
            else -1
        )
        expected_required_policy_points = (
            expected_inference_budget_policy_points + blend_policy_points + replan_margin_policy_points
        )
        if (
            summary.get("inference_schedule") != "async_remaining_ratio_absolute_step_lipo"
            or scheduler.get("mode") != "remaining_ratio_absolute_step_lipo"
            or not np.isfinite(replan_threshold)
            or not 0.0 < replan_threshold < 1.0
            or not 1 <= trigger_policy_points < execution_shape[0]
            or trigger_policy_points != expected_trigger_policy_points
            or trigger_control_points != trigger_policy_points * sample_factor
            or not 1 <= blend_policy_points <= trigger_policy_points
            or blend_control_points != blend_policy_points * sample_factor
            or not np.isfinite(policy_hz)
            or policy_hz <= 0.0
            or float(scheduler.get("control_hz", 0.0)) != control_hz
            or scheduler.get("asynchronous") is not True
            or not np.isfinite(simulated_inference_ms)
            or simulated_inference_ms < 0.0
            or not np.isfinite(inference_budget_ms)
            or inference_budget_ms < 0.0
            or inference_budget_policy_points != expected_inference_budget_policy_points
            or replan_margin_policy_points < 0
            or required_policy_points != expected_required_policy_points
            or available_policy_points != trigger_policy_points - required_policy_points
            or available_policy_points < 0
            or int(scheduler.get("max_in_flight", 0)) != 1
            or int(scheduler.get("max_candidates", 0)) != 1
            or int(scheduler.get("body_dimensions", 0)) != 17
            or scheduler.get("gripper_blended") is not False
            or processed_shape != [execution_shape[0] * sample_factor, 19]
        ):
            raise AssertionError("LIPO bridge timing, trigger, blend, or state contract is invalid")
    if summary.get("joint_limit_violations") != 0:
        raise AssertionError("Run reported joint limit violations")
    timing = summary["timing"]
    for name in ("quality_wait_seconds", "scoring_seconds"):
        if not np.isfinite(float(timing.get(name, -1.0))) or float(timing[name]) < 0.0:
            raise AssertionError(f"Invalid post-rollout timing field: {name}")
    if timing["realtime_requested"]:
        minimum_fps = float(summary["control_hz"]) * 0.96
        cycle_multiplier = 3.0 if action_pipeline == "raw" else 1.5
        maximum_p95_ms = 1000.0 / float(summary["control_hz"]) * cycle_multiplier
        if timing["effective_fps"] < minimum_fps:
            raise AssertionError(f"Realtime rollout was too slow: {timing['effective_fps']:.3f} FPS")
        if timing["p95_cycle_ms"] > maximum_p95_ms:
            raise AssertionError(
                f"Realtime p95 cycle exceeded {maximum_p95_ms:.3f} ms: {timing['p95_cycle_ms']:.3f} ms"
            )
    image_keys = tuple(summary["policy_contract"]["images"])
    expected_camera_streams = [key.removeprefix("observation.images.") for key in image_keys]
    expected_primary_views = {
        "standard": ["mujoco_robot_3d"],
        "eye": ["mujoco_eye_camera"],
        "both": ["mujoco_robot_3d", "mujoco_eye_camera"],
    }[rerun_view_mode]
    if (
        rerun_layout.get("primary_views") != expected_primary_views
        or rerun_layout.get("dataset_camera_streams") != expected_camera_streams
        or rerun_layout.get("dataset_camera_column_visible") is not True
    ):
        raise AssertionError("Rerun primary view or persistent dataset-camera column is invalid")
    camera_sources = summary.get("camera_sources")
    if not isinstance(camera_sources, dict) or set(camera_sources) != set(image_keys):
        raise AssertionError("Summary camera sources do not match the ACT image contract")
    camera_sources = {key: camera_sources[key] for key in image_keys}

    trajectory_path = Path(summary["trajectory"])
    recording_path = Path(summary["rerun_recording"])
    runtime_model = Path(summary["runtime_model"])
    if sha256_file(trajectory_path) != summary["trajectory_sha256"]:
        raise AssertionError("Trajectory hash mismatch")
    if sha256_file(recording_path) != summary["rerun_recording_sha256"]:
        raise AssertionError("Rerun recording hash mismatch")
    if sha256_file(runtime_model) != summary["runtime_model_sha256"]:
        raise AssertionError("Runtime model hash mismatch")
    runtime_excludes = {
        (element.get("body1"), element.get("body2"))
        for element in ET.parse(runtime_model).getroot().findall("./contact/exclude")
    }
    summary_excludes = {tuple(pair) for pair in summary.get("self_collision_excludes", [])}
    if runtime_excludes != summary_excludes or runtime_excludes != set(SELF_COLLISION_EXCLUDES):
        raise AssertionError("Runtime self-collision excludes do not match the validated contract")
    runtime_root = ET.parse(runtime_model).getroot()
    runtime_cameras = list(runtime_root.iter("camera"))
    if eye_camera_enabled:
        if (
            eye_camera.get("used_by_policy") is not False
            or eye_camera.get("source") != "mujoco_named_camera"
            or eye_camera.get("camera_mount_source") != "source_urdf_link"
            or eye_camera.get("robot_geometry_source") != "source_urdf_visual_mesh"
            or eye_camera.get("intrinsics_source") != "visualization_default"
            or eye_camera.get("scene") not in {"robot", "grid"}
            or not isinstance(eye_camera.get("resolution"), list)
            or len(eye_camera["resolution"]) != 2
            or int(eye_camera.get("frames_submitted", -1)) != int(eye_camera.get("frames_rendered", -2))
            or int(eye_camera.get("frames_dropped", -1)) != 0
        ):
            raise AssertionError("MuJoCo eye-camera summary is incomplete or invalid")
        named_cameras = [camera for camera in runtime_cameras if camera.get("name") == eye_camera.get("name")]
        parent_matches = [
            body
            for body in runtime_root.iter("body")
            if body.get("name") == eye_camera.get("parent_body")
            and any(camera in list(body) for camera in named_cameras)
        ]
        visual_global = runtime_root.find("./visual/global")
        width, height = (int(value) for value in eye_camera["resolution"])
        if (
            len(named_cameras) != 1
            or len(parent_matches) != 1
            or named_cameras[0].get("pos") != "0 0 0"
            or named_cameras[0].get("quat") != "1 0 0 0"
            or not np.isclose(
                float(named_cameras[0].get("fovy", "nan")),
                float(eye_camera.get("fovy_degrees", "nan")),
            )
            or visual_global is None
            or int(visual_global.get("offwidth", 0)) != width
            or int(visual_global.get("offheight", 0)) != height
        ):
            raise AssertionError("Runtime MJCF eye camera differs from its recorded configuration")
        headlight = runtime_root.find("./visual/headlight")
        indoor_lights = runtime_root.findall("./worldbody/light")
        expected_indoor_lights = [
            {
                "name": "eye_camera_key",
                "pos": "-2.4 -1.6 3.8",
                "dir": "0.5963 0.3975 -0.6957",
                "directional": "false",
                "castshadow": "false",
                "ambient": "0.05 0.042 0.032",
                "diffuse": "1.0 0.85 0.65",
                "specular": "0.20 0.16 0.12",
                "attenuation": "1 0.03 0.02",
                "cutoff": "55",
                "exponent": "6",
            },
            {
                "name": "eye_camera_fill",
                "pos": "2.0 -1.8 2.8",
                "dir": "-0.6283 0.5654 -0.5341",
                "directional": "false",
                "castshadow": "false",
                "ambient": "0.03 0.04 0.055",
                "diffuse": "0.40 0.52 0.72",
                "specular": "0.07 0.10 0.14",
                "attenuation": "1 0.04 0.03",
                "cutoff": "75",
                "exponent": "2",
            },
            {
                "name": "eye_camera_rim",
                "pos": "0 2.4 3.2",
                "dir": "0 -0.7682 -0.6402",
                "directional": "false",
                "castshadow": "false",
                "ambient": "0.01 0.013 0.02",
                "diffuse": "0.55 0.70 0.95",
                "specular": "0.14 0.18 0.24",
                "attenuation": "1 0.02 0.025",
                "cutoff": "45",
                "exponent": "8",
            },
        ]
        if (
            headlight is None
            or headlight.get("ambient") != "0.22 0.22 0.22"
            or headlight.get("diffuse") != "0.45 0.45 0.45"
            or headlight.get("specular") != "0.07 0.07 0.07"
            or [dict(light.attrib) for light in indoor_lights] != expected_indoor_lights
        ):
            raise AssertionError("MuJoCo eye-camera lighting contract is invalid")
        if eye_camera.get("scene") == "grid":
            grid = runtime_root.find("./worldbody/geom[@name='eye_camera_grid_floor']")
            if grid is None or grid.get("contype") != "0" or grid.get("conaffinity") != "0":
                raise AssertionError("Eye-camera grid must remain visual-only")
        adapted_urdf = runtime_model.with_name("robot_mujoco.urdf")
        adapted_root = ET.parse(adapted_urdf).getroot()
        source_root = ET.parse(Path(summary["source_urdf"])).getroot()
        source_visuals = source_root.findall("./link/visual/geometry/mesh")
        adapted_visuals = adapted_root.findall("./link/visual/geometry/mesh")
        adapted_collisions = {
            mesh.get("filename") for mesh in adapted_root.findall("./link/collision/geometry/mesh")
        }
        visual_mesh_root = runtime_model.parent / "urdf_visual_meshes"
        visual_mesh_paths = [Path(mesh.get("filename", "")) for mesh in adapted_visuals]
        visual_colors = adapted_root.findall("./link/visual/material/color")
        rgba_values = [
            tuple(float(value) for value in color.get("rgba", "").split()) for color in visual_colors
        ]
        source_visual_roots = [
            visual_mesh_root / Path(mesh.get("filename", "")).with_suffix("") for mesh in source_visuals
        ]
        if (
            len(adapted_visuals) < len(source_visuals)
            or not visual_mesh_paths
            or len(set(visual_mesh_paths)) != len(visual_mesh_paths)
            or len(visual_colors) != len(adapted_visuals)
            or any(
                len(rgba) != 4 or any(not np.isfinite(value) or value < 0.0 or value > 1.0 for value in rgba)
                for rgba in rgba_values
            )
            or len({tuple(round(value, 6) for value in rgba) for rgba in rgba_values}) < 2
            or any(
                not any(
                    path.parent == visual_root.parent
                    and path.stem.startswith(f"{visual_root.name}_primitive_")
                    for path in visual_mesh_paths
                )
                for visual_root in source_visual_roots
            )
            or any(
                path.suffix != ".msh"
                or not path.is_file()
                or not path.is_relative_to(visual_mesh_root)
                or str(path) in adapted_collisions
                for path in visual_mesh_paths
            )
        ):
            raise AssertionError("MuJoCo eye camera is not using converted source-URDF visual meshes")
    elif runtime_cameras:
        raise AssertionError("Disabled eye-camera mode unexpectedly modified the runtime model")
    hand_mapping_path = Path(summary["hand_mapping_file"])
    if sha256_file(hand_mapping_path) != summary["hand_mapping_file_sha256"]:
        raise AssertionError("Hand mapping file hash mismatch")
    if ActHandGestureConfig.from_json(hand_mapping_path).as_dict() != summary["hand_mapping"]:
        raise AssertionError("Hand mapping summary differs from the configured endpoint table")
    for script in summary["deployment_scripts"].values():
        if sha256_file(Path(script["path"])) != script["sha256"]:
            raise AssertionError(f"Deployment script hash mismatch: {script['path']}")
    subprocess.run(
        ["rerun", "rrd", "verify", str(recording_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    locked_joint_values = {name: float(value) for name, value in dict(summary["locked_joint_values"]).items()}
    simulator = W1Simulator(
        runtime_model,
        SimulationConfig(locked_joint_values=locked_joint_values),
    )
    mapper = ActJointMapper(
        simulator.model,
        gestures=ActHandGestureConfig.from_dict(summary["hand_mapping"]),
        selected_body_names=body_command_names,
    )
    behavior_metrics: dict[str, float] = {}
    quality_report = summary.get(
        "quality_evaluation",
        {"enabled": False, "metrics": [], "tensorboard_tags": []},
    )
    quality_metrics = validate_quality_metrics(tuple(quality_report.get("metrics", [])))
    if bool(quality_report.get("enabled")) != bool(quality_metrics):
        raise AssertionError("Quality evaluation enabled flag differs from its metric selection")
    max_reference_delta_ms = 0.0
    with np.load(trajectory_path, allow_pickle=False) as trajectory:
        expected_frames = int(summary["frames"])
        if trajectory["act_state"].shape != (expected_frames, 19):
            raise AssertionError("ACT state trajectory shape mismatch")
        if trajectory["act_action"].shape != (expected_frames, 19):
            raise AssertionError("ACT action trajectory shape mismatch")
        if trajectory["body_position_command"].shape != (
            expected_frames,
            len(body_command_names),
        ):
            raise AssertionError("W1 body position-command trajectory shape mismatch")
        recorded_body_names = tuple(
            value.decode("utf-8") for value in trajectory["body_position_command_names"]
        )
        if recorded_body_names != body_command_names:
            raise AssertionError("W1 body position-command names differ from the command contract")
        if trajectory["left_hand_position_command"].shape != (expected_frames, 6):
            raise AssertionError("W1 left-hand position-command trajectory shape mismatch")
        if trajectory["right_hand_position_command"].shape != (expected_frames, 6):
            raise AssertionError("W1 right-hand position-command trajectory shape mismatch")
        if trajectory["sim_target"].shape != (expected_frames, 29):
            raise AssertionError("Simulator target trajectory shape mismatch")
        if trajectory["runtime_cycle_ms"].shape != (expected_frames,):
            raise AssertionError("Runtime cycle trajectory shape mismatch")
        if trajectory["runtime_deadline_misses"].shape != (expected_frames,):
            raise AssertionError("Runtime deadline trajectory shape mismatch")
        if trajectory["feedback_act_state"].shape != (expected_frames, 19):
            raise AssertionError("ACT feedback-state trajectory shape mismatch")
        numeric_keys = (
            "sim_qpos_before",
            "sim_qpos_after",
            "feedback_act_state",
            "act_state",
            "act_action",
            "body_position_command",
            "left_hand_position_command",
            "right_hand_position_command",
            "sim_target",
        )
        if not all(np.isfinite(trajectory[key]).all() for key in numeric_keys):
            raise AssertionError("Trajectory contains non-finite values")
        cycle_values = np.asarray(trajectory["runtime_cycle_ms"], dtype=np.float64)
        deadline_values = np.asarray(trajectory["runtime_deadline_misses"], dtype=np.int64)
        if (
            np.any(cycle_values < 0.0)
            or np.any(np.diff(deadline_values) < 0)
            or int(deadline_values[-1]) != int(timing["deadline_misses"])
            or not np.isclose(np.mean(cycle_values), timing["mean_cycle_ms"], atol=1e-4, rtol=0.0)
            or not np.isclose(np.percentile(cycle_values, 95), timing["p95_cycle_ms"], atol=1e-4, rtol=0.0)
        ):
            raise AssertionError("Runtime timing summary differs from the per-step trajectory")
        before = np.asarray(trajectory["sim_qpos_before"], dtype=np.float64)
        after = np.asarray(trajectory["sim_qpos_after"], dtype=np.float64)
        if np.any(after < simulator.lower - 1e-6) or np.any(after > simulator.upper + 1e-6):
            raise AssertionError("Trajectory exceeds MuJoCo joint limits")
        reconstructed_states = np.asarray([mapper.act_state_from_sim(row) for row in before])
        np.testing.assert_allclose(
            reconstructed_states, trajectory["feedback_act_state"], atol=1e-5, rtol=0.0
        )
        if action_pipeline == "raw":
            np.testing.assert_allclose(reconstructed_states, trajectory["act_state"], atol=1e-5, rtol=0.0)
        else:
            np.testing.assert_allclose(
                trajectory["act_state"][0], trajectory["feedback_act_state"][0], atol=1e-5, rtol=0.0
            )
            if expected_frames > 1:
                body_index = {name: index for index, name in enumerate(BODY_JOINTS)}
                selected_indices = np.asarray(
                    [body_index[name] for name in body_command_names],
                    dtype=np.int64,
                )
                expected_bridge_states = np.repeat(
                    trajectory["act_state"][[0]],
                    expected_frames - 1,
                    axis=0,
                )
                expected_bridge_states[:, selected_indices] = trajectory["act_action"][:-1, selected_indices]
                expected_bridge_states[:, len(BODY_JOINTS) :] = trajectory["act_action"][
                    :-1, len(BODY_JOINTS) :
                ]
                np.testing.assert_allclose(
                    trajectory["act_state"][1:],
                    expected_bridge_states,
                    atol=1e-5,
                    rtol=0.0,
                )
        reconstructed_commands = [mapper.act_action_to_command(row) for row in trajectory["act_action"]]
        np.testing.assert_allclose(
            np.asarray([command.body.position for command in reconstructed_commands]),
            trajectory["body_position_command"],
            atol=1e-5,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray([command.left_hand.value for command in reconstructed_commands]),
            trajectory["left_hand_position_command"],
            atol=1e-5,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray([command.right_hand.value for command in reconstructed_commands]),
            trajectory["right_hand_position_command"],
            atol=1e-5,
            rtol=0.0,
        )
        simulator.reset(before[0], before[0])
        reconstructed_target_rows = []
        for command in reconstructed_commands:
            target = simulator.target_from_command(command)
            reconstructed_target_rows.append(target)
            simulator.set_target(target)
        reconstructed_targets = np.asarray(reconstructed_target_rows)
        np.testing.assert_allclose(reconstructed_targets, trajectory["sim_target"], atol=1e-5, rtol=0.0)
        legacy_targets = np.asarray([mapper.act_action_to_target(row) for row in trajectory["act_action"]])
        selected_body_indices = mapper.act_adapter.selected_body_indices
        np.testing.assert_allclose(
            legacy_targets[:, selected_body_indices],
            reconstructed_targets[:, selected_body_indices],
            atol=1e-5,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            legacy_targets[:, len(BODY_JOINTS) :],
            reconstructed_targets[:, len(BODY_JOINTS) :],
            atol=1e-5,
            rtol=0.0,
        )
        omitted_body_indices = np.asarray(
            [index for index in range(len(BODY_JOINTS)) if index not in selected_body_indices],
            dtype=np.int64,
        )
        if len(omitted_body_indices):
            previous_targets = np.vstack((before[[0]], reconstructed_targets[:-1]))
            np.testing.assert_allclose(
                reconstructed_targets[:, omitted_body_indices],
                previous_targets[:, omitted_body_indices],
                atol=1e-5,
                rtol=0.0,
            )
        if control_mode == "kinematic":
            np.testing.assert_allclose(after, trajectory["sim_target"], atol=1e-6, rtol=0.0)

        _verify_action_pipeline(trajectory, summary, mapper.gestures)

        replay = OriginReplay(Path(summary["origin"]), camera_sources=camera_sources)
        frames = replay.select(
            int(summary["start_frame"]), int(summary.get("source_frames", expected_frames))
        )
        source_frame_indices = np.asarray(trajectory["source_frame_index"], dtype=np.int32)
        evaluation_timestamps = np.asarray(trajectory["timestamp"], dtype=np.float64)
        if state_aligned_images:
            evaluation_timestamps = np.asarray(
                [frames[index].timestamp for index in source_frame_indices],
                dtype=np.float64,
            )
        reference, max_reference_delta_ms = _reference_states(Path(summary["origin"]), evaluation_timestamps)
        actual_states = np.asarray([mapper.act_state_from_sim(row) for row in after])
        state_span = np.concatenate(
            (
                np.maximum(
                    mapper.upper[: len(BODY_JOINTS)] - mapper.lower[: len(BODY_JOINTS)],
                    1e-9,
                ),
                np.asarray([100.0, 100.0]),
            )
        )
        behavior_metrics = _behavior_metrics(reference, actual_states)
        if max_reference_delta_ms > 50.0:
            raise AssertionError(f"Origin pose alignment exceeded 50 ms: {max_reference_delta_ms:.3f} ms")
        is_full_recording = int(summary.get("start_frame", -1)) == 0 and int(
            summary.get("source_frames", -1)
        ) == int(summary.get("available_synchronized_frames", -2))
        if control_mode == "kinematic" and is_full_recording:
            _enforce_kinematic_behavior(behavior_metrics)

        expected_quality_arrays: dict[str, np.ndarray] = {}
        score_reference, _, _ = load_reference_states(Path(summary["origin"]), evaluation_timestamps)
        if quality_metrics:
            evaluator = MotionQualityEvaluator(
                Path(summary["origin"]),
                evaluation_timestamps,
                mapper,
                before[0],
                quality_metrics,
            )
            for step, qpos in enumerate(after):
                evaluator.evaluate(step, qpos)
            expected_quality = evaluator.summary()
            expected_quality["reference_timebase"] = (
                "selected_image_frame" if state_aligned_images else "control_timestamp"
            )
            expected_quality_arrays = evaluator.trajectory_arrays()
            for name, expected in expected_quality_arrays.items():
                if name not in trajectory:
                    raise AssertionError(f"Missing quality trajectory: {name}")
                np.testing.assert_allclose(trajectory[name], expected, atol=1e-5, rtol=0.0)
            for key in (
                "enabled",
                "metrics",
                "weights",
                "reference_file",
                "reference_sha256",
                "reference_use",
                "reference_timebase",
                "tensorboard_tags",
                "score_contract",
            ):
                if quality_report.get(key) != expected_quality[key]:
                    raise AssertionError(f"Quality summary contract mismatch: {key}")
            for group in ("mean", "final"):
                for name, expected in expected_quality[group].items():
                    if not np.isclose(quality_report[group][name], expected, atol=1e-5, rtol=0.0):
                        raise AssertionError(f"Quality summary value mismatch: {group}.{name}")
            if not np.isclose(
                quality_report["minimum_score"], expected_quality["minimum_score"], atol=1e-5, rtol=0.0
            ):
                raise AssertionError("Quality minimum score mismatch")
            if quality_report.get("execution") != "asynchronous_during_rollout":
                raise AssertionError("Process quality evaluation is not asynchronous")

        expected_score = compute_run_score(
            quality_arrays=expected_quality_arrays,
            actual_states=actual_states,
            reference_states=score_reference,
            state_span=state_span,
            timing=timing,
            control_hz=control_hz,
            frames=expected_frames,
            joint_limit_violations=int(summary["joint_limit_violations"]),
            target_step_errors=np.asarray(trajectory["target_step_error"]),
            action_pipeline=action_pipeline,
            enable_smoothness=bool(score_requested["smoothness"]),
            enable_realtime=bool(score_requested["realtime"]),
        )
        for name, expected in expected_score.trajectory_arrays.items():
            if name not in trajectory:
                raise AssertionError(f"Missing run-score trajectory: {name}")
            np.testing.assert_allclose(trajectory[name], expected, atol=1e-5, rtol=0.0)
        for key in (
            "enabled",
            "valid",
            "base_weights",
            "normalized_weights",
            "selected_components",
            "disabled_reasons",
            "requested",
            "comparable",
            "safety_status",
            "safety_checks",
            "execution",
            "tensorboard_tags",
            "contract",
        ):
            if score_report.get(key) != expected_score.summary[key]:
                raise AssertionError(f"Run score contract mismatch: {key}")
        for name, expected in expected_score.summary["components"].items():
            actual = score_report["components"].get(name)
            if expected is None:
                if actual is not None:
                    raise AssertionError(f"Run score component should be disabled: {name}")
            elif not np.isclose(float(actual), float(expected), atol=1e-5, rtol=0.0):
                raise AssertionError(f"Run score component mismatch: {name}")
        expected_average = expected_score.summary["average_score"]
        actual_average = score_report.get("average_score")
        if expected_average is None:
            if actual_average is not None:
                raise AssertionError("Invalid run must not report an official average score")
        elif not np.isclose(float(actual_average), float(expected_average), atol=1e-5, rtol=0.0):
            raise AssertionError("Run average score mismatch")

        clock_source_frame_indices = np.asarray(trajectory["clock_source_frame_index"], dtype=np.int32)
        expected_source_indices: list[int] = []
        source_index = 0
        for timestamp in np.asarray(trajectory["timestamp"], dtype=np.float64):
            while (
                source_index + 1 < len(frames)
                and float(frames[source_index + 1].timestamp) <= timestamp + 1e-9
            ):
                source_index += 1
            expected_source_indices.append(source_index)
        np.testing.assert_array_equal(clock_source_frame_indices, expected_source_indices)
        image_match_distances = np.asarray(trajectory["image_match_distance"], dtype=np.float32)
        image_match_frozen = np.asarray(trajectory["image_match_frozen"], dtype=np.bool_)
        image_selection_updated = np.asarray(trajectory["image_selection_updated"], dtype=np.bool_)
        expected_image_updates = np.concatenate(([True], np.diff(clock_source_frame_indices) != 0))
        if (
            image_match_distances.shape != (expected_frames,)
            or image_match_frozen.shape != (expected_frames,)
            or image_selection_updated.shape != (expected_frames,)
        ):
            raise AssertionError("State-aligned image diagnostics have invalid shapes")
        if not np.isfinite(image_match_distances).all() or np.any(image_match_distances < 0.0):
            raise AssertionError("State-aligned image distance contains invalid values")
        np.testing.assert_array_equal(image_selection_updated, expected_image_updates)
        if state_aligned_images:
            if action_pipeline != "bridge":
                raise AssertionError("State-aligned image replay is only valid for bridge runs")
            if image_replay.get("selection_state_source") != "mujoco_feedback_19d_for_frame_selection_only":
                raise AssertionError("State-aligned image replay uses the wrong selection state")
            if image_replay.get("reference_joint_use") != "frame_selection_only_not_policy_state":
                raise AssertionError("State-aligned image replay leaked reference state into the policy")
            if image_replay.get("allow_backward") is not False:
                raise AssertionError("State-aligned image replay must be forward-only")
            if np.any(np.diff(source_frame_indices) < 0):
                raise AssertionError("State-aligned image replay moved backward")
            max_advance = int(image_replay["max_advance_frames"])
            if np.any(np.diff(source_frame_indices) > max_advance):
                raise AssertionError("State-aligned image replay exceeded its advance limit")
            image_reference_states, _, _ = load_reference_states(
                Path(summary["origin"]),
                np.asarray([frame.timestamp for frame in frames], dtype=np.float64),
            )
            selector = StateAlignedFrameSelector(
                frames,
                image_reference_states,
                state_span,
                search_ahead_frames=int(image_replay["search_ahead_frames"]),
                max_advance_frames=max_advance,
                match_threshold=float(image_replay["match_threshold"]),
                similarity_slack=float(image_replay["similarity_slack"]),
            )
            expected_selected_indices: list[int] = []
            expected_match_distances: list[float] = []
            expected_frozen_flags: list[bool] = []
            selected_index = 0
            selected_distance = 0.0
            selected_frozen = False
            selection_count = 0
            feedback_states = np.asarray(trajectory["feedback_act_state"], dtype=np.float64)
            for step, selection_updated in enumerate(expected_image_updates):
                if selection_updated:
                    selection = (
                        selector.current(feedback_states[step])
                        if selection_count == 0
                        else selector.select(feedback_states[step])
                    )
                    selection_count += 1
                    selected_index = selection.index
                    selected_distance = selection.match_distance
                    selected_frozen = selection.frozen
                expected_selected_indices.append(selected_index)
                expected_match_distances.append(selected_distance)
                expected_frozen_flags.append(selected_frozen)
            np.testing.assert_array_equal(source_frame_indices, expected_selected_indices)
            np.testing.assert_allclose(
                image_match_distances,
                expected_match_distances,
                atol=1e-5,
                rtol=0.0,
            )
            np.testing.assert_array_equal(image_match_frozen, expected_frozen_flags)
            selector_summary = selector.summary()
            for key in ("final_frame_index", "freeze_count", "repeat_count"):
                if int(image_replay[key]) != int(selector_summary[key]):
                    raise AssertionError(f"State-aligned image selector summary mismatch: {key}")
            for key in ("mean_match_distance", "p95_match_distance", "max_match_distance"):
                if not np.isclose(
                    float(image_replay[key]),
                    float(selector_summary[key]),
                    atol=1e-5,
                    rtol=0.0,
                ):
                    raise AssertionError(f"State-aligned image selector summary mismatch: {key}")
        else:
            np.testing.assert_array_equal(source_frame_indices, expected_source_indices)
            if np.any(image_match_distances != 0.0) or np.any(image_match_frozen):
                raise AssertionError("Time-based image replay recorded state-alignment diagnostics")
        expected_visualization_steps = np.flatnonzero(expected_image_updates)
        np.testing.assert_array_equal(trajectory["visualization_step"], expected_visualization_steps)
        np.testing.assert_array_equal(
            summary.get("visualization_source_frames", []),
            source_frame_indices[expected_visualization_steps],
        )
        cached_source_index = -1
        cached_hashes: tuple[dict[str, str], dict[str, str]] | None = None
        for index, source_index in enumerate(source_frame_indices):
            frame = frames[int(source_index)]
            if int(source_index) != cached_source_index:
                _, source_hashes, input_hashes = frame.load_images()
                cached_hashes = source_hashes, input_hashes
                cached_source_index = int(source_index)
            if cached_hashes is None:
                raise AssertionError("Origin image hash cache was not initialized")
            source_hashes, input_hashes = cached_hashes
            expected_source = [value.decode("ascii") for value in trajectory["source_image_sha256"][index]]
            expected_input = [value.decode("ascii") for value in trajectory["model_input_sha256"][index]]
            if expected_source != [source_hashes[key] for key in image_keys]:
                raise AssertionError(f"Source image hash mismatch at rollout frame {index}")
            if expected_input != [input_hashes[key] for key in image_keys]:
                raise AssertionError(f"Model input image hash mismatch at rollout frame {index}")
        if eye_camera_enabled:
            effective_camera_fps = float(eye_camera.get("effective_fps", 0.0))
            schedule = EyeCameraSchedule(effective_camera_fps, control_hz)
            expected_eye_steps = np.asarray(
                [step for step in range(expected_frames) if schedule.due(step)], dtype=np.int32
            )
        else:
            expected_eye_steps = np.asarray([], dtype=np.int32)
        np.testing.assert_array_equal(trajectory["eye_camera_step"], expected_eye_steps)
        expected_rerun_steps = np.union1d(expected_visualization_steps, expected_eye_steps).astype(np.int32)
        np.testing.assert_array_equal(trajectory["rerun_state_step"], expected_rerun_steps)
        if int(summary["rerun_state_frames"]) != len(expected_rerun_steps):
            raise AssertionError("Rerun state-frame count differs from its trajectory")
        if eye_camera_enabled and (
            int(eye_camera["frames_submitted"]) != len(expected_eye_steps)
            or int(eye_camera["frames_rendered"]) != len(expected_eye_steps)
        ):
            raise AssertionError("MuJoCo eye-camera frame count differs from its schedule")
    simulator.close()

    expected_input_rows = int(summary["visualization_frames"])
    expected_state_rows = int(summary["rerun_state_frames"])
    rrd_rows = {
        "joints/qpos": _rrd_rows(recording_path, "joints/qpos"),
        **{
            f"observation/{key.removeprefix('observation.images.')}": _rrd_rows(
                recording_path, f"observation/{key.removeprefix('observation.images.')}"
            )
            for key in image_keys
        },
        **{
            f"observation/{key.removeprefix('observation.images.')}/sha256": _rrd_rows(
                recording_path,
                f"observation/{key.removeprefix('observation.images.')}/sha256",
            )
            for key in image_keys
        },
    }
    if rrd_rows["joints/qpos"] != expected_state_rows or any(
        rows != expected_input_rows for entity, rows in rrd_rows.items() if entity != "joints/qpos"
    ):
        raise AssertionError(
            "Rerun visualization row mismatch: "
            f"{rrd_rows}, state_expected={expected_state_rows}, input_expected={expected_input_rows}"
        )
    if eye_camera_enabled:
        camera_entity = f"world/robot/{eye_camera['parent_body']}/{eye_camera['name']}"
        rrd_rows[f"{camera_entity}/rgb"] = _rrd_rows(recording_path, f"{camera_entity}/rgb")
        rrd_rows[f"{camera_entity}/calibration"] = _rrd_rows(recording_path, camera_entity)
        rrd_rows["metrics/eye_camera/render_ms"] = _rrd_rows(recording_path, "metrics/eye_camera/render_ms")
        expected_eye_rows = int(eye_camera["frames_rendered"])
        if (
            rrd_rows[f"{camera_entity}/rgb"] != expected_eye_rows
            or rrd_rows[f"{camera_entity}/calibration"] < 2
            or rrd_rows["metrics/eye_camera/render_ms"] != expected_eye_rows
        ):
            raise AssertionError(
                f"MuJoCo eye-camera Rerun row mismatch: {rrd_rows}, expected={expected_eye_rows}"
            )

    event_files = sorted(Path(summary["tensorboard"]).glob("events.out.tfevents.*"))
    if len(event_files) != 1:
        raise AssertionError(f"Expected one TensorBoard event file, got {len(event_files)}")
    accumulator = EventAccumulator(str(event_files[0]))
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags().get("scalars", []))
    required_tensorboard_tags = (
        REQUIRED_TENSORBOARD_TAGS
        | set(quality_report["tensorboard_tags"])
        | set(score_report["tensorboard_tags"])
    )
    if action_pipeline == "bridge":
        required_tensorboard_tags |= {
            "scheduler/lipo_blend_active",
            "scheduler/lipo_blend_alpha",
        }
    missing_tags = required_tensorboard_tags - scalar_tags
    if missing_tags:
        raise AssertionError(f"Missing TensorBoard tags: {sorted(missing_tags)}")
    for tag in required_tensorboard_tags:
        events = accumulator.Scalars(tag)
        if not events or not all(np.isfinite(event.value) for event in events):
            raise AssertionError(f"TensorBoard tag has no finite values: {tag}")

    report_path = paths["verification"]
    save_verification_report(
        report_path,
        {
            "status": "passed",
            "run_name": run_name,
            "summary": str(summary_path.resolve()),
            "frames": expected_frames,
            "state_source": summary["state_source_after_reset"],
            "control_mode": control_mode,
            "action_pipeline": action_pipeline,
            "action_processor": summary["action_processor"],
            "robot_command_contract": summary["robot_command_contract"],
            "inference_schedule": summary["inference_schedule"],
            "reference_pose_use": (
                "asynchronous_process_quality_and_offline_run_score_verification"
                if quality_metrics
                else "offline_run_score_verification_only"
            ),
            "max_reference_alignment_delta_ms": max_reference_delta_ms,
            "behavior_metrics": behavior_metrics,
            "quality_evaluation": quality_report,
            "run_score": score_report,
            "behavior_limits": KINEMATIC_BEHAVIOR_LIMITS if control_mode == "kinematic" else None,
            "origin_pose_reads": summary["origin_pose_reads"],
            "joint_limit_violations": summary["joint_limit_violations"],
            "rrd_rows": rrd_rows,
            "rerun_view_mode": rerun_view_mode,
            "rerun_layout": rerun_layout,
            "eye_camera": eye_camera,
            "tensorboard_tags": sorted(scalar_tags),
        },
    )
    print(f"ACT_SIM_VERIFICATION={report_path}")
    print("ACT_SIM_VERIFICATION_STATUS=passed")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify ACT simulation artifacts")
    parser.add_argument("--run-directory", type=Path, required=True)
    args = parser.parse_args()
    verify_run(args.run_directory)


if __name__ == "__main__":
    main()
