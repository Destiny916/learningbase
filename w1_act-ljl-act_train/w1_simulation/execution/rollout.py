from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from w1_simulation.artifacts import (
    ensure_simulation_artifact_dirs,
    sha256_file,
    simulation_run_directory,
)
from w1_simulation.cli import camera_name as _camera_name
from w1_simulation.cli import parse_camera_sources as _parse_camera_sources
from w1_simulation.control.bridge import (
    ActionChunkController,
    LipoActionChunkController,
    LipoControllerConfig,
)
from w1_simulation.control.processing import (
    build_action_processor,
    processor_manifest,
)
from w1_simulation.control.scheduling import SOURCE_IMAGE_HZ
from w1_simulation.control.scheduling import control_ticks as _control_ticks
from w1_simulation.control.scheduling import format_bridge_inference_log as _format_bridge_inference_log
from w1_simulation.control.scheduling import raw_control_ticks as _raw_control_ticks
from w1_simulation.evaluation.quality import (
    AsyncMotionQualityEvaluator,
    MotionQualityEvaluator,
    load_reference_states,
    validate_quality_metrics,
)
from w1_simulation.evaluation.scoring import compute_run_score
from w1_simulation.execution.recording import save_trajectory
from w1_simulation.execution.summary import save_summary
from w1_simulation.inference.contract import resolve_execution_horizon as _resolve_execution_horizon
from w1_simulation.inference.direct import ActPolicyRuntime, inspect_checkpoint_contract
from w1_simulation.inference.subprocess import ScriptPolicyRuntime
from w1_simulation.observability.system import gpu_metrics
from w1_simulation.replay.origin import InitialPoseLoader, OriginReplay, StateAlignedFrameSelector
from w1_simulation.robot.commands import position_command_contract
from w1_simulation.robot.mapping import ActHandGestureConfig, ActJointMapper
from w1_simulation.simulation.camera import RERUN_VIEW_MODES, EyeCameraConfig, EyeCameraSchedule
from w1_simulation.simulation.config import (
    BODY_JOINTS,
    SELF_COLLISION_EXCLUDES,
    ActSimulationConfig,
    SimulationConfig,
)
from w1_simulation.simulation.model import build_runtime_model
from w1_simulation.simulation.simulator import W1Simulator
from w1_simulation.simulation.telemetry import AsyncRerunTelemetry
from w1_simulation.w1_profile import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_BRIDGE_SCRIPT,
    DEFAULT_CHECKPOINT,
    DEFAULT_ORIGIN,
    DEFAULT_POLICY_SCRIPT,
    DEFAULT_PROFILE,
    DEFAULT_WHOLE_SCRIPT,
    W1Profile,
)


def run_act_simulation(
    checkpoint: Path,
    origin_root: Path,
    artifact_root: Path,
    run_name: str,
    profile: W1Profile = DEFAULT_PROFILE,
    start_frame: int = 0,
    max_frames: int = 0,
    device: str = "cuda:0",
    rerun_url: str | None = None,
    save_artifacts: bool = True,
    realtime: bool = True,
    policy_backend: str = "direct",
    policy_script: Path = DEFAULT_POLICY_SCRIPT,
    bridge_script: Path = DEFAULT_BRIDGE_SCRIPT,
    whole_script: Path = DEFAULT_WHOLE_SCRIPT,
    camera_sources: dict[str, str] | None = None,
    control_mode: str = "kinematic",
    action_pipeline: str = "raw",
    execution_horizon: int = 0,
    replan_interval: int = 0,
    bridge_simulated_inference_ms: float = 200.0,
    bridge_inference_budget_ms: float = 300.0,
    bridge_policy_hz: float = 20.0,
    bridge_replan_threshold: float = 0.5,
    bridge_lipo_blend_policy_points: int = 5,
    bridge_replan_margin_policy_points: int = 2,
    bridge_sample_factor: int = 2,
    image_replay_mode: str = "time",
    image_search_ahead_frames: int = 15,
    image_max_advance_frames: int = 2,
    image_match_threshold: float = 0.18,
    image_similarity_slack: float = 0.005,
    quality_metrics: tuple[str, ...] | list[str] = (),
    score_smoothness: bool = True,
    score_realtime: bool = True,
    rerun_view_mode: str = "eye",
    eye_camera_width: int = 1280,
    eye_camera_height: int = 720,
    eye_camera_fps: float = 30.0,
    eye_camera_fovy: float = 70.0,
    eye_camera_scene: str = "grid",
) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("run_name may contain only letters, digits, underscores, and hyphens")
    if not torch.cuda.is_available() or not device.startswith("cuda"):
        raise RuntimeError("ACT simulation requires CUDA and does not fall back to CPU")
    for script in (policy_script, bridge_script, whole_script):
        if not Path(script).resolve().is_file():
            raise FileNotFoundError(Path(script).resolve())
    if action_pipeline not in {"raw", "bridge"}:
        raise ValueError(f"Unknown action pipeline: {action_pipeline}")
    if rerun_view_mode not in RERUN_VIEW_MODES:
        raise ValueError(f"Unknown Rerun view mode: {rerun_view_mode}")
    if image_replay_mode not in {"time", "state"}:
        raise ValueError(f"Unknown image replay mode: {image_replay_mode}")
    if image_replay_mode == "state" and action_pipeline != "bridge":
        raise ValueError("State-aligned image replay is reserved for the bridge validation path")
    quality_metrics = validate_quality_metrics(quality_metrics)
    paths = ensure_simulation_artifact_dirs(artifact_root)
    summary_path = paths["summary"]
    trajectory_path = paths["trajectory"]
    recording_path = paths["rerun"] if save_artifacts else None
    for output in (summary_path, trajectory_path, recording_path):
        if output is not None and output.exists():
            raise FileExistsError(f"Refusing to overwrite existing run artifact: {output}")

    checkpoint_contract = inspect_checkpoint_contract(checkpoint)
    image_shapes = checkpoint_contract.image_shapes
    prediction_horizon = checkpoint_contract.prediction_horizon
    checkpoint_execution_horizon = checkpoint_contract.execution_horizon
    execution_horizon, execution_horizon_source = _resolve_execution_horizon(
        prediction_horizon,
        checkpoint_execution_horizon,
        execution_horizon,
    )
    lipo_config = (
        LipoControllerConfig(
            simulated_inference_ms=bridge_simulated_inference_ms,
            inference_budget_ms=bridge_inference_budget_ms,
            replan_threshold=bridge_replan_threshold,
            lipo_blend_policy_points=bridge_lipo_blend_policy_points,
            replan_margin_policy_points=bridge_replan_margin_policy_points,
            policy_hz=bridge_policy_hz,
            sample_factor=bridge_sample_factor,
            execution_horizon=execution_horizon,
        )
        if action_pipeline == "bridge"
        else None
    )
    effective_replan_interval = execution_horizon if replan_interval == 0 else replan_interval
    if effective_replan_interval <= 0:
        raise ValueError("replan_interval must be zero (automatic) or positive")
    if camera_sources is None:
        missing_defaults = set(image_shapes) - set(profile.camera_sources)
        if missing_defaults:
            raise ValueError(
                "Checkpoint image inputs have no default replay source; configure --camera-source for: "
                f"{sorted(missing_defaults)}"
            )
        requested_sources = {key: profile.camera_sources[key] for key in image_shapes}
    else:
        requested_sources = dict(camera_sources)
    if set(requested_sources) != set(image_shapes):
        raise ValueError(
            "Configured camera inputs must exactly match the ACT checkpoint: "
            f"configured={sorted(requested_sources)}, checkpoint={sorted(image_shapes)}"
        )
    camera_sources = {key: requested_sources[key] for key in image_shapes}
    image_keys = tuple(camera_sources)
    gestures = ActHandGestureConfig.from_dict(profile.hands)
    active_sample_factor = bridge_sample_factor if action_pipeline == "bridge" else 1
    action_processor = build_action_processor(action_pipeline, active_sample_factor)
    action_processor_manifest = processor_manifest(action_processor)
    control_hz = bridge_policy_hz * active_sample_factor if action_pipeline == "bridge" else SOURCE_IMAGE_HZ
    eye_camera = EyeCameraConfig(
        enabled=rerun_view_mode != "standard",
        width=eye_camera_width,
        height=eye_camera_height,
        fps=eye_camera_fps,
        fovy_degrees=eye_camera_fovy,
        scene=eye_camera_scene,
    )
    effective_eye_camera_fps = eye_camera.effective_fps(control_hz) if eye_camera.enabled else 0.0
    eye_camera_schedule = (
        EyeCameraSchedule(effective_eye_camera_fps, control_hz) if eye_camera.enabled else None
    )
    base_config = ActSimulationConfig()
    frame_skip = base_config.frame_skip
    simulation_timestep = 1.0 / (control_hz * frame_skip)
    act_config = ActSimulationConfig(
        control_hz=control_hz,
        timestep=simulation_timestep,
        frame_skip=frame_skip,
        replan_interval=effective_replan_interval,
        control_mode=control_mode,
    )
    replay = OriginReplay(
        origin_root,
        max_camera_skew_ms=act_config.max_camera_skew_ms,
        camera_sources=camera_sources,
    )
    source_frames = replay.select(start_frame, max_frames)
    control_ticks = (
        _raw_control_ticks(source_frames)
        if action_pipeline == "raw"
        else _control_ticks(source_frames, control_hz)
    )
    pose_loader = InitialPoseLoader(origin_root)
    initial_pose, pose_match = pose_loader.read_nearest(source_frames[0].timestamp)
    locked_values = profile.locked_joint_values
    simulation_config = SimulationConfig(
        timestep=act_config.timestep,
        frame_skip=act_config.frame_skip,
        body_kp=act_config.body_kp,
        hand_kp=act_config.hand_kp,
        locked_joint_values=locked_values,
    )
    runtime_model = build_runtime_model(
        paths["generated"],
        source=profile.urdf,
        config=simulation_config,
        locked_joint_values=locked_values,
        eye_camera=eye_camera,
    )
    simulator = W1Simulator(runtime_model, simulation_config)
    mapper = ActJointMapper(
        simulator.model,
        gestures=gestures,
        selected_body_names=profile.body_command_names,
    )
    initial_target = mapper.initial_target_from_pose(initial_pose)
    simulator.reset(initial_target, initial_target)
    del initial_pose
    if pose_loader.read_count != 1:
        raise AssertionError("Origin pose must be read exactly once during reset")
    state_span = np.concatenate(
        (
            np.maximum(
                mapper.upper[: len(BODY_JOINTS)] - mapper.lower[: len(BODY_JOINTS)],
                1e-9,
            ),
            np.asarray([100.0, 100.0]),
        )
    )
    image_selector: StateAlignedFrameSelector | None = None
    image_reference_states: np.ndarray | None = None
    if image_replay_mode == "state":
        image_reference_states, _, _ = load_reference_states(
            origin_root,
            np.asarray([frame.timestamp for frame in source_frames], dtype=np.float64),
        )
        image_selector = StateAlignedFrameSelector(
            source_frames,
            image_reference_states,
            state_span,
            search_ahead_frames=image_search_ahead_frames,
            max_advance_frames=image_max_advance_frames,
            match_threshold=image_match_threshold,
            similarity_slack=image_similarity_slack,
        )
    quality_core = (
        MotionQualityEvaluator(
            origin_root,
            np.asarray([tick_timestamp for _, tick_timestamp, _ in control_ticks], dtype=np.float64),
            mapper,
            simulator.current_qpos,
            quality_metrics,
            execution="asynchronous_during_rollout",
        )
        if quality_metrics
        else None
    )

    if policy_backend == "direct":
        policy = ActPolicyRuntime(checkpoint, device=device, execution_horizon=execution_horizon)
    elif policy_backend == "script":
        policy = ScriptPolicyRuntime(
            checkpoint,
            script=policy_script,
            device=device,
            image_shapes=image_shapes,
            execution_horizon=execution_horizon,
        )
    else:
        raise ValueError(f"Unknown policy backend: {policy_backend}")
    script_path = Path(policy.script_path) if policy_backend == "script" else None
    script_pid = policy.server_pid if policy_backend == "script" else None
    if action_pipeline == "raw":
        controller = ActionChunkController(
            policy,
            action_processor,
            replan_interval=act_config.replan_interval,
            asynchronous=False,
        )
        inference_schedule = "synchronous_latest"
    else:
        if lipo_config is None:
            raise AssertionError("Bridge action pipeline has no LIPO controller configuration")
        controller = LipoActionChunkController(
            policy,
            action_processor,
            config=lipo_config,
            asynchronous=True,
            published_body_indices=mapper.act_adapter.selected_body_indices,
        )
        inference_schedule = "async_remaining_ratio_absolute_step_lipo"
    bootstrap_images, _, _ = source_frames[0].load_images()
    controller.reset(mapper.act_state_from_sim(simulator.current_qpos), bootstrap_images)
    telemetry = AsyncRerunTelemetry(
        simulator.model,
        grpc_url=rerun_url,
        recording_path=recording_path,
        source_urdf=profile.urdf,
        application_id="dexforce_w1_simulation",
        camera_streams=tuple(_camera_name(key) for key in image_keys),
        view_mode=rerun_view_mode,
        eye_camera=eye_camera,
    )
    tensorboard_dir = paths["tensorboard"]
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    quality_evaluator = AsyncMotionQualityEvaluator(quality_core) if quality_core is not None else None
    for name, value in gpu_metrics().items():
        writer.add_scalar(f"system/{name}", value, 0)

    sim_state_before: list[np.ndarray] = []
    sim_state_after: list[np.ndarray] = []
    feedback_act_states: list[np.ndarray] = []
    act_states: list[np.ndarray] = []
    act_actions: list[np.ndarray] = []
    body_position_commands: list[np.ndarray] = []
    left_hand_position_commands: list[np.ndarray] = []
    right_hand_position_commands: list[np.ndarray] = []
    processed_queue_actions: list[np.ndarray] = []
    sim_targets: list[np.ndarray] = []
    frame_ids: list[int] = []
    source_frame_indices: list[int] = []
    clock_source_frame_indices: list[int] = []
    source_frame_ids: list[list[int]] = []
    timestamps: list[float] = []
    camera_skews: list[float] = []
    image_match_distances: list[float] = []
    image_frozen_flags: list[bool] = []
    image_selection_updated_flags: list[bool] = []
    source_hashes_all: list[list[str]] = []
    input_hashes_all: list[list[str]] = []
    policy_input_flags: list[bool] = []
    action_indices: list[int] = []
    chunk_origins: list[int] = []
    chunk_install_steps: list[int] = []
    replan_installed_flags: list[bool] = []
    processor_record_indices: list[int] = []
    held_command_flags: list[bool] = []
    queue_sizes: list[int] = []
    blend_active_flags: list[bool] = []
    blend_alphas: list[float] = []
    lipo_old_record_indices: list[int] = []
    lipo_new_record_indices: list[int] = []
    candidate_counts: list[int] = []
    observation_age_steps: list[int] = []
    target_step_errors: list[int] = []
    discarded_prefix_steps: list[int] = []
    control_errors: list[float] = []
    cycle_ms: list[float] = []
    deadline_miss_counts: list[int] = []
    visualization_steps: list[int] = []
    rerun_state_steps: list[int] = []
    eye_camera_steps: list[int] = []
    logged_bridge_inferences: set[int] = set()
    deadline_misses = 0
    limit_violations = 0
    started_unix = time.time()
    started_monotonic = time.monotonic()
    next_deadline = started_monotonic
    rollout_finished_monotonic = started_monotonic
    quality_wait_seconds = 0.0
    telemetry_flush_seconds = 0.0
    deadline_tolerance_s = 0.002

    cached_source_index = -1
    cached_images: tuple[dict[str, np.ndarray], dict[str, str], dict[str, str]] | None = None
    last_clock_source_index = -1
    selected_frame = source_frames[0]
    selected_source_index = 0
    selected_match_distance = 0.0
    selected_frozen = False
    try:
        for step, (clock_frame, tick_timestamp, clock_source_frame_index) in enumerate(control_ticks):
            cycle_started = time.monotonic()
            before = simulator.current_qpos
            feedback_state = mapper.act_state_from_sim(before)
            image_tick = clock_source_frame_index != last_clock_source_index
            if image_tick:
                last_clock_source_index = clock_source_frame_index
                if image_selector is None:
                    selected_frame = clock_frame
                    selected_source_index = clock_source_frame_index
                    selected_match_distance = 0.0
                    selected_frozen = False
                else:
                    selection = (
                        image_selector.current(feedback_state)
                        if step == 0
                        else image_selector.select(feedback_state)
                    )
                    selected_frame = selection.frame
                    selected_source_index = selection.index
                    selected_match_distance = selection.match_distance
                    selected_frozen = selection.frozen
            frame = selected_frame
            source_frame_index = selected_source_index
            image_match_distance = selected_match_distance
            image_frozen = selected_frozen
            source_frame_changed = source_frame_index != cached_source_index
            if source_frame_changed:
                cached_images = frame.load_images()
                cached_source_index = source_frame_index
            if cached_images is None:
                raise RuntimeError("Origin image cache was not initialized")
            images, source_hashes, input_hashes = cached_images
            state = controller.observation_state(feedback_state)
            control = controller.step(step, feedback_state, images)
            queue_action = control.queue_action
            processor_record_index = control.record_index
            held_command = control.held_last_command
            queue_size = control.queue_size
            if processor_record_index >= 0:
                active_record = controller.inference_records[processor_record_index]
                chunk_origin_step = active_record.submit_step
                chunk_install_step = active_record.install_step
            else:
                chunk_origin_step = chunk_origins[-1] if chunk_origins else 0
                chunk_install_step = chunk_install_steps[-1] if chunk_install_steps else 0
            command = mapper.act_action_to_command(control.action)
            if act_config.control_mode == "kinematic":
                after = simulator.step_kinematic(command)
            else:
                after = simulator.step(command)
            target = simulator.target.copy()
            control_rmse = float(np.sqrt(np.mean(np.square(after - target))))
            if quality_evaluator is not None:
                quality_evaluator.submit(
                    step,
                    after,
                    (
                        image_reference_states[source_frame_index]
                        if image_reference_states is not None
                        else None
                    ),
                )
            violations = int(
                np.count_nonzero((after < simulator.lower - 1e-6) | (after > simulator.upper + 1e-6))
            )
            limit_violations += violations
            is_policy_input = step == 0 or control.replan_submitted
            metrics = {
                "control/tracking_rmse": control_rmse,
                "control/joint_limit_violations": float(violations),
                "scheduler/replan_submitted": float(is_policy_input),
                "scheduler/replan_installed": float(control.replan_installed),
                "scheduler/action_index": float(control.action_index),
                "scheduler/chunk_origin_step": float(chunk_origin_step),
                "scheduler/chunk_install_step": float(chunk_install_step),
                "scheduler/queue_size": float(queue_size),
                "scheduler/held_last_command": float(held_command),
                "scheduler/candidate_count": float(control.candidate_count),
                "scheduler/observation_age_steps": float(control.observation_age_steps),
                "scheduler/target_step_error": float(control.target_step_error),
                "scheduler/discarded_prefix_steps": float(control.discarded_prefix_steps),
                "scheduler/lipo_blend_active": float(control.blend_active),
                "scheduler/lipo_blend_alpha": float(control.blend_alpha),
                "inference/e2e_ms": float(control.policy_latency_ms),
                "input/camera_skew_ms": frame.camera_skew_ms,
                "input/state_match_distance": image_match_distance,
                "input/state_match_frozen": float(image_frozen),
                "input/state_aligned_frame_index": float(source_frame_index),
                "input/state_selection_updated": float(image_tick),
            }
            render_eye = eye_camera_schedule is not None and eye_camera_schedule.due(step)
            if image_tick or render_eye:
                telemetry.log_state(
                    step,
                    simulator.data,
                    target,
                    control.action,
                    metrics,
                    images=(
                        {_camera_name(key): image for key, image in images.items()} if image_tick else None
                    ),
                    time_seconds=tick_timestamp - float(source_frames[0].timestamp),
                    render_eye=render_eye,
                )
                rerun_state_steps.append(step)
            if image_tick:
                visualization_steps.append(step)
            if render_eye:
                eye_camera_steps.append(step)
            writer.add_scalar("control/tracking_rmse", control_rmse, step)
            writer.add_scalar("control/joint_limit_violations", violations, step)
            writer.add_scalar("scheduler/replan_count", controller.replan_count, step)
            writer.add_scalar(
                "scheduler/low_watermark",
                controller.low_watermark if controller.low_watermark is not None else -1,
                step,
            )
            writer.add_scalar("scheduler/replan_interval", controller.replan_interval or -1, step)
            writer.add_scalar("scheduler/action_index", control.action_index, step)
            writer.add_scalar("scheduler/queue_size", queue_size, step)
            writer.add_scalar("scheduler/held_last_command", held_command, step)
            writer.add_scalar("scheduler/candidate_count", control.candidate_count, step)
            writer.add_scalar("scheduler/observation_age_steps", control.observation_age_steps, step)
            writer.add_scalar("scheduler/target_step_error", control.target_step_error, step)
            writer.add_scalar("scheduler/discarded_prefix_steps", control.discarded_prefix_steps, step)
            writer.add_scalar("scheduler/lipo_blend_active", control.blend_active, step)
            writer.add_scalar("scheduler/lipo_blend_alpha", control.blend_alpha, step)
            writer.add_scalar("inference/e2e_ms", control.policy_latency_ms, step)
            writer.add_scalar("input/camera_skew_ms", frame.camera_skew_ms, step)
            writer.add_scalar("input/state_match_distance", image_match_distance, step)
            writer.add_scalar("input/state_match_frozen", image_frozen, step)
            writer.add_scalar("input/state_aligned_frame_index", source_frame_index, step)
            writer.add_scalar("input/state_selection_updated", image_tick, step)
            sim_state_before.append(before)
            sim_state_after.append(after)
            feedback_act_states.append(feedback_state)
            act_states.append(state)
            act_actions.append(control.action)
            body_position_commands.append(command.body.position)
            left_hand_position_commands.append(command.left_hand.value)
            right_hand_position_commands.append(command.right_hand.value)
            processed_queue_actions.append(queue_action)
            sim_targets.append(target)
            frame_ids.append(frame.frame_id)
            source_frame_indices.append(source_frame_index)
            clock_source_frame_indices.append(clock_source_frame_index)
            source_frame_ids.append([frame.records[key].frame_id for key in image_keys])
            timestamps.append(tick_timestamp)
            camera_skews.append(frame.camera_skew_ms)
            image_match_distances.append(image_match_distance)
            image_frozen_flags.append(image_frozen)
            image_selection_updated_flags.append(image_tick)
            source_hashes_all.append([source_hashes[key] for key in image_keys])
            input_hashes_all.append([input_hashes[key] for key in image_keys])
            policy_input_flags.append(is_policy_input)
            action_indices.append(control.action_index)
            chunk_origins.append(chunk_origin_step)
            chunk_install_steps.append(chunk_install_step)
            replan_installed_flags.append(control.replan_installed)
            processor_record_indices.append(processor_record_index)
            held_command_flags.append(held_command)
            queue_sizes.append(queue_size)
            blend_active_flags.append(control.blend_active)
            blend_alphas.append(control.blend_alpha)
            lipo_old_record_indices.append(control.old_record_index)
            lipo_new_record_indices.append(control.new_record_index)
            candidate_counts.append(control.candidate_count)
            observation_age_steps.append(control.observation_age_steps)
            target_step_errors.append(control.target_step_error)
            discarded_prefix_steps.append(control.discarded_prefix_steps)
            control_errors.append(control_rmse)

            if realtime:
                next_deadline += 1.0 / act_config.control_hz
                remaining = next_deadline - time.monotonic()
                if remaining < -deadline_tolerance_s:
                    deadline_misses += 1
                elif remaining > 0.0:
                    time.sleep(remaining)
            cycle_duration_ms = (time.monotonic() - cycle_started) * 1000.0
            cycle_ms.append(cycle_duration_ms)
            deadline_miss_counts.append(deadline_misses)
            writer.add_scalar("runtime/cycle_ms", cycle_duration_ms, step)
            writer.add_scalar("runtime/deadline_misses", deadline_misses, step)
            quality_fragment = (
                f" {quality_evaluator.terminal_fragment()}" if quality_evaluator is not None else ""
            )
            if action_pipeline == "bridge" and (step == 0 or control.replan_installed):
                installed_indices = [
                    index
                    for index, record in enumerate(controller.inference_records)
                    if record.install_step == step
                ]
                if len(installed_indices) != 1:
                    raise AssertionError(
                        f"Expected one bridge inference installed at step {step}, got {installed_indices}"
                    )
                installed_index = installed_indices[0]
                print(
                    _format_bridge_inference_log(
                        installed_index,
                        controller.inference_records[installed_index],
                        control.action_index,
                    )
                    + quality_fragment,
                    flush=True,
                )
                logged_bridge_inferences.add(installed_index)
            elif action_pipeline == "raw" and (
                step == 0
                or control.replan_submitted
                or control.action_index == execution_horizon - 1
                or (step + 1) % int(act_config.control_hz) == 0
                or step + 1 == len(control_ticks)
            ):
                print(
                    f"ACT_SIM step={step + 1}/{len(control_ticks)} frame={frame.frame_id} "
                    f"action_index={control.action_index} replan={is_policy_input} "
                    f"e2e_ms={control.policy_latency_ms:.2f} track_rmse={control_rmse:.6f}"
                    f"{quality_fragment}",
                    flush=True,
                )
        rollout_finished_monotonic = time.monotonic()
    finally:
        try:
            controller.close()
        finally:
            try:
                quality_wait_started = time.monotonic()
                if quality_evaluator is not None:
                    quality_evaluator.close()
                quality_wait_seconds = time.monotonic() - quality_wait_started
            finally:
                try:
                    writer.flush()
                    telemetry_flush_started = time.monotonic()
                    telemetry.close()
                    telemetry_flush_seconds = time.monotonic() - telemetry_flush_started
                finally:
                    simulator.close()
                    if hasattr(policy, "close"):
                        policy.close()

    eye_camera_summary = telemetry.eye_camera_summary(effective_eye_camera_fps)
    rerun_layout_summary = telemetry.layout_summary()

    if pose_loader.read_count != 1:
        raise AssertionError("Origin pose was accessed after simulator reset")
    if limit_violations != 0:
        raise AssertionError(f"MuJoCo rollout produced {limit_violations} joint-limit violations")
    inference_records = controller.inference_records
    if action_pipeline == "bridge":
        for record_index, record in enumerate(inference_records):
            if record_index not in logged_bridge_inferences:
                print(_format_bridge_inference_log(record_index, record, -1), flush=True)
    raw_chunks = np.asarray([record.trace.raw for record in inference_records], dtype=np.float32)
    processed_chunks = np.asarray([record.trace.processed for record in inference_records], dtype=np.float32)
    stage_names = tuple(inference_records[0].trace.stages)
    if any(tuple(record.trace.stages) != stage_names for record in inference_records):
        raise AssertionError("Action processor stage schema changed during the rollout")
    stage_arrays = {
        f"processor_stage_{name}": np.asarray(
            [record.trace.stages[name] for record in inference_records], dtype=np.float32
        )
        for name in stage_names
    }
    replan_seed_states = np.asarray([record.seed_state for record in inference_records], dtype=np.float32)
    replan_image_hashes = np.asarray([record.image_sha256 for record in inference_records], dtype="S64")
    interpolated_chunks = stage_arrays.get("processor_stage_interpolated", raw_chunks)
    despiked_chunks = stage_arrays.get("processor_stage_despiked", processed_chunks)
    median_chunks = stage_arrays.get("processor_stage_median", processed_chunks)
    blended_chunks = stage_arrays.get("processor_stage_blended", processed_chunks)
    aligned_reference_states = (
        image_reference_states[np.asarray(source_frame_indices, dtype=np.int32)]
        if image_reference_states is not None
        else None
    )
    if quality_core is not None and aligned_reference_states is not None:
        quality_core.reference_states = aligned_reference_states
        _, _, quality_core.max_reference_delta_ms = load_reference_states(
            origin_root,
            np.asarray(
                [source_frames[index].timestamp for index in source_frame_indices],
                dtype=np.float64,
            ),
        )
    quality_arrays = quality_evaluator.trajectory_arrays() if quality_evaluator is not None else {}
    quality_summary = (
        quality_evaluator.summary()
        if quality_evaluator is not None
        else {
            "enabled": False,
            "metrics": [],
            "weights": {},
            "reference_use": "disabled",
            "tensorboard_tags": [],
            "execution": "disabled",
        }
    )
    quality_summary["reference_timebase"] = (
        "selected_image_frame" if image_selector is not None else "control_timestamp"
    )
    actual_states_for_score = np.asarray(
        [mapper.act_state_from_sim(qpos) for qpos in np.asarray(sim_state_after, dtype=np.float32)],
        dtype=np.float64,
    )
    if quality_core is not None:
        reference_states_for_score = quality_core.reference_states
        state_span_for_score = quality_core.state_span
    elif score_smoothness:
        reference_states_for_score = aligned_reference_states
        if reference_states_for_score is None:
            reference_states_for_score, _, _ = load_reference_states(
                origin_root,
                np.asarray(timestamps, dtype=np.float64),
            )
        state_span_for_score = np.concatenate(
            (
                np.maximum(
                    mapper.upper[: len(BODY_JOINTS)] - mapper.lower[: len(BODY_JOINTS)],
                    1e-9,
                ),
                np.asarray([100.0, 100.0]),
            )
        )
    else:
        reference_states_for_score = actual_states_for_score
        state_span_for_score = np.ones(19, dtype=np.float64)
    timing = {
        "realtime_requested": realtime,
        "rollout_duration_seconds": rollout_finished_monotonic - started_monotonic,
        "effective_fps": len(control_ticks) / max(rollout_finished_monotonic - started_monotonic, 1e-9),
        "telemetry_flush_seconds": telemetry_flush_seconds,
        "mean_cycle_ms": float(np.mean(cycle_ms)),
        "p95_cycle_ms": float(np.percentile(cycle_ms, 95)),
        "completed_policy_calls": len(inference_records),
        "mean_policy_ms": float(np.mean([record.latency_ms for record in inference_records])),
        "p95_policy_ms": float(np.percentile([record.latency_ms for record in inference_records], 95)),
        "deadline_misses": deadline_misses,
        "deadline_tolerance_ms": deadline_tolerance_s * 1000.0,
        "quality_wait_seconds": quality_wait_seconds,
    }
    scoring_started = time.monotonic()
    try:
        run_score_result = compute_run_score(
            quality_arrays=quality_arrays,
            actual_states=actual_states_for_score,
            reference_states=reference_states_for_score,
            state_span=state_span_for_score,
            timing=timing,
            control_hz=act_config.control_hz,
            frames=len(control_ticks),
            joint_limit_violations=limit_violations,
            target_step_errors=np.asarray(target_step_errors, dtype=np.int16),
            action_pipeline=action_pipeline,
            enable_smoothness=score_smoothness,
            enable_realtime=score_realtime,
        )
        for name, values in quality_arrays.items():
            tag = f"quality/{name.removeprefix('quality_')}"
            for step, value in enumerate(values):
                writer.add_scalar(tag, float(value), step)
        final_step = len(control_ticks) - 1
        for name, values in run_score_result.trajectory_arrays.items():
            component = name.removeprefix("score_")
            tag = f"score/component/{component}"
            if len(values) == len(control_ticks):
                for step, value in enumerate(values):
                    writer.add_scalar(tag, float(value), step)
            else:
                writer.add_scalar(tag, float(values[-1]), final_step)
        writer.add_scalar(
            "score/safety_pass",
            float(run_score_result.summary["safety_status"] == "passed"),
            final_step,
        )
        if run_score_result.summary["average_score"] is not None:
            writer.add_scalar("score/average", float(run_score_result.summary["average_score"]), final_step)
    finally:
        timing["scoring_seconds"] = time.monotonic() - scoring_started
        writer.flush()
        writer.close()
    finished_unix = time.time()
    timing["wall_duration_seconds"] = finished_unix - started_unix
    score_arrays = run_score_result.trajectory_arrays
    save_trajectory(
        trajectory_path,
        sim_qpos_before=np.asarray(sim_state_before, dtype=np.float32),
        sim_qpos_after=np.asarray(sim_state_after, dtype=np.float32),
        feedback_act_state=np.asarray(feedback_act_states, dtype=np.float32),
        act_state=np.asarray(act_states, dtype=np.float32),
        act_action=np.asarray(act_actions, dtype=np.float32),
        body_position_command=np.asarray(body_position_commands, dtype=np.float32),
        body_position_command_names=np.asarray(profile.body_command_names, dtype="S32"),
        left_hand_position_command=np.asarray(left_hand_position_commands, dtype=np.float32),
        right_hand_position_command=np.asarray(right_hand_position_commands, dtype=np.float32),
        processed_queue_action=np.asarray(processed_queue_actions, dtype=np.float32),
        bridge_queue_action=np.asarray(processed_queue_actions, dtype=np.float32),
        sim_target=np.asarray(sim_targets, dtype=np.float32),
        frame_id=np.asarray(frame_ids, dtype=np.int32),
        source_frame_index=np.asarray(source_frame_indices, dtype=np.int32),
        clock_source_frame_index=np.asarray(clock_source_frame_indices, dtype=np.int32),
        source_frame_id=np.asarray(source_frame_ids, dtype=np.int32),
        timestamp=np.asarray(timestamps, dtype=np.float64),
        camera_skew_ms=np.asarray(camera_skews, dtype=np.float32),
        image_match_distance=np.asarray(image_match_distances, dtype=np.float32),
        image_match_frozen=np.asarray(image_frozen_flags, dtype=np.bool_),
        image_selection_updated=np.asarray(image_selection_updated_flags, dtype=np.bool_),
        source_image_sha256=np.asarray(source_hashes_all, dtype="S64"),
        model_input_sha256=np.asarray(input_hashes_all, dtype="S64"),
        policy_input=np.asarray(policy_input_flags, dtype=np.bool_),
        action_index=np.asarray(action_indices, dtype=np.int16),
        chunk_origin_step=np.asarray(chunk_origins, dtype=np.int32),
        active_chunk_install_step=np.asarray(chunk_install_steps, dtype=np.int32),
        replan_installed=np.asarray(replan_installed_flags, dtype=np.bool_),
        processor_record_index=np.asarray(processor_record_indices, dtype=np.int32),
        bridge_record_index=np.asarray(processor_record_indices, dtype=np.int32),
        held_last_command=np.asarray(held_command_flags, dtype=np.bool_),
        action_queue_size=np.asarray(queue_sizes, dtype=np.int32),
        bridge_queue_size=np.asarray(queue_sizes, dtype=np.int32),
        blend_active=np.asarray(blend_active_flags, dtype=np.bool_),
        blend_alpha=np.asarray(blend_alphas, dtype=np.float32),
        lipo_old_record_index=np.asarray(lipo_old_record_indices, dtype=np.int32),
        lipo_new_record_index=np.asarray(lipo_new_record_indices, dtype=np.int32),
        candidate_count=np.asarray(candidate_counts, dtype=np.int16),
        observation_age_steps=np.asarray(observation_age_steps, dtype=np.int16),
        target_step_error=np.asarray(target_step_errors, dtype=np.int16),
        discarded_prefix_steps=np.asarray(discarded_prefix_steps, dtype=np.int16),
        runtime_cycle_ms=np.asarray(cycle_ms, dtype=np.float32),
        runtime_deadline_misses=np.asarray(deadline_miss_counts, dtype=np.int32),
        candidate_chunks=raw_chunks,
        raw_candidate_chunks=raw_chunks,
        processed_candidate_chunks=processed_chunks,
        interpolated_chunks=interpolated_chunks,
        despiked_chunks=despiked_chunks,
        median_chunks=median_chunks,
        blended_chunks=blended_chunks,
        replan_seed_state=replan_seed_states,
        replan_image_sha256=replan_image_hashes,
        chunk_submit_step=np.asarray([record.submit_step for record in inference_records], dtype=np.int32),
        chunk_install_step=np.asarray([record.install_step for record in inference_records], dtype=np.int32),
        chunk_latency_ms=np.asarray([record.latency_ms for record in inference_records], dtype=np.float32),
        visualization_step=np.asarray(visualization_steps, dtype=np.int32),
        rerun_state_step=np.asarray(rerun_state_steps, dtype=np.int32),
        eye_camera_step=np.asarray(eye_camera_steps, dtype=np.int32),
        chunk_discarded_prefix_steps=np.asarray(
            [
                max(record.install_step - record.submit_step, 0) if record.install_step >= 0 else -1
                for record in inference_records
            ],
            dtype=np.int16,
        ),
        **stage_arrays,
        **quality_arrays,
        **score_arrays,
    )
    summary = {
        "status": "completed",
        "run_name": run_name,
        "artifacts_persistent": save_artifacts,
        "run_directory": str(paths["root"].resolve()) if save_artifacts else None,
        "started_unix": started_unix,
        "finished_unix": finished_unix,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_model_sha256": sha256_file(checkpoint.resolve() / "model.safetensors"),
        "checkpoint_train_config_sha256": sha256_file(checkpoint.resolve() / "train_config.json"),
        "policy_type": "act",
        "policy_backend": policy_backend,
        "policy_script": str(script_path) if script_path is not None else None,
        "policy_script_sha256": sha256_file(script_path) if script_path is not None else None,
        "policy_script_server_pid": script_pid,
        "deployment_scripts": {
            "policy_server": {
                "path": str(Path(policy_script).resolve()),
                "sha256": sha256_file(Path(policy_script).resolve()),
                "executed": policy_backend == "script",
            },
            "ros_bridge_profile": {
                "path": str(Path(bridge_script).resolve()),
                "sha256": sha256_file(Path(bridge_script).resolve()),
                "executed": False,
                "simulated_contract": "runtime_bridge_async",
                "simulated_contract_executed": action_pipeline == "bridge",
            },
            "hardware_launcher_profile": {
                "path": str(Path(whole_script).resolve()),
                "sha256": sha256_file(Path(whole_script).resolve()),
                "executed": False,
            },
        },
        "policy_contract": {
            "images": list(image_keys),
            "image_shapes_hwc": {key: list(shape) for key, shape in image_shapes.items()},
            "state_dim": 19,
            "prediction_horizon": prediction_horizon,
            "checkpoint_execution_horizon": checkpoint_execution_horizon,
            "execution_horizon": execution_horizon,
            "execution_horizon_source": execution_horizon_source,
            "prediction_chunk_shape": [prediction_horizon, 19],
            "chunk_shape": [execution_horizon, 19],
            "execution_chunk_shape": [execution_horizon, 19],
            "processed_chunk_shape": [execution_horizon * active_sample_factor, 19],
        },
        "robot_command_contract": position_command_contract(
            profile.endpoints,
            profile.body_command_names,
        ),
        "w1_profile": profile.manifest(),
        "camera_sources": camera_sources,
        "image_replay": {
            "mode": image_replay_mode,
            "selection_state_source": (
                "mujoco_feedback_19d_for_frame_selection_only"
                if image_selector is not None
                else "origin_timestamp"
            ),
            "reference_joint_use": (
                "frame_selection_only_not_policy_state" if image_selector is not None else "none"
            ),
            "search_ahead_frames": image_search_ahead_frames if image_selector is not None else None,
            "max_advance_frames": image_max_advance_frames if image_selector is not None else None,
            "allow_backward": False,
            "match_threshold": image_match_threshold if image_selector is not None else None,
            "similarity_slack": image_similarity_slack if image_selector is not None else None,
            "output_hz": SOURCE_IMAGE_HZ,
            "three_cameras_share_frame": True,
            **(image_selector.summary() if image_selector is not None else {}),
        },
        "origin": str(origin_root.resolve()),
        "origin_metadata_sha256": replay.metadata_sha256,
        "origin_pose_sha256": pose_loader.file_sha256,
        "origin_pose_reads": pose_loader.read_count,
        "origin_pose_use": (
            "reset_quality_evaluation_and_frame_selection"
            if image_selector is not None and (quality_evaluator is not None or score_smoothness)
            else "reset_and_frame_selection"
            if image_selector is not None
            else "reset_and_quality_evaluation"
            if quality_evaluator is not None or score_smoothness
            else "reset_only"
        ),
        "reset_pose_match": pose_match,
        "state_source_first_inference": "mujoco_controlled_qpos",
        "state_source_after_reset": (
            "session_initial_unpublished_and_last_published_selected_command"
            if action_pipeline == "bridge"
            else "mujoco_controlled_qpos"
        ),
        "recorded_joint_data_used_after_reset": (
            quality_evaluator is not None or score_smoothness or image_selector is not None
        ),
        "recorded_joint_data_use_after_reset": (
            "quality_evaluation_and_frame_selection_only"
            if image_selector is not None and (quality_evaluator is not None or score_smoothness)
            else "frame_selection_only"
            if image_selector is not None
            else "quality_evaluation_only"
            if quality_evaluator is not None or score_smoothness
            else "none"
        ),
        "recorded_joint_data_used_by_policy": False,
        "recorded_joint_data_used_for_image_selection": image_selector is not None,
        "quality_evaluation": quality_summary,
        "run_score": run_score_result.summary,
        "control_mode": act_config.control_mode,
        "action_pipeline": action_pipeline,
        "action_processor": action_processor_manifest,
        "action_processor_stages": list(stage_names),
        "inference_schedule": inference_schedule,
        "action_scheduler": (
            {
                "mode": controller.schedule_mode,
                "replan_interval": controller.replan_interval,
                "replan_frequency_hz": act_config.control_hz / controller.replan_interval,
                "requested_replan_frequency_hz": None,
                "replan_interval_override": replan_interval if replan_interval > 0 else None,
                "replan_interval_source": (
                    "cli"
                    if replan_interval > 0
                    else "runtime_execution_horizon"
                    if execution_horizon_source == "runtime_override"
                    else "checkpoint_execution_horizon"
                ),
                "low_watermark": controller.low_watermark,
                "asynchronous": controller.asynchronous,
                "minimum_inference_ms": None,
                "max_candidates": None,
                "handoff_weights": None,
                "handoff_duration_ms": None,
            }
            if action_pipeline == "raw"
            else {
                "mode": controller.schedule_mode,
                "replan_threshold": controller.config.replan_threshold,
                "trigger_policy_points": controller.config.trigger_policy_points,
                "trigger_control_points": controller.config.trigger_control_points,
                "lipo_blend_policy_points": controller.config.lipo_blend_policy_points,
                "lipo_blend_control_points": controller.config.lipo_blend_control_points,
                "policy_hz": controller.config.policy_hz,
                "control_hz": act_config.control_hz,
                "asynchronous": controller.asynchronous,
                "simulated_inference_ms": controller.config.simulated_inference_ms,
                "inference_budget_ms": controller.config.inference_budget_ms,
                "inference_budget_policy_points": controller.config.inference_budget_policy_points,
                "replan_margin_policy_points": controller.config.replan_margin_policy_points,
                "required_policy_points": controller.config.required_policy_points,
                "available_policy_points": controller.config.available_policy_points,
                "max_in_flight": 1,
                "max_candidates": 1,
                "body_dimensions": controller.config.body_dimensions,
                "gripper_blended": False,
            }
        ),
        "bridge_profile": {
            **dict(profile.simulation["bridge"]),
            "sample_factor": active_sample_factor,
        },
        "bridge_profile_source": dict(profile.simulation["bridge"]),
        "self_collision_excludes": [list(pair) for pair in SELF_COLLISION_EXCLUDES],
        "runtime_model": str(runtime_model.resolve()),
        "runtime_model_sha256": sha256_file(runtime_model),
        "source_urdf": str(profile.urdf.resolve()),
        "locked_joint_values": locked_values,
        "hand_mapping": mapper.gestures.as_dict(),
        "hand_mapping_file": str(profile.source),
        "hand_mapping_file_sha256": sha256_file(profile.source),
        "control_hz": act_config.control_hz,
        "source_image_hz": SOURCE_IMAGE_HZ,
        "action_sample_factor": active_sample_factor,
        "rerun_view_mode": rerun_view_mode,
        "rerun_layout": rerun_layout_summary,
        "eye_camera": eye_camera_summary,
        "visualization_hz": SOURCE_IMAGE_HZ,
        "visualization_frames": len(visualization_steps),
        "rerun_state_frames": len(rerun_state_steps),
        "visualization_source_frames": [source_frame_indices[step] for step in visualization_steps],
        "visualization_state_source": "mujoco_qpos_after_control_step",
        "visualization_timebase": (
            "control_clock_relative_seconds"
            if image_selector is not None
            else "origin_timestamp_relative_seconds"
        ),
        "simulation_dt": simulator.control_dt,
        "replan_interval": controller.replan_interval,
        "replan_frequency_hz": (
            act_config.control_hz / controller.replan_interval if action_pipeline == "raw" else None
        ),
        "action_queue_low_watermark": controller.low_watermark,
        "frames": len(control_ticks),
        "source_frames": len(source_frames),
        "control_ticks_per_source_frame": 1 if action_pipeline == "raw" else None,
        "control_to_image_rate_ratio": act_config.control_hz / SOURCE_IMAGE_HZ,
        "start_frame": start_frame,
        "available_synchronized_frames": len(replay.frames),
        "dropped_for_camera_skew": replay.dropped_for_skew,
        "max_camera_skew_ms": float(np.max(camera_skews)),
        "pose_to_first_image_delta_ms": float(pose_match["delta_ms"]),
        "replan_count": controller.replan_count,
        "joint_limit_violations": limit_violations,
        "mean_tracking_rmse": float(np.mean(control_errors)),
        "max_tracking_rmse": float(np.max(control_errors)),
        "timing": timing,
        "device": str(policy.device),
        "cuda_device": torch.cuda.get_device_name(torch.device(device)),
        "trajectory": str(trajectory_path.resolve()),
        "trajectory_sha256": sha256_file(trajectory_path),
        "rerun_recording_enabled": recording_path is not None,
        "rerun_recording": str(recording_path.resolve()) if recording_path is not None else None,
        "rerun_recording_sha256": sha256_file(recording_path) if recording_path is not None else None,
        "tensorboard": str(tensorboard_dir.resolve()),
    }
    save_summary(summary_path, summary)
    score = run_score_result.summary["average_score"]
    component_text = " ".join(
        f"{name}={value:.1f}"
        for name, value in run_score_result.summary["components"].items()
        if value is not None
    )
    print(
        f"ACT_SIM_RUN_SCORE={'invalid' if score is None else f'{score:.1f}'} {component_text}",
        flush=True,
    )
    print(f"ACT_SIM_RERUN_RECORDING={recording_path or 'disabled'}", flush=True)
    print(f"ACT_SIM_SUMMARY={summary_path if save_artifacts else 'disabled'}", flush=True)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained ACT checkpoint against W1 MuJoCo feedback")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--origin", type=Path, default=DEFAULT_ORIGIN)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--policy-backend", choices=("direct", "script"), default="direct")
    parser.add_argument("--policy-script", type=Path, default=DEFAULT_POLICY_SCRIPT)
    parser.add_argument("--bridge-script", type=Path, default=DEFAULT_BRIDGE_SCRIPT)
    parser.add_argument("--whole-script", type=Path, default=DEFAULT_WHOLE_SCRIPT)
    parser.add_argument(
        "--camera-source",
        action="append",
        default=None,
        metavar="MODEL_INPUT=SOURCE",
        help=(
            "Repeat once per ACT image input; SOURCE is camera_type, relative directory, "
            "or absolute directory"
        ),
    )
    parser.add_argument("--control-mode", choices=("kinematic", "dynamic"), default="kinematic")
    parser.add_argument("--action-pipeline", choices=("raw", "bridge"), default="raw")
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=0,
        help="ACT execution horizon; zero uses checkpoint n_action_steps",
    )
    parser.add_argument(
        "--replan-interval",
        type=int,
        default=0,
        help="Raw ACT replan interval; zero uses the effective execution horizon",
    )
    parser.add_argument("--bridge-simulated-inference-ms", type=float, default=200.0)
    parser.add_argument("--bridge-inference-budget-ms", type=float, default=300.0)
    parser.add_argument("--bridge-policy-hz", type=float, default=20.0)
    parser.add_argument("--bridge-replan-threshold", type=float, default=0.5)
    parser.add_argument("--bridge-lipo-blend-policy-points", type=int, default=5)
    parser.add_argument("--bridge-replan-margin-policy-points", type=int, default=2)
    parser.add_argument("--bridge-sample-factor", type=int, default=2)
    parser.add_argument("--image-replay-mode", choices=("time", "state"), default="time")
    parser.add_argument("--image-search-ahead-frames", type=int, default=15)
    parser.add_argument("--image-max-advance-frames", type=int, default=2)
    parser.add_argument("--image-match-threshold", type=float, default=0.18)
    parser.add_argument("--image-similarity-slack", type=float, default=0.005)
    parser.add_argument(
        "--quality-metric",
        action="append",
        default=None,
        choices=("pose", "end_effector", "motion_direction", "amplitude"),
    )
    parser.add_argument("--score-smoothness", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--score-realtime", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rerun-url", default=None)
    parser.add_argument("--save-artifacts", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rerun-view-mode", choices=RERUN_VIEW_MODES, default="eye")
    parser.add_argument("--eye-camera-width", type=int, default=1280)
    parser.add_argument("--eye-camera-height", type=int, default=720)
    parser.add_argument("--eye-camera-fps", type=float, default=30.0)
    parser.add_argument("--eye-camera-fovy", type=float, default=70.0)
    parser.add_argument("--eye-camera-scene", choices=("robot", "grid"), default="grid")
    parser.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    run_name = args.run_name or f"act_{args.action_pipeline}"
    with simulation_run_directory(args.artifacts, run_name, args.save_artifacts) as run_directory:
        run_act_simulation(
            checkpoint=args.checkpoint,
            origin_root=args.origin,
            artifact_root=run_directory,
            run_name=run_name,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
            device=args.device,
            rerun_url=args.rerun_url,
            save_artifacts=args.save_artifacts,
            realtime=args.realtime,
            policy_backend=args.policy_backend,
            policy_script=args.policy_script,
            bridge_script=args.bridge_script,
            whole_script=args.whole_script,
            camera_sources=_parse_camera_sources(args.camera_source),
            profile=DEFAULT_PROFILE,
            control_mode=args.control_mode,
            action_pipeline=args.action_pipeline,
            execution_horizon=args.execution_horizon,
            replan_interval=args.replan_interval,
            bridge_simulated_inference_ms=args.bridge_simulated_inference_ms,
            bridge_inference_budget_ms=args.bridge_inference_budget_ms,
            bridge_policy_hz=args.bridge_policy_hz,
            bridge_replan_threshold=args.bridge_replan_threshold,
            bridge_lipo_blend_policy_points=args.bridge_lipo_blend_policy_points,
            bridge_replan_margin_policy_points=args.bridge_replan_margin_policy_points,
            bridge_sample_factor=args.bridge_sample_factor,
            image_replay_mode=args.image_replay_mode,
            image_search_ahead_frames=args.image_search_ahead_frames,
            image_max_advance_frames=args.image_max_advance_frames,
            image_match_threshold=args.image_match_threshold,
            image_similarity_slack=args.image_similarity_slack,
            quality_metrics=tuple(args.quality_metric or ()),
            score_smoothness=args.score_smoothness,
            score_realtime=args.score_realtime,
            rerun_view_mode=args.rerun_view_mode,
            eye_camera_width=args.eye_camera_width,
            eye_camera_height=args.eye_camera_height,
            eye_camera_fps=args.eye_camera_fps,
            eye_camera_fovy=args.eye_camera_fovy,
            eye_camera_scene=args.eye_camera_scene,
        )


if __name__ == "__main__":
    main()
