from __future__ import annotations

import argparse
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from w1_simulation.artifacts import ensure_simulation_artifact_dirs, simulation_run_directory
from w1_simulation.cli import parse_camera_sources as _parse_camera_sources
from w1_simulation.evaluation.verification import verify_run
from w1_simulation.execution.rollout import run_act_simulation
from w1_simulation.simulation.camera import RERUN_VIEW_MODES
from w1_simulation.w1_profile import (
    DEFAULT_PROFILE_PATH,
    W1Profile,
)

RERUN_MEMORY_LIMIT = "10GB"


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.25):
            return True
    except OSError:
        return False


def _resolve_port(port: int, excluded: set[int] | None = None) -> int:
    if not 0 <= port <= 65535:
        raise ValueError(f"Port must be in [0, 65535], got {port}")
    if port != 0:
        return port
    excluded = excluded or set()
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            selected = int(sock.getsockname()[1])
        if selected not in excluded:
            return selected


def _wait_for_port(process: subprocess.Popen[bytes], port: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Service exited with code {process.returncode} before port {port} opened")
        if _port_open(port):
            return
        time.sleep(0.1)
    raise TimeoutError(f"Service did not open port {port} within {timeout_s:.1f}s")


def _stop_processes(processes: list[subprocess.Popen[bytes]], handles: list[object]) -> None:
    for process in reversed(processes):
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
    for process in reversed(processes):
        if process.poll() is not None:
            continue
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    for handle in handles:
        handle.close()


def _start_visualization_services(
    run_directory: Path,
    rerun_port: int,
    tensorboard_port: int,
) -> tuple[list[subprocess.Popen[bytes]], list[object]]:
    occupied = [port for port in (rerun_port, tensorboard_port) if _port_open(port)]
    if occupied:
        raise RuntimeError(f"Visualization ports are already in use: {occupied}")
    paths = ensure_simulation_artifact_dirs(run_directory)
    processes: list[subprocess.Popen[bytes]] = []
    handles: list[object] = []
    try:
        rerun_log = (paths["logs"] / "rerun.log").open("ab")
        handles.append(rerun_log)
        rerun_process = subprocess.Popen(
            [
                "rerun",
                "--port",
                str(rerun_port),
                "--renderer",
                "gl",
                "--window-size",
                "1600x900",
                "--memory-limit",
                RERUN_MEMORY_LIMIT,
                "--hide-welcome-screen",
            ],
            stdin=subprocess.DEVNULL,
            stdout=rerun_log,
            stderr=subprocess.STDOUT,
        )
        processes.append(rerun_process)
        _wait_for_port(rerun_process, rerun_port, 20.0)

        tensorboard_log = (paths["logs"] / "tensorboard.log").open("ab")
        handles.append(tensorboard_log)
        tensorboard_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tensorboard.main",
                "--logdir",
                str(paths["tensorboard"]),
                "--host",
                "127.0.0.1",
                "--port",
                str(tensorboard_port),
            ],
            stdin=subprocess.DEVNULL,
            stdout=tensorboard_log,
            stderr=subprocess.STDOUT,
        )
        processes.append(tensorboard_process)
        _wait_for_port(tensorboard_process, tensorboard_port, 20.0)
        return processes, handles
    except BaseException:
        _stop_processes(processes, handles)
        raise


def launch(
    *,
    checkpoint: Path,
    origin: Path,
    artifacts: Path,
    run_name: str,
    profile: W1Profile,
    start_frame: int,
    max_frames: int,
    device: str,
    policy_backend: str,
    policy_script: Path,
    bridge_script: Path,
    whole_script: Path,
    camera_sources: dict[str, str] | None,
    rerun_port: int,
    tensorboard_port: int,
    save_artifacts: bool,
    realtime: bool,
    keep_open: bool,
    control_mode: str,
    action_pipeline: str,
    execution_horizon: int,
    replan_interval: int,
    bridge_simulated_inference_ms: float,
    bridge_inference_budget_ms: float,
    bridge_policy_hz: float,
    bridge_replan_threshold: float,
    bridge_lipo_blend_policy_points: int,
    bridge_replan_margin_policy_points: int,
    bridge_sample_factor: int,
    image_replay_mode: str,
    image_search_ahead_frames: int,
    image_max_advance_frames: int,
    image_match_threshold: float,
    image_similarity_slack: float,
    quality_metrics: tuple[str, ...],
    score_smoothness: bool,
    score_realtime: bool,
    rerun_view_mode: str,
    eye_camera_width: int,
    eye_camera_height: int,
    eye_camera_fps: float,
    eye_camera_fovy: float,
    eye_camera_scene: str,
    strict_verification: bool,
) -> Path | None:
    if strict_verification and not save_artifacts:
        raise ValueError("Strict verification requires SAVE_ARTIFACTS=true")
    rerun_port = _resolve_port(rerun_port)
    tensorboard_port = _resolve_port(tensorboard_port, {rerun_port})
    print(f"ACT_SIM_RERUN_PORT={rerun_port}", flush=True)
    print(f"ACT_SIM_RERUN_MEMORY_LIMIT={RERUN_MEMORY_LIMIT}", flush=True)
    print(f"ACT_SIM_TENSORBOARD_PORT={tensorboard_port}", flush=True)
    with simulation_run_directory(artifacts, run_name, save_artifacts) as run_directory:
        print(f"ACT_SIM_RUN_DIRECTORY={run_directory if save_artifacts else 'ephemeral'}", flush=True)
        processes, handles = _start_visualization_services(run_directory, rerun_port, tensorboard_port)
        summary_path = run_directory / "summary.json"
        try:
            summary_path = run_act_simulation(
                checkpoint=checkpoint,
                origin_root=origin,
                artifact_root=run_directory,
                run_name=run_name,
                profile=profile,
                start_frame=start_frame,
                max_frames=max_frames,
                device=device,
                rerun_url=f"rerun+http://127.0.0.1:{rerun_port}/proxy",
                save_artifacts=save_artifacts,
                realtime=realtime,
                policy_backend=policy_backend,
                policy_script=policy_script,
                bridge_script=bridge_script,
                whole_script=whole_script,
                camera_sources=camera_sources,
                control_mode=control_mode,
                action_pipeline=action_pipeline,
                execution_horizon=execution_horizon,
                replan_interval=replan_interval,
                bridge_simulated_inference_ms=bridge_simulated_inference_ms,
                bridge_inference_budget_ms=bridge_inference_budget_ms,
                bridge_policy_hz=bridge_policy_hz,
                bridge_replan_threshold=bridge_replan_threshold,
                bridge_lipo_blend_policy_points=bridge_lipo_blend_policy_points,
                bridge_replan_margin_policy_points=bridge_replan_margin_policy_points,
                bridge_sample_factor=bridge_sample_factor,
                image_replay_mode=image_replay_mode,
                image_search_ahead_frames=image_search_ahead_frames,
                image_max_advance_frames=image_max_advance_frames,
                image_match_threshold=image_match_threshold,
                image_similarity_slack=image_similarity_slack,
                quality_metrics=quality_metrics,
                score_smoothness=score_smoothness,
                score_realtime=score_realtime,
                rerun_view_mode=rerun_view_mode,
                eye_camera_width=eye_camera_width,
                eye_camera_height=eye_camera_height,
                eye_camera_fps=eye_camera_fps,
                eye_camera_fovy=eye_camera_fovy,
                eye_camera_scene=eye_camera_scene,
            )
            verification_result = "skipped-artifacts-disabled"
            if save_artifacts:
                verification_result = "not-passed"
                try:
                    verification_result = str(verify_run(run_directory))
                except AssertionError as exc:
                    if strict_verification:
                        raise
                    print(f"ACT_SIM_VERIFICATION_WARNING={exc}", flush=True)
            print(f"ACT_SIM_NATIVE_VIEWER_PID={processes[0].pid}", flush=True)
            print(f"ACT_SIM_TENSORBOARD_URL=http://127.0.0.1:{tensorboard_port}", flush=True)
            print(f"ACT_SIM_SUMMARY={summary_path if save_artifacts else 'disabled'}", flush=True)
            print(f"ACT_SIM_VERIFICATION={verification_result}", flush=True)
            if keep_open:
                print("ACT_SIM_READY=True；原生 Rerun 窗口保持运行，按 Ctrl+C 关闭。", flush=True)
                while all(process.poll() is None for process in processes):
                    time.sleep(1.0)
                raise RuntimeError("A visualization service exited unexpectedly")
            return summary_path if save_artifacts else None
        except KeyboardInterrupt:
            return summary_path if save_artifacts and summary_path.is_file() else None
        finally:
            _stop_processes(processes, handles)


def main() -> None:
    profile_parser = argparse.ArgumentParser(add_help=False)
    profile_parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    profile_args, _ = profile_parser.parse_known_args()
    profile = W1Profile.load(profile_args.profile)
    parser = argparse.ArgumentParser(
        description="Launch native Rerun, TensorBoard, ACT simulation, and verification"
    )
    parser.add_argument("--profile", type=Path, default=profile.source)
    parser.add_argument("--checkpoint", type=Path, default=profile.checkpoint)
    parser.add_argument("--origin", type=Path, default=profile.origin)
    parser.add_argument("--artifacts", type=Path, default=profile.artifacts)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--policy-backend", choices=("direct", "script"), default="script")
    parser.add_argument("--policy-script", type=Path, default=profile.policy_script)
    parser.add_argument("--bridge-script", type=Path, default=profile.bridge_script)
    parser.add_argument("--whole-script", type=Path, default=profile.whole_script)
    parser.add_argument(
        "--control-mode",
        choices=("kinematic", "dynamic"),
        default=str(profile.simulation["control_mode"]),
    )
    parser.add_argument("--action-pipeline", choices=("raw", "bridge"), default="raw")
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=int(profile.act["n_action_steps"]),
        help="ACT execution horizon; zero uses checkpoint n_action_steps",
    )
    parser.add_argument(
        "--replan-interval",
        type=int,
        default=0,
        help="Raw ACT replan interval; zero uses the effective execution horizon",
    )
    bridge = profile.simulation["bridge"]
    parser.add_argument(
        "--bridge-simulated-inference-ms", type=float, default=float(bridge["simulated_inference_ms"])
    )
    parser.add_argument(
        "--bridge-inference-budget-ms", type=float, default=float(bridge["inference_budget_ms"])
    )
    parser.add_argument("--bridge-policy-hz", type=float, default=float(bridge["policy_hz"]))
    parser.add_argument("--bridge-replan-threshold", type=float, default=float(bridge["replan_threshold"]))
    parser.add_argument(
        "--bridge-lipo-blend-policy-points",
        type=int,
        default=int(bridge["lipo_blend_policy_points"]),
    )
    parser.add_argument(
        "--bridge-replan-margin-policy-points",
        type=int,
        default=int(bridge["replan_margin_policy_points"]),
    )
    parser.add_argument("--bridge-sample-factor", type=int, default=int(bridge["sample_factor"]))
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
    parser.add_argument("--camera-source", action="append", default=None, metavar="MODEL_INPUT=SOURCE")
    parser.add_argument("--rerun-port", type=int, default=9876)
    parser.add_argument("--tensorboard-port", type=int, default=6006)
    parser.add_argument(
        "--save-artifacts",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    visualization = profile.simulation["visualization"]
    parser.add_argument(
        "--rerun-view-mode", choices=RERUN_VIEW_MODES, default=str(visualization["rerun_view_mode"])
    )
    parser.add_argument("--eye-camera-width", type=int, default=int(visualization["eye_camera_width"]))
    parser.add_argument("--eye-camera-height", type=int, default=int(visualization["eye_camera_height"]))
    parser.add_argument("--eye-camera-fps", type=float, default=float(visualization["eye_camera_fps"]))
    parser.add_argument("--eye-camera-fovy", type=float, default=float(visualization["eye_camera_fovy"]))
    parser.add_argument(
        "--eye-camera-scene", choices=("robot", "grid"), default=str(visualization["eye_camera_scene"])
    )
    parser.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-open", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-verification", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    launch(
        checkpoint=args.checkpoint,
        origin=args.origin,
        artifacts=args.artifacts,
        run_name=args.run_name or f"act_{args.action_pipeline}",
        profile=profile,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        device=args.device,
        policy_backend=args.policy_backend,
        policy_script=args.policy_script,
        bridge_script=args.bridge_script,
        whole_script=args.whole_script,
        camera_sources=_parse_camera_sources(args.camera_source),
        rerun_port=args.rerun_port,
        tensorboard_port=args.tensorboard_port,
        save_artifacts=args.save_artifacts,
        realtime=args.realtime,
        keep_open=args.keep_open,
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
        strict_verification=args.strict_verification,
    )


if __name__ == "__main__":
    main()
