from __future__ import annotations

from w1_simulation.control.bridge import ActionInferenceRecord

SOURCE_IMAGE_HZ = 30.0


def format_bridge_inference_log(
    record_index: int,
    record: ActionInferenceRecord,
    action_index: int,
) -> str:
    discarded_prefix_steps = (
        max(record.install_step - record.submit_step, 0) if record.install_step >= 0 else -1
    )
    return (
        f"ACT_SIM_INFERENCE pipeline=bridge inference_index={record_index} "
        f"submit_step={record.submit_step} install_step={record.install_step} "
        f"raw_points={len(record.trace.raw)} control_points={len(record.trace.processed)} "
        f"discarded_prefix_steps={discarded_prefix_steps} action_index={action_index} "
        f"e2e_ms={record.latency_ms:.2f}"
    )


def control_ticks(frames: list[object], control_hz: float) -> list[tuple[object, float, int]]:
    if not frames or control_hz <= 0.0:
        raise ValueError("frames must be non-empty and control_hz must be positive")
    ticks: list[tuple[object, float, int]] = []
    start = float(frames[0].timestamp)
    stop = float(frames[-1].timestamp) + 1.0 / SOURCE_IMAGE_HZ
    source_index = 0
    tick_index = 0
    while True:
        tick_timestamp = start + tick_index / control_hz
        if tick_timestamp >= stop - 1e-9:
            break
        while (
            source_index + 1 < len(frames)
            and float(frames[source_index + 1].timestamp) <= tick_timestamp + 1e-9
        ):
            source_index += 1
        ticks.append((frames[source_index], tick_timestamp, source_index))
        tick_index += 1
    return ticks


def raw_control_ticks(frames: list[object]) -> list[tuple[object, float, int]]:
    if not frames:
        raise ValueError("frames must be non-empty")
    return [(frame, float(frame.timestamp), index) for index, frame in enumerate(frames)]
