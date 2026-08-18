from __future__ import annotations

import numpy as np
import pytest
from w1_simulation.control.processing import (
    ActionChunkTrace,
    BridgeActionChunkProcessor,
    IdentityActionChunkProcessor,
    validate_trace,
)


def test_raw_processor_preserves_chunk_and_emitted_action_bit_exact() -> None:
    processor = IdentityActionChunkProcessor()
    chunk = np.arange(30 * 19, dtype=np.float32).reshape(30, 19) / 7.0

    trace = validate_trace(processor.process_chunk(chunk, np.ones(19, dtype=np.float32)), chunk)

    np.testing.assert_array_equal(trace.raw, chunk)
    np.testing.assert_array_equal(trace.processed, chunk)
    np.testing.assert_array_equal(processor.process_action(chunk[7]), chunk[7])


def test_bridge_processor_linearly_interpolates_thirty_policy_points_to_sixty_control_points() -> None:
    processor = BridgeActionChunkProcessor(sample_factor=2)
    chunk = np.zeros((30, 19), dtype=np.float32)
    chunk[:, 0] = np.linspace(0.0, 29.0, 30)
    chunk[:, 17] = np.linspace(100.0, 129.0, 30)

    trace = validate_trace(processor.process_chunk(chunk, None), chunk)

    expected = np.linspace(0.0, 29.0, 60, dtype=np.float32)
    assert trace.raw.shape == (30, 19)
    assert trace.processed.shape == (60, 19)
    assert tuple(trace.stages) == ("raw", "interpolated", "processed")
    np.testing.assert_array_equal(trace.raw, chunk)
    np.testing.assert_allclose(trace.processed[:, 0], expected, atol=1e-6)
    np.testing.assert_allclose(trace.processed[:, 17], expected + 100.0, atol=1e-6)


def test_bridge_processor_with_sample_factor_one_preserves_all_actions() -> None:
    processor = BridgeActionChunkProcessor(sample_factor=1)
    chunk = np.arange(30 * 19, dtype=np.float32).reshape(30, 19)

    trace = validate_trace(processor.process_chunk(chunk, None), chunk)

    np.testing.assert_array_equal(trace.processed, chunk)


def test_trace_rejects_processors_that_change_action_dimension() -> None:
    raw = np.zeros((30, 19), dtype=np.float32)
    wrong_dimension = np.zeros((60, 18), dtype=np.float32)
    trace = ActionChunkTrace(
        raw=raw,
        processed=wrong_dimension,
        stages={"raw": raw, "processed": wrong_dimension},
    )

    with pytest.raises(ValueError, match="preserve the action dimension"):
        validate_trace(trace, raw)
