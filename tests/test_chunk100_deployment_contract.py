import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "start"
sys.path.insert(0, str(START))


def test_chunk100_contract_requires_full_100_frame_chunks():
    from runtime_chunk100 import validate_action_chunk_100

    actions = np.zeros((100, 19), dtype=np.float32)
    assert validate_action_chunk_100(actions).shape == (100, 19)
    with pytest.raises(ValueError, match="100.*19"):
        validate_action_chunk_100(np.zeros((16, 19), dtype=np.float32))


def test_chunk100_launcher_config_declares_chunk_and_execution_horizon_100():
    config = json.loads((START / "client_runtime_200000_chunk100.json").read_text())
    assert config["action_horizon"] == 100
    assert config["max_steps"] == 100
    assert config["chunk_size"] == 100
    assert config["n_action_steps"] == 100
