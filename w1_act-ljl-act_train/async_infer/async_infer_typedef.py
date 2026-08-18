from dataclasses import dataclass, field
from typing import Dict, Any, List, Union, Optional
import numpy as np

# For type annotation
NpArray1d = np.ndarray
NpArray2d = np.ndarray


# Make sure an array from numpy is not writable, clone else
def ensure_immutable_numpy(array_in: np.ndarray):
    # The input is not writable
    if not array_in.flags.writeable:
        return array_in

    # Make a copy
    array_immutable = np.copy(array_in)
    array_immutable.flags.writeable = False
    return array_immutable


# For observation
@dataclass(frozen=True)
class ObservationMap(object):
    # Meta
    observation_time: float

    # State info, should be immutable after construction
    state: Optional[NpArray1d] = None

    # For tensor data
    tensor_dict: Dict[str, Optional[np.ndarray]] = field(default_factory=dict)

    # For misc data
    misc_dict: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.state is None or (not self.state.flags.writeable):
            return

        immutable_state = ensure_immutable_numpy(self.state)
        object.__setattr__(self, 'state', immutable_state)

    @property
    def sync_time(self) -> float:
        return self.observation_time

    # Clone without copying tensor
    def new_observation_from_replacement(
            self,
            new_state: Optional[NpArray1d] = None,
            new_tensor_dict: Optional[Dict[str, Optional[np.ndarray]]] = None,
            new_misc_dict: Optional[Dict[str, Any]] = None,
            shallow_clone_tensor_misc_dict: bool = False) -> 'ObservationMap':
        if shallow_clone_tensor_misc_dict:
            return ObservationMap(
                observation_time=self.observation_time,
                state=self.state if new_state is None else new_state,
                tensor_dict=self.tensor_dict.copy() if new_tensor_dict is None else new_tensor_dict,
                misc_dict=self.misc_dict.copy() if new_misc_dict is None else new_misc_dict
            )

        # Not cloned
        return ObservationMap(
            observation_time=self.observation_time,
            state=self.state if new_state is None else new_state,
            tensor_dict=self.tensor_dict if new_tensor_dict is None else new_tensor_dict,
            misc_dict=self.misc_dict if new_misc_dict is None else new_misc_dict
        )


@dataclass(frozen=True)
class RolloutClientObservation(object):
    """
    This class is used to accommodate the observation that are directly represented
    as Dict (instead of ObservationMap)
    """
    sync_time: float = -1.0
    observation_map: Union[ObservationMap, Dict[str, Any], None] = None
    process_obs_status_str: Optional[str] = None
    explicit_state: Optional[NpArray1d] = None  # Should be immutable

    @staticmethod
    def invalid() -> 'RolloutClientObservation':
        return RolloutClientObservation(sync_time=-1, observation_map=None,
                                        process_obs_status_str="Invalid By Construction")

    @staticmethod
    def from_observation_map(observation_map: ObservationMap,
                             status_str: Optional[str] = None) -> 'RolloutClientObservation':
        return RolloutClientObservation(sync_time=observation_map.sync_time, observation_map=observation_map,
                                        process_obs_status_str=status_str,
                                        explicit_state=ensure_immutable_numpy(observation_map.state))

    @property
    def is_valid(self) -> bool:
        return (self.explicit_state is not None) and (self.observation_map is not None) and (self.sync_time >= -1e-3)

    @property
    def state(self) -> Optional[np.ndarray]:
        return self.explicit_state


# Config
@dataclass(frozen=True)
class AsyncInferObservationKeys(object):
    state_key: str
    rgb_images: List[str] = field(default_factory=list)
    depth_images: List[str] = field(default_factory=list)
    other_keys: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class AsyncInferStateDimensionConfig(object):
    # Dimension of the state, > 0
    state_dim: int

    # [0, state_dim) indicates some state are actually tool discrete_tool_state_indices,
    # These are integral of binary, and should NOT be continuously interpolated
    discrete_tool_state_indices: List[int] = field(default_factory=list)

    # weight to compute norm
    state_distance_weight: Optional[NpArray1d] = None


@dataclass(frozen=True)
class TimePointAndSequenceIndex(object):
    time: float
    seq_index: int

    @staticmethod
    def invalid() -> 'TimePointAndSequenceIndex':
        return TimePointAndSequenceIndex(time=-1.0, seq_index=-1)

    @property
    def is_valid(self):
        return self.time >= 0.0 and self.seq_index >= 0


__all__ = [
    "NpArray1d",
    "NpArray2d",
    "ensure_immutable_numpy",
    "ObservationMap",
    "RolloutClientObservation",
    "AsyncInferObservationKeys",
    "AsyncInferStateDimensionConfig",
    "TimePointAndSequenceIndex"
]
