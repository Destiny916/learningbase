from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from async_infer.async_infer_typedef import *
from async_infer.timed_sequence_array import TimedSequenceArray


@dataclass(frozen=True)
class PolicyClientRequestMeta(object):
    reqeust_time: float
    reqeust_seq_index: int
    observation_sync_time: float = -1.0

    def __post_init__(self):
        # Explicit sync time
        if self.observation_sync_time >= -1e-3:
            return

        # Set the sync time as request time
        sync_time = self.reqeust_time
        object.__setattr__(self, 'observation_sync_time', sync_time)

    @staticmethod
    def invalid() -> 'PolicyClientRequestMeta':
        return PolicyClientRequestMeta(reqeust_time=-1.0, reqeust_seq_index=-1)

    @property
    def is_valid(self):
        return self.reqeust_time >= 0.0 and self.reqeust_seq_index >= 0 and self.observation_sync_time >= 0.0


@dataclass(frozen=True)
class PolicyClientInvokeInfo(object):
    meta: PolicyClientRequestMeta
    reqeust_id: int

    @staticmethod
    def invalid() -> 'PolicyClientInvokeInfo':
        return PolicyClientInvokeInfo(meta=PolicyClientRequestMeta.invalid(), reqeust_id=-1)


@dataclass(frozen=True)
class PolicyClientResponse(object):
    # Meta info
    request_meta: PolicyClientRequestMeta

    # State trajectory, None implies invalid response
    state_trajectory: Optional[TimedSequenceArray]

    @property
    def is_valid(self):
        return self.state_trajectory is not None and self.request_meta.is_valid

    # Other info
    error_str: Optional[str] = None
    misc_dict: Optional[Dict[str, Any]] = None


class PolicyClientInterface(object):

    def __init__(self):
        pass

    def invoke_async(self,
                     meta: PolicyClientRequestMeta,
                     observation: RolloutClientObservation,
                     current_cmd_trajectory: TimedSequenceArray) -> int:
        """
        Invoke the policy client asynchronously.

        Args:
            meta: The meta info of the request.
            observation: The observation map.
            current_cmd_trajectory: The trajectory that is currently used for rollout

        Returns:
            The request id.
        """
        return -1

    def try_take_response(self, request_id: int,
                          current_timepoint: TimePointAndSequenceIndex) -> Optional[PolicyClientResponse]:
        """
        Try to take the response of the request.

        Args:
            request_id: The request id.
            current_timepoint: current time and seq index

        Returns:
            The response of the request, None if not ready.
        """
        return None


__all__ = [
    'PolicyClientInterface',
    'PolicyClientRequestMeta',
    'PolicyClientInvokeInfo',
    'PolicyClientResponse'
]
