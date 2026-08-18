from typing import Dict, Callable, Union, Optional
from concurrent.futures import Future, Executor
from async_infer.timed_sequence_array import *
from async_infer.async_infer_typedef import *
from async_infer.policy_client_interface import *


# Interface for synchronized srv, which can be turned into async one
class SynchronizedPolicyClientInterface(object):

    def __init__(self):
        pass

    def __call__(self, invoke_info: PolicyClientInvokeInfo,
                 observation: RolloutClientObservation,
                 current_cmd_trajectory: TimedSequenceArray) -> PolicyClientResponse:
        return PolicyClientResponse(request_meta=invoke_info.meta, state_trajectory=None, error_str='Not Implemented')


# Functor that combine interface and callable
SynchronizedPolicyFunctor = Union[
    SynchronizedPolicyClientInterface, Callable[
        [PolicyClientInvokeInfo, RolloutClientObservation], PolicyClientResponse]]


class PolicyClientByFunctor(PolicyClientInterface):
    """
    A policy client that use a functor to build async service. It maintains
    a future map to store the future of each request. Once try_take_response is called,
    it will check if the future is done, if so, it will return the response.
    """

    def __init__(self, func: SynchronizedPolicyFunctor, executor_now_owned: Executor):
        super().__init__()
        self._func = func
        self._executor = executor_now_owned
        self._future_map: Dict[int, Future] = {}
        self._request_id_counter: int = 0

    def invoke_async(self, meta: PolicyClientRequestMeta, observation: RolloutClientObservation,
                     current_cmd_trajectory: TimedSequenceArray) -> int:
        self._request_id_counter += 1
        request_id = self._request_id_counter
        invoke_info = PolicyClientInvokeInfo(meta=meta, reqeust_id=request_id)
        future = self._executor.submit(self._func, invoke_info, observation, current_cmd_trajectory)
        self._future_map[request_id] = future
        return request_id

    def try_take_response(self, request_id: int,
                          current_timepoint: TimePointAndSequenceIndex) -> Optional[PolicyClientResponse]:
        if request_id not in self._future_map:
            return None
        future = self._future_map[request_id]
        if not future.done():
            return None
        response = future.result()
        del self._future_map[request_id]
        return response


class PolicyClientByFunctorOneActive(PolicyClientInterface):
    """
    A policy client that use a functor to build async service. It maintains
    a future map to store the future of each request. Once try_take_response is called,
    it will check if the future is done, if so, it will return the response.
    Only one active request is allowed in this class.
    """

    def __init__(self, func: SynchronizedPolicyFunctor, executor_now_owned: Executor):
        super().__init__()
        self._func = func
        self._executor = executor_now_owned
        self._active_future: Optional[Future] = None
        self._request_id_counter: int = 0

    def invoke_async(self,
                     meta: PolicyClientRequestMeta,
                     observation: RolloutClientObservation,
                     current_cmd_trajectory: TimedSequenceArray) -> int:
        # Check active
        if self._active_future is not None:
            return -1

        # Invoke
        self._request_id_counter += 1
        request_id = self._request_id_counter
        invoke_info = PolicyClientInvokeInfo(meta=meta, reqeust_id=request_id)
        future = self._executor.submit(self._func, invoke_info, observation, current_cmd_trajectory)
        self._active_future = future
        return request_id

    def try_take_response(self, request_id: int,
                          current_timepoint: TimePointAndSequenceIndex) -> Optional[PolicyClientResponse]:
        # Check active
        if self._active_future is None or (request_id != self._request_id_counter):
            return None

        # Now it must be a valid one
        assert request_id == self._request_id_counter
        assert self._active_future is not None
        if not self._active_future.done():
            return None

        # Now the future is done, we can return the response
        response = self._active_future.result()
        self._active_future = None
        return response


__all__ = [
    'SynchronizedPolicyClientInterface',
    'SynchronizedPolicyFunctor',
    'PolicyClientByFunctor',
    'PolicyClientByFunctorOneActive',
]
