from typing import Generic, TypeVar, List, Callable, Dict, Any, Union, Optional
from dataclasses import dataclass, field
import queue
import numpy as np
from async_infer.async_infer_typedef import NpArray1d

# Internal type-var
_T = TypeVar("_T")


@dataclass
class GatherDataResponse(Generic[_T]):
    data: _T
    obtained_time: float


class GatherDataFunctorBase(Generic[_T]):

    def __init__(self):
        pass

    def __call__(self, request_time: float) -> Union[GatherDataResponse[_T], List[GatherDataResponse[_T]]]:
        pass


# Reqeust time -> Response
GatherDataFunctor = Union[GatherDataFunctorBase[_T], Callable[[float], GatherDataFunctorBase[_T]]]


@dataclass
class GatherDataFunctors(object):
    # For gathering of state
    state_gather_fn: Optional[GatherDataFunctor[NpArray1d]] = None

    # For gathering of tensor data
    gather_tensor_fn_dict: Dict[str, GatherDataFunctor[np.ndarray]] = field(default_factory=dict)

    # For gather of other data
    gather_misc_fn_dict: Dict[str, GatherDataFunctor[Any]] = field(default_factory=dict)


class SingleReaderGatherDataInterfaceAsyncQueue(GatherDataFunctorBase[_T]):

    def __init__(self, rough_max_size: int = -1):
        super().__init__()
        assert rough_max_size < 0 or rough_max_size >= 4
        self._rough_max_size = rough_max_size
        self._fifo_queue = queue.SimpleQueue()  # This queue is thread-safe

    def __call__(self, request_time: float) -> List[GatherDataResponse[_T]]:
        return self._gather_take_all(request_time=request_time)

    def rough_size(self) -> int:
        return self._fifo_queue.qsize()

    def append(self, data: _T, data_time: float):
        # This never block
        response_data = GatherDataResponse[_T](data=data, obtained_time=data_time)
        self._fifo_queue.put(response_data)
        if self._rough_max_size < 0: return

        # Pop at most one
        if self._fifo_queue.qsize() > self._rough_max_size:
            self.take_one()

    def take_one(self) -> Optional[GatherDataResponse[_T]]:
        try:
            out = self._fifo_queue.get_nowait()
            return out
        except queue.Empty:
            return None

    def _gather_take_all(self, request_time: float) -> List[GatherDataResponse[_T]]:
        output_list: List[GatherDataResponse[_T]] = list()
        while True:
            # Take one
            obtained_instance = self.take_one()
            if obtained_instance is None:
                return output_list

            # Check time
            if obtained_instance.obtained_time < 0:
                obtained_instance.obtained_time = request_time

            # Append into list
            output_list.append(obtained_instance)


__all__ = [
    "GatherDataResponse",
    "GatherDataFunctors",
    "GatherDataFunctorBase",
    "SingleReaderGatherDataInterfaceAsyncQueue"
]
