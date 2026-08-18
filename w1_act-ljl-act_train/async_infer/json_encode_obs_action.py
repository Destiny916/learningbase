import numpy as np
import json_numpy
from enum import Enum
from typing import Set, Dict, Any, Tuple, Callable, List, Union
from dataclasses import dataclass, field
from async_infer.constants import DefaultObservationKeys, DefaultActionKeys
from async_infer.async_infer_typedef import ObservationMap, AsyncInferObservationKeys
from async_infer.policy_client_interface import PolicyClientRequestMeta, PolicyClientResponse
from async_infer.timed_sequence_array import TimedSequenceArray


class NpArrayEncodeOption(Enum):
    DoNothing = 0
    ToList = 1
    ToNumpyJsonStr = 2


@dataclass
class EntryEncodeOption(object):
    # General options
    new_key: Union[str, None] = None
    remove_me_in_encoded: bool = False
    np_array_encode_option: NpArrayEncodeOption = NpArrayEncodeOption.ToNumpyJsonStr


@dataclass
class EncodeOption(object):
    # General options
    state_key: Union[str, None] = None
    global_np_array_encode_option: NpArrayEncodeOption = NpArrayEncodeOption.ToNumpyJsonStr
    option_dict: Dict[str, EntryEncodeOption] = field(default_factory=dict)
    removed_entry_keys: Set[str] = field(default_factory=set)


def _encode_numpy_array(arr: np.ndarray, arr_option: NpArrayEncodeOption) -> Any:
    if arr_option == NpArrayEncodeOption.ToNumpyJsonStr:
        return json_numpy.dumps(arr)
    elif arr_option == NpArrayEncodeOption.ToList:
        return arr.tolist()
    else:
        return arr


def _encode_tensor_generic(
        tensor_key: str,
        tensor_dict: Dict[str, np.ndarray],
        option: EncodeOption,
        tensor_proc: Union[Callable[[np.ndarray], np.ndarray], None] = None,
) -> Tuple[Union[str, None], Union[Any, None]]:
    # Check exist
    if (tensor_key not in tensor_dict) or (tensor_key in option.removed_entry_keys):
        return None, None

    # Get option
    this_key_option: Union[EntryEncodeOption, None] = option.option_dict.get(tensor_key, None)
    this_key_option_valid = this_key_option is not None
    if this_key_option_valid and this_key_option.remove_me_in_encoded:
        return None, None

    # Get tensor
    np_tensor = tensor_dict[tensor_key]
    if tensor_proc is not None:
        np_tensor = tensor_proc(np_tensor)

    # Check type
    if not isinstance(np_tensor, np.ndarray):
        return None, None

    # This is a valid tensor
    encoded_key = this_key_option.new_key if (
            this_key_option_valid and this_key_option.new_key is not None) else tensor_key
    arr_option = this_key_option.np_array_encode_option if this_key_option_valid else option.global_np_array_encode_option
    encoded = _encode_numpy_array(np_tensor, arr_option)
    return encoded_key, encoded


def encode_observation_into_json_dict(
        observation: ObservationMap,
        config_keys: AsyncInferObservationKeys,
        option_in: Union[EncodeOption, None] = None) -> Union[Dict[str, Any], None]:
    # Basic check
    if observation is None or observation.sync_time < 0.0:
        return None

    # For state
    option: EncodeOption = option_in if option_in is not None else EncodeOption()
    output_dict = dict()
    if observation.state is not None:
        state_key = option.state_key if option.state_key is not None else DefaultObservationKeys.state_key()
        output_dict[state_key] = _encode_numpy_array(observation.state, option.global_np_array_encode_option)

    # For rgb image
    for rgb_key in config_keys.rgb_images:
        # Try encode
        key, value = _encode_tensor_generic(rgb_key, observation.tensor_dict, option, None)
        if key is None or value is None:
            continue

        # Into output
        output_dict[key] = value

    # For depth image
    for depth_key in config_keys.depth_images:
        # Try encode
        key, value = _encode_tensor_generic(depth_key, observation.tensor_dict, option, None)
        if key is None or value is None:
            continue

        # Into output
        output_dict[key] = value

    # For misc info
    for misc_key in config_keys.other_keys:
        # Check if key exists in misc_dict
        if (misc_key not in observation.misc_dict) or (misc_key in option.removed_entry_keys):
            continue

        # Get option
        this_key_option: Union[EntryEncodeOption, None] = option.option_dict.get(misc_key, None)
        this_key_option_valid = this_key_option is not None
        if this_key_option_valid and this_key_option.remove_me_in_encoded:
            continue

        # Get value
        misc_value = observation.misc_dict[misc_key]

        # Encode numpy arrays if needed
        if isinstance(misc_value, np.ndarray):
            arr_option = this_key_option.np_array_encode_option if this_key_option_valid else option.global_np_array_encode_option
            misc_value = _encode_numpy_array(misc_value, arr_option)

        # Determine encoded key
        encoded_key = this_key_option.new_key if (
                this_key_option_valid and this_key_option.new_key is not None) else misc_key
        output_dict[encoded_key] = misc_value

    # Done
    return output_dict


@dataclass
class DecodePolicyResponseOption(object):
    trajectory_key: str
    trajectory_time_key: str
    error_str_key: str
    nominal_trajectory_time: float = 1.0

    @staticmethod
    def default_option() -> 'DecodePolicyResponseOption':
        return DecodePolicyResponseOption(trajectory_key=DefaultActionKeys.trajectory_key(),
                                          trajectory_time_key=DefaultActionKeys.trajectory_time_key(),
                                          error_str_key=DefaultActionKeys.error_str_key(),
                                          nominal_trajectory_time=1.0)


def _decode_into_numpy_array(
        decode_from: Union[str, List, np.ndarray],
        dtype: np.dtype = np.float32) -> Union[np.ndarray, None]:
    """
    Decode a numpy array from a string, list or numpy array itself. Can be used in the
    decoding of trajectory or trajectory_time below.
    """
    if isinstance(decode_from, np.ndarray):
        if decode_from.dtype == dtype: return decode_from
        return decode_from.astype(dtype)
    elif isinstance(decode_from, str):
        # json_numpy string
        try:
            loaded_data = json_numpy.loads(decode_from)
            if loaded_data.dtype == dtype:
                return loaded_data
            return loaded_data.astype(dtype)
        except Exception:
            return None
    elif isinstance(decode_from, list):
        # List from tolist()
        try:
            return np.array(decode_from, dtype=dtype)
        except Exception:
            return None
    else:
        # Invalid type
        return None


def _combine_error_str(
        error_str_from_response_dict: Union[str, None], decode_error_str: Union[str, None]) -> Union[str, None]:
    if error_str_from_response_dict is None: return decode_error_str
    if decode_error_str is None: return error_str_from_response_dict
    return f'Response error: {error_str_from_response_dict}; Decode error: {decode_error_str}'


def decode_policy_response(
        request_meta: PolicyClientRequestMeta,
        response_dict: Dict[str, Any],
        option_in: Union[DecodePolicyResponseOption, None] = None) -> PolicyClientResponse:
    """
    Decode a response_json_dict into a PolicyClientResponse object, using the provided or the default option and the response_json_dict.
    The response_json_dict should contain the following keys:
    - option.trajectory_key: A 2D numpy array of shape (num_timesteps, action_dim)
    The response_json_dict might contain the following keys:
    - option.trajectory_time_key: A 1D numpy array of shape (num_timesteps,). If not presented, the trajectory time is evenly spaced with nominal_trajectory_time.
    - option.error_str_key: A string
    The numpy array might be represented as: 1) json_numpy str (e.g., from json_numpy.dumps); 2) List from np.ndarray.tolist() or 3) it is just np.ndarray (without any encoding)
    """
    # Check type
    if response_dict is None or (not isinstance(response_dict, dict)):
        return PolicyClientResponse(request_meta=request_meta, error_str='Invalid response_dict',
                                    state_trajectory=None, misc_dict=response_dict)

    # option
    option = option_in if option_in is not None else DecodePolicyResponseOption.default_option()
    error_str_dict = response_dict.get(option.error_str_key, None)

    # Get trajectory data
    if option.trajectory_key not in response_dict:
        error_str_out = _combine_error_str(error_str_dict, f'Missing trajectory key: {option.trajectory_key}')
        return PolicyClientResponse(request_meta=request_meta, state_trajectory=None,
                                    error_str=error_str_out, misc_dict=response_dict)

    # Process trajectory data
    trajectory_data = response_dict[option.trajectory_key]

    # Convert to numpy array
    trajectory_data = _decode_into_numpy_array(trajectory_data, dtype=np.float64)
    if trajectory_data is None:
        error_str_out = _combine_error_str(error_str_dict, 'Failed to decode trajectory data')
        return PolicyClientResponse(request_meta=request_meta, state_trajectory=None,
                                    error_str=error_str_out, misc_dict=response_dict)

    # Check trajectory shape
    if len(trajectory_data.shape) != 2:
        error_str_out = _combine_error_str(error_str_dict, 'Invalid trajectory shape: ' + str(trajectory_data.shape))
        return PolicyClientResponse(request_meta=request_meta, state_trajectory=None,
                                    error_str=error_str_out, misc_dict=response_dict)

    # Get trajectory time
    num_timesteps = trajectory_data.shape[0]
    if option.trajectory_time_key in response_dict:
        # Convert to numpy array
        time_data = response_dict[option.trajectory_time_key]
        time_data = _decode_into_numpy_array(time_data, dtype=np.float64)
        if time_data is None:
            error_str_out = _combine_error_str(error_str_dict, 'Failed to decode time data')
            return PolicyClientResponse(request_meta=request_meta, state_trajectory=None,
                                        error_str=error_str_out, misc_dict=response_dict)

        # Resize for two-dimension with one of them is 1
        if len(time_data.shape) == 2 and (time_data.shape[0] == 1 or time_data.shape[1] == 1):
            time_data = time_data.reshape(-1)

        # Check time shape
        if len(time_data.shape) != 1 or time_data.shape[0] != num_timesteps:
            error_str_out = _combine_error_str(error_str_dict, 'Invalid time shape: ' + str(time_data.shape))
            return PolicyClientResponse(request_meta=request_meta, state_trajectory=None,
                                        error_str=error_str_out, misc_dict=response_dict)
    else:
        # Generate evenly spaced time
        time_data = np.linspace(0, option.nominal_trajectory_time, num_timesteps, dtype=np.float64)

    # These data are no-longer mutable after this
    trajectory_data.flags.writeable = False
    time_data.flags.writeable = False

    # Create TimedSequenceArray
    state_trajectory = TimedSequenceArray(data=trajectory_data, time=time_data)
    return PolicyClientResponse(request_meta=request_meta, state_trajectory=state_trajectory,
                                error_str=error_str_dict, misc_dict=response_dict)


__all__ = [
    'EncodeOption',
    'NpArrayEncodeOption',
    'EntryEncodeOption',
    'encode_observation_into_json_dict',
    'DecodePolicyResponseOption',
    'decode_policy_response'
]

# sandbox below
# Please refer to test_json_encode_obs_action.py
