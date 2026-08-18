from typing import Dict, Any, Union, Optional
from async_infer.policy_client_async import SynchronizedPolicyClientInterface
from async_infer.json_encode_obs_action import *
from async_infer.policy_client_interface import *
from async_infer.async_infer_typedef import *
from async_infer.timed_sequence_array import TimedSequenceArray


class SynchronizedPolicyInterfaceJson(SynchronizedPolicyClientInterface):

    def __init__(self,
                 config_keys: AsyncInferObservationKeys,
                 encode_option: EncodeOption,
                 decode_option: DecodePolicyResponseOption):
        super().__init__()
        self._config_keys = config_keys
        self._encode_option = encode_option
        self._decode_option = decode_option

    def _invoke_dict_io(self, payload: Dict[str, Any]) -> Union[Dict[str, Any], None]:
        return None

    # Impl of interface
    def __call__(self, invoke_info: PolicyClientInvokeInfo,
                 observation: RolloutClientObservation,
                 current_cmd_trajectory: TimedSequenceArray) -> PolicyClientResponse:
        # Check input
        if observation is None:
            return PolicyClientResponse(request_meta=invoke_info.meta, state_trajectory=None,
                                        error_str='Invalid Observation')

        # Encode into json
        obs_json_dict: Optional[Dict[str, Any]] = None
        if isinstance(observation.observation_map, ObservationMap):
            obs_json_dict = encode_observation_into_json_dict(
                observation.observation_map, config_keys=self._config_keys, option_in=self._encode_option)
        else:
            obs_json_dict = observation.observation_map

        # Check json
        if obs_json_dict is None:
            return PolicyClientResponse(request_meta=invoke_info.meta, state_trajectory=None,
                                        error_str='Invalid Observation ToJson/JsonDict')

        # Call it
        response_json = self._invoke_dict_io(payload=obs_json_dict)

        # Check output
        if response_json is None:
            return PolicyClientResponse(request_meta=invoke_info.meta, state_trajectory=None,
                                        error_str='Server failed (return None).')

        # Into response
        response = decode_policy_response(request_meta=invoke_info.meta,
                                          response_dict=response_json, option_in=self._decode_option)
        return response


class SynchronizedPolicyInterfaceJsonRequestsPost(SynchronizedPolicyInterfaceJson):

    def __init__(self,
                 url: str,
                 config_keys: AsyncInferObservationKeys,
                 encode_option: EncodeOption,
                 decode_option: DecodePolicyResponseOption):
        super().__init__(config_keys=config_keys, encode_option=encode_option, decode_option=decode_option)
        self._url = url

    def _invoke_dict_io(self, payload: Dict[str, Any]) -> Union[Dict[str, Any], None]:
        try:
            import requests
            resp = requests.post(self._url, json=payload)
            resp.raise_for_status()
            response_json = resp.json()
            return response_json
        except Exception as e:
            error_str = f"Policy server request failed: {e}"
            return {self._decode_option.error_str_key: error_str}


__all__ = [
    'SynchronizedPolicyInterfaceJson',
    'SynchronizedPolicyInterfaceJsonRequestsPost'
]
