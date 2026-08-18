import os
from pathlib import Path


class DefaultObservationKeys:

    @staticmethod
    def observation_prefix() -> str:
        return 'observation'

    @staticmethod
    def state_key() -> str:
        return DefaultObservationKeys.observation_prefix() + '.state'

    @staticmethod
    def rgb_image_prefix() -> str:
        return DefaultObservationKeys.observation_prefix() + '.rgb'

    @staticmethod
    def depth_image_prefix() -> str:
        return DefaultObservationKeys.observation_prefix() + '.depth'


class DefaultActionKeys:

    @staticmethod
    def action_prefix() -> str:
        return 'action'

    @staticmethod
    def trajectory_key() -> str:
        return DefaultActionKeys.action_prefix() + '.trajectory'

    @staticmethod
    def trajectory_time_key():
        return DefaultActionKeys.action_prefix() + '.trajectory_time'

    @staticmethod
    def error_str_key():
        return DefaultActionKeys.action_prefix() + '.error'