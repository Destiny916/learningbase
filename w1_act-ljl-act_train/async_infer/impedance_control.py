from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import copy
from async_infer.async_infer_typedef import NpArray1d


@dataclass
class ImpedanceControlParameter(object):
    # Dimension of the state, > 0
    state_dim: int

    # Weight
    K_p: NpArray1d = field(default_factory=NpArray1d)
    K_v: NpArray1d = field(default_factory=NpArray1d)
    nominal_mass: NpArray1d = field(default_factory=NpArray1d)

    # [0, state_dim) indicates some state are actually tool discrete_tool_state_indices,
    # These are integral of binary, and should NOT be continuously interpolated
    discrete_tool_state_indices: List[int] = field(default_factory=list)


class ImpedanceControl(object):

    def __init__(self, parameter: ImpedanceControlParameter):
        # Parameter
        self._parameter = parameter

        # State
        self._x: Optional[NpArray1d] = None
        self._dot_x: Optional[NpArray1d] = None
        self._ddot_x: Optional[NpArray1d] = None
        self._t: float = 0.0

    @property
    def x(self):
        return self._x

    @property
    def dot_x(self):
        return self._dot_x

    @property
    def ddot_x(self):
        return self._ddot_x

    @property
    def discrete_tool_state_indices(self):
        return self._parameter.discrete_tool_state_indices

    def reset(self,
              x: NpArray1d,
              dot_x: Optional[NpArray1d],
              ddot_x: Optional[NpArray1d]):
        self._x = copy.deepcopy(x)
        self._dot_x = copy.deepcopy(dot_x) if (dot_x is not None) else np.zeros_like(x)
        self._ddot_x = copy.deepcopy(ddot_x) if (ddot_x is not None) else np.zeros_like(x)
        self._t = 0.0

    def step(self, delta_t: float, x_target: NpArray1d):
        # Calculate acceleration using impedance control equation
        # nominal_mass * ddot(x) + K_v * dot(x) = K_p (x_target - x)
        # Rearranged: ddot(x) = (K_p * (x_target - x) - K_v * dot(x)) / nominal_mass
        self._ddot_x = (self._parameter.K_p * (
                x_target - self._x) - self._parameter.K_v * self._dot_x) / self._parameter.nominal_mass

        # Update velocity using Euler integration
        self._dot_x += self._ddot_x * delta_t

        # Update position using Euler integration
        self._x += self._dot_x * delta_t

        # Update time
        self._t += delta_t

        # Update discrete state
        if len(self.discrete_tool_state_indices) > 0:
            self._x[self.discrete_tool_state_indices] = x_target[self.discrete_tool_state_indices]
            self._dot_x[self.discrete_tool_state_indices] = 0.0
            self._ddot_x[self.discrete_tool_state_indices] = 0.0

# sandbox below
# Please refer to test_impedance_ctrl.py
