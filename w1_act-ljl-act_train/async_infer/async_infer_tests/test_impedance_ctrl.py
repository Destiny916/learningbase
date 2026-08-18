import numpy as np
import unittest
from async_infer.impedance_control import ImpedanceControl, ImpedanceControlParameter


class TestImpedanceControl(unittest.TestCase):
    """Test cases for ImpedanceControl class"""
    
    def test_1d_impedance_control(self):
        """Test 1D impedance control convergence to target"""
        # Create parameter with tuned values for faster convergence
        param_1d = ImpedanceControlParameter(
            state_dim=1,
            K_p=np.array([200.0]),
            K_v=np.array([20.0]),
            nominal_mass=np.array([1.0])
        )
        
        # Create controller
        ctrl_1d = ImpedanceControl(param_1d)
        
        # Reset with initial conditions
        initial_x = np.array([0.0])
        initial_dot_x = np.array([0.0])
        initial_ddot_x = np.array([0.0])
        ctrl_1d.reset(initial_x, initial_dot_x, initial_ddot_x)
        
        # Target position
        x_target = np.array([1.0])
        delta_t = 0.01
        simulation_time = 5.0
        steps = int(simulation_time / delta_t)
        
        # Run simulation
        for i in range(steps):
            ctrl_1d.step(delta_t, x_target)
        
        # Check if position converges to target
        self.assertTrue(np.isclose(ctrl_1d.x[0], x_target[0], atol=1e-3), 
                       f"Position did not converge to target: {ctrl_1d.x[0]} vs {x_target[0]}")
        
        # Check if velocity and acceleration are close to zero
        self.assertTrue(np.isclose(ctrl_1d.dot_x[0], 0.0, atol=1e-3), 
                       f"Velocity did not converge to zero: {ctrl_1d.dot_x[0]}")
        self.assertTrue(np.isclose(ctrl_1d.ddot_x[0], 0.0, atol=1e-3), 
                       f"Acceleration did not converge to zero: {ctrl_1d.ddot_x[0]}")

    def test_2d_impedance_control(self):
        """Test 2D impedance control convergence to target"""
        # Create 2D parameter with tuned values for faster convergence
        param_2d = ImpedanceControlParameter(
            state_dim=2,
            K_p=np.array([250.0, 200.0]),
            K_v=np.array([10.0, 8.944]),
            nominal_mass=np.array([1.0, 1.0])
        )
        
        # Create controller
        ctrl_2d = ImpedanceControl(param_2d)
        
        # Reset with initial conditions
        initial_x_2d = np.array([0.0, 0.0])
        initial_dot_x_2d = np.array([0.0, 0.0])
        initial_ddot_x_2d = np.array([0.0, 0.0])
        ctrl_2d.reset(initial_x_2d, initial_dot_x_2d, initial_ddot_x_2d)
        
        # Target position
        x_target_2d = np.array([1.0, 2.0])
        delta_t = 0.01
        simulation_time = 5.0
        steps = int(simulation_time / delta_t)
        
        # Run simulation
        for i in range(steps):
            ctrl_2d.step(delta_t, x_target_2d)
        
        # Check if position converges to target
        self.assertTrue(np.allclose(ctrl_2d.x, x_target_2d, atol=1e-3), 
                       f"Position did not converge to target: {ctrl_2d.x} vs {x_target_2d}")
        
        # Check if velocity and acceleration are close to zero
        self.assertTrue(np.allclose(ctrl_2d.dot_x, np.zeros_like(x_target_2d), atol=1e-3), 
                       f"Velocity did not converge to zero: {ctrl_2d.dot_x}")
        self.assertTrue(np.allclose(ctrl_2d.ddot_x, np.zeros_like(x_target_2d), atol=1e-3), 
                       f"Acceleration did not converge to zero: {ctrl_2d.ddot_x}")

    def test_discrete_state_handling(self):
        """Test that discrete states are handled correctly"""
        # Create parameter with discrete state index
        param = ImpedanceControlParameter(
            state_dim=2,
            K_p=np.array([25.0, 25.0]),
            K_v=np.array([10.0, 10.0]),
            nominal_mass=np.array([1.0, 1.0]),
            discrete_tool_state_indices=[1]  # Second state is discrete
        )
        
        # Create controller
        ctrl = ImpedanceControl(param)
        
        # Reset with initial conditions
        initial_x = np.array([0.0, 0.0])
        initial_dot_x = np.array([0.0, 0.0])
        initial_ddot_x = np.array([0.0, 0.0])
        ctrl.reset(initial_x, initial_dot_x, initial_ddot_x)
        
        # Target position
        x_target = np.array([1.0, 1.0])
        delta_t = 0.01
        
        # Run one step
        ctrl.step(delta_t, x_target)
        
        # Check that continuous state is updated
        self.assertTrue(ctrl.x[0] > 0.0, "Continuous state should be updated")
        
        # Check that discrete state is set directly to target
        self.assertEqual(ctrl.x[1], x_target[1], 
                         f"Discrete state should be set to target: {ctrl.x[1]} vs {x_target[1]}")
        self.assertEqual(ctrl.dot_x[1], 0.0, 
                         f"Discrete state velocity should be zero: {ctrl.dot_x[1]}")
        self.assertEqual(ctrl.ddot_x[1], 0.0, 
                         f"Discrete state acceleration should be zero: {ctrl.ddot_x[1]}")


if __name__ == "__main__":
    unittest.main()
