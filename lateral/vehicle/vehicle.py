"""
Vehicle class implementing 2-DOF bicycle model for lateral dynamics.
VERSION 2.0 - Clean implementation with rate limiting
"""

import numpy as np
from scipy.linalg import expm
from typing import Tuple
from .components import VehicleParameters, VehicleState


class Vehicle:
    """2-DOF Bicycle model for lateral dynamics control."""
    
    def __init__(self, 
                 initial_y: float = 0.0,
                 initial_psi: float = 0.0,
                 initial_x: float = 0.0,
                 vehicle_id: str = "Vehicle",
                 longitudinal_velocity: float = 20.0):
        
        self.params = VehicleParameters()
        self.state = VehicleState()
        self.vehicle_id = vehicle_id
        
        self.state.y = initial_y
        self.state.psi = initial_psi
        self.state.x = initial_x
        self.state.vx = longitudinal_velocity
        self.vx = longitudinal_velocity
        self.L = self.params.length
        
        # Steering history for rate limiting
        self.delta_prev = 0.0
        self.ay_prev = 0.0
        
        # State-space matrices
        self.A_c = None
        self.B_c = None
        self.A_d = None
        self.B_d = None
        self.C = None
        self.A = None
        self.B1 = None
        self.B2 = None
        
        self.joined_platoon = False
        self._build_state_space_matrices()
        
        print(f"🚗 Vehicle '{vehicle_id}' initialized at y={initial_y:.1f}m")
        
    def _build_state_space_matrices(self, dt: float = 0.01):
        """Build continuous and discrete-time state-space matrices."""
        p = self.params
        vx = max(self.vx, 0.001)  # Avoid division by zero
        
        # Continuous-time A matrix (2-DOF bicycle model with kinematic coupling)
        a22 = -2 * (p.Cf + p.Cr) / (p.mass * vx)
        a24 = -vx - (2 * p.Cf * p.lf - 2 * p.Cr * p.lr) / (p.mass * vx)
        a42 = -(2 * p.lf * p.Cf - 2 * p.lr * p.Cr) / (p.Iz * vx)
        a44 = -(2 * p.lf**2 * p.Cf + 2 * p.lr**2 * p.Cr) / (p.Iz * vx)
        
        # Row 0: dy/dt = y_dot + vx*psi (kinematic coupling for global Y position)
        self.A_c = np.array([
            [0.0, 1.0, vx,  0.0],
            [0.0, a22, 0.0, a24],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, a42, 0.0, a44]
        ])
        
        b2 = 2 * p.Cf / p.mass
        b4 = 2 * (p.lf * p.Cf) / p.Iz
        self.B_c = np.array([[0.0], [b2], [0.0], [b4]])
        
        self.C = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ])
        
        self._discretize_zoh(dt)
        
    def _discretize_zoh(self, dt: float):
        """Discretize using Zero-Order Hold (matrix exponential)."""
        n = self.A_c.shape[0]
        m = self.B_c.shape[1]
        
        augmented = np.zeros((n + m, n + m))
        augmented[0:n, 0:n] = self.A_c * dt
        augmented[0:n, n:n+m] = self.B_c * dt
        
        exp_augmented = expm(augmented)
        
        self.A_d = exp_augmented[0:n, 0:n]
        self.B_d = exp_augmented[0:n, n:n+m]
        
        self.A = self.A_d
        self.B1 = self.B_d
        self.B2 = self.B_d.copy()
    
    def get_state_space_matrices(self, dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get discrete-time state-space matrices for Nash solver."""
        self._discretize_zoh(dt)
        return self.A_d, self.B_d, self.C
    
    def update_dynamics(self, dt: float, delta_commanded: float):
        """Update vehicle state with rate limiting."""
        if self.A_d is None:
            self._build_state_space_matrices(dt)
        
        # 1. Steering angle limits
        delta_commanded = np.clip(delta_commanded, 
                                  -self.params.max_steering_angle,
                                  self.params.max_steering_angle)
        
        # 2. Rate limiting
        max_delta_change = self.params.max_steering_rate * dt
        delta_change = np.clip(delta_commanded - self.delta_prev, 
                               -max_delta_change, max_delta_change)
        delta_actual = self.delta_prev + delta_change
        self.delta_prev = delta_actual
        self.state.delta = delta_actual
        
        # 3. State update: x[k+1] = A_d * x[k] + B_d * u[k]
        x_current = self.state.get_state_vector().reshape(-1, 1)
        u = np.array([[delta_actual]])
        x_next = self.A_d @ x_current + self.B_d @ u
        
        # 4. State clamping for numerical stability
        # y_dot: limit lateral velocity
        x_next[1, 0] = np.clip(x_next[1, 0], -2.0, 2.0)   # ±2 m/s max
        
        # psi: limit heading angle (CRITICAL - must stay small for bicycle model validity)
        x_next[2, 0] = np.clip(x_next[2, 0], -np.radians(30), np.radians(30))  # ±30 deg max
        
        # psi_dot: limit yaw rate  
        x_next[3, 0] = np.clip(x_next[3, 0], -0.3, 0.3)   # ±17 deg/s max
        
        self.state.set_state_vector(x_next.flatten())
        
        # 5. Lateral acceleration with jerk limiting
        ay_new = self.vx * self.state.psi_dot
        max_ay_change = self.params.max_lateral_jerk * dt
        ay_change = np.clip(ay_new - self.ay_prev, -max_ay_change, max_ay_change)
        self.state.ay = self.ay_prev + ay_change
        self.ay_prev = self.state.ay
        
        # Update longitudinal position
        self.state.x += self.vx * dt
        
        # NaN check
        if not np.isfinite(self.state.get_state_vector()).all():
            print(f"⚠️ {self.vehicle_id}: NaN detected, resetting derivatives")
            self.state.y_dot = 0.0
            self.state.psi_dot = 0.0
    
    def get_state_vector(self) -> np.ndarray:
        return self.state.get_state_vector()
    
    def reset_steering(self):
        self.delta_prev = 0.0
        self.ay_prev = 0.0
        self.state.delta = 0.0


__all__ = ['Vehicle']
