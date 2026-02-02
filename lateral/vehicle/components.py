"""
Vehicle components module for lateral control simulation.
VERSION 2.0
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class VehicleParameters:
    """Audi TT 2.0 TFSI parameters."""
    mass: float = 1305.0
    length: float = 4.177
    width: float = 1.832
    wheelbase: float = 2.505
    lf: float = 1.2525
    lr: float = 1.2525
    Cf: float = 80000.0
    Cr: float = 80000.0
    Iz: float = 2500.0
    max_steering_angle: float = np.radians(25.0)
    max_steering_rate: float = np.radians(15.0)
    max_lateral_acceleration: float = 3.0
    max_lateral_jerk: float = 2.5
    nominal_velocity: float = 20.0


class VehicleState:
    """Vehicle state for lateral dynamics: [y, y_dot, psi, psi_dot]"""
    
    def __init__(self):
        self.y = 0.0
        self.y_dot = 0.0
        self.psi = 0.0
        self.psi_dot = 0.0
        self.x = 0.0
        self.vx = 20.0
        self.delta = 0.0
        self.ay = 0.0
        self.beta = 0.0
    
    def get_state_vector(self) -> np.ndarray:
        return np.array([self.y, self.y_dot, self.psi, self.psi_dot])
    
    def set_state_vector(self, state: np.ndarray):
        self.y = float(state[0])
        self.y_dot = float(state[1])
        self.psi = float(state[2])
        self.psi_dot = float(state[3])


__all__ = ['VehicleParameters', 'VehicleState']
