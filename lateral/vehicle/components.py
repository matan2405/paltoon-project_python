"""
Vehicle components module for lateral control simulation.
VERSION 2.0
"""

import numpy as np
from dataclasses import dataclass

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NOMINAL_VELOCITY


@dataclass
class VehicleParameters:
    """Audi TT Coupé 2.0 TFSI (230PS) parameters - from official technical data."""
    # Vehicle dimensions
    mass: float = 1305.0  # kg (kerb weight without driver)
    length: float = 4.177  # m (overall length)
    width: float = 1.832  # m (overall width)
    height: float = 1.353  # m (overall height)
    wheelbase: float = 2.505  # m
    track_width_front: float = 1.572  # m (front track)
    track_width_rear: float = 1.555  # m (rear track)
    
    # Center of gravity distances (assumed 50/50 distribution)
    lf: float = 1.2525  # distance from CG to front axle
    lr: float = 1.2525  # distance from CG to rear axle
    
    # Tire cornering stiffness
    Cf: float = 80000.0  # Front axle cornering stiffness N/rad
    Cr: float = 80000.0  # Rear axle cornering stiffness N/rad
    
    # Moment of inertia
    Iz: float = 2500.0  # kg·m² (estimated)
    
    # Steering constraints (Audi TT specs)
    max_steering_angle: float = np.radians(25.0)  # rad (25°)
    max_steering_rate: float = np.radians(15.0)   # rad/s (15°/s)
    steering_ratio: float = 14.6  # from technical data
    
    # Lateral dynamics constraints
    max_lateral_acceleration: float = 3.0  # m/s²
    max_lateral_jerk: float = 2.5  # m/s³
    
    # Performance
    nominal_velocity: float = NOMINAL_VELOCITY  # m/s
    max_velocity: float = 250.0 / 3.6  # m/s (250 km/h electronically limited)


class VehicleState:
    """Vehicle state for lateral dynamics: [y, y_dot, psi, psi_dot]"""
    
    def __init__(self):
        self.y = 0.0
        self.y_dot = 0.0
        self.psi = 0.0
        self.psi_dot = 0.0
        self.x = 0.0
        self.vx = NOMINAL_VELOCITY
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
