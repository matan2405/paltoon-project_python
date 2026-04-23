"""
Human Driver Model for Lateral Control.

Research basis and adopted elements:
- Stanley steering structure (cross-track + heading regulation).
- Driver-personality parameterization through gain selection from config.
- Filtered steering command to reduce high-frequency actuation chatter.
"""

import numpy as np
from dataclasses import dataclass

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DRIVER_PARAMS, STANLEY_K_SOFT, STANLEY_FILTER_ALPHA,
                    NASH_NP, SIMULATION_DT)
from vehicle.components import VehicleParameters


@dataclass
class StanleyParams:
    """Stanley controller parameters."""
    k_e: float = DRIVER_PARAMS['normal']['stanley_k_e']
    k_psi: float = DRIVER_PARAMS['normal']['stanley_k_psi']
    k_soft: float = STANLEY_K_SOFT
    delta_max: float = VehicleParameters.max_steering_angle  # Default matches vehicle (Audi TT: 25°)


class HumanDriver:
    """Human driver model using Stanley controller."""

    def __init__(self, vehicle, target_lane_y: float = 0.0, dt: float = SIMULATION_DT,
                 driver_type: str = 'normal'):
        self.vehicle = vehicle
        self.target_lane_y = target_lane_y
        self.dt = dt
        self.Np = NASH_NP

        p = DRIVER_PARAMS.get(driver_type, DRIVER_PARAMS['normal'])
        self.stanley = StanleyParams(k_e=p['stanley_k_e'], k_psi=p['stanley_k_psi'])
        self.stanley.delta_max = vehicle.params.max_steering_angle

        self.delta_filtered = 0.0
        self.filter_alpha = STANLEY_FILTER_ALPHA

        print(f"🧑‍✈️ Human Driver (Stanley) Initialized - driver_type={driver_type}, k_e={self.stanley.k_e:.4f}")

    def set_driver_type(self, driver_type: str):
        """Set driver personality."""
        p = DRIVER_PARAMS.get(driver_type, DRIVER_PARAMS['normal'])
        self.stanley.k_e   = p['stanley_k_e']
        self.stanley.k_psi = p['stanley_k_psi']
        print(f"👤 Driver type: {driver_type} (k_e={self.stanley.k_e:.3f}, k_psi={self.stanley.k_psi:.2f})")

    def compute_stanley_steering(self, y_error: float, psi_error: float, velocity: float) -> float:
        """Compute Stanley steering angle."""
        s = self.stanley
        delta_raw = -s.k_e * y_error - s.k_psi * psi_error
        delta_filtered = self.filter_alpha * delta_raw + (1 - self.filter_alpha) * self.delta_filtered
        self.delta_filtered = delta_filtered
        return np.clip(delta_filtered, -s.delta_max, s.delta_max)

    def get_human_steering_input(self, ego_vehicle, target_y: float) -> float:
        """Get instantaneous steering command."""
        y_error = ego_vehicle.state.y - target_y
        psi_error = ego_vehicle.state.psi
        vx = ego_vehicle.vx
        return self.compute_stanley_steering(y_error, psi_error, vx)

    def reset(self):
        self.delta_filtered = 0.0


__all__ = ['HumanDriver', 'StanleyParams']
