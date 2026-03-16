"""
Human Driver Model for Lateral Control.
VERSION 2.0 - Stanley Controller with moderate gains
"""

import numpy as np
from typing import List, Dict
from dataclasses import dataclass
from enum import Enum

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DRIVER_PARAMS, STANLEY_K_SOFT, STANLEY_FILTER_ALPHA,
                    NASH_NP, SIMULATION_DT)
from vehicle.components import VehicleParameters, VehicleState
class DriverType(Enum):
    CAUTIOUS = "cautious"
    NORMAL = "normal"
    AGGRESSIVE = "aggressive"


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
        self.Np = NASH_NP  # From config.py

        # Initialize Stanley params from DRIVER_PARAMS
        p = DRIVER_PARAMS.get(driver_type, DRIVER_PARAMS['normal'])
        self.stanley = StanleyParams(k_e=p['stanley_k_e'], k_psi=p['stanley_k_psi'])
        self.driver_type = DriverType(driver_type) if driver_type in ('cautious', 'normal', 'aggressive') else DriverType.NORMAL

        # Override steering limit from vehicle.params
        self.stanley.delta_max = vehicle.params.max_steering_angle

        # Low-pass filter for steering
        self.delta_filtered = 0.0
        self.filter_alpha = STANLEY_FILTER_ALPHA

        print(f"🧑‍✈️ Human Driver (Stanley) Initialized - driver_type={driver_type}, k_e={self.stanley.k_e:.4f}")
    
    def set_driver_type(self, driver_type: str):
        """Set driver personality."""
        p = DRIVER_PARAMS.get(driver_type, DRIVER_PARAMS['normal'])
        self.driver_type = DriverType(driver_type) if driver_type in ('cautious', 'normal', 'aggressive') else DriverType.NORMAL
        self.stanley.k_e   = p['stanley_k_e']
        self.stanley.k_psi = p['stanley_k_psi']
        print(f"👤 Driver type: {driver_type} (k_e={self.stanley.k_e:.3f}, k_psi={self.stanley.k_psi:.2f})")
    
    def compute_stanley_steering(self, y_error: float, psi_error: float, velocity: float) -> float:
        """
        Compute Stanley steering angle.
        
        Using a simpler, more stable formulation:
        delta = -k_y * y_error - k_psi * psi_error
        
        This directly matches the stable PD controller we tested.
        """
        s = self.stanley
        
        # Direct proportional control (like the working PD controller)
        # Note: k_e here acts as k_y, k_psi acts as k_psi
        delta_raw = -s.k_e * y_error - s.k_psi * psi_error
        
        # Low-pass filter
        delta_filtered = self.filter_alpha * delta_raw + (1 - self.filter_alpha) * self.delta_filtered
        self.delta_filtered = delta_filtered
        
        # Limits
        delta = np.clip(delta_filtered, -s.delta_max, s.delta_max)
        
        return delta
    
    def generate_human_reference_trajectory(self, ego_vehicle, target_y: float,
                                            obstacles: List[Dict] = None) -> np.ndarray:
        """
        Generate human's intended trajectory.
        
        IMPORTANT: Human should target the SAME goal as system (target_y, psi=0).
        The difference is in HOW they get there (different gains), not WHERE.
        """
        trajectory = np.zeros((self.Np, 2))
        
        # Human targets the same goal as system
        for i in range(self.Np):
            # Target is always: y=target_y, psi=0
            trajectory[i, 0] = target_y
            trajectory[i, 1] = 0.0
        
        return trajectory
    
    def get_human_steering_input(self, ego_vehicle, target_y: float) -> float:
        """Get instantaneous steering command."""
        y_error = ego_vehicle.state.y - target_y
        psi_error = ego_vehicle.state.psi
        vx = ego_vehicle.vx
        return self.compute_stanley_steering(y_error, psi_error, vx)
    
    def reset(self):
        self.delta_filtered = 0.0


__all__ = ['HumanDriver', 'DriverType', 'StanleyParams']
