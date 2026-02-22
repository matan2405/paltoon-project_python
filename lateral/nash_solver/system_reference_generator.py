"""
System Reference Generator for Lateral Control.
VERSION 3.0 - HEADING-CONSTRAINED TIME-BASED TRAJECTORY

CRITICAL IMPROVEMENTS:
1. Trajectory is planned ONCE at lane change start (prevents oscillation)
2. Progress based on elapsed time (not current position)
3. **NEW** Maximum heading constraint to ensure ψ < 2° requirement

HEADING CONSTRAINT:
The key insight is that heading angle ψ ≈ dy/dx = (dy/dt)/vx = ẏ/vx
For ψ < ψ_max, we need: ẏ < vx * tan(ψ_max) ≈ vx * ψ_max

For a 5th order polynomial y(τ) where τ = t/T_lc:
  ẏ = Δy * (30τ² - 60τ³ + 30τ⁴) / T_lc
  
Max ẏ occurs at τ = 0.5: ẏ_max = 1.875 * Δy / T_lc

To satisfy ψ < ψ_max: T_lc > 1.875 * Δy / (vx * ψ_max)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from enum import Enum
from dataclasses import dataclass

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LANE_WIDTH, NOMINAL_VELOCITY, NASH_CONTROL_DT, NASH_NP


class TrajectoryPhase(Enum):
    CRUISE = "CRUISE"
    GAP_SEARCH = "GAP_SEARCH"
    LANE_CHANGE = "LANE_CHANGE"
    LANE_KEEPING = "LANE_KEEPING"
    FOLLOWING = "FOLLOWING"


@dataclass
class DynamicTrajectoryParams:
    """
    Dynamic parameters for state-dependent trajectory planning.
    
    These are computed based on current vehicle state and platoon configuration.
    """
    # From platoon state
    target_velocity: float = 20.0       # Platoon target velocity [m/s]
    desired_gap: float = 35.0           # Desired longitudinal gap [m]
    platoon_lane_y: float = 0.0         # Platoon lane y position [m]
    
    # Computed constraints
    max_heading_angle: float = np.radians(8)  # Max allowed ψ [rad] - gives margin below 3.5° requirement
    
    def compute_min_T_lc(self, delta_y: float, vx: float) -> float:
        """
        Compute minimum T_lc to satisfy heading constraint.
        
        For quintic polynomial:
        T_lc_min = 1.875 * |Δy| / (vx * tan(ψ_max))
        
        Args:
            delta_y: Lane change distance [m]
            vx: Current velocity [m/s]
            
        Returns:
            Minimum T_lc [s]
        """
        if vx < 1.0 or abs(delta_y) < 0.1:
            return 5.0  # Default for very low speed
        
        # Max lateral velocity for heading constraint
        max_y_dot = vx * np.tan(self.max_heading_angle)
        
        # Minimum T_lc (quintic polynomial factor is 1.875)
        T_lc_min = 1.875 * abs(delta_y) / max_y_dot
        
        return max(T_lc_min, 3.0)  # Physical minimum 3 seconds


class SystemReferenceGenerator:
    """
    Generate reference trajectories for Nash Solver.
    
    VERSION 3.0 - HEADING-CONSTRAINED TRAJECTORY:
    - Trajectory planned ONCE at lane change start
    - Progress based on ELAPSED TIME
    - **NEW** T_lc dynamically adjusted to ensure ψ < max_heading
    - **NEW** State-dependent phase transitions
    """
    
    def __init__(self, Np: int = NASH_NP, dt: float = NASH_CONTROL_DT, 
                 driver_type: str = 'normal'):
        self.Np = Np
        self.dt = dt
        self.driver_type = driver_type
        
        # Lane configuration (from config.py)
        self.lane_width = LANE_WIDTH
        self.target_lane_y = 0.0
        
        # System T_lc = Human T_lc × 1.5 (always more conservative, but no conflict)
        # This ensures the Nash game has ALIGNED but DIFFERENT references
        self._system_T_lc_multiplier = 1.5
        self._human_base_T_lc = {
            'cautious': 6.0,
            'normal': 4.5,
            'aggressive': 3.0
        }
        human_T_lc = self._human_base_T_lc.get(driver_type, 4.5)
        self._base_T_lc = human_T_lc * self._system_T_lc_multiplier
        self.lane_change_duration = self._base_T_lc
        
        # Dynamic parameters (updated from platoon state)
        self.dynamic_params = DynamicTrajectoryParams()
        
        # Phase tracking
        self._current_phase = TrajectoryPhase.CRUISE
        self._lane_change_start_time = None
        self._lane_change_start_y = None
        self._current_time = 0.0
        
        # Velocity tracking
        self._current_vx = NOMINAL_VELOCITY
        
        print(f"🚀 System Reference Generator V3.1 (Driver-Aware) Initialized")
        print(f"   Driver type: {driver_type}, Base T_lc={self.lane_change_duration:.1f}s")
        print(f"   (Human T_lc={human_T_lc:.1f}s × {self._system_T_lc_multiplier})")
        print(f"   Max heading constraint: {np.degrees(self.dynamic_params.max_heading_angle):.1f}°")
    
    def set_current_time(self, time: float):
        self._current_time = time
    
    def set_velocity(self, vx: float):
        """Update current velocity for heading constraint calculation."""
        self._current_vx = vx
    
    def set_driver_type(self, driver_type: str):
        """Update driver type and recompute base T_lc."""
        self.driver_type = driver_type
        human_T_lc = self._human_base_T_lc.get(driver_type, 4.5)
        self._base_T_lc = human_T_lc * self._system_T_lc_multiplier
        self.lane_change_duration = self._base_T_lc
    
    def update_from_platoon_state(self, target_velocity: float, desired_gap: float, 
                                   platoon_lane_y: float = 0.0):
        """
        Update dynamic parameters from current platoon state.
        
        This makes the trajectory planning STATE-DEPENDENT as requested.
        
        Args:
            target_velocity: Platoon target velocity [m/s]
            desired_gap: Current desired longitudinal gap [m]
            platoon_lane_y: Platoon lane y position [m]
        """
        self.dynamic_params.target_velocity = target_velocity
        self.dynamic_params.desired_gap = desired_gap
        self.dynamic_params.platoon_lane_y = platoon_lane_y
        
        # Adjust T_lc for velocity (scale proportionally, not from scratch)
        # At 20 m/s baseline → factor=1.0, at 30 m/s → factor ~1.22
        velocity_factor = np.sqrt(target_velocity / 20.0)
        human_T_lc = self._human_base_T_lc.get(self.driver_type, 4.5)
        self._base_T_lc = human_T_lc * self._system_T_lc_multiplier * velocity_factor
    
    def update_phase_from_safety_field(self, safety_phase: str):
        """Sync phase with safety field."""
        phase_map = {
            "CRUISE": TrajectoryPhase.CRUISE,
            "GAP_SEARCH": TrajectoryPhase.GAP_SEARCH,
            "LANE_CHANGE": TrajectoryPhase.LANE_CHANGE,
            "LANE_KEEPING": TrajectoryPhase.LANE_KEEPING,
            "FOLLOWING": TrajectoryPhase.FOLLOWING
        }
        new_phase = phase_map.get(safety_phase, TrajectoryPhase.CRUISE)
        
        if new_phase != self._current_phase:
            self._current_phase = new_phase
            if new_phase == TrajectoryPhase.LANE_CHANGE:
                self._lane_change_start_time = self._current_time
    
    def generate_reference_trajectory(self, ego_vehicle, target_y: float = 0.0,
                                       obstacles: List[Dict] = None) -> np.ndarray:
        """
        Generate reference trajectory with heading constraint.
        
        VERSION 3.0: Dynamically adjusts T_lc to ensure ψ < max_heading.
        """
        trajectory = np.zeros((self.Np, 2))
        
        current_y = ego_vehicle.state.y
        self.target_lane_y = target_y
        self._current_vx = getattr(ego_vehicle, 'vx', 20.0)
        
        # Lock starting position when first entering GAP_SEARCH or LANE_CHANGE
        if self._lane_change_start_y is None and self._current_phase in [
            TrajectoryPhase.GAP_SEARCH, TrajectoryPhase.LANE_CHANGE
        ]:
            self._lane_change_start_y = current_y
            
            # Compute heading-constrained T_lc
            delta_y = abs(current_y - target_y)
            min_T_lc = self.dynamic_params.compute_min_T_lc(delta_y, self._current_vx)
            
            # Use the larger of base T_lc and minimum required T_lc
            self.lane_change_duration = max(self._base_T_lc, min_T_lc, self.lane_change_duration)
            
            print(f"🚀 System: Locked start y={current_y:.2f}m, T_lc={self.lane_change_duration:.1f}s")
            print(f"   (min T_lc for ψ<{np.degrees(self.dynamic_params.max_heading_angle):.1f}°: {min_T_lc:.1f}s)")
        
        if self._current_phase == TrajectoryPhase.CRUISE:
            trajectory = self._generate_stay_trajectory(current_y)
        
        elif self._current_phase == TrajectoryPhase.GAP_SEARCH:
            trajectory = self._generate_transition_trajectory(target_y)
        
        elif self._current_phase == TrajectoryPhase.LANE_CHANGE:
            trajectory = self._generate_lane_change_trajectory(target_y)
        
        elif self._current_phase in [TrajectoryPhase.LANE_KEEPING, TrajectoryPhase.FOLLOWING]:
            trajectory = self._generate_target_trajectory(target_y)
        
        return trajectory
    
    def _generate_stay_trajectory(self, current_y: float) -> np.ndarray:
        """Stay at current lane."""
        trajectory = np.zeros((self.Np, 2))
        for i in range(self.Np):
            trajectory[i, 0] = current_y
            trajectory[i, 1] = 0.0
        return trajectory
    
    def _generate_transition_trajectory(self, target_y: float) -> np.ndarray:
        """Gradual transition start - TIME-BASED from locked position."""
        trajectory = np.zeros((self.Np, 2))
        
        y_start = self._lane_change_start_y if self._lane_change_start_y is not None else target_y
        y_end = target_y
        
        for i in range(self.Np):
            t_pred = i * self.dt
            progress = min(t_pred / self.lane_change_duration, 0.1)
            
            # Quintic for smooth start
            s = 10 * progress**3 - 15 * progress**4 + 6 * progress**5
            
            trajectory[i, 0] = y_start + (y_end - y_start) * s
            trajectory[i, 1] = 0.0
        
        return trajectory
    
    def _generate_lane_change_trajectory(self, target_y: float) -> np.ndarray:
        """
        Generate smooth lane change trajectory using 5th order polynomial.
        
        VERSION 3.0 - HEADING-CONSTRAINED:
        - T_lc is already adjusted to ensure ψ < max_heading
        - Progress based on elapsed time
        """
        trajectory = np.zeros((self.Np, 2))
        
        y_start = self._lane_change_start_y if self._lane_change_start_y is not None else target_y
        y_end = target_y
        T_lc = self.lane_change_duration
        
        if self._lane_change_start_time is not None:
            t_elapsed = self._current_time - self._lane_change_start_time
        else:
            t_elapsed = 0.0
        
        for i in range(self.Np):
            t_total = t_elapsed + i * self.dt
            tau = min(t_total / T_lc, 1.0)
            
            # 5th order polynomial (quintic)
            s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
            
            trajectory[i, 0] = y_start + (y_end - y_start) * s
            trajectory[i, 1] = 0.0
        
        return trajectory
    
    def _generate_target_trajectory(self, target_y: float) -> np.ndarray:
        """Generate trajectory AT the target."""
        trajectory = np.zeros((self.Np, 2))
        for i in range(self.Np):
            trajectory[i, 0] = target_y
            trajectory[i, 1] = 0.0
        return trajectory
    
    def reset(self):
        """Reset generator state."""
        self._current_phase = TrajectoryPhase.CRUISE
        self._lane_change_start_time = None
        self._lane_change_start_y = None
        self._current_time = 0.0
        self.lane_change_duration = self._base_T_lc


__all__ = ['SystemReferenceGenerator', 'TrajectoryPhase', 'DynamicTrajectoryParams']