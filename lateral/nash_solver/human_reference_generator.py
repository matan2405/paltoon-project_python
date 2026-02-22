"""
Human Reference Generator for Lateral Control.
VERSION 3.0 - HEADING-CONSTRAINED TIME-BASED TRAJECTORY

Based on Li et al. 2019:
- R2(k) = human driver's previewed target path
- Different from R1 (system path) in shape and timing
- Same final target (y=0, psi=0) but faster/more direct trajectory

Key differences from System Reference:
- Uses 3rd order polynomial (faster transition, less smooth)
- Lane change duration depends on driver personality
- More aggressive path shape

VERSION 3.0 IMPROVEMENTS:
- **NEW** T_lc dynamically adjusted based on heading constraint
- **NEW** State-dependent parameters from platoon configuration
"""

import numpy as np
from typing import List, Dict, Optional
from enum import Enum
from dataclasses import dataclass

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LANE_WIDTH, NOMINAL_VELOCITY, NASH_CONTROL_DT, NASH_NP


class HumanTrajectoryPhase(Enum):
    CRUISE = "CRUISE"
    GAP_SEARCH = "GAP_SEARCH"
    LANE_CHANGE = "LANE_CHANGE"
    LANE_KEEPING = "LANE_KEEPING"
    FOLLOWING = "FOLLOWING"


@dataclass
class HumanDynamicParams:
    """Dynamic parameters for human driver trajectory planning."""
    target_velocity: float = 20.0
    desired_gap: float = 35.0
    platoon_lane_y: float = 0.0
    
    def compute_min_T_lc(self, delta_y: float, vx: float, max_heading: float) -> float:
        """
        Compute minimum T_lc for cubic polynomial with heading constraint.
        
        For cubic polynomial y(τ) = y0 + Δy*(3τ² - 2τ³):
        ẏ = Δy * (6τ - 6τ²) / T_lc
        Max ẏ at τ = 0.5: ẏ_max = 1.5 * Δy / T_lc
        
        For ψ < max_heading: T_lc > 1.5 * |Δy| / (vx * tan(max_heading))
        """
        if vx < 1.0 or abs(delta_y) < 0.1:
            return 8.0
        
        max_y_dot = vx * np.tan(max_heading)
        T_lc_min = 1.5 * abs(delta_y) / max_y_dot
        
        return max(T_lc_min, 2.0)  # Physical minimum only (no artificial floor)


class HumanReferenceGenerator:
    """
    Generate reference trajectories representing human driver's desired path.
    
    VERSION 3.0 - HEADING-CONSTRAINED TRAJECTORY:
    - Trajectory planned ONCE at lane change start
    - Progress based on ELAPSED TIME
    - **NEW** T_lc dynamically adjusted to ensure ψ < max_heading for each driver type
    - **NEW** State-dependent parameters from platoon configuration
    """
    
    def __init__(self, Np: int = NASH_NP, dt: float = NASH_CONTROL_DT, driver_type: str = 'normal'):
        self.Np = Np
        self.dt = dt
        self.driver_type = driver_type
        
        # Base lane change durations (will be adjusted dynamically)
        self._base_lane_change_durations = {
            'cautious': 6.0,
            'normal': 4.5,
            'aggressive': 3.0
        }
        self.lane_change_duration = self._base_lane_change_durations.get(driver_type, 12.0)
        
        # Maximum heading angles per driver type
        # Based on empirical data: Toledo & Zohar (2007), Song et al. (2013)
        # Required ψ_max for cubic polynomial: tan(ψ) ≥ 1.5·Δy/(vx·T_lc)
        #   cautious  T_lc=6.0s → ψ_min=2.5° → allow 3.0°
        #   normal    T_lc=4.5s → ψ_min=3.3° → allow 4.0°
        #   aggressive T_lc=3.0s → ψ_min=5.0° → allow 6.0°
        self.max_heading_angles = {
            'cautious': np.radians(3.0),    # Conservative - allows T_lc=6.0s
            'normal': np.radians(4.0),      # Moderate - allows T_lc=4.5s
            'aggressive': np.radians(6.0)   # Tolerant - allows T_lc=3.0s
        }
        self.max_heading_angle = self.max_heading_angles.get(driver_type, np.radians(4.0))
        
        # Dynamic parameters
        self.dynamic_params = HumanDynamicParams()
        
        # Lane configuration (from config.py)
        self.lane_width = LANE_WIDTH
        self.target_lane_y = 0.0
        
        # Phase tracking
        self._current_phase = HumanTrajectoryPhase.CRUISE
        self._lane_change_start_time = None
        self._lane_change_start_y = None
        self._current_time = 0.0
        self._current_vx = NOMINAL_VELOCITY
        
        print(f"👤 Human Reference Generator V3.0 (Heading-Constrained) Initialized")
        print(f"   Driver type: {driver_type}, Base T_lc={self.lane_change_duration}s")
        print(f"   Max heading: {np.degrees(self.max_heading_angle):.1f}°")
    
    def set_driver_type(self, driver_type: str):
        """Update driver type and associated parameters."""
        self.driver_type = driver_type
        self.lane_change_duration = self._base_lane_change_durations.get(driver_type, 12.0)
        self.max_heading_angle = self.max_heading_angles.get(driver_type, np.radians(4.0))
    
    def set_current_time(self, time: float):
        self._current_time = time
    
    def set_velocity(self, vx: float):
        """Update current velocity for heading constraint calculation."""
        self._current_vx = vx
    
    def update_from_platoon_state(self, target_velocity: float, desired_gap: float,
                                   platoon_lane_y: float = 0.0):
        """
        Update dynamic parameters from current platoon state.
        
        This makes the trajectory planning STATE-DEPENDENT.
        """
        self.dynamic_params.target_velocity = target_velocity
        self.dynamic_params.desired_gap = desired_gap
        self.dynamic_params.platoon_lane_y = platoon_lane_y
    
    def update_phase_from_safety_field(self, safety_phase: str):
        """Sync phase with safety field."""
        phase_map = {
            "CRUISE": HumanTrajectoryPhase.CRUISE,
            "GAP_SEARCH": HumanTrajectoryPhase.GAP_SEARCH,
            "LANE_CHANGE": HumanTrajectoryPhase.LANE_CHANGE,
            "LANE_KEEPING": HumanTrajectoryPhase.LANE_KEEPING,
            "FOLLOWING": HumanTrajectoryPhase.FOLLOWING
        }
        new_phase = phase_map.get(safety_phase, HumanTrajectoryPhase.CRUISE)
        
        if new_phase != self._current_phase:
            self._current_phase = new_phase
            if new_phase == HumanTrajectoryPhase.LANE_CHANGE:
                self._lane_change_start_time = self._current_time
    
    def generate_reference_trajectory(self, ego_vehicle, target_y: float = 0.0,
                                       obstacles: List[Dict] = None) -> np.ndarray:
        """
        Generate human driver's desired trajectory with heading constraint.
        
        VERSION 3.0: Dynamically adjusts T_lc to ensure ψ < max_heading.
        """
        trajectory = np.zeros((self.Np, 2))
        
        current_y = ego_vehicle.state.y
        self.target_lane_y = target_y
        self._current_vx = getattr(ego_vehicle, 'vx', 20.0)
        
        # Lock starting position when first entering GAP_SEARCH or LANE_CHANGE
        if self._lane_change_start_y is None and self._current_phase in [
            HumanTrajectoryPhase.GAP_SEARCH, HumanTrajectoryPhase.LANE_CHANGE
        ]:
            self._lane_change_start_y = current_y
            
            # Compute heading-constrained T_lc
            delta_y = abs(current_y - target_y)
            min_T_lc = self.dynamic_params.compute_min_T_lc(
                delta_y, self._current_vx, self.max_heading_angle
            )
            
            # Use the larger of base T_lc and minimum required T_lc
            base_T_lc = self._base_lane_change_durations.get(self.driver_type, 12.0)
            self.lane_change_duration = max(base_T_lc, min_T_lc, self.lane_change_duration)
            
            print(f"👤 Human: Locked start y={current_y:.2f}m, T_lc={self.lane_change_duration:.1f}s")
        
        if self._current_phase == HumanTrajectoryPhase.CRUISE:
            trajectory = self._generate_stay_trajectory(current_y)
        
        elif self._current_phase == HumanTrajectoryPhase.GAP_SEARCH:
            trajectory = self._generate_eager_transition(target_y)
        
        elif self._current_phase == HumanTrajectoryPhase.LANE_CHANGE:
            trajectory = self._generate_human_lane_change(target_y)
        
        elif self._current_phase in [HumanTrajectoryPhase.LANE_KEEPING, 
                                      HumanTrajectoryPhase.FOLLOWING]:
            trajectory = self._generate_target_trajectory(target_y)
        
        return trajectory
    
    def _generate_stay_trajectory(self, current_y: float) -> np.ndarray:
        """Stay at current lane."""
        trajectory = np.zeros((self.Np, 2))
        for i in range(self.Np):
            trajectory[i, 0] = current_y
            trajectory[i, 1] = 0.0
        return trajectory
    
    def _generate_eager_transition(self, target_y: float) -> np.ndarray:
        """Human driver wants to start lane change sooner."""
        trajectory = np.zeros((self.Np, 2))
        
        y_start = self._lane_change_start_y if self._lane_change_start_y is not None else target_y
        y_end = target_y
        
        for i in range(self.Np):
            t_pred = i * self.dt
            progress = min(t_pred / self.lane_change_duration, 0.15)
            s = 3 * progress**2 - 2 * progress**3
            
            trajectory[i, 0] = y_start + (y_end - y_start) * s
            trajectory[i, 1] = 0.0
        
        return trajectory
    
    def _generate_human_lane_change(self, target_y: float) -> np.ndarray:
        """
        Generate human's preferred lane change trajectory.
        
        VERSION 3.0 - HEADING-CONSTRAINED:
        - T_lc already adjusted to ensure ψ < max_heading
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
            
            # 3rd order polynomial (cubic) - human preference
            s = 3 * tau**2 - 2 * tau**3
            
            trajectory[i, 0] = y_start + (y_end - y_start) * s
            trajectory[i, 1] = 0.0
        
        return trajectory
    
    def _generate_target_trajectory(self, target_y: float) -> np.ndarray:
        """Generate trajectory at target."""
        trajectory = np.zeros((self.Np, 2))
        for i in range(self.Np):
            trajectory[i, 0] = target_y
            trajectory[i, 1] = 0.0
        return trajectory
    
    def reset(self):
        """Reset generator state."""
        self._current_phase = HumanTrajectoryPhase.CRUISE
        self._lane_change_start_time = None
        self._lane_change_start_y = None
        self._current_time = 0.0
        self.lane_change_duration = self._base_lane_change_durations.get(self.driver_type, 12.0)


__all__ = ['HumanReferenceGenerator', 'HumanTrajectoryPhase', 'HumanDynamicParams']