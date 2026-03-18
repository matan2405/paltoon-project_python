"""
Human Reference Generator for Lateral Control.
VERSION 4.0 - HEADING-CONSISTENT + SOFT LANE-KEEPING TRANSITION

CRITICAL FIXES (V4.0):
======================
1. HEADING REFERENCE: Now provides ψ_ref from trajectory derivative during lane change.
   Previously ψ_ref=0 always, which caused Nash to fight the lane change heading.

2. SOFT LANE-KEEPING TRANSITION: When entering LANE_KEEPING, generates a settling
   trajectory from current state to target, preventing reference discontinuity.

Based on Li et al. 2019:
- R2(k) = human driver's previewed target path
- Different from R1 (system path) in shape and timing
- Same final target (y=0, psi=0) but faster/more direct trajectory

Key differences from System Reference:
- Uses 3rd order polynomial (faster transition, less smooth)
- Lane change duration depends on driver personality
- More aggressive path shape
"""

import numpy as np
from typing import List, Dict, Optional
from enum import Enum
from dataclasses import dataclass

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DRIVER_PARAMS, LANE_WIDTH, NOMINAL_VELOCITY, NASH_CONTROL_DT, NASH_NP,
    REFGEN_MIN_TLC_CUBIC, REFGEN_MIN_TLC_FREE_ROAD_HUMAN,
)


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
            return REFGEN_MIN_TLC_FREE_ROAD_HUMAN
        max_y_dot = vx * np.tan(max_heading)
        T_lc_min = 1.5 * abs(delta_y) / max_y_dot
        return max(T_lc_min, REFGEN_MIN_TLC_CUBIC)


class HumanReferenceGenerator:
    """
    Generate reference trajectories representing human driver's desired path.
    
    VERSION 4.0 - HEADING-CONSISTENT + SOFT LANE-KEEPING TRANSITION
    
    Key improvements over V3.0:
    1. ψ_ref computed from trajectory derivative (not always 0)
    2. Soft settling trajectory when entering LANE_KEEPING
    """
    
    def __init__(self, Np: int = NASH_NP, dt: float = NASH_CONTROL_DT, driver_type: str = 'normal'):
        self.Np = Np
        self.dt = dt
        self.driver_type = driver_type
        
        # Per-driver lookup tables built from DRIVER_PARAMS
        self._base_lane_change_durations = {k: v['tlc'] for k, v in DRIVER_PARAMS.items()}
        self.lane_change_duration = self._base_lane_change_durations.get(driver_type, 4.5)

        self.max_heading_angles = {
            k: np.radians(v['max_heading_deg']) for k, v in DRIVER_PARAMS.items()
        }
        self.max_heading_angle = self.max_heading_angles.get(driver_type, np.radians(4.0))
        
        # Dynamic parameters
        self.dynamic_params = HumanDynamicParams()
        
        # Lane configuration
        self.lane_width = LANE_WIDTH
        self.target_lane_y = 0.0
        
        # Phase tracking
        self._current_phase = HumanTrajectoryPhase.CRUISE
        self._previous_phase = HumanTrajectoryPhase.CRUISE
        self._lane_change_start_time = None
        self._lane_change_start_y = None
        self._current_time = 0.0
        self._current_vx = NOMINAL_VELOCITY
        
        # =====================================================================
        # V4.0 NEW: Soft lane-keeping transition
        # =====================================================================
        self._lane_keeping_entry_time = None
        self._lane_keeping_entry_y = None
        self._lane_keeping_entry_psi = None
        self._lane_keeping_entry_y_dot = None
        
        # Settling time for lane-keeping entry (from DRIVER_PARAMS)
        self._settle_time = {k: v['human_settle_time'] for k, v in DRIVER_PARAMS.items()}
        self._T_settle = self._settle_time.get(driver_type, 3.0)
        # =====================================================================
        
        print(f"👤 Human Reference Generator V4.0 (Heading-Consistent) Initialized")
        print(f"   Driver type: {driver_type}, Base T_lc={self.lane_change_duration}s")
        print(f"   Max heading: {np.degrees(self.max_heading_angle):.1f}°")
    
    def set_driver_type(self, driver_type: str):
        self.driver_type = driver_type
        self.lane_change_duration = self._base_lane_change_durations.get(driver_type, 12.0)
        self.max_heading_angle = self.max_heading_angles.get(driver_type, np.radians(4.0))
        self._T_settle = self._settle_time.get(driver_type, 3.0)
    
    def set_current_time(self, time: float):
        self._current_time = time
    
    def set_velocity(self, vx: float):
        self._current_vx = vx
    
    def update_from_platoon_state(self, target_velocity: float, desired_gap: float,
                                   platoon_lane_y: float = 0.0):
        self.dynamic_params.target_velocity = target_velocity
        self.dynamic_params.desired_gap = desired_gap
        self.dynamic_params.platoon_lane_y = platoon_lane_y
    
    def update_phase_from_safety_field(self, safety_phase: str):
        """Sync phase with safety field — with V4.0 soft transition detection."""
        phase_map = {
            "CRUISE": HumanTrajectoryPhase.CRUISE,
            "GAP_SEARCH": HumanTrajectoryPhase.GAP_SEARCH,
            "LANE_CHANGE": HumanTrajectoryPhase.LANE_CHANGE,
            "LANE_KEEPING": HumanTrajectoryPhase.LANE_KEEPING,
            "FOLLOWING": HumanTrajectoryPhase.FOLLOWING
        }
        new_phase = phase_map.get(safety_phase, HumanTrajectoryPhase.CRUISE)
        
        if new_phase != self._current_phase:
            self._previous_phase = self._current_phase
            self._current_phase = new_phase
            
            if new_phase == HumanTrajectoryPhase.LANE_CHANGE:
                self._lane_change_start_time = self._current_time
            
            # V4.0: Capture entry time for soft transition
            if new_phase in [HumanTrajectoryPhase.LANE_KEEPING, HumanTrajectoryPhase.FOLLOWING]:
                self._lane_keeping_entry_time = self._current_time
    
    def generate_reference_trajectory(self, ego_vehicle, target_y: float = 0.0,
                                       obstacles: List[Dict] = None) -> np.ndarray:
        """
        Generate human driver's desired trajectory.
        
        V4.0: Heading-consistent references + soft lane-keeping transition.
        """
        trajectory = np.zeros((self.Np, 2))
        
        current_y = ego_vehicle.state.y
        current_psi = ego_vehicle.state.psi
        current_y_dot = ego_vehicle.state.y_dot
        self.target_lane_y = target_y
        self._current_vx = getattr(ego_vehicle, 'vx', 20.0)
        
        # Lock starting position when first entering GAP_SEARCH or LANE_CHANGE
        if self._lane_change_start_y is None and self._current_phase in [
            HumanTrajectoryPhase.GAP_SEARCH, HumanTrajectoryPhase.LANE_CHANGE
        ]:
            self._lane_change_start_y = current_y
            
            delta_y = abs(current_y - target_y)
            min_T_lc = self.dynamic_params.compute_min_T_lc(
                delta_y, self._current_vx, self.max_heading_angle
            )
            base_T_lc = self._base_lane_change_durations.get(self.driver_type, 12.0)
            self.lane_change_duration = max(base_T_lc, min_T_lc, self.lane_change_duration)
            
            print(f"👤 Human: Locked start y={current_y:.2f}m, T_lc={self.lane_change_duration:.1f}s")
        
        # V4.0: Capture entry state for soft transition
        if self._lane_keeping_entry_time is not None and self._lane_keeping_entry_y is None:
            if self._current_phase in [HumanTrajectoryPhase.LANE_KEEPING, HumanTrajectoryPhase.FOLLOWING]:
                self._lane_keeping_entry_y = current_y
                self._lane_keeping_entry_psi = current_psi
                self._lane_keeping_entry_y_dot = current_y_dot
        
        # Generate trajectory based on phase
        if self._current_phase == HumanTrajectoryPhase.CRUISE:
            trajectory = self._generate_stay_trajectory(current_y)
        
        elif self._current_phase == HumanTrajectoryPhase.GAP_SEARCH:
            trajectory = self._generate_eager_transition(target_y)
        
        elif self._current_phase == HumanTrajectoryPhase.LANE_CHANGE:
            trajectory = self._generate_human_lane_change(target_y)
        
        elif self._current_phase in [HumanTrajectoryPhase.LANE_KEEPING, 
                                      HumanTrajectoryPhase.FOLLOWING]:
            # V4.0: Use soft transition if within settling period
            if self._lane_keeping_entry_time is not None and self._lane_keeping_entry_y is not None:
                t_in_phase = self._current_time - self._lane_keeping_entry_time
                if t_in_phase < self._T_settle:
                    trajectory = self._generate_settling_trajectory(
                        target_y, current_y, current_psi, current_y_dot
                    )
                else:
                    trajectory = self._generate_target_trajectory(target_y, current_psi)
            else:
                trajectory = self._generate_target_trajectory(target_y, current_psi)
        
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
            
            # V4.0: Compute heading reference from derivative
            if progress < 0.15:
                ds_dprog = 6 * progress - 6 * progress**2
                dprog_dt = 1.0 / self.lane_change_duration
                y_dot_ref = (y_end - y_start) * ds_dprog * dprog_dt
                psi_ref = y_dot_ref / max(self._current_vx, 1.0)
            else:
                psi_ref = 0.0
            
            trajectory[i, 0] = y_start + (y_end - y_start) * s
            trajectory[i, 1] = psi_ref
        
        return trajectory
    
    def _generate_human_lane_change(self, target_y: float) -> np.ndarray:
        """
        Generate human's preferred lane change trajectory.
        
        V4.0: Now includes heading reference from trajectory derivative.
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
            
            # V4.0: Heading reference from derivative
            # ds/dτ = 6τ - 6τ², ẏ = Δy * ds/dτ / T_lc
            if tau < 1.0:
                ds_dtau = 6 * tau - 6 * tau**2
                y_dot_ref = (y_end - y_start) * ds_dtau / T_lc
                psi_ref = y_dot_ref / max(self._current_vx, 1.0)
            else:
                psi_ref = 0.0
            
            trajectory[i, 0] = y_start + (y_end - y_start) * s
            trajectory[i, 1] = psi_ref
        
        return trajectory
    
    def _generate_settling_trajectory(self, target_y: float,
                                       current_y: float,
                                       current_psi: float,
                                       current_y_dot: float = 0.0) -> np.ndarray:
        """
        V4.4: Generate soft settling trajectory for LANE_KEEPING entry.

        Receding-horizon cubic polynomial from current state to target.

        FIXES applied:
        ==============
        V4.3: Receding horizon — τ always starts at 0 for the current step,
              ensuring y_ref(i=0) = current_y always (no phantom reference jump).

        V4.4: Two additional fixes for the body-frame bicycle model (Eq. 2.31):

        1. CORRECT psi_ref SIGN:
           In the body-frame model with centripetal term a24 ≈ -vx, the
           steady-state relationship is ẏ_body ≈ -vx · ψ.  Therefore the
           heading needed to achieve a desired lateral rate is:
               ψ_ref = -ẏ_ref / vx    (note the MINUS sign)
           The previous formula  ψ_ref = +ẏ_ref / vx  had the opposite sign,
           causing the Nash solver to steer the vehicle AWAY from the target.

        2. CLIP INITIAL VELOCITY to prevent polynomial overshoot:
           For a cubic polynomial with non-zero endpoint velocity, large y_dot_0
           relative to T_remaining causes the trajectory to pass through the
           target and return, driving the vehicle past y=0.  Clip y_dot_0 so
           that the trajectory is monotone (no overshoot).
        """
        trajectory = np.zeros((self.Np, 2))

        t_in_phase = self._current_time - self._lane_keeping_entry_time
        # Remaining settling time — never less than one control step
        T_remaining = max(self._T_settle - t_in_phase, self.dt)

        y0 = current_y
        y_target = target_y
        delta_y = y_target - y0
        vx = max(self._current_vx, 1.0)

        # Clip initial velocity to prevent polynomial overshoot.
        # For a cubic with boundary conditions y(0)=y0, y(T)=y_target,
        # y'(0)=y_dot_0, y'(T)=0, the trajectory is monotone when
        #   |y_dot_0| <= 1.5 * |delta_y| / T_remaining
        if abs(delta_y) > 1e-6:
            y_dot_max = 1.5 * abs(delta_y) / T_remaining
            y_dot_0 = float(np.clip(current_y_dot, -y_dot_max, y_dot_max))
        else:
            y_dot_0 = 0.0

        # 3rd order polynomial coefficients (τ ∈ [0, 1], τ = t_pred / T_remaining)
        a0 = y0
        a1 = y_dot_0 * T_remaining
        a2 = 3 * delta_y - 2 * y_dot_0 * T_remaining
        a3 = -2 * delta_y + y_dot_0 * T_remaining

        for i in range(self.Np):
            t_pred = i * self.dt  # always starts at 0 (receding horizon)
            tau = min(t_pred / T_remaining, 1.0)

            if tau < 1.0:
                y_ref = a0 + a1 * tau + a2 * tau**2 + a3 * tau**3
                y_dot_ref = (a1 + 2 * a2 * tau + 3 * a3 * tau**2) / T_remaining
                # Body-frame sign convention: ψ ≈ ẏ_world / vx
                # Ẏ_world = vx·sin(ψ) ≈ vx·ψ  → ψ_ref = y_dot_ref / vx
                psi_ref = y_dot_ref / vx
            else:
                y_ref = y_target
                psi_ref = 0.0

            trajectory[i, 0] = y_ref
            trajectory[i, 1] = psi_ref

        return trajectory
    
    def _generate_target_trajectory(self, target_y: float, current_psi: float) -> np.ndarray:
        """Generate trajectory at target (y=0, psi=0)."""
        trajectory = np.zeros((self.Np, 2))
        for i in range(self.Np):
            trajectory[i, 0] = target_y
            trajectory[i, 1] = 0.0
        return trajectory
    
    def reset(self):
        """Reset generator state."""
        self._current_phase = HumanTrajectoryPhase.CRUISE
        self._previous_phase = HumanTrajectoryPhase.CRUISE
        self._lane_change_start_time = None
        self._lane_change_start_y = None
        self._current_time = 0.0
        self.lane_change_duration = self._base_lane_change_durations.get(self.driver_type, 12.0)
        
        # V4.0: Reset settling state
        self._lane_keeping_entry_time = None
        self._lane_keeping_entry_y = None
        self._lane_keeping_entry_psi = None
        self._lane_keeping_entry_y_dot = None


__all__ = ['HumanReferenceGenerator', 'HumanTrajectoryPhase', 'HumanDynamicParams']