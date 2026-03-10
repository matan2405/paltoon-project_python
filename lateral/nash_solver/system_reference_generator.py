"""
System Reference Generator for Lateral Control.
VERSION 4.0 - HEADING-CONSISTENT + SOFT LANE-KEEPING TRANSITION

CRITICAL FIX (V4.0):
====================
V3.0 had a catastrophic bug: when transitioning from LANE_CHANGE to LANE_KEEPING,
the reference trajectory jumped instantaneously from the planned trajectory to
(y=0, ψ=0). If the vehicle hadn't finished the lane change (e.g., cautious driver
with T_lc=9s but phase transition at t=10s), this created a massive heading error
(up to 15°), causing saturation and oscillations.

V4.0 introduces a SOFT TRANSITION mechanism:
- When entering LANE_KEEPING, capture the current state (y, ψ)
- Generate a settling trajectory from current state to target (y=0, ψ=0)
- Use a 3rd order polynomial with settling time T_settle
- After T_settle, switch to pure target reference

This ensures the reference trajectory is ALWAYS continuous with the vehicle state,
regardless of when the phase transition occurs.

HEADING CONSTRAINT (preserved from V3.0):
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
    """Dynamic parameters for state-dependent trajectory planning."""
    target_velocity: float = 20.0
    desired_gap: float = 35.0
    platoon_lane_y: float = 0.0
    max_heading_angle: float = np.radians(8)
    
    def compute_min_T_lc(self, delta_y: float, vx: float) -> float:
        """Compute minimum T_lc to satisfy heading constraint."""
        if vx < 1.0 or abs(delta_y) < 0.1:
            return 5.0
        max_y_dot = vx * np.tan(self.max_heading_angle)
        T_lc_min = 1.875 * abs(delta_y) / max_y_dot
        return max(T_lc_min, 3.0)


class SystemReferenceGenerator:
    """
    Generate reference trajectories for Nash Solver.
    
    VERSION 4.0 - HEADING-CONSISTENT + SOFT LANE-KEEPING TRANSITION
    
    Key improvement: Soft transition when entering LANE_KEEPING phase.
    Instead of jumping to (y=0, ψ=0), generates a settling trajectory
    from current state to target, preventing heading saturation.
    """
    
    def __init__(self, Np: int = NASH_NP, dt: float = NASH_CONTROL_DT, 
                 driver_type: str = 'normal'):
        self.Np = Np
        self.dt = dt
        self.driver_type = driver_type
        
        # Lane configuration
        self.lane_width = LANE_WIDTH
        self.target_lane_y = 0.0
        
        # System T_lc = Human T_lc × multiplier (aggressive gets 2.0× for smoother QP tracking)
        self._system_T_lc_multiplier = 2.0 if driver_type == 'aggressive' else 1.5
        self._human_base_T_lc = {
            'cautious': 6.0,
            'normal': 4.5,
            'aggressive': 3.0
        }
        human_T_lc = self._human_base_T_lc.get(driver_type, 4.5)
        self._base_T_lc = human_T_lc * self._system_T_lc_multiplier
        self.lane_change_duration = self._base_T_lc
        
        # Dynamic parameters
        self.dynamic_params = DynamicTrajectoryParams()
        
        # Phase tracking
        self._current_phase = TrajectoryPhase.CRUISE
        self._previous_phase = TrajectoryPhase.CRUISE
        self._lane_change_start_time = None
        self._lane_change_start_y = None
        self._current_time = 0.0
        
        # =====================================================================
        # V4.0 NEW: Soft lane-keeping transition
        # =====================================================================
        self._lane_keeping_entry_time = None
        self._lane_keeping_entry_y = None
        self._lane_keeping_entry_psi = None
        self._lane_keeping_entry_y_dot = None
        
        # Settling time for lane-keeping entry (adapts to driver type)
        self._settle_time = {
            'cautious': 5.0,    # Slow, gentle settling
            'normal': 3.0,      # Medium
            'aggressive': 2.0   # Quick settling
        }
        self._T_settle = self._settle_time.get(driver_type, 3.0)
        # =====================================================================
        
        # Velocity tracking
        self._current_vx = NOMINAL_VELOCITY
        
        print(f"🚀 System Reference Generator V4.0 (Heading-Consistent) Initialized")
        print(f"   Driver type: {driver_type}, Base T_lc={self.lane_change_duration:.1f}s")
        print(f"   (Human T_lc={human_T_lc:.1f}s × {self._system_T_lc_multiplier})")
        print(f"   Max heading constraint: {np.degrees(self.dynamic_params.max_heading_angle):.1f}°")
    
    def set_current_time(self, time: float):
        self._current_time = time
    
    def set_velocity(self, vx: float):
        self._current_vx = vx
    
    def set_driver_type(self, driver_type: str):
        self.driver_type = driver_type
        human_T_lc = self._human_base_T_lc.get(driver_type, 4.5)
        self._base_T_lc = human_T_lc * self._system_T_lc_multiplier
        self.lane_change_duration = self._base_T_lc
        self._T_settle = self._settle_time.get(driver_type, 3.0)
    
    def update_from_platoon_state(self, target_velocity: float, desired_gap: float, 
                                   platoon_lane_y: float = 0.0):
        self.dynamic_params.target_velocity = target_velocity
        self.dynamic_params.desired_gap = desired_gap
        self.dynamic_params.platoon_lane_y = platoon_lane_y
        
        velocity_factor = np.sqrt(target_velocity / 20.0)
        human_T_lc = self._human_base_T_lc.get(self.driver_type, 4.5)
        self._base_T_lc = human_T_lc * self._system_T_lc_multiplier * velocity_factor
    
    def update_phase_from_safety_field(self, safety_phase: str):
        """Sync phase with safety field — with V4.0 soft transition detection."""
        phase_map = {
            "CRUISE": TrajectoryPhase.CRUISE,
            "GAP_SEARCH": TrajectoryPhase.GAP_SEARCH,
            "LANE_CHANGE": TrajectoryPhase.LANE_CHANGE,
            "LANE_KEEPING": TrajectoryPhase.LANE_KEEPING,
            "FOLLOWING": TrajectoryPhase.FOLLOWING
        }
        new_phase = phase_map.get(safety_phase, TrajectoryPhase.CRUISE)
        
        if new_phase != self._current_phase:
            self._previous_phase = self._current_phase
            self._current_phase = new_phase
            
            if new_phase == TrajectoryPhase.LANE_CHANGE:
                self._lane_change_start_time = self._current_time
            
            # V4.0: Capture state at LANE_KEEPING entry for soft transition
            if new_phase in [TrajectoryPhase.LANE_KEEPING, TrajectoryPhase.FOLLOWING]:
                self._lane_keeping_entry_time = self._current_time
                # Note: actual y, psi will be set in generate_reference_trajectory
                # from the ego_vehicle state
    
    def generate_reference_trajectory(self, ego_vehicle, target_y: float = 0.0,
                                       obstacles: List[Dict] = None) -> np.ndarray:
        """
        Generate reference trajectory with heading constraint and soft transitions.
        
        V4.0: Soft transition when entering LANE_KEEPING or FOLLOWING.
        """
        trajectory = np.zeros((self.Np, 2))
        
        current_y = ego_vehicle.state.y
        current_psi = ego_vehicle.state.psi
        current_y_dot = ego_vehicle.state.y_dot
        self.target_lane_y = target_y
        self._current_vx = getattr(ego_vehicle, 'vx', 20.0)
        
        # Lock starting position when first entering GAP_SEARCH or LANE_CHANGE
        if self._lane_change_start_y is None and self._current_phase in [
            TrajectoryPhase.GAP_SEARCH, TrajectoryPhase.LANE_CHANGE
        ]:
            self._lane_change_start_y = current_y
            
            delta_y = abs(current_y - target_y)
            min_T_lc = self.dynamic_params.compute_min_T_lc(delta_y, self._current_vx)
            self.lane_change_duration = max(self._base_T_lc, min_T_lc, self.lane_change_duration)
            
            print(f"🚀 System: Locked start y={current_y:.2f}m, T_lc={self.lane_change_duration:.1f}s")
            print(f"   (min T_lc for ψ<{np.degrees(self.dynamic_params.max_heading_angle):.1f}°: {min_T_lc:.1f}s)")
        
        # V4.0: Capture entry state for soft transition
        if self._lane_keeping_entry_time is not None and self._lane_keeping_entry_y is None:
            if self._current_phase in [TrajectoryPhase.LANE_KEEPING, TrajectoryPhase.FOLLOWING]:
                self._lane_keeping_entry_y = current_y
                self._lane_keeping_entry_psi = current_psi
                self._lane_keeping_entry_y_dot = current_y_dot
        
        # Generate trajectory based on phase
        if self._current_phase == TrajectoryPhase.CRUISE:
            trajectory = self._generate_stay_trajectory(current_y)
        
        elif self._current_phase == TrajectoryPhase.GAP_SEARCH:
            trajectory = self._generate_transition_trajectory(target_y)
        
        elif self._current_phase == TrajectoryPhase.LANE_CHANGE:
            trajectory = self._generate_lane_change_trajectory(target_y)
        
        elif self._current_phase in [TrajectoryPhase.LANE_KEEPING, TrajectoryPhase.FOLLOWING]:
            # V4.0: Use soft transition if within settling period
            if self._lane_keeping_entry_time is not None and self._lane_keeping_entry_y is not None:
                t_in_phase = self._current_time - self._lane_keeping_entry_time
                if t_in_phase < self._T_settle:
                    trajectory = self._generate_settling_trajectory(
                        target_y, current_y, current_psi
                    )
                else:
                    trajectory = self._generate_target_trajectory(target_y)
            else:
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
            s = 10 * progress**3 - 15 * progress**4 + 6 * progress**5
            trajectory[i, 0] = y_start + (y_end - y_start) * s
            trajectory[i, 1] = 0.0
        
        return trajectory
    
    def _generate_lane_change_trajectory(self, target_y: float) -> np.ndarray:
        """
        Generate smooth lane change trajectory using 5th order polynomial.
        Heading-constrained: T_lc adjusted to ensure ψ < max_heading.
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
            
            # Also compute the heading reference from the derivative
            # ψ_ref = ẏ_ref / vx where ẏ = Δy * (30τ² - 60τ³ + 30τ⁴) / T_lc
            if tau < 1.0:
                ds_dtau = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
                y_dot_ref = (y_end - y_start) * ds_dtau / T_lc
                psi_ref = y_dot_ref / max(self._current_vx, 1.0)
            else:
                psi_ref = 0.0
            
            trajectory[i, 0] = y_start + (y_end - y_start) * s
            trajectory[i, 1] = psi_ref
        
        return trajectory
    
    def _generate_settling_trajectory(self, target_y: float,
                                       current_y: float,
                                       current_psi: float) -> np.ndarray:
        """
        V4.0 NEW: Generate soft settling trajectory for LANE_KEEPING entry.
        
        Uses a 3rd order polynomial to smoothly bring the vehicle from its
        current state (y, ψ) to the target (target_y, 0).
        
        This prevents the reference from jumping instantaneously,
        which would create a massive heading error and cause saturation.
        
        The settling polynomial satisfies:
          y(0) = current_y,  y(T) = target_y
          ẏ(0) = vx * sin(current_psi) ≈ vx * current_psi,  ẏ(T) = 0
        
        From which: ψ_ref(t) = ẏ(t) / vx
        """
        trajectory = np.zeros((self.Np, 2))
        
        T_settle = self._T_settle
        t_in_phase = self._current_time - self._lane_keeping_entry_time
        
        # Boundary conditions
        y0 = current_y
        y_target = target_y
        y_dot_0 = self._current_vx * np.sin(current_psi)  # Current lateral velocity
        y_dot_T = 0.0  # Want zero lateral velocity at end
        
        # 3rd order polynomial: y(τ) = a0 + a1*τ + a2*τ² + a3*τ³
        # where τ = (t - t_entry) / T_settle ∈ [0, 1]
        # y(0) = y0,  y(1) = y_target
        # ẏ(0) = y_dot_0 * T_settle,  ẏ(1) = 0
        
        a0 = y0
        a1 = y_dot_0 * T_settle
        a2 = 3 * (y_target - y0) - 2 * y_dot_0 * T_settle
        a3 = -2 * (y_target - y0) + y_dot_0 * T_settle
        
        for i in range(self.Np):
            t_pred = t_in_phase + i * self.dt
            tau = min(t_pred / T_settle, 1.0)
            
            if tau < 1.0:
                # Position
                y_ref = a0 + a1 * tau + a2 * tau**2 + a3 * tau**3
                # Velocity (derivative)
                y_dot_ref = (a1 + 2 * a2 * tau + 3 * a3 * tau**2) / T_settle
                # Heading reference
                psi_ref = np.arctan2(y_dot_ref, self._current_vx)
            else:
                y_ref = y_target
                psi_ref = 0.0
            
            trajectory[i, 0] = y_ref
            trajectory[i, 1] = psi_ref
        
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
        self._previous_phase = TrajectoryPhase.CRUISE
        self._lane_change_start_time = None
        self._lane_change_start_y = None
        self._current_time = 0.0
        self.lane_change_duration = self._base_T_lc
        
        # V4.0: Reset settling state
        self._lane_keeping_entry_time = None
        self._lane_keeping_entry_y = None
        self._lane_keeping_entry_psi = None
        self._lane_keeping_entry_y_dot = None


__all__ = ['SystemReferenceGenerator', 'TrajectoryPhase', 'DynamicTrajectoryParams']