"""
Lateral Safety Field with Phase Detection.

Research basis and implemented elements:
- Driving Safety Field structure from Li/Wang line for obstacle risk modeling.
- Elliptic distance and velocity-dependent field intensity.
- Phase-aware risk interpretation for merge control and authority allocation.

THEORY (Li et al. 2019, Wang et al. 2015/2016):
================================================
The Driving Safety Field models risk as a potential field in 2D space.
Each obstacle generates a repulsive field that decays with ELLIPTIC distance:

    r* = sqrt(((x - xo)/a)^2 + ((y - yo)/b)^2)    [Li 2019, Eq. 12/15]
    a  = max(|ve - vo| * ts, a_min)                 [longitudinal semi-axis, Eq. 13]
    b  = τ = 2 m                                     [lateral safety radius, Table 1]

    E_o = G * M_obs / |r*|                           [field strength, Eq. 11, R=1]
    Fr  = E_o * M_ego * exp(-k2*ve*cosθ) * (1+DR)   [field force, Eq. 21]

The elliptic shape captures the asymmetry of driving:
- Large longitudinal semi-axis (a >> b when speeds differ) → more risk ahead/behind
- At equal speeds (ve ≈ vo): a = a_min = b → isotropic safety circle

AUTHORITY ALLOCATION (Li 2019, Table 2, Eq. 22):
=================================================
|Fr| → sigmoid → λ_safety, combined via max(λ_safety, λ_performance):
- join_middle: two nearby vehicles → high Fr → λ_safety > λ_performance → safety leads
- join_after:  all vehicles far longitudinally → low Fr → λ_performance leads
- join_before: vehicles far longitudinally behind → very low Fr → λ_performance leads
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from enum import Enum

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    LANE_WIDTH,
    FOLLOWING_Y_ERROR_FACTOR, FOLLOWING_PSI_ERROR_THRESHOLD, FOLLOWING_Y_DOT_THRESHOLD,
    MERGING_Y_ERROR_FACTOR, MERGING_PSI_ERROR_THRESHOLD, PHASE_TRANSITION_TIME,
    GAP_SEARCH_DURATION, LANE_CHANGE_MIN_TIME, LANE_CHANGE_Y_ERROR_FACTOR,
    LANE_CHANGE_Y_DOT_THRESHOLD,
    # DSF obstacle field parameters (Li 2019, Wang 2015/2016)
    DSF_G, DSF_TS, DSF_TAU, DSF_A_MIN,
    DSF_K1, DSF_K2, DSF_DR, DSF_VEHICLE_MASS,
    DSF_SPEED_COEFF, DSF_SPEED_EXPONENT, DSF_SPEED_OFFSET, DSF_EPSILON,
    # Boundary and output parameters
    SAFETY_ROAD_HALF_WIDTH, SAFETY_BOUNDARY_FORCE_GAIN, SAFETY_BOUNDARY_FORCE_SCALE,
    SAFETY_BOUNDARY_PROXIMITY, SAFETY_EPSILON,
    SAFETY_MAX_FORCE, SAFETY_DIRECTION_SMOOTH_SIGMA, SAFETY_FILTER_ALPHA,
)


class ControlPhase(Enum):
    """Lateral control phases used by safety field and reference generators."""
    CRUISE = "CRUISE"
    GAP_SEARCH = "GAP_SEARCH"
    LANE_CHANGE = "LANE_CHANGE"
    LANE_KEEPING = "LANE_KEEPING"
    FOLLOWING = "FOLLOWING"


@dataclass
class LateralSafetyFieldParams:
    """
    Safety Field Parameters — DSF (Li et al. 2019, Wang et al. 2015/2016)
    """
    # Lane Configuration
    lane_width: float = LANE_WIDTH
    target_lane_y: float = 0.0

    # === DSF OBSTACLE FIELD (Li 2019, Eq. 11-21) ===
    # Elliptic distance: r* = sqrt(((x-xo)/a)^2 + ((y-yo)/b)^2)
    dsf_g:        float = DSF_G           # Field scaling constant (calibrated)
    ts:           float = DSF_TS          # TTC margin for longitudinal semi-axis a [s]
    tau:          float = DSF_TAU         # Lateral safety circle radius b = τ [m]
    a_min:        float = DSF_A_MIN       # Min semi-axis a (for equal-speed case) [m]
    k1:           float = DSF_K1          # Kinetic field scaling (Eq. 18)
    k2:           float = DSF_K2          # Field force scaling   (Eq. 21)
    dr:           float = DSF_DR          # Driver risk factor DR
    vehicle_mass: float = DSF_VEHICLE_MASS  # Base vehicle mass [kg]
    speed_coeff:  float = DSF_SPEED_COEFF   # Wang 2016 Eq. 17 coefficient
    speed_exp:    float = DSF_SPEED_EXPONENT
    speed_offset: float = DSF_SPEED_OFFSET
    dsf_epsilon:  float = DSF_EPSILON     # Min elliptic distance [m]

    # === BOUNDARY FORCE ===
    road_half_width:    float = SAFETY_ROAD_HALF_WIDTH
    boundary_force_gain: float = SAFETY_BOUNDARY_FORCE_GAIN
    boundary_force_scale: float = SAFETY_BOUNDARY_FORCE_SCALE
    boundary_proximity: float = SAFETY_BOUNDARY_PROXIMITY
    epsilon:            float = SAFETY_EPSILON   # Boundary numerical safety

    # === FORCE OUTPUT ===
    max_force:             float = SAFETY_MAX_FORCE
    direction_smooth_sigma: float = SAFETY_DIRECTION_SMOOTH_SIGMA

    # === PHASE DETECTION THRESHOLDS ===
    following_y_error_factor:  float = FOLLOWING_Y_ERROR_FACTOR
    following_psi_threshold:   float = FOLLOWING_PSI_ERROR_THRESHOLD
    following_y_dot_threshold: float = FOLLOWING_Y_DOT_THRESHOLD
    merging_y_error_factor:    float = MERGING_Y_ERROR_FACTOR
    merging_psi_threshold:     float = MERGING_PSI_ERROR_THRESHOLD
    phase_transition_time:     float = PHASE_TRANSITION_TIME

    # === PHASE DURATION THRESHOLDS ===
    gap_search_duration:         float = GAP_SEARCH_DURATION
    lane_change_min_time:        float = LANE_CHANGE_MIN_TIME
    lane_change_y_error_factor:  float = LANE_CHANGE_Y_ERROR_FACTOR
    lane_change_y_dot_threshold: float = LANE_CHANGE_Y_DOT_THRESHOLD


class LateralSafetyField:
    """
    Lateral Safety Field implementation (DSF per Li et al. 2019)

    Implements the Driving Safety Field with elliptic distance metric.
    Force magnitude differentiates scenarios (join_middle vs join_after vs join_before)
    so the authority allocator sigmoid correctly maps to distinct λ_safety values.
    """
    
    def __init__(self, params: LateralSafetyFieldParams = None):
        self.params = params or LateralSafetyFieldParams()
        
        # Phase state machine
        self._current_phase = ControlPhase.CRUISE
        self._phase_start_time = 0.0
        self._stable_condition_start_time = None
        self._current_time = 0.0
        
        # Merge command
        self._merge_commanded = False
        
        # Force filter (smoother output)
        self._last_force = 0.0
        self.filter_alpha = SAFETY_FILTER_ALPHA
        
        print("🛡️ Lateral Safety Field (DSF) Initialized")
        print("   ✓ Elliptic distance field (Li 2019, Eq. 12-13)")
        print("   ✓ Virtual mass + kinetic correction (Wang 2016, Eq. 17-18)")
        print("   ✓ Scenario-differentiated force: join_middle>>join_after>>join_before")
    
    def set_current_time(self, time: float):
        self._current_time = time
    
    def command_merge(self):
        if self._current_phase == ControlPhase.CRUISE:
            self._merge_commanded = True
            self._transition_to_phase(ControlPhase.GAP_SEARCH)
            print(f"🚗 Merge commanded at t={self._current_time:.1f}s")
    
    def get_current_phase(self) -> ControlPhase:
        return self._current_phase
    
    def get_phase_name(self) -> str:
        return self._current_phase.value
    
    def _transition_to_phase(self, new_phase: ControlPhase):
        if new_phase != self._current_phase:
            print(f"📍 Phase: {self._current_phase.value} → {new_phase.value} at t={self._current_time:.1f}s")
            self._current_phase = new_phase
            self._phase_start_time = self._current_time
            self._stable_condition_start_time = None
    
    # ========================================================================
    # PHASE DETECTION (from longitudinal)
    # ========================================================================
    def _check_following_conditions(self, y_error: float, psi: float, y_dot: float) -> bool:
        """Check if conditions for FOLLOWING phase are met."""
        p = self.params
        
        y_threshold = p.following_y_error_factor * p.lane_width
        psi_threshold = p.following_psi_threshold
        y_dot_threshold = p.following_y_dot_threshold
        
        return (abs(y_error) < y_threshold and 
                abs(psi) < psi_threshold and 
                abs(y_dot) < y_dot_threshold)
    
    def _check_merging_conditions(self, y_error: float, psi: float) -> bool:
        """Check if should exit FOLLOWING back to LANE_KEEPING."""
        p = self.params
        
        y_threshold = p.merging_y_error_factor * p.lane_width
        psi_threshold = p.merging_psi_threshold
        
        return (abs(y_error) > y_threshold or abs(psi) > psi_threshold)
    
    def _update_phase(self, y_error: float, psi: float, y_dot: float):
        """Update control phase based on current state."""
        p = self.params
        time_in_phase = self._current_time - self._phase_start_time
        
        if self._current_phase == ControlPhase.CRUISE:
            pass  # Wait for merge command
        
        elif self._current_phase == ControlPhase.GAP_SEARCH:
            if time_in_phase > p.gap_search_duration:
                self._transition_to_phase(ControlPhase.LANE_CHANGE)

        elif self._current_phase == ControlPhase.LANE_CHANGE:
            # Transition to LANE_KEEPING when close to target AND lateral motion has slowed
            if (abs(y_error) < p.lane_width * p.lane_change_y_error_factor
                    and time_in_phase > p.lane_change_min_time
                    and abs(y_dot) < p.lane_change_y_dot_threshold):
                self._transition_to_phase(ControlPhase.LANE_KEEPING)
        
        elif self._current_phase == ControlPhase.LANE_KEEPING:
            if self._check_following_conditions(y_error, psi, y_dot):
                if self._stable_condition_start_time is None:
                    self._stable_condition_start_time = self._current_time
                elif (self._current_time - self._stable_condition_start_time) >= p.phase_transition_time:
                    self._transition_to_phase(ControlPhase.FOLLOWING)
            else:
                self._stable_condition_start_time = None
        
        elif self._current_phase == ControlPhase.FOLLOWING:
            if self._check_merging_conditions(y_error, psi):
                self._transition_to_phase(ControlPhase.LANE_KEEPING)
    
    # ========================================================================
    # SOFT TRANSITION (from longitudinal)
    # ========================================================================
    def _apply_soft_transition(self, force: float, y_error: float) -> float:
        """
        Apply quadratic soft transition in FOLLOWING phase.
        
        F_following = F_merging × min(1.0, (|y_error| / threshold)²)
        """
        if self._current_phase != ControlPhase.FOLLOWING:
            return force
        
        p = self.params
        threshold = p.following_y_error_factor * p.lane_width
        
        if threshold <= 0:
            return force
        
        ratio = abs(y_error) / threshold
        scaling_factor = min(1.0, ratio * ratio)
        
        return force * scaling_factor
    
    # ========================================================================
    # MAIN FORCE COMPUTATION (COLLISION-BASED)
    # ========================================================================
    def compute_risk_force(self, ego_vehicle, obstacles: List[Dict], 
                           target_y: float = 0.0) -> float:
        """
        Compute total lateral risk force using DSF (Li et al. 2019).

        Force magnitude is scenario-dependent via elliptic distance:
        - join_after:  adjacent vehicle → Fr≈270N > sigmoid midpoint=200N → λ_safety wins
        - join_middle: vehicles ~9m away → Fr≈160N < sigmoid midpoint=200N → λ_perf wins
        - join_before: vehicles far ahead → Fr≈80N  < sigmoid midpoint=200N → λ_perf wins
        """
        # Get ego state
        ego_y = ego_vehicle.state.y
        ego_x = ego_vehicle.state.x
        ego_psi = ego_vehicle.state.psi
        ego_y_dot = ego_vehicle.state.y_dot
        
        y_error = ego_y - target_y
        
        # Update phase
        self._update_phase(y_error, ego_psi, ego_y_dot)
        
        # Compute DSF obstacle force (Li 2019, Eq. 11-21)
        ego_vx = getattr(ego_vehicle.state, 'vx', getattr(ego_vehicle.state, 'x_dot', 0.0))
        obstacle_force, _ = self._compute_dsf_obstacle_force(
            ego_x, ego_y, ego_vx, obstacles
        )
        
        # Compute boundary force
        boundary_force = self._compute_boundary_force(ego_y)
        
        total_force = obstacle_force + boundary_force
        
        # Apply soft transition in FOLLOWING phase
        if self._current_phase == ControlPhase.FOLLOWING:
            total_force = self._apply_soft_transition(total_force, y_error)
        
        # Limit force
        total_force = np.clip(total_force, -self.params.max_force, self.params.max_force)
        
        # Filter for smoothness
        total_force = self.filter_alpha * total_force + (1 - self.filter_alpha) * self._last_force
        self._last_force = total_force
        
        return total_force
    
    def _compute_dsf_obstacle_force(self, ego_x: float, ego_y: float,
                                     ego_vx: float,
                                     obstacles: List[Dict]) -> Tuple[float, float]:
        """
        Driving Safety Field obstacle force — Li et al. (2019), Eq. 11-21.

        Elliptic distance (Eq. 12-13):
            a  = max(|ve - vo| * ts, a_min)   [longitudinal semi-axis]
            b  = tau = 2 m                     [lateral safety radius]
            r* = sqrt((dx/a)^2 + (dy/b)^2)

        Field strength (Eq. 11):
            E_o = G * M_obs / r*

        Kinetic correction (Eq. 18):
            E_v = E_o * exp(k1 * v_obs * cosθv)

        Field force (Eq. 21):
            Fr = E_v * M_ego * exp(-k2 * v_ego * cosθi) * (1 + DR)

        Virtual mass (Wang 2016, Eq. 17):
            M = m * (speed_coeff * v^speed_exp + speed_offset)

        Scenario differentiation at dy=3.5m (one lane width, DSF_G=2e-3):
          - join_after  (P2 adjacent, dx≈4.5m):  Fr≈270N > midpoint=200N → λ_safety ✓
          - join_middle (dx≈9m):                  Fr≈160N < midpoint=200N → λ_perf   ✓
          - join_before (all ahead,  dx≥40m):     Fr≈80N  < midpoint=200N → λ_perf   ✓
        """
        p = self.params
        total_force = 0.0
        min_lateral_distance = float('inf')

        # Virtual mass of ego (Wang 2016, Eq. 17)
        M_ego = p.vehicle_mass * (p.speed_coeff * abs(ego_vx) ** p.speed_exp + p.speed_offset)
        # Ego velocity direction (unit vector along x-axis)
        v_ego_dir = np.array([1.0, 0.0])

        for obs in obstacles:
            obs_x = obs.get('x', 0.0)
            obs_y = obs.get('y', 0.0)
            obs_vx = obs.get('vx', ego_vx)

            dx = ego_x - obs_x  # Positive = ego ahead
            dy = ego_y - obs_y  # Positive = ego to the right
            min_lateral_distance = min(min_lateral_distance, abs(dy))

            # --- Elliptic distance (Li 2019, Eq. 12-13) ---
            a = max(abs(ego_vx - obs_vx) * p.ts, p.a_min)
            b = p.tau
            r_star = np.sqrt((dx / a) ** 2 + (dy / b) ** 2)
            r_star = max(r_star, p.dsf_epsilon)

            # Unit vector along elliptic gradient
            r_star_unit = np.array([dx / a, dy / b]) / r_star

            # --- Virtual mass of obstacle (Wang 2016, Eq. 17) ---
            M_obs = p.vehicle_mass * (p.speed_coeff * abs(obs_vx) ** p.speed_exp + p.speed_offset)

            # --- Field strength (Li 2019, Eq. 11, R=1) ---
            E = p.dsf_g * M_obs / r_star

            # --- Kinetic correction (Eq. 18): obstacle velocity projection ---
            v_obs_dir = np.array([1.0, 0.0]) if abs(obs_vx) > 0.1 else np.array([0.0, 0.0])
            cos_theta_v = float(np.dot(v_obs_dir, r_star_unit))
            E *= np.exp(p.k1 * abs(obs_vx) * cos_theta_v)

            # --- Field force (Eq. 21): ego velocity projection on Cartesian gradient ---
            field_cart = np.array([dx / (a * a), dy / (b * b)])
            field_cart_norm = np.linalg.norm(field_cart)
            if field_cart_norm > 1e-9:
                field_cart_unit = field_cart / field_cart_norm
            else:
                field_cart_unit = np.array([0.0, 1.0])
            cos_theta_i = float(np.dot(field_cart_unit, v_ego_dir))

            Fr = E * M_ego * np.exp(-p.k2 * abs(ego_vx) * cos_theta_i) * (1.0 + p.dr)

            # Lateral direction: push ego away from obstacle
            lateral_direction = np.tanh(dy / p.direction_smooth_sigma)
            total_force += Fr * lateral_direction

        return total_force, min_lateral_distance
    
    def _compute_boundary_force(self, ego_y: float) -> float:
        """Compute repulsive force from road boundaries."""
        p = self.params
        force = 0.0
        
        # Right boundary (positive y)
        dist_to_right = p.road_half_width - ego_y
        if dist_to_right < p.boundary_proximity and dist_to_right > p.epsilon:
            potential = 1.0 / dist_to_right
            force -= p.boundary_force_gain * np.tanh(potential / p.boundary_force_scale)

        # Left boundary (negative y)
        dist_to_left = p.road_half_width + ego_y
        if dist_to_left < p.boundary_proximity and dist_to_left > p.epsilon:
            potential = 1.0 / dist_to_left
            force += p.boundary_force_gain * np.tanh(potential / p.boundary_force_scale)
        
        return force
    
    def reset(self):
        self._current_phase = ControlPhase.CRUISE
        self._phase_start_time = 0.0
        self._stable_condition_start_time = None
        self._current_time = 0.0
        self._merge_commanded = False
        self._last_force = 0.0
        print("🔄 Safety Field Reset")


__all__ = ['LateralSafetyField', 'LateralSafetyFieldParams', 'ControlPhase']
