"""
Lateral Safety Field with Phase Detection.
VERSION 2.1 - INSPIRED BY WORKING LONGITUDINAL ARCHITECTURE

KEY PRINCIPLES (from longitudinal):
1. Safety Field provides REPULSIVE force only when vehicle is TOO CLOSE
2. Force = 0 when gap > safe_distance (no force when safe)
3. Phase-aware soft transition (MERGING → LANE_KEEPING → FOLLOWING)
4. Quadratic soft transition in FOLLOWING phase

KEY DIFFERENCE from longitudinal:
- Longitudinal: gap_error = gap_actual - desired_gap (negative = danger)
- Lateral: y_error = ego_y - target_y (approaching target is GOOD, not danger!)

SOLUTION: Safety force only from ACTUAL proximity to other vehicles,
NOT from approaching the target lane!
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from enum import Enum

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (LANE_WIDTH, FOLLOWING_Y_ERROR_FACTOR, FOLLOWING_PSI_ERROR_THRESHOLD,
                    FOLLOWING_Y_DOT_THRESHOLD, MERGING_Y_ERROR_FACTOR,
                    MERGING_PSI_ERROR_THRESHOLD, PHASE_TRANSITION_TIME,
                    GAP_SEARCH_DURATION, LANE_CHANGE_MIN_TIME, LANE_CHANGE_Y_ERROR_FACTOR)


class ControlPhase(Enum):
    """Control phases - matching longitudinal controller."""
    CRUISE = "CRUISE"
    GAP_SEARCH = "GAP_SEARCH"
    LANE_CHANGE = "LANE_CHANGE"
    LANE_KEEPING = "LANE_KEEPING"
    FOLLOWING = "FOLLOWING"


@dataclass
class LateralSafetyFieldParams:
    """
    Safety Field Parameters - V2.1
    Inspired by longitudinal but adapted for lateral dynamics.
    """
    # Lane Configuration
    lane_width: float = LANE_WIDTH
    target_lane_y: float = 0.0
    
    # === COLLISION DETECTION THRESHOLDS ===
    # Only trigger force when ACTUALLY close to collision
    collision_lateral_threshold: float = 2.0    # meters - lateral distance for "danger"
    collision_longitudinal_threshold: float = 10.0  # meters - longitudinal overlap
    safe_lateral_distance: float = 3.0          # meters - no force if > this

    # Force Parameters - MODERATE (safety, not tracking!)
    obstacle_force_gain: float = 150.0
    obstacle_force_scale: float = 5.0     # Scale for tanh

    # Boundary Parameters
    road_half_width: float = 7.0
    boundary_force_gain: float = 150.0
    boundary_force_scale: float = 2.0
    
    # Distance Decay
    distance_decay_factor: float = 20.0
    epsilon: float = 0.1
    
    # Force Limits
    max_force: float = 500.0
    
    # === PHASE DETECTION THRESHOLDS (from longitudinal) ===
    following_y_error_factor: float = FOLLOWING_Y_ERROR_FACTOR      # 15%
    following_psi_threshold: float = FOLLOWING_PSI_ERROR_THRESHOLD  # ~3 deg
    following_y_dot_threshold: float = FOLLOWING_Y_DOT_THRESHOLD    # 0.3 m/s

    merging_y_error_factor: float = MERGING_Y_ERROR_FACTOR          # 25%
    merging_psi_threshold: float = MERGING_PSI_ERROR_THRESHOLD      # ~6 deg

    phase_transition_time: float = PHASE_TRANSITION_TIME            # 5 seconds

    # === PHASE DURATION THRESHOLDS ===
    gap_search_duration: float = GAP_SEARCH_DURATION                # 0.5 seconds
    lane_change_min_time: float = LANE_CHANGE_MIN_TIME              # 3.0 seconds
    lane_change_y_error_factor: float = LANE_CHANGE_Y_ERROR_FACTOR  # 30%


class LateralSafetyField:
    """
    Lateral Safety Field - VERSION 2.1
    
    INSPIRED BY LONGITUDINAL:
    - Force = 0 when no collision risk (vehicle is safe)
    - Force > 0 only when ACTUALLY close to another vehicle
    - Soft transition in FOLLOWING phase
    
    KEY INSIGHT: Safety field should NOT interfere with Nash tracking!
    It only prevents actual collisions.
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
        self.filter_alpha = 0.3
        
        print("🛡️ Lateral Safety Field V2.1 Initialized")
        print("   ✓ Collision-based force (not proximity-based)")
        print("   ✓ Phase-aware soft transition")
        print("   ✓ Force = 0 when safe (no interference with Nash)")
    
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
            # Transition to LANE_KEEPING when close to target
            if abs(y_error) < p.lane_width * p.lane_change_y_error_factor and time_in_phase > p.lane_change_min_time:
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
    # MAIN FORCE COMPUTATION - V2.1 (COLLISION-BASED)
    # ========================================================================
    def compute_risk_force(self, ego_vehicle, obstacles: List[Dict], 
                           target_y: float = 0.0) -> float:
        """
        Compute total lateral risk force.
        
        VERSION 2.1 - COLLISION-BASED APPROACH:
        - Force = 0 when no collision risk (vehicle is safe)
        - Force > 0 only when ACTUALLY close to another vehicle
        - This allows Nash to control the trajectory without interference
        """
        # Get ego state
        ego_y = ego_vehicle.state.y
        ego_x = ego_vehicle.state.x
        ego_vx = ego_vehicle.vx
        ego_psi = ego_vehicle.state.psi
        ego_y_dot = ego_vehicle.state.y_dot
        
        y_error = ego_y - target_y
        
        # Update phase
        self._update_phase(y_error, ego_psi, ego_y_dot)
        
        # Compute collision-based obstacle force
        obstacle_force, min_lateral_dist = self._compute_collision_force(
            ego_x, ego_y, ego_vx, ego_y_dot, obstacles
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
    
    def _compute_collision_force(self, ego_x: float, ego_y: float, 
                                  ego_vx: float, ego_y_dot: float,
                                  obstacles: List[Dict]) -> Tuple[float, float]:
        """
        Compute force based on ACTUAL collision risk.
        
        KEY PRINCIPLE (from longitudinal):
        - If lateral distance > safe_distance → Force = 0 (SAFE!)
        - If lateral distance < collision_threshold → Force > 0 (DANGER!)
        
        This is the key difference from V2.0:
        - V2.0: Force always present when near obstacles
        - V2.1: Force = 0 when safe, regardless of proximity
        """
        p = self.params
        total_force = 0.0
        min_lateral_distance = float('inf')
        
        for obs in obstacles:
            obs_x = obs.get('x', 0.0)
            obs_y = obs.get('y', 0.0)
            obs_vx = obs.get('vx', ego_vx)
            
            # Relative positions
            dx = ego_x - obs_x  # Positive = ego ahead
            dy = ego_y - obs_y  # Positive = ego to the right
            
            abs_dy = abs(dy)
            abs_dx = abs(dx)
            
            min_lateral_distance = min(min_lateral_distance, abs_dy)
            
            # ================================================================
            # COLLISION-BASED FORCE LOGIC (like longitudinal gap_error logic)
            # ================================================================
            # If lateral distance > safe_distance → NO FORCE (we're safe!)
            if abs_dy > p.safe_lateral_distance:
                continue  # No force from this obstacle
            
            # If not longitudinally overlapping → reduced concern
            if abs_dx > p.collision_longitudinal_threshold * 2:
                continue  # Too far longitudinally
            
            # Calculate "lateral gap error" (like longitudinal gap_error)
            # Negative = danger (too close), Positive = safe
            lateral_gap_error = abs_dy - p.collision_lateral_threshold
            
            if lateral_gap_error > 0:
                # Still safe (lateral distance > threshold) → minimal force
                # Use exponential decay like longitudinal
                decay = np.exp(-abs_dy / p.distance_decay_factor)
                force_magnitude = p.obstacle_force_gain * 0.1 * decay
            else:
                # DANGER! Too close laterally
                # Force increases as we get closer (inverse relationship)
                danger_level = abs(lateral_gap_error) / p.collision_lateral_threshold
                danger_level = min(danger_level, 2.0)  # Cap at 2x
                
                # Longitudinal overlap factor
                if abs_dx < p.collision_longitudinal_threshold:
                    # Direct overlap - maximum danger
                    overlap_factor = 1.5
                else:
                    # Partial overlap
                    overlap_factor = 1.0 - (abs_dx - p.collision_longitudinal_threshold) / p.collision_longitudinal_threshold
                    overlap_factor = max(overlap_factor, 0.3)
                
                # Calculate force using tanh (smooth saturation)
                raw_force = danger_level * overlap_factor * p.obstacle_force_gain
                force_magnitude = p.obstacle_force_gain * np.tanh(raw_force / p.obstacle_force_scale)
            
            # Direction: push away from obstacle (positive dy = push more positive)
            if abs(dy) > 0.01:
                lateral_direction = np.sign(dy)
            else:
                lateral_direction = 1.0 if ego_y > obs_y else -1.0
            
            total_force += force_magnitude * lateral_direction
        
        return total_force, min_lateral_distance
    
    def _compute_boundary_force(self, ego_y: float) -> float:
        """Compute repulsive force from road boundaries."""
        p = self.params
        force = 0.0
        
        # Right boundary (positive y)
        dist_to_right = p.road_half_width - ego_y
        if dist_to_right < 3.0 and dist_to_right > 0.1:
            potential = 1.0 / dist_to_right
            force -= p.boundary_force_gain * np.tanh(potential / p.boundary_force_scale)
        
        # Left boundary (negative y)
        dist_to_left = p.road_half_width + ego_y
        if dist_to_left < 3.0 and dist_to_left > 0.1:
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
