#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: longitudinal_safety_field.py
Description: Pure Repulsive Safety Field with Dynamic Phase Detection.

VERSION 3.0 - MERGING/FOLLOWING PHASE SYSTEM
=============================================
Key improvements:
1. Dynamic phase detection based on simulation state (not timers)
2. MERGING phase: Aggressive safety field for convergence
3. FOLLOWING phase: Soft transition for steady-state comfort
4. Hysteresis to prevent oscillation between phases

The phase transition is based on:
- Gap error relative to desired_gap
- Relative velocity relative to target_velocity
- Acceleration magnitude
- All conditions must hold for 5 seconds to transition to FOLLOWING
"""

import numpy as np
from dataclasses import dataclass
import os
import sys
from typing import Optional, Dict, TYPE_CHECKING

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vehicle.vehicle import Vehicle
from control.platoon_control import PlatoonManager


# ============================================================================
# PHASE ENUMERATION
# ============================================================================
class ControlPhase:
    """Control phase constants"""
    MERGING = "MERGING"
    FOLLOWING = "FOLLOWING"


# ============================================================================
# DATA CLASSES
# ============================================================================
@dataclass
class PlatoonContext:
    """Complete context for dynamic parameter computation"""
    has_leader: bool
    has_follower: bool
    ego_index: int
    platoon_size: int
    gap_to_leader: float = float('inf')
    gap_to_follower: float = float('inf')
    rel_vel_leader: float = 0.0
    rel_vel_follower: float = 0.0
    ttc_leader: float = float('inf')
    ttc_follower: float = float('inf')
    ego_velocity: float = 0.0
    velocity_error: float = 0.0
    high_risk_situation: bool = False
    emergency_brake_active: bool = False
    follower_is_joining: bool = False
    follower_decelerating: bool = False
    road_condition: str = "dry"
    visibility: float = 1.0
    target_velocity: float = 120.0
    # NEW: Phase information
    current_phase: str = ControlPhase.MERGING


@dataclass
class EllipseLongitudinalParams:
    """Parameters for pure repulsive safety field with phase detection"""
    
    # === Hard Constraints ===
    min_safe_distance: float = 5.0
    emergency_brake_distance: float = 8.0
    max_decel: float = 3.5
    
    # === Base Parameters ===
    base_safety_radius: float = 10.0
    base_obstacle_mass: float = 400.0
    base_influence_factor: float = 1.0
    base_driver_risk: float = 0.5
    epsilon: float = 0.1
    
    # === Position Multipliers ===
    leader_position_multiplier: float = 0.8
    middle_position_multiplier: float = 1.2
    follower_position_multiplier: float = 1.0
    
    # === Risk Multipliers ===
    normal_risk_multiplier: float = 1.0
    moderate_risk_multiplier: float = 1.5
    high_risk_multiplier: float = 2.0
    emergency_risk_multiplier: float = 2.5
    
    # === Velocity Scaling ===
    velocity_reference: float = 120.0
    velocity_scaling_factor: float = 0.8
    
    # === Distance Decay ===
    distance_decay_factor: float = 15.0
    
    # === Force Limits ===
    max_repulsive_force: float = 800.0
    
    # === Follower Weights ===
    leader_weight: float = 1.0
    follower_weight: float = 0.5
    joining_follower_weight: float = 0.2
    decelerating_follower_weight: float = 0.4
    
    # === Platoon Factors ===
    platoon_coherence_factor: float = 1.5
    merging_multiplier: float = 1.2
    
    # === NEW: Phase Detection Parameters (relative to platoon) ===
    # Entry to FOLLOWING thresholds (must ALL be satisfied)
    following_gap_error_factor: float = 0.15      # 15% of desired_gap
    following_velocity_error_factor: float = 0.05  # 5% of target_velocity
    following_acceleration_threshold: float = 0.5  # m/s² absolute
    
    # Exit from FOLLOWING thresholds (ANY triggers exit)
    merging_gap_error_factor: float = 0.25        # 25% of desired_gap (hysteresis)
    merging_velocity_error_factor: float = 0.10   # 10% of target_velocity (hysteresis)
    
    # Time to confirm phase transition
    phase_transition_time: float = 5.0  # seconds
    
    # === Road Conditions ===
    road_condition_factors: Dict[str, float] = None
    
    def __post_init__(self):
        if self.road_condition_factors is None:
            self.road_condition_factors = {
                "dry": 1.0,
                "wet": 0.7,
                "icy": 0.5
            }


# ============================================================================
# MAIN CLASS
# ============================================================================
class EllipseLongitudinalSafetyField:
    """
    Ellipse-based longitudinal safety field with dynamic phase detection.
    
    Two operating modes:
    - MERGING: Aggressive response for convergence (full force)
    - FOLLOWING: Soft response for steady-state (reduced force for small errors)
    
    Phase transitions are based on actual system state, not timers.
    """
    
    def __init__(self, params: EllipseLongitudinalParams = None):
        self.params = params or EllipseLongitudinalParams()
        self._last_context = None
        self._last_breakdown = None
        
        # Filter state
        self.filter_alpha = 0.2
        self._s_filtered = self.params.base_safety_radius
        self._Mo_filtered = self.params.base_obstacle_mass
        self._Ri_filtered = self.params.base_influence_factor
        self._DRi_filtered = self.params.base_driver_risk
        self._last_u_applied = 0.0
        self.ignoring_follower = False
        self.leader_gap_error = 0.0
        
        # === NEW: Phase detection state ===
        self._current_phase = ControlPhase.MERGING
        self._stable_condition_start_time = None
        self._current_time = 0.0
        self._last_gap_error = 0.0
        self._last_rel_velocity = 0.0
        self._last_acceleration = 0.0
        
        print("🛡️ Safety Field V3.0 (Phase Detection) Initialized")
        print(f"   📊 Phases: MERGING → FOLLOWING")
        print(f"   ⏱️ Transition time: {self.params.phase_transition_time}s")
        print(f"   📏 FOLLOWING entry: gap_error < {self.params.following_gap_error_factor*100}% of desired_gap")

    # ========================================================================
    # PUBLIC: Time update (call from main simulation loop)
    # ========================================================================
    def set_current_time(self, time: float):
        """Update current simulation time for phase tracking."""
        self._current_time = time

    def get_current_phase(self) -> str:
        """Get current control phase."""
        return self._current_phase

    # ========================================================================
    # PHASE DETECTION LOGIC
    # ========================================================================
    def _check_following_conditions(self, gap_error: float, rel_velocity: float, 
                                     acceleration: float, desired_gap: float,
                                     target_velocity: float) -> bool:
        """
        Check if all conditions for FOLLOWING phase are met.
        
        Conditions (ALL must be true):
        1. |gap_error| < 15% × desired_gap
        2. |rel_velocity| < 5% × target_velocity
        3. |acceleration| < 0.5 m/s²
        """
        gap_threshold = self.params.following_gap_error_factor * desired_gap
        vel_threshold = self.params.following_velocity_error_factor * target_velocity
        acc_threshold = self.params.following_acceleration_threshold
        
        gap_ok = abs(gap_error) < gap_threshold
        vel_ok = abs(rel_velocity) < vel_threshold
        acc_ok = abs(acceleration) < acc_threshold
        
        return gap_ok and vel_ok and acc_ok

    def _check_merging_conditions(self, gap_error: float, rel_velocity: float,
                                   desired_gap: float, target_velocity: float) -> bool:
        """
        Check if ANY condition for returning to MERGING phase is met.
        
        Conditions (ANY triggers return to MERGING):
        1. |gap_error| > 25% × desired_gap
        2. |rel_velocity| > 10% × target_velocity
        """
        gap_threshold = self.params.merging_gap_error_factor * desired_gap
        vel_threshold = self.params.merging_velocity_error_factor * target_velocity
        
        gap_violated = abs(gap_error) > gap_threshold
        vel_violated = abs(rel_velocity) > vel_threshold
        
        return gap_violated or vel_violated

    def _update_phase(self, gap_error: float, rel_velocity: float,
                      acceleration: float, desired_gap: float,
                      target_velocity: float):
        """
        Update control phase based on current conditions.
        
        MERGING → FOLLOWING: All conditions met for 5 seconds
        FOLLOWING → MERGING: Any exit condition violated (immediate)
        """
        if self._current_phase == ControlPhase.MERGING:
            # Check if we can transition to FOLLOWING
            if self._check_following_conditions(gap_error, rel_velocity, 
                                                 acceleration, desired_gap,
                                                 target_velocity):
                # Conditions met - start or continue timing
                if self._stable_condition_start_time is None:
                    self._stable_condition_start_time = self._current_time
                
                # Check if enough time has passed
                elapsed = self._current_time - self._stable_condition_start_time
                if elapsed >= self.params.phase_transition_time:
                    self._current_phase = ControlPhase.FOLLOWING
                    self._stable_condition_start_time = None
                    print(f"✅ Phase transition: MERGING → FOLLOWING at t={self._current_time:.1f}s")
                    print(f"   📏 Gap error: {gap_error:.2f}m, Rel vel: {rel_velocity:.2f}m/s")
            else:
                # Conditions not met - reset timer
                self._stable_condition_start_time = None
                
        elif self._current_phase == ControlPhase.FOLLOWING:
            # Check if we need to return to MERGING
            if self._check_merging_conditions(gap_error, rel_velocity,
                                               desired_gap, target_velocity):
                self._current_phase = ControlPhase.MERGING
                self._stable_condition_start_time = None
                print(f"⚠️ Phase transition: FOLLOWING → MERGING at t={self._current_time:.1f}s")
                print(f"   📏 Gap error: {gap_error:.2f}m, Rel vel: {rel_velocity:.2f}m/s")

    # ========================================================================
    # SOFT TRANSITION FORCE CALCULATION
    # ========================================================================
    def _apply_soft_transition(self, force: float, gap_error: float, 
                                desired_gap: float) -> float:
        """
        Apply soft transition to force in FOLLOWING phase.
        
        F_following = F_merging × min(1.0, (|gap_error| / threshold)²)
        
        This creates a smooth reduction of force for small errors.
        """
        if self._current_phase != ControlPhase.FOLLOWING:
            return force  # No modification in MERGING phase
        
        threshold = self.params.following_gap_error_factor * desired_gap
        
        if threshold <= 0:
            return force
        
        # Quadratic soft transition
        ratio = abs(gap_error) / threshold
        scaling_factor = min(1.0, ratio * ratio)
        
        return force * scaling_factor

    # ========================================================================
    # MAIN FORCE COMPUTATION
    # ========================================================================
    def compute_risk_force_from_platoon(self, ego_vehicle: 'Vehicle',
                                        platoon_manager: 'PlatoonManager',
                                        desired_gap: float) -> float:
        """
        Compute total risk force with phase-aware behavior.
        """
        context = self.extract_platoon_context(ego_vehicle, platoon_manager)
        context.target_velocity = platoon_manager.target_velocity
        context.current_phase = self._current_phase
        self._last_context = context
        
        leader = self._get_leader(ego_vehicle, platoon_manager)
        follower = self._get_follower(ego_vehicle, platoon_manager)
        
        # Calculate gap error and relative velocity for phase detection
        gap_error = 0.0
        rel_velocity = 0.0
        acceleration = getattr(ego_vehicle.state, 'ax', 0.0)
        
        if leader is not None:
            gap_actual = leader.state.x - ego_vehicle.state.x
            gap_error = gap_actual - desired_gap
            rel_velocity = ego_vehicle.state.vx - leader.state.vx
        
        # Update phase based on conditions
        self._update_phase(gap_error, rel_velocity, acceleration,
                          desired_gap, platoon_manager.target_velocity)
        
        # Store for logging
        self._last_gap_error = gap_error
        self._last_rel_velocity = rel_velocity
        self._last_acceleration = acceleration
        
        F_leader = 0.0
        F_follower = 0.0
        
        # === Leader Risk (Repulsive only) ===
        if leader is not None:
            F_leader_raw = self._compute_force_to_vehicle(
                ego_vehicle, leader, desired_gap, context, is_leader=True
            )
            # Apply soft transition in FOLLOWING phase
            F_leader = self._apply_soft_transition(F_leader_raw, gap_error, desired_gap)
            
            # Logic to ignore follower if we are chasing a far leader
            leader_gap = leader.state.x - ego_vehicle.state.x
            if leader_gap > desired_gap * 2.0:
                self.ignoring_follower = True
            else:
                self.ignoring_follower = False
        
        # === Follower Risk (Repulsive only) ===
        if follower is not None and not self.ignoring_follower:
            follower_weight = self._get_follower_weight(context)
            
            # Calculate follower gap error for soft transition
            follower_gap = ego_vehicle.state.x - follower.state.x
            follower_gap_error = follower_gap - desired_gap
            
            F_follower_raw = follower_weight * self._compute_force_to_vehicle(
                ego_vehicle, follower, desired_gap, context, is_leader=False
            )
            # Apply soft transition in FOLLOWING phase
            F_follower = self._apply_soft_transition(F_follower_raw, follower_gap_error, desired_gap)
        
        F_total = F_leader + F_follower
        F_total = min(F_total, self.params.max_repulsive_force)
        
        self._last_breakdown = {
            'context': context,
            'F_leader': F_leader,
            'F_follower': F_follower,
            'F_total': F_total,
            'follower_weight': self._get_follower_weight(context),
            'leader': {'total_force': F_leader},
            'follower': {'total_force': F_follower},
            # NEW: Phase information
            'phase': self._current_phase,
            'gap_error': gap_error,
            'rel_velocity': rel_velocity
        }
        
        return F_total

    def get_force_breakdown_from_platoon(self, ego_vehicle: 'Vehicle',
                                         platoon_manager: 'PlatoonManager',
                                         desired_gap: float) -> Dict:
        """Get detailed force breakdown for debugging."""
        total_force = self.compute_risk_force_from_platoon(
            ego_vehicle, platoon_manager, desired_gap
        )

        if self._last_breakdown is None:
            return {'error': 'No breakdown available'}
        
        breakdown = self._last_breakdown.copy()
        breakdown['total_force'] = total_force
        
        return breakdown

    # ========================================================================
    # FORCE CALCULATION (unchanged from original)
    # ========================================================================
    def _compute_force_to_vehicle(self, ego: 'Vehicle', other: 'Vehicle',
                                   desired_gap: float, context: PlatoonContext,
                                   is_leader: bool) -> float:
        """Compute PURE REPULSIVE force (Risk only)."""
        if is_leader:
            gap_actual = other.state.x - ego.state.x
            v_rel = ego.state.vx - other.state.vx
        else:
            gap_actual = ego.state.x - other.state.x
            v_rel = other.state.vx - ego.state.vx
        
        gap_error = gap_actual - desired_gap
        
        # If gap > desired, we are safe. Force = 0.
        if gap_error > 0:
            return 0.0
            
        # Compute dynamic parameters
        s = self._compute_dynamic_safety_radius(context, is_leader)
        Mo = self._compute_dynamic_obstacle_mass(context, is_leader)
        Ri = self._compute_dynamic_influence_factor(context, is_leader)
        DRi = self._compute_dynamic_driving_risk(context, is_leader)
        
        # Repulsive Force Logic (Li et al.)
        r_elliptic = abs(gap_error) / (s + self.params.epsilon)
        potential = (Mo * Ri) / ((r_elliptic + self.params.epsilon) ** 2)
        
        weight_factor = np.exp(-abs(gap_error) / self.params.distance_decay_factor)
        velocity_factor = np.exp(max(0, v_rel) / 5.0)
        behavior_factor = 1.0 + DRi
        
        force = potential * weight_factor * velocity_factor * behavior_factor
        
        return min(force, self.params.max_repulsive_force)

    def apply_hard_constraint(self, u_shared, ego_vehicle, platoon_manager):
        """Apply constraints and rate limiting."""
        max_jerk = 2.0
        dt = 0.02
        
        if not hasattr(self, '_last_u_applied'):
            self._last_u_applied = u_shared
            
        delta_u = u_shared - self._last_u_applied
        max_delta = max_jerk * dt
        delta_u = np.clip(delta_u, -max_delta, max_delta)
        u_out = self._last_u_applied + delta_u
        
        u_out = np.clip(u_out, -3.5, 2.5)
        
        self._last_u_applied = u_out
        return u_out

    # ========================================================================
    # HELPER FUNCTIONS (unchanged from original)
    # ========================================================================
    def _apply_lowpass_filter(self, target_value: float, current_filtered: float, 
                               alpha: float = None) -> float:
        if alpha is None:
            alpha = self.filter_alpha
        return alpha * target_value + (1 - alpha) * current_filtered

    def _get_position_multiplier(self, context: PlatoonContext) -> float:
        if not context.has_leader:
            return self.params.leader_position_multiplier
        elif context.has_leader and context.has_follower:
            return self.params.middle_position_multiplier
        else:
            return self.params.follower_position_multiplier
    
    def _get_follower_weight(self, context: PlatoonContext) -> float:
        if context.follower_is_joining:
            return self.params.joining_follower_weight
        elif context.follower_decelerating:
            return self.params.decelerating_follower_weight
        else:
            return self.params.follower_weight

    def _compute_dynamic_safety_radius(self, context: PlatoonContext, 
                                        is_leader: bool) -> float:
        s_target = self.params.base_safety_radius * self._get_position_multiplier(context)
        velocity_factor = 1.0 + self.params.velocity_scaling_factor * (
            context.ego_velocity / self.params.velocity_reference - 1.0
        )
        s_target *= max(0.5, min(2.0, velocity_factor))
        return self._apply_lowpass_filter(s_target, self._s_filtered)

    def _compute_dynamic_obstacle_mass(self, context: PlatoonContext, 
                                        is_leader: bool) -> float:
        Mo_target = self.params.base_obstacle_mass * self._get_position_multiplier(context)
        return self._apply_lowpass_filter(Mo_target, self._Mo_filtered)

    def _compute_dynamic_influence_factor(self, context: PlatoonContext, 
                                           is_leader: bool) -> float:
        Ri_target = self.params.base_influence_factor
        road_factor = self.params.road_condition_factors.get(context.road_condition, 1.0)
        Ri_target *= road_factor
        if context.has_leader and context.has_follower:
            Ri_target *= self.params.platoon_coherence_factor
        return self._apply_lowpass_filter(Ri_target, self._Ri_filtered)

    def _compute_dynamic_driving_risk(self, context: PlatoonContext, 
                                       is_leader: bool) -> float:
        DRi_target = self.params.base_driver_risk
        if context.high_risk_situation:
            DRi_target *= self.params.high_risk_multiplier
        return self._apply_lowpass_filter(DRi_target, self._DRi_filtered)

    def _get_leader(self, ego: 'Vehicle', 
                    platoon_mgr: 'PlatoonManager') -> Optional['Vehicle']:
        try:
            ego_idx = platoon_mgr.vehicles.index(ego)
            if ego_idx > 0:
                return platoon_mgr.vehicles[ego_idx - 1]
        except (ValueError, IndexError):
            pass
        return None
    
    def _get_follower(self, ego: 'Vehicle', 
                      platoon_mgr: 'PlatoonManager') -> Optional['Vehicle']:
        try:
            ego_idx = platoon_mgr.vehicles.index(ego)
            if ego_idx < len(platoon_mgr.vehicles) - 1:
                return platoon_mgr.vehicles[ego_idx + 1]
        except (ValueError, IndexError):
            pass
        return None
    
    def extract_platoon_context(self, ego: 'Vehicle', 
                                 platoon_mgr: 'PlatoonManager') -> PlatoonContext:
        try:
            ego_idx = platoon_mgr.vehicles.index(ego)
        except ValueError:
            ego_idx = -1
        
        leader = self._get_leader(ego, platoon_mgr)
        follower = self._get_follower(ego, platoon_mgr)
        
        gap_leader = (leader.state.x - ego.state.x) if leader else float('inf')
        gap_follower = (ego.state.x - follower.state.x) if follower else float('inf')
        
        rel_vel_leader = (ego.state.vx - leader.state.vx) if leader else 0.0
        rel_vel_follower = (follower.state.vx - ego.state.vx) if follower else 0.0
        
        ttc_leader = (gap_leader / rel_vel_leader) if (leader and rel_vel_leader > 0.1) else float('inf')
        ttc_follower = (gap_follower / rel_vel_follower) if (follower and rel_vel_follower > 0.1) else float('inf')
        
        return PlatoonContext(
            has_leader=(leader is not None),
            has_follower=(follower is not None),
            ego_index=ego_idx,
            platoon_size=len(platoon_mgr.vehicles),
            gap_to_leader=gap_leader,
            gap_to_follower=gap_follower,
            rel_vel_leader=rel_vel_leader,
            rel_vel_follower=rel_vel_follower,
            ttc_leader=ttc_leader,
            ttc_follower=ttc_follower,
            ego_velocity=ego.state.vx,
            velocity_error=ego.state.vx - platoon_mgr.target_velocity,
            current_phase=self._current_phase
        )


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================
LongitudinalSafetyField = EllipseLongitudinalSafetyField
LongitudinalSafetyFieldParams = EllipseLongitudinalParams

__all__ = ['EllipseLongitudinalSafetyField', 'EllipseLongitudinalParams', 
           'PlatoonContext', 'ControlPhase']


# ============================================================================
# UNIT TEST
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Safety Field V3.0 - Phase Detection Unit Test")
    print("="*60)
    
    params = EllipseLongitudinalParams()
    field = EllipseLongitudinalSafetyField(params)
    
    # Test parameters
    desired_gap = 54.0  # meters
    target_velocity = 33.3  # m/s (120 km/h)
    
    print(f"\n📊 Test Parameters:")
    print(f"   Desired gap: {desired_gap}m")
    print(f"   Target velocity: {target_velocity}m/s ({target_velocity*3.6:.0f}km/h)")
    
    # Calculate thresholds
    following_gap_thresh = params.following_gap_error_factor * desired_gap
    following_vel_thresh = params.following_velocity_error_factor * target_velocity
    merging_gap_thresh = params.merging_gap_error_factor * desired_gap
    merging_vel_thresh = params.merging_velocity_error_factor * target_velocity
    
    print(f"\n📏 FOLLOWING Entry Thresholds:")
    print(f"   Gap error < {following_gap_thresh:.1f}m")
    print(f"   Rel velocity < {following_vel_thresh:.2f}m/s")
    print(f"   Acceleration < {params.following_acceleration_threshold}m/s²")
    
    print(f"\n⚠️ MERGING Return Thresholds (Hysteresis):")
    print(f"   Gap error > {merging_gap_thresh:.1f}m")
    print(f"   Rel velocity > {merging_vel_thresh:.2f}m/s")
    
    # Test soft transition
    print(f"\n🔄 Soft Transition Test (in FOLLOWING phase):")
    field._current_phase = ControlPhase.FOLLOWING
    
    test_errors = [1, 2, 4, 6, 8, 10, 12]
    test_force = 400.0
    
    for err in test_errors:
        scaled = field._apply_soft_transition(test_force, err, desired_gap)
        print(f"   Gap error {err:2d}m: {test_force:.0f}N → {scaled:.1f}N ({scaled/test_force*100:.1f}%)")
    
    print("\n" + "="*60)
    print("✅ Unit test complete")
    print("="*60)