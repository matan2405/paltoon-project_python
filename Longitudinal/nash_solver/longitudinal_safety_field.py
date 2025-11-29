#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: longitudinal_safety_field.py
Description: UPDATED - Pure Repulsive Safety Field (Collision Avoidance Only).
Removed 'Attractive' forces to prevent interference with Navigation logic.
Includes 'get_force_breakdown_from_platoon' for logging.
"""

import numpy as np
from dataclasses import dataclass
import os
import sys
from typing import Optional, Dict, TYPE_CHECKING

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vehicle.vehicle import Vehicle
from control.platoon_control import PlatoonManager

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

@dataclass
class EllipseLongitudinalParams():
    """Parameters for pure repulsive safety field"""
    # Hard Constraints
    min_safe_distance: float = 5.0
    emergency_brake_distance: float = 8.0
    max_decel: float = 3.5
    
    # Base Parameters
    base_safety_radius: float = 10.0
    base_obstacle_mass: float = 400.0#1305.0
    base_influence_factor: float = 1.0
    base_driver_risk: float = 0.5
    epsilon: float = 0.1
    
    # Multipliers
    leader_position_multiplier: float = 0.8
    middle_position_multiplier: float = 1.2
    follower_position_multiplier: float = 1.0
    
    # Risk Multipliers
    normal_risk_multiplier: float = 1.0
    moderate_risk_multiplier: float = 1.5
    high_risk_multiplier: float = 2.0
    emergency_risk_multiplier: float = 2.5
    
    # Velocity Scaling
    velocity_reference: float = 120.0
    velocity_scaling_factor: float = 0.8
    
    # Distance Decay
    distance_decay_factor: float = 15.0
    
    # Force Limits
    max_repulsive_force: float = 800.0#1500.0
    # Removed max_attractive_force - Field is purely repulsive now
    
    # Weights
    leader_weight: float = 1.0
    follower_weight: float = 0.5
    joining_follower_weight: float = 0.2
    decelerating_follower_weight: float = 0.4
    
    platoon_coherence_factor: float = 1.5
    merging_multiplier: float = 1.2
    
    road_condition_factors: Dict[str, float] = None
    
    def __post_init__(self):
        if self.road_condition_factors is None:
            self.road_condition_factors = {
                "dry": 1.0,
                "wet": 0.7,
                "icy": 0.5
            }

class EllipseLongitudinalSafetyField:
    """
    Ellipse-based longitudinal safety field.
    Calculates repulsive forces to prevent collisions.
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
        
        print("🛡️ Pure Repulsive Safety Field Initialized (Fixed Version)")

    def compute_risk_force_from_platoon(self, ego_vehicle: 'Vehicle',
                                       platoon_manager: 'PlatoonManager',
                                       desired_gap: float) -> float:
        """
        Compute total risk force.
        """
        context = self.extract_platoon_context(ego_vehicle, platoon_manager)
        context.target_velocity = platoon_manager.target_velocity
        self._last_context = context
        
        leader = self._get_leader(ego_vehicle, platoon_manager)
        follower = self._get_follower(ego_vehicle, platoon_manager)
        
        F_leader = 0.0
        F_follower = 0.0
        
        # Leader Risk (Repulsive only)
        if leader is not None:
            F_leader = self._compute_force_to_vehicle(
                ego_vehicle, leader, desired_gap, context, is_leader=True
            )
            
            # Logic to ignore follower if we are chasing a far leader
            leader_gap = leader.state.x - ego_vehicle.state.x
            if leader_gap > desired_gap * 2.0:
                self.ignoring_follower = True
            else:
                self.ignoring_follower = False
        
        # Follower Risk (Repulsive only)
        if follower is not None and not self.ignoring_follower:
            follower_weight = self._get_follower_weight(context)
            F_follower = follower_weight * self._compute_force_to_vehicle(
                ego_vehicle, follower, desired_gap, context, is_leader=False
            )
        
        F_total = F_leader + F_follower
        
        self._last_breakdown = {
            'context': context,
            'F_leader': F_leader,
            'F_follower': F_follower,
            'F_total': F_total,
            'follower_weight': self._get_follower_weight(context),
            'leader': {'total_force': F_leader},     # Structure for main_with_nash logging
            'follower': {'total_force': F_follower}  # Structure for main_with_nash logging
        }
        
        return F_total

    def get_force_breakdown_from_platoon(self, ego_vehicle: 'Vehicle',
                                        platoon_manager: 'PlatoonManager',
                                        desired_gap: float) -> Dict:
        """Get detailed force breakdown for debugging."""
        # Ensure calculation is up to date
        total_force = self.compute_risk_force_from_platoon(ego_vehicle, platoon_manager, desired_gap)

        if self._last_breakdown is None:
            return {'error': 'No breakdown available'}
        
        breakdown = self._last_breakdown.copy()
        # Ensure total_force is in the breakdown dictionary
        breakdown['total_force'] = total_force
        
        return breakdown

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
        
        # ELEGANT FIX: If gap > desired, we are safe. Force = 0.
        # The Navigation (Nash Reference) handles closing the gap.
        # Safety Field only handles "Too Close" (gap_error < 0).
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
        # Only increase risk if closing in (v_rel > 0)
        velocity_factor = np.exp(max(0, v_rel) / 5.0) 
        behavior_factor = 1.0 + DRi
        
        force = potential * weight_factor * velocity_factor * behavior_factor
        
        return min(force, self.params.max_repulsive_force)

    def apply_hard_constraint(self, u_shared, ego_vehicle, platoon_manager):
        """Apply constraints and rate limiting."""
        # Rate Limiting (Jerk control)
        max_jerk = 2.0
        dt = 0.02
        
        if not hasattr(self, '_last_u_applied'):
            self._last_u_applied = u_shared
            
        delta_u = u_shared - self._last_u_applied
        max_delta = max_jerk * dt
        delta_u = np.clip(delta_u, -max_delta, max_delta)
        u_out = self._last_u_applied + delta_u
        
        # Absolute Limits
        u_out = np.clip(u_out, -3.5, 2.5)
        
        self._last_u_applied = u_out
        return u_out

    # ==================== HELPER FUNCTIONS ====================

    def _apply_lowpass_filter(self, target_value: float, current_filtered: float, alpha: float = None) -> float:
        if alpha is None: alpha = self.filter_alpha
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

    def _compute_dynamic_safety_radius(self, context: PlatoonContext, is_leader: bool) -> float:
        s_target = self.params.base_safety_radius * self._get_position_multiplier(context)
        # Simple velocity scaling
        velocity_factor = 1.0 + self.params.velocity_scaling_factor * (context.ego_velocity / self.params.velocity_reference - 1.0)
        s_target *= max(0.5, min(2.0, velocity_factor))
        return self._apply_lowpass_filter(s_target, self._s_filtered)

    def _compute_dynamic_obstacle_mass(self, context: PlatoonContext, is_leader: bool) -> float:
        Mo_target = self.params.base_obstacle_mass * self._get_position_multiplier(context)
        return self._apply_lowpass_filter(Mo_target, self._Mo_filtered)

    def _compute_dynamic_influence_factor(self, context: PlatoonContext, is_leader: bool) -> float:
        Ri_target = self.params.base_influence_factor
        road_factor = self.params.road_condition_factors.get(context.road_condition, 1.0)
        Ri_target *= road_factor
        if context.has_leader and context.has_follower:
            Ri_target *= self.params.platoon_coherence_factor
        return self._apply_lowpass_filter(Ri_target, self._Ri_filtered)

    def _compute_dynamic_driving_risk(self, context: PlatoonContext, is_leader: bool) -> float:
        DRi_target = self.params.base_driver_risk
        if context.high_risk_situation:
            DRi_target *= self.params.high_risk_multiplier
        return self._apply_lowpass_filter(DRi_target, self._DRi_filtered)

    def _get_leader(self, ego: 'Vehicle', platoon_mgr: 'PlatoonManager') -> Optional['Vehicle']:
        try:
            ego_idx = platoon_mgr.vehicles.index(ego)
            if ego_idx > 0:
                return platoon_mgr.vehicles[ego_idx - 1]
        except (ValueError, IndexError):
            pass
        return None
    
    def _get_follower(self, ego: 'Vehicle', platoon_mgr: 'PlatoonManager') -> Optional['Vehicle']:
        try:
            ego_idx = platoon_mgr.vehicles.index(ego)
            if ego_idx < len(platoon_mgr.vehicles) - 1:
                return platoon_mgr.vehicles[ego_idx + 1]
        except (ValueError, IndexError):
            pass
        return None
    
    def extract_platoon_context(self, ego: 'Vehicle', platoon_mgr: 'PlatoonManager') -> PlatoonContext:
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
            velocity_error=ego.state.vx - platoon_mgr.target_velocity
        )

# Backward compatibility aliases
LongitudinalSafetyField = EllipseLongitudinalSafetyField
LongitudinalSafetyFieldParams = EllipseLongitudinalParams

__all__ = ['EllipseLongitudinalSafetyField', 'EllipseLongitudinalParams', 'PlatoonContext']