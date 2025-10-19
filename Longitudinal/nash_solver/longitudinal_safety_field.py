#!/usr/bin/env python3
"""
File: longitudinal_safety_field.py
Description: Contains the driving safety field model for longitudinal risk assessment.

⭐ VERSION 2.0: Five-component safety field with CORRECTED sign conventions
    1. TTC (Time-To-Collision) - collision avoidance (repulsive)
    2. Headway - time gap maintenance (repulsive)
    3. Gap Error - spacing maintenance (BIDIRECTIONAL)
    4. Relative Velocity - closing speed safety (repulsive)
    5. Velocity Error - platoon speed maintenance (BIDIRECTIONAL) ⭐ SIGN FIXED

Sign Convention (consistent across all bidirectional forces):
    - Positive error (deficiency) → Negative force (attractive)
    - Negative error (excess)     → Positive force (repulsive)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from vehicle.vehicle import Vehicle

if TYPE_CHECKING:
    from control.platoon_control import PlatoonManager

@dataclass
class LongitudinalSafetyFieldParams:
    """Parameters for the longitudinal safety field with FIVE force components."""
    
    # Time-To-Collision (TTC) parameters - exponential computation
    ttc_critical: float = 2.0      # [s] Critical TTC threshold
    ttc_weight: float = 60.0       # Weight for TTC force component
    ttc_decay_rate: float = 2.0    # Exponential decay rate for TTC

    # Headway parameters - exponential computation
    headway_critical: float = 1.5  # [s] Critical headway threshold
    headway_weight: float = 40.0   # Weight for headway force component
    headway_decay_rate: float = 1.5 # Exponential decay rate for headway

    # Spacing error parameters - BIDIRECTIONAL
    gap_error_weight: float = 100.0 # Weight for spacing error force component
    gap_error_threshold: float = 2.0 # [m] Dead zone: ±2m from desired gap
    gap_error_decay_rate: float = 1.5 # Exponential decay rate for spacing error

    # Relative velocity parameters - safety only
    rel_vel_weight: float = 40.0   # Weight for relative velocity force component
    rel_vel_threshold: float = 2.0 # [m/s] Minimum relative velocity to consider
    rel_vel_decay_rate: float = 1.0 # Exponential decay rate for relative velocity

    # ⭐ Velocity error parameters - BIDIRECTIONAL (SIGN FIXED)
    velocity_error_weight: float = 50.0  # Weight for velocity maintenance
    velocity_error_threshold: float = 1.0  # [m/s] Dead zone (±1 m/s ≈ ±3.6 km/h)
    velocity_error_decay_rate: float = 1.2  # Exponential decay rate

    # Force limits - BIDIRECTIONAL
    max_force: float = 1000.0       # [N] Maximum REPULSIVE force
    max_attractive_force: float = 300.0  # [N] Maximum ATTRACTIVE force
    
    # Follower parameters
    follower_weight: float = 0.5   # Weight multiplier for follower risk


class LongitudinalSafetyField:
    """
    Calculates longitudinal risk force based on FIVE components:
    
    Safety Components (repulsive only):
    1. Time-To-Collision (TTC) - exponential
    2. Headway - exponential  
    4. Relative velocity - exponential
    
    Control Components (bidirectional):
    3. Spacing error - exponential (BIDIRECTIONAL)
    5. ⭐ Velocity error - exponential (BIDIRECTIONAL) - SIGN FIXED
    
    Sign Convention for bidirectional forces:
        error = target - actual
        force = -sign(error) * magnitude
        
        Positive error (deficiency) → Negative force (attractive)
        Negative error (excess)     → Positive force (repulsive)
    
    Supports bidirectional safety: considers both leader and follower vehicles.
    Works directly with Vehicle objects and PlatoonManager.
    """

    def __init__(self, params: LongitudinalSafetyFieldParams = LongitudinalSafetyFieldParams()):
        self.params = params
        print("🛡️ Longitudinal Safety Field V2.0 (5 Components - Sign Fixed) initialized.")
        print(f"   Safety: TTC, Headway, RelVel (repulsive only)")
        print(f"   Control: Gap ±{self.params.gap_error_threshold}m, "
              f"Velocity ±{self.params.velocity_error_threshold}m/s (bidirectional)")
        print(f"   Force limits: +{self.params.max_force}N / -{self.params.max_attractive_force}N")

    def get_leader_and_follower(self, ego_vehicle: Vehicle, platoon_manager: 'PlatoonManager') -> tuple[Optional[Vehicle], Optional[Vehicle]]:
        """
        Get the leader and follower vehicles for the ego vehicle from the platoon.
        
        Args:
            ego_vehicle: The ego Vehicle object
            platoon_manager: PlatoonManager instance
            
        Returns:
            tuple: (leader_vehicle, follower_vehicle) - either can be None
        """
        if not platoon_manager.vehicles:
            return None, None
        
        # Find ego vehicle index in platoon
        ego_index = None
        for i, vehicle in enumerate(platoon_manager.vehicles):
            if vehicle.vehicle_id == ego_vehicle.vehicle_id:
                ego_index = i
                break
        
        if ego_index is None:
            # Ego vehicle not in platoon yet - find where it would be
            ego_index = platoon_manager.get_new_vehicle_index(ego_vehicle)
            
            # Get leader and follower based on hypothetical position
            leader = platoon_manager.vehicles[ego_index - 1] if ego_index > 0 else None
            follower = platoon_manager.vehicles[ego_index] if ego_index < len(platoon_manager.vehicles) else None
            
            return leader, follower
        
        # Ego vehicle is in platoon
        leader = platoon_manager.vehicles[ego_index - 1] if ego_index > 0 else None
        follower = platoon_manager.vehicles[ego_index + 1] if ego_index < len(platoon_manager.vehicles) - 1 else None
        
        return leader, follower

    def compute_ttc_force(self, ttc: float) -> float:
        """Computes TTC force component using exponential decay (SAFETY ONLY)."""
        if ttc >= self.params.ttc_critical or ttc == float('inf'):
            return 0.0
        
        # Exponential force: increases exponentially as TTC decreases
        normalized_ttc = ttc / self.params.ttc_critical
        force = self.params.ttc_weight * np.exp(-self.params.ttc_decay_rate * normalized_ttc)
        return force

    def compute_headway_force(self, headway: float) -> float:
        """Computes headway force component using exponential decay (SAFETY ONLY)."""
        if headway >= self.params.headway_critical:
            return 0.0
        
        # Exponential force: increases exponentially as headway decreases
        normalized_headway = headway / self.params.headway_critical
        force = self.params.headway_weight * np.exp(-self.params.headway_decay_rate * normalized_headway)
        return force

    def compute_gap_error_force(self, gap_error: float, desired_gap: float) -> float:
        """
        Computes spacing error force component (BIDIRECTIONAL).
        
        Physics:
        - gap_error < 0 (too close)  → force > 0 (repulsive, brake)
        - gap_error > 0 (too far)    → force < 0 (attractive, accelerate)  
        - gap_error ≈ 0 (perfect)    → force ≈ 0 (equilibrium)
        
        Args:
            gap_error: actual_gap - desired_gap [m]
            desired_gap: desired spacing [m]
        
        Returns:
            Force [N]: positive = repel, negative = attract
        """
        # Dead zone - comfortable range around desired gap
        if np.abs(gap_error) < self.params.gap_error_threshold:
            return 0.0
        
        # Calculate force MAGNITUDE (always positive)
        normalized_error = np.abs(gap_error) / max(desired_gap, 1.0)
        force_magnitude = self.params.gap_error_weight * (
            np.exp(self.params.gap_error_decay_rate * normalized_error) - 1.0
        )
        
        # Apply sign: negative error (too close) → positive force (repel)
        #            positive error (too far)   → negative force (attract)
        force = -np.sign(gap_error) * force_magnitude
        force = (1+normalized_error) * force  if force > 0 else force  # Slightly more aggressive on gap errors
        return force

    def compute_relative_velocity_force(self, relative_vel: float) -> float:
        """
        Computes relative velocity force component (SAFETY ONLY).
        
        Only penalizes closing in (positive relative_vel).
        Opening or stable spacing produces zero force.
        
        Args:
            relative_vel: vel_ego - vel_other [m/s]
                         positive = closing (dangerous)
                         negative = opening (safe)
        
        Returns:
            Force [N]: always ≥ 0 (repulsive or zero)
        """
        # Only penalize closing in
        if relative_vel <= 0:
            return 0.0
        
        if relative_vel < self.params.rel_vel_threshold:
            return 0.0
        
        # Exponential force: increases exponentially as relative velocity increases
        excess_velocity = relative_vel - self.params.rel_vel_threshold
        normalized_excess = excess_velocity / self.params.rel_vel_threshold
        force = self.params.rel_vel_weight * (
            np.exp(self.params.rel_vel_decay_rate * normalized_excess) - 1.0
        )
        return force

    def compute_velocity_error_force(self, velocity_error: float, target_velocity: float) -> float:
        """
        ⭐ FIXED: Computes velocity error force component (BIDIRECTIONAL).
        
        Purpose: Maintain platoon target velocity
        
        Physics (CORRECTED):
        - velocity_error > 0 (lacking speed) → force < 0 (attractive, accelerate)
        - velocity_error < 0 (excess speed)  → force > 0 (repulsive, brake)
        - velocity_error ≈ 0 (perfect)       → force ≈ 0 (cruise)
        
        ⭐ IMPORTANT: velocity_error = target_velocity - actual_velocity
                     This makes it consistent with gap_error convention:
                     - Positive error = deficiency → attractive force (negative)
                     - Negative error = excess → repulsive force (positive)
        
        Args:
            velocity_error: target_velocity - actual_velocity [m/s]  ← REVERSED!
            target_velocity: desired platoon velocity [m/s]
        
        Returns:
            Force [N]: positive = brake, negative = accelerate
        """
        # Dead zone - comfortable range around target velocity
        if abs(velocity_error) < self.params.velocity_error_threshold:
            return 0.0
        
        # Calculate force magnitude
        normalized_error = abs(velocity_error) / max(target_velocity, 1.0)
        force_magnitude = self.params.velocity_error_weight * (
            np.exp(self.params.velocity_error_decay_rate * normalized_error) - 1.0
        )
        
        # ⭐ FIXED: Apply sign (consistent with gap_error)
        # Positive velocity_error (lacking speed) → negative force (attract/accelerate)
        # Negative velocity_error (excess speed)  → positive force (repel/brake)
        force = -np.sign(velocity_error) * force_magnitude
        
        return force

    def compute_risk_force_from_vehicle(self,
                                       ego_vehicle: Vehicle,
                                       other_vehicle: Vehicle,
                                       desired_gap: float,
                                       target_velocity: float,
                                       is_leader: bool = True) -> float:
        """
        ⭐ FIXED: Computes risk force from a single vehicle (5 components).
        
        Force components:
        1. TTC - collision avoidance (repulsive)
        2. Headway - time gap safety (repulsive)
        3. Gap error - spacing maintenance (bidirectional)
        4. Relative velocity - closing speed safety (repulsive)
        5. ⭐ Velocity error - speed maintenance (bidirectional) - SIGN FIXED
        
        Args:
            ego_vehicle: The ego Vehicle object
            other_vehicle: The other Vehicle object (leader or follower)
            desired_gap: desired gap distance [m]
            target_velocity: desired platoon velocity [m/s]
            is_leader: True if other vehicle is ahead, False if behind
            
        Returns:
            Risk force in Newtons (can be positive or negative)
        """
        pos_ego = ego_vehicle.state.x
        vel_ego = ego_vehicle.state.vx
        pos_other = other_vehicle.state.x
        vel_other = other_vehicle.state.vx
        
        if is_leader:
            # Leader ahead
            actual_gap = pos_other - pos_ego
            relative_vel = vel_ego - vel_other  # Positive when closing in
        else:
            # Follower behind
            actual_gap = pos_ego - pos_other
            relative_vel = vel_other - vel_ego  # Positive when follower closing in

        # Calculate TTC when closing in
        if np.abs(relative_vel) > 1e-3 and actual_gap > 0:
            ttc = actual_gap / np.abs(relative_vel)
        else:
            ttc = float('inf')

        # Calculate headway (time gap)
        if is_leader and vel_ego > 1e-3:
            headway = actual_gap / vel_ego
        elif not is_leader and vel_other > 1e-3:
            headway = actual_gap / vel_other
        else:
            headway = float('inf')

        # Gap error: positive means "too far", negative means "too close"
        gap_error = actual_gap - desired_gap
        
        # ⭐ FIXED: Velocity error (reversed definition)
        velocity_error = target_velocity - vel_ego  # Was: vel_ego - target_velocity

        # --- Calculate individual force components ---
        
        # Safety forces (always repulsive, ≥ 0)
        force_ttc = self.compute_ttc_force(ttc)
        force_headway = self.compute_headway_force(headway)
        force_rel_vel = self.compute_relative_velocity_force(relative_vel)
        force_safety = force_ttc + force_headway + force_rel_vel
        
        # Gap maintenance (bidirectional)
        force_gap_error = self.compute_gap_error_force(gap_error, desired_gap)
        
        # ⭐ Velocity maintenance (bidirectional) - SIGN FIXED
        force_velocity_error = self.compute_velocity_error_force(velocity_error, target_velocity)

        # Total risk force = safety + gap control + velocity control
        total_force = force_safety + force_gap_error + force_velocity_error

        return total_force

    def compute_risk_force(self,
                           ego_vehicle: Vehicle,
                           leader_vehicle: Optional[Vehicle],
                           desired_gap: float,
                           target_velocity: float,
                           follower_vehicle: Optional[Vehicle] = None) -> float:
        """
        ⭐ FIXED: Computes total longitudinal risk force (5 components).
        
        Args:
            ego_vehicle: The ego Vehicle object
            leader_vehicle: Leader Vehicle object (can be None)
            desired_gap: desired gap distance [m]
            target_velocity: desired platoon velocity [m/s]
            follower_vehicle: Follower Vehicle object (can be None)
        
        Returns:
            Total risk force with bidirectional limits [N]
        """
        total_force = 0.0
        
        # Compute risk from leader
        if leader_vehicle is not None and desired_gap != float('inf'):
            force_from_leader = self.compute_risk_force_from_vehicle(
                ego_vehicle, leader_vehicle, desired_gap, target_velocity, is_leader=True
            )
            total_force += force_from_leader
        
        # Compute risk from follower
        if follower_vehicle is not None and desired_gap != float('inf'):
            force_from_follower = self.compute_risk_force_from_vehicle(
                ego_vehicle, follower_vehicle, desired_gap, target_velocity, is_leader=False
            )
            # Apply weight multiplier for follower risk
            total_force += force_from_follower * self.params.follower_weight

        # ⭐ Bidirectional limits
        if total_force > 0:  # Repulsive
            return min(total_force, self.params.max_force)
        else:  # Attractive
            return max(total_force, -self.params.max_attractive_force)

    def compute_risk_force_from_platoon(self,
                                       ego_vehicle: Vehicle,
                                       platoon_manager: 'PlatoonManager',
                                       desired_gap: float) -> float:
        """
        ⭐ FIXED: Computes risk force automatically from platoon manager.
        Automatically gets target_velocity from platoon_manager.
        
        Args:
            ego_vehicle: The ego Vehicle object
            platoon_manager: PlatoonManager instance
            desired_gap: desired gap distance [m]
            
        Returns:
            Total risk force (with bidirectional limits) [N]
        """
        leader, follower = self.get_leader_and_follower(ego_vehicle, platoon_manager)
        target_velocity = platoon_manager.target_velocity
        return self.compute_risk_force(ego_vehicle, leader, desired_gap, target_velocity, follower)

    def get_force_breakdown(self,
                           ego_vehicle: Vehicle,
                           leader_vehicle: Optional[Vehicle],
                           desired_gap: float,
                           target_velocity: float,
                           follower_vehicle: Optional[Vehicle] = None) -> dict:
        """
        ⭐ FIXED: Returns detailed breakdown of all FIVE force components.
        """
        breakdown = {
            'leader': {
                'ttc_force': 0.0,
                'headway_force': 0.0,
                'gap_error_force': 0.0,
                'rel_vel_force': 0.0,
                'velocity_error_force': 0.0,  # ⭐
                'total_force': 0.0,
                'ttc': float('inf'),
                'headway': float('inf'),
                'gap_error': 0.0,
                'relative_vel': 0.0,
                'velocity_error': 0.0  # ⭐
            },
            'follower': {
                'ttc_force': 0.0,
                'headway_force': 0.0,
                'gap_error_force': 0.0,
                'rel_vel_force': 0.0,
                'velocity_error_force': 0.0,  # ⭐
                'total_force': 0.0,
                'ttc': float('inf'),
                'headway': float('inf'),
                'gap_error': 0.0,
                'relative_vel': 0.0,
                'velocity_error': 0.0  # ⭐
            },
            'total_force': 0.0
        }

        pos_ego = ego_vehicle.state.x
        vel_ego = ego_vehicle.state.vx
        
        # ⭐ FIXED: Velocity error (reversed)
        velocity_error = target_velocity - vel_ego  # Was: vel_ego - target_velocity
        
        # Process leader
        if leader_vehicle is not None and desired_gap != float('inf'):
            pos_leader = leader_vehicle.state.x
            vel_leader = leader_vehicle.state.vx
            actual_gap = pos_leader - pos_ego
            relative_vel = vel_ego - vel_leader

            if relative_vel > 1e-3 and actual_gap > 0:
                ttc = actual_gap / relative_vel
            else:
                ttc = float('inf')

            if vel_ego > 1e-3:
                headway = actual_gap / vel_ego
            else:
                headway = float('inf')

            gap_error = actual_gap - desired_gap

            breakdown['leader'] = {
                'ttc_force': self.compute_ttc_force(ttc),
                'headway_force': self.compute_headway_force(headway),
                'gap_error_force': self.compute_gap_error_force(gap_error, desired_gap),
                'rel_vel_force': self.compute_relative_velocity_force(relative_vel),
                'velocity_error_force': self.compute_velocity_error_force(velocity_error, target_velocity),  # ⭐
                'ttc': ttc,
                'headway': headway,
                'gap_error': gap_error,
                'relative_vel': relative_vel,
                'velocity_error': velocity_error  # ⭐
            }
            breakdown['leader']['total_force'] = sum([
                breakdown['leader']['ttc_force'],
                breakdown['leader']['headway_force'],
                breakdown['leader']['gap_error_force'],
                breakdown['leader']['rel_vel_force'],
                breakdown['leader']['velocity_error_force']  # ⭐
            ])

        # Process follower
        if follower_vehicle is not None and desired_gap != float('inf'):
            pos_follower = follower_vehicle.state.x
            vel_follower = follower_vehicle.state.vx
            actual_gap = pos_ego - pos_follower
            relative_vel = vel_follower - vel_ego

            if relative_vel > 1e-3 and actual_gap > 0:
                ttc = actual_gap / relative_vel
            else:
                ttc = float('inf')

            if vel_follower > 1e-3:
                headway = actual_gap / vel_follower
            else:
                headway = float('inf')

            gap_error = actual_gap - desired_gap

            breakdown['follower'] = {
                'ttc_force': self.compute_ttc_force(ttc),
                'headway_force': self.compute_headway_force(headway),
                'gap_error_force': self.compute_gap_error_force(gap_error, desired_gap),
                'rel_vel_force': self.compute_relative_velocity_force(relative_vel),
                'velocity_error_force': self.compute_velocity_error_force(velocity_error, target_velocity),  # ⭐
                'ttc': ttc,
                'headway': headway,
                'gap_error': gap_error,
                'relative_vel': relative_vel,
                'velocity_error': velocity_error  # ⭐
            }
            breakdown['follower']['total_force'] = sum([
                breakdown['follower']['ttc_force'],
                breakdown['follower']['headway_force'],
                breakdown['follower']['gap_error_force'],
                breakdown['follower']['rel_vel_force'],
                breakdown['follower']['velocity_error_force']  # ⭐
            ]) * self.params.follower_weight

        # Total force with bidirectional limit
        total_raw = breakdown['leader']['total_force'] + breakdown['follower']['total_force']
        
        if total_raw > 0:
            breakdown['total_force'] = min(total_raw, self.params.max_force)
        else:
            breakdown['total_force'] = max(total_raw, -self.params.max_attractive_force)

        return breakdown

    def get_force_breakdown_from_platoon(self,
                                        ego_vehicle: Vehicle,
                                        platoon_manager: 'PlatoonManager',
                                        desired_gap: float) -> dict:
        """
        ⭐ FIXED: Get force breakdown automatically from platoon manager.
        """
        leader, follower = self.get_leader_and_follower(ego_vehicle, platoon_manager)
        target_velocity = platoon_manager.target_velocity
        return self.get_force_breakdown(ego_vehicle, leader, desired_gap, target_velocity, follower)


# ============================================================================
# DEMONSTRATION EXAMPLES
# ============================================================================

def run_example_demonstration():
    """⭐ FIXED: Demonstrates 5-component safety field with correct signs."""
    print("\n" + "="*70)
    print("🚗 LONGITUDINAL SAFETY FIELD V2.0 - 5 COMPONENTS (SIGN FIXED)")
    print("="*70)
    os.system('cls' if os.name == 'nt' else 'clear')
    
    from control.platoon_control import PlatoonManager
    
    # Initialize safety field
    safety_field = LongitudinalSafetyField()
    desired_gap = 30.0  # meters
    target_velocity = 20.0  # m/s = 72 km/h
    
    # Create test platoon
    vehicles = [
        Vehicle(initial_x=0.0, initial_velocity=20.0, vehicle_id="Leader"),
        Vehicle(initial_x=50.0, initial_velocity=20.0, vehicle_id="Vehicle_2"),
    ]
    
    platoon_manager = PlatoonManager(vehicles)
    platoon_manager.target_velocity = target_velocity
    
    print(f"\nPlatoon Configuration:")
    print(f"  Desired Gap: {desired_gap:.1f}m")
    print(f"  ⭐ Target Velocity: {target_velocity:.1f}m/s ({target_velocity*3.6:.0f} km/h)\n")
    
    # ⭐ Test scenarios covering all 5 components
    test_scenarios = [
        {
            'name': 'Perfect State',
            'leader_x': 80.0,
            'ego_x': 50.0,
            'ego_velocity': 20.0,
            'description': '✅ Perfect gap AND perfect velocity'
        },
        {
            'name': 'Too Slow',
            'leader_x': 80.0,
            'ego_x': 50.0,
            'ego_velocity': 15.0,
            'description': '🐌 Good spacing but 5 m/s too slow'
        },
        {
            'name': 'Too Fast',
            'leader_x': 80.0,
            'ego_x': 50.0,
            'ego_velocity': 25.0,
            'description': '🚀 Good spacing but 5 m/s too fast'
        },
        {
            'name': 'Too Close',
            'leader_x': 60.0,
            'ego_x': 50.0,
            'ego_velocity': 20.0,
            'description': '⚠️ Dangerously close (10m gap)'
        },
        {
            'name': 'Too Far',
            'leader_x': 120.0,
            'ego_x': 50.0,
            'ego_velocity': 20.0,
            'description': '📏 Too far (70m gap, 40m too much)'
        },
        {
            'name': 'Close AND Slow',
            'leader_x': 60.0,
            'ego_x': 50.0,
            'ego_velocity': 15.0,
            'description': '🔴 Double problem: safety + speed'
        },
        {
            'name': 'Far AND Fast',
            'leader_x': 100.0,
            'ego_x': 50.0,
            'ego_velocity': 25.0,
            'description': '🟡 Catching up but too fast'
        },
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{'='*70}")
        print(f"{i}. {scenario['name']:40}")
        print(f"   {scenario['description']}")
        print("-" * 70)
        
        # Setup scenario
        leader = vehicles[0]
        ego = vehicles[1]
        
        leader.state.x = scenario['leader_x']
        ego.state.x = scenario['ego_x']
        ego.state.vx = scenario['ego_velocity']
        
        actual_gap = leader.state.x - ego.state.x
        gap_error = actual_gap - desired_gap
        velocity_error = target_velocity - ego.state.vx  # ⭐ FIXED: reversed
        
        # Get breakdown
        breakdown = safety_field.get_force_breakdown(
            ego, leader, desired_gap, target_velocity, None
        )
        
        # Display state
        print(f"\n   📊 Current State:")
        print(f"      Gap:      {actual_gap:.1f}m (desired: {desired_gap:.1f}m, error: {gap_error:+.1f}m)")
        print(f"      Velocity: {ego.state.vx:.1f}m/s (target: {target_velocity:.1f}m/s, error: {velocity_error:+.1f}m/s)")
        
        # Display forces
        print(f"\n   🔧 Force Components:")
        print(f"      1. TTC:           {breakdown['leader']['ttc_force']:+8.2f}N  (collision risk)")
        print(f"      2. Headway:       {breakdown['leader']['headway_force']:+8.2f}N  (time gap)")
        print(f"      3. Gap Error:     {breakdown['leader']['gap_error_force']:+8.2f}N  {'⬅️ REPEL' if breakdown['leader']['gap_error_force'] > 0 else '➡️ ATTRACT' if breakdown['leader']['gap_error_force'] < 0 else '⚖️ OK'}")
        print(f"      4. RelVel:        {breakdown['leader']['rel_vel_force']:+8.2f}N  (closing speed)")
        print(f"      5. ⭐ Vel Error:  {breakdown['leader']['velocity_error_force']:+8.2f}N  {'🚀 ACCEL' if breakdown['leader']['velocity_error_force'] < 0 else '🔴 BRAKE' if breakdown['leader']['velocity_error_force'] > 0 else '⚪ OK'}")
        
        print(f"\n   🎯 TOTAL FORCE: {breakdown['total_force']:+8.2f}N")
        
        # ⭐ FIXED: Correct interpretation
        if breakdown['total_force'] > 200:
            action = "🔴 STRONG DECELERATION"
            safety = "CRITICAL"
        elif breakdown['total_force'] > 50:
            action = "🟡 MODERATE DECELERATION"
            safety = "CAUTION"
        elif breakdown['total_force'] > 10:
            action = "🟢 LIGHT DECELERATION"
            safety = "SAFE"
        elif breakdown['total_force'] > -10:
            action = "⚪ CRUISE/MAINTAIN"
            safety = "OPTIMAL"
        elif breakdown['total_force'] > -50:
            action = "🟢 LIGHT ACCELERATION"
            safety = "ADJUST"
        elif breakdown['total_force'] > -100:
            action = "🟡 MODERATE ACCELERATION"
            safety = "CATCH UP"
        else:
            action = "🚀 STRONG ACCELERATION"
            safety = "CATCH UP"
        
        print(f"   {action:30} [{safety}]")
    
    print(f"\n{'='*70}")
    print("✅ 5-Component Safety Field Demonstration Completed!")
    print("\n📌 Component Summary:")
    print("   1-2-4: Safety Forces (TTC, Headway, RelVel) - repulsive only")
    print("   3:     Gap Control (bidirectional) - maintains spacing")
    print("   5:     ⭐ Velocity Control (bidirectional) - maintains platoon speed [SIGN FIXED]")
    print(f"{'='*70}\n")


def run_velocity_focus_demo():
    """⭐ FIXED: Focus specifically on velocity error component with correct signs."""
    print("\n" + "="*70)
    print("⭐ VELOCITY CONTROL COMPONENT - FOCUSED DEMONSTRATION (SIGN FIXED)")
    print("="*70)
    
    from control.platoon_control import PlatoonManager
    
    safety_field = LongitudinalSafetyField()
    target_velocity = 20.0  # 72 km/h
    
    print(f"\nTarget Velocity: {target_velocity:.1f}m/s ({target_velocity*3.6:.0f} km/h)")
    print(f"Dead Zone: ±{safety_field.params.velocity_error_threshold:.1f}m/s (±{safety_field.params.velocity_error_threshold*3.6:.1f} km/h)\n")
    
    # Test different velocities
    test_velocities = [10, 15, 18, 19, 20, 21, 22, 25, 30]
    
    print(f"{'Actual':>10} | {'Error':>8} | {'Force':>10} | {'Action':>25}")
    print("-" * 65)
    
    for vel in test_velocities:
        # ⭐ FIXED: Reversed definition
        velocity_error = target_velocity - vel  # Was: vel - target_velocity
        force = safety_field.compute_velocity_error_force(velocity_error, target_velocity)
        
        if abs(velocity_error) < safety_field.params.velocity_error_threshold:
            action = "✅ CRUISE (dead zone)"
        elif force < -100:
            action = "🚀🚀 STRONG ACCELERATE"
        elif force < 0:
            action = "🚀 LIGHT ACCELERATE"
        elif force > 100:
            action = "🔴🔴 STRONG BRAKE"
        else:
            action = "🔴 LIGHT BRAKE"
        
        print(f"{vel:>7.0f} m/s | {velocity_error:+7.1f} | {force:+9.2f}N | {action}")
    
    print("\n💡 Insights (CORRECTED):")
    print("   • Within ±1 m/s (±3.6 km/h): No force (comfortable cruise)")
    print("   • Too slow (positive error): NEGATIVE force → accelerate ✓")
    print("   • Too fast (negative error): POSITIVE force → brake ✓")
    print("   • Force grows exponentially with speed error")
    print("="*70 + "\n")


__all__ = ['LongitudinalSafetyField', 'LongitudinalSafetyFieldParams']

if __name__ == "__main__":
    run_example_demonstration()
    run_velocity_focus_demo()