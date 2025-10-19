#!/usr/bin/env python3
"""
File: longitudinal_safety_field.py
Description: Contains the driving safety field model for longitudinal risk assessment.
Computes risk force based on four components: TTC, headway, spacing error, and relative velocity.
The first two components (TTC and headway) use exponential computation, 
while the last two (spacing error and relative velocity) use linear computation.
Supports bidirectional safety assessment (leader and follower).
Uses Vehicle class and PlatoonManager for better integration.
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
    """Parameters for the longitudinal safety field with four force components."""
    # Time-To-Collision (TTC) parameters - exponential computation
    ttc_critical: float = 2.0      # [s] Critical TTC threshold
    ttc_weight: float = 60.0       # Weight for TTC force component
    ttc_decay_rate: float = 2.0    # Exponential decay rate for TTC

    # Headway parameters - exponential computation
    headway_critical: float = 1.5  # [s] Critical headway threshold
    headway_weight: float = 40.0   # Weight for headway force component
    headway_decay_rate: float = 1.5 # Exponential decay rate for headway

    # Spacing error parameters - linear computation
    gap_error_weight: float = 100.0 # Weight for spacing error force component
    gap_error_threshold: float = 2.0 # [m] Minimum gap error to consider
    gap_error_decay_rate: float = 1.5 # Exponential decay rate for spacing error

    # Relative velocity parameters - linear computation
    rel_vel_weight: float = 40.0   # Weight for relative velocity force component
    rel_vel_threshold: float = 2.0 # [m/s] Minimum relative velocity to consider
    rel_vel_decay_rate: float = 1.0 # Exponential decay rate for relative velocity

    max_force: float = 1000.0       # [N] Maximum allowable risk force
    
    # Follower parameters
    follower_weight: float = 0.5   # Weight multiplier for follower risk (relative to leader)

class LongitudinalSafetyField:
    """
    Calculates longitudinal risk force based on four components:
    1. Time-To-Collision (TTC) - exponential
    2. Headway - exponential  
    3. Spacing error - linear
    4. Relative velocity - linear
    
    Supports bidirectional safety: considers both leader and follower vehicles.
    Works directly with Vehicle objects and PlatoonManager.
    """

    def __init__(self, params: LongitudinalSafetyFieldParams = LongitudinalSafetyFieldParams()):
        self.params = params
        print("🛡️  Longitudinal Safety Field (Four Components + Bidirectional + Platoon Integration) initialized.")

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
        """Computes TTC force component using exponential decay."""
        if ttc >= self.params.ttc_critical or ttc == float('inf'):
            return 0.0
        
        # Exponential force: increases exponentially as TTC decreases
        normalized_ttc = ttc / self.params.ttc_critical
        force = self.params.ttc_weight * np.exp(-self.params.ttc_decay_rate * normalized_ttc)
        return force

    def compute_headway_force(self, headway: float) -> float:
        """Computes headway force component using exponential decay."""
        if headway >= self.params.headway_critical:
            return 0.0
        
        # Exponential force: increases exponentially as headway decreases
        normalized_headway = headway / self.params.headway_critical
        force = self.params.headway_weight * np.exp(-self.params.headway_decay_rate * normalized_headway)
        return force

    def compute_gap_error_force(self, gap_error: float, desired_gap: float) -> float:
        """Computes spacing error force component using linear computation."""
        # Only penalize when too close (negative gap_error)
        if np.abs(gap_error) < self.params.gap_error_threshold:
            return 0.0
        
        # # Linear force: proportional to the gap error magnitude
        # normalized_error = abs(gap_error) / max(desired_gap, 1.0)
        # force = self.params.gap_error_weight * normalized_error
        
        # Exponential force: increases exponentially as gap error increases
        normalized_error = abs(gap_error) / max(desired_gap, 1.0)
        force = self.params.gap_error_weight * (np.exp(self.params.gap_error_decay_rate * normalized_error) - 1.0)
        return force

    def compute_relative_velocity_force(self, relative_vel: float) -> float:
        """Computes relative velocity force component using linear computation."""
        # Only consider when closing in too fast
        relative_vel = np.abs(relative_vel)
        if relative_vel < self.params.rel_vel_threshold:
            return 0.0
        
        # # Linear force: proportional to excess relative velocity
        # excess_velocity = relative_vel - self.params.rel_vel_threshold
        # force = self.params.rel_vel_weight * excess_velocity
        
        # Exponential force: increases exponentially as relative velocity increases
        excess_velocity = relative_vel - self.params.rel_vel_threshold
        normalized_excess = excess_velocity / self.params.rel_vel_threshold
        force = self.params.rel_vel_weight * (np.exp(self.params.rel_vel_decay_rate * normalized_excess) - 1.0)
        return force

    def compute_risk_force_from_vehicle(self,
                                       ego_vehicle: Vehicle,
                                       other_vehicle: Vehicle,
                                       desired_gap: float,
                                       is_leader: bool = True) -> float:
        """
        Computes risk force from a single vehicle (leader or follower).
        
        Args:
            ego_vehicle: The ego Vehicle object
            other_vehicle: The other Vehicle object (leader or follower)
            desired_gap: desired gap distance
            is_leader: True if other vehicle is ahead, False if behind
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
        if np.abs(relative_vel) > 1e-3 and actual_gap != desired_gap:
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

        # --- Calculate individual force components ---
        force_ttc = self.compute_ttc_force(ttc)
        force_headway = self.compute_headway_force(headway)
        force_gap_error = self.compute_gap_error_force(gap_error, desired_gap)
        force_rel_vel = self.compute_relative_velocity_force(relative_vel)

        # Total risk force is the sum of all components
        total_force = force_ttc + force_headway + force_gap_error + force_rel_vel

        return max(0.0, total_force)  # Ensure non-negative

    def compute_risk_force(self,
                           ego_vehicle: Vehicle,
                           leader_vehicle: Optional[Vehicle],
                           desired_gap: float,
                           follower_vehicle: Optional[Vehicle] = None) -> float:
        """
        Computes the total longitudinal risk force considering both leader and follower.
        
        Args:
            ego_vehicle: The ego Vehicle object
            leader_vehicle: Leader Vehicle object (can be None)
            desired_gap: desired gap distance
            follower_vehicle: Follower Vehicle object (can be None)
        
        Returns:
            Total risk force (capped at max_force)
        """
        total_force = 0.0
        
        # Compute risk from leader
        if leader_vehicle is not None and desired_gap != float('inf'):
            force_from_leader = self.compute_risk_force_from_vehicle(
                ego_vehicle, leader_vehicle, desired_gap, is_leader=True
            )
            total_force += force_from_leader
        
        # Compute risk from follower
        if follower_vehicle is not None and desired_gap != float('inf'):
            force_from_follower = self.compute_risk_force_from_vehicle(
                ego_vehicle, follower_vehicle, desired_gap, is_leader=False
            )
            # Apply weight multiplier for follower risk
            total_force += force_from_follower * self.params.follower_weight

        return min(total_force, self.params.max_force)  # Cap at maximum

    def compute_risk_force_from_platoon(self,
                                       ego_vehicle: Vehicle,
                                       platoon_manager: 'PlatoonManager',
                                       desired_gap: float) -> float:
        """
        Computes risk force automatically from platoon manager.
        Automatically finds leader and follower vehicles.
        
        Args:
            ego_vehicle: The ego Vehicle object
            platoon_manager: PlatoonManager instance
            desired_gap: desired gap distance
            
        Returns:
            Total risk force (capped at max_force)
        """
        leader, follower = self.get_leader_and_follower(ego_vehicle, platoon_manager)
        return self.compute_risk_force(ego_vehicle, leader, desired_gap, follower)

    def get_force_breakdown(self,
                           ego_vehicle: Vehicle,
                           leader_vehicle: Optional[Vehicle],
                           desired_gap: float,
                           follower_vehicle: Optional[Vehicle] = None) -> dict:
        """Returns detailed breakdown of all force components for analysis."""
        breakdown = {
            'leader': {
                'ttc_force': 0.0,
                'headway_force': 0.0,
                'gap_error_force': 0.0,
                'rel_vel_force': 0.0,
                'total_force': 0.0,
                'ttc': float('inf'),
                'headway': float('inf'),
                'gap_error': 0.0,
                'relative_vel': 0.0
            },
            'follower': {
                'ttc_force': 0.0,
                'headway_force': 0.0,
                'gap_error_force': 0.0,
                'rel_vel_force': 0.0,
                'total_force': 0.0,
                'ttc': float('inf'),
                'headway': float('inf'),
                'gap_error': 0.0,
                'relative_vel': 0.0
            },
            'total_force': 0.0
        }

        pos_ego = ego_vehicle.state.x
        vel_ego = ego_vehicle.state.vx
        
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
                'ttc': ttc,
                'headway': headway,
                'gap_error': gap_error,
                'relative_vel': relative_vel
            }
            breakdown['leader']['total_force'] = sum([
                breakdown['leader']['ttc_force'],
                breakdown['leader']['headway_force'],
                breakdown['leader']['gap_error_force'],
                breakdown['leader']['rel_vel_force']
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
                'ttc': ttc,
                'headway': headway,
                'gap_error': gap_error,
                'relative_vel': relative_vel
            }
            breakdown['follower']['total_force'] = sum([
                breakdown['follower']['ttc_force'],
                breakdown['follower']['headway_force'],
                breakdown['follower']['gap_error_force'],
                breakdown['follower']['rel_vel_force']
            ]) * self.params.follower_weight

        # Total force
        breakdown['total_force'] = min(
            breakdown['leader']['total_force'] + breakdown['follower']['total_force'],
            self.params.max_force
        )

        return breakdown

    def get_force_breakdown_from_platoon(self,
                                        ego_vehicle: Vehicle,
                                        platoon_manager: 'PlatoonManager',
                                        desired_gap: float) -> dict:
        """
        Get force breakdown automatically from platoon manager.
        Automatically finds leader and follower vehicles.
        
        Args:
            ego_vehicle: The ego Vehicle object
            platoon_manager: PlatoonManager instance
            desired_gap: desired gap distance
            
        Returns:
            Detailed breakdown dictionary
        """
        leader, follower = self.get_leader_and_follower(ego_vehicle, platoon_manager)
        return self.get_force_breakdown(ego_vehicle, leader, desired_gap, follower)

def run_example_demonstration():
    """Demonstrates the longitudinal safety field with platoon integration."""
    print("\n" + "="*70)
    print("🚗 LONGITUDINAL SAFETY FIELD - PLATOON INTEGRATION DEMONSTRATION")
    print("="*70)
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Import here to avoid circular dependency
    from control.platoon_control import PlatoonManager
    
    # Initialize safety field
    safety_field = LongitudinalSafetyField()
    desired_gap = 30.0  # meters
    
    # Create a simple platoon
    vehicles = [
        Vehicle(initial_x=0.0, initial_velocity=20.0, vehicle_id="Leader"),
        Vehicle(initial_x=50.0, initial_velocity=20.0, vehicle_id="Vehicle_2"),
        Vehicle(initial_x=100.0, initial_velocity=20.0, vehicle_id="Vehicle_3"),
    ]
    
    platoon_manager = PlatoonManager(vehicles)
    
    print(f"Platoon created with {len(vehicles)} vehicles\n")
    print(f"Desired Gap: {desired_gap:.1f} m\n")
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'Check Vehicle_2 (middle vehicle)',
            'ego_id': 'Vehicle_2',
            'description': 'Should have both leader and follower'
        },
        {
            'name': 'Check Leader (front vehicle)',
            'ego_id': 'Leader',
            'description': 'Should have only follower'
        },
        {
            'name': 'Check Vehicle_3 (rear vehicle)',
            'ego_id': 'Vehicle_3',
            'description': 'Should have only leader'
        },
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"{i}. {scenario['name']:40}")
        print(f"   {scenario['description']}")
        print("-" * 70)
        
        # Find ego vehicle
        ego_vehicle = next(v for v in vehicles if v.vehicle_id == scenario['ego_id'])
        
        # Get leader and follower automatically
        leader, follower = safety_field.get_leader_and_follower(ego_vehicle, platoon_manager)
        
        print(f"   EGO:      {ego_vehicle.vehicle_id:10} | Pos={ego_vehicle.state.x:6.1f}m, Vel={ego_vehicle.state.vx:5.1f}m/s")
        print(f"   LEADER:   {leader.vehicle_id if leader else 'None':10} | Pos={leader.state.x if leader else 'N/A':>6}m, Vel={leader.state.vx if leader else 'N/A':>5}m/s")
        print(f"   FOLLOWER: {follower.vehicle_id if follower else 'None':10} | Pos={follower.state.x if follower else 'N/A':>6}m, Vel={follower.state.vx if follower else 'N/A':>5}m/s")
        
        # Get risk force using platoon integration
        total_force = safety_field.compute_risk_force_from_platoon(ego_vehicle, platoon_manager, desired_gap)
        breakdown = safety_field.get_force_breakdown_from_platoon(ego_vehicle, platoon_manager, desired_gap)
        
        print(f"\n   📍 LEADER RISK:   Force = {breakdown['leader']['total_force']:8.2f}N")
        print(f"   📍 FOLLOWER RISK: Force = {breakdown['follower']['total_force']:8.2f}N")
        print(f"   🎯 TOTAL RISK:    Force = {total_force:8.2f}N\n")

    print("="*70)
    print("✅ Platoon integration demonstration completed!")
    print("   - Automatically finds leader and follower from platoon")
    print("   - Uses get_new_vehicle_index for vehicle positioning")
    print("   - Handles edge cases (front/rear vehicles)")
    print("="*70)

__all__=['LongitudinalSafetyField', 'LongitudinalSafetyFieldParams']

if __name__ == "__main__":
    run_example_demonstration()

