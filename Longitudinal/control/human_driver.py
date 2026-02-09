#!/usr/bin/env python3
"""
Human driver module containing the HumanDriver class.
VERSION 2.0 - LEADER-AWARE PREDICTION

Key Fix:
========
The get_human_acceleration_and_state_sequence() method now accepts an optional
leader parameter. When provided, the human driver prediction will use IDM
car-following behavior instead of free-road driving.

This fixes the "Join Middle" and "Join After" scenarios where the human driver
prediction was ignoring the leader, causing the Nash solver to receive conflicting
references (System wants to close gap, Human wants to accelerate to target speed).

The fix ensures R2_ref (human reference) is consistent with actual car-following
behavior, leading to cooperative Nash equilibrium solutions.
"""

import numpy as np
from typing import List, Tuple, Optional
import sys
import copy
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vehicle.vehicle import Vehicle


class HumanDriver:
    def __init__(self, vehicle, target_speed: float = 120.0 / 3.6, dt: float = 0.1):
        self.vehicle = vehicle
        self.target_speed = target_speed
        
        # Standard Parameters (Execution)
        self.max_acceleration = 2.5
        self.desired_time_headway = 1.2
        self.min_spacing = 2.0
        self.delta_IDM = 4.0
        self.comfortable_deceleration = 2.0
        
        # Planning Parameters (More tolerant for Nash prediction)
        self.plan_time_headway = 0.8  # Shorter headway for planning
        self.plan_decel = 4.0  # Harder braking allowed in planning
        
        self.dt = dt
        self.merging = False
        self.lane_change_progress = 0.0
        self.ignore_platoon_before_merge = True

        
    def set_motion_model(self, use_kinematic: bool, use_state_space: bool = False):
        self.vehicle.set_motion_model(use_kinematic, use_state_space)
        
    def _get_leader(self, platoon_vehicles: List):
        if not platoon_vehicles: 
            return None
        my_x = self.vehicle.state.x
        min_dist = float('inf')
        leader = None
        for v in platoon_vehicles:
            if v.vehicle_id == self.vehicle.vehicle_id: 
                continue
            dist = v.state.x - my_x
            if 0 < dist < min_dist:
                min_dist = dist
                leader = v
        return leader

    def update(self, dt: float, platoon_vehicles: List, mode='execution'):
        """
        Update using IDM logic.
        mode: 'execution' (real driving) or 'planning' (prediction)
        """
        if self.ignore_platoon_before_merge and not self.merging:
            # Before t_merge: drive as if alone
            leader = None
        else:
            # After t_merge: normal IDM with platoon awareness
            leader = self._get_leader(platoon_vehicles)
        v = self.vehicle.state.vx
        v0 = self.target_speed
        
        # Choose parameters based on mode
        T = self.desired_time_headway if mode == 'execution' else self.plan_time_headway
        b = self.comfortable_deceleration if mode == 'execution' else self.plan_decel
        
        # Free road term
        if v0 > 0.1:
            free_road_term = 1 - (v / v0) ** self.delta_IDM
        else:
            free_road_term = 0
            
        # Interaction term
        interaction_term = 0.0
        if leader:
            s = leader.state.x - self.vehicle.state.x - leader.L
            delta_v = v - leader.state.vx
            s_star = (self.min_spacing + v * T + 
                      (v * delta_v) / (2 * np.sqrt(self.max_acceleration * b)))
            s = max(0.1, s)
            interaction_term = - (s_star / s) ** 2
        
        # Total Accel
        desired_accel = self.max_acceleration * (free_road_term + interaction_term)
        
        # Constraints
        desired_accel = np.clip(desired_accel, -4.0, 2.5)
        
        if mode == 'execution':
            # Apply to vehicle (Throttle/Brake logic)
            if desired_accel > 0.1:
                throttle = np.clip(desired_accel / self.max_acceleration, 0, 1)
                brake = 0.0
            elif desired_accel < -0.1:
                throttle = 0.0
                brake = np.clip(abs(desired_accel) / 4.0, 0, 1)
            else:
                throttle = 0.0
                brake = 0.0
                
            steering = 0.0
            if self.merging and self.lane_change_progress < 1.0:
                self.lane_change_progress += dt * 0.25
            self.vehicle.set_manual_inputs(throttle, brake, steering)
            
        return desired_accel

    def get_human_acceleration_and_state_sequence(
        self, 
        dt: float, 
        Np: int, 
        vehicle: Vehicle,
        leader: Optional[Vehicle] = None  # NEW: Optional leader for gap-aware prediction
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predicted acceleration and state sequence for the human driver.
        
        VERSION 2.0: Now accepts an optional leader for gap-aware prediction.
        
        Args:
            dt: Time step for prediction
            Np: Number of prediction steps
            vehicle: Current ego vehicle state
            leader: Optional leader vehicle for car-following prediction
            
        Returns:
            Tuple of (acceleration_sequence, state_sequence)
            - acceleration_sequence: Shape (Np,)
            - state_sequence: Shape (Np, 2) with [position, velocity]
        """
        accel_sequence = np.zeros(Np)
        state_sequence = np.zeros((Np, 2))
        
        sim_veh = copy.deepcopy(vehicle)
        sim_driver = copy.deepcopy(self)
        sim_driver.vehicle = sim_veh
        
        # === NEW: Create simulated leader if provided ===
        sim_leader = None
        if leader is not None:
            sim_leader = copy.deepcopy(leader)
        
        dt_sim = 0.02
        steps_per_dt = max(1, int(dt / dt_sim))
        
        for i in range(Np):
            # === NEW: If we have a leader, update its position (constant velocity assumption) ===
            if sim_leader is not None:
                # Leader moves at constant velocity during prediction
                sim_leader.state.x += sim_leader.state.vx * dt
            
            # Predict using PLANNING mode
            # Pass the leader in a list if available, empty list otherwise
            if sim_leader is not None:
                platoon_list = [sim_leader]
                # Force merging=True so IDM uses the leader
                old_merging = sim_driver.merging
                old_ignore = sim_driver.ignore_platoon_before_merge
                sim_driver.merging = True
                sim_driver.ignore_platoon_before_merge = False
                
                a_human = sim_driver.update(dt, platoon_vehicles=platoon_list, mode='planning')
                
                sim_driver.merging = old_merging
                sim_driver.ignore_platoon_before_merge = old_ignore
            else:
                # No leader - free road driving
                a_human = sim_driver.update(dt, platoon_vehicles=[], mode='planning')
            
            sim_veh.a_desired = a_human
            for _ in range(steps_per_dt):
                sim_veh.update_dynamics(dt_sim)
            
            accel_sequence[i] = a_human
            state_sequence[i, 0] = sim_veh.state.x
            state_sequence[i, 1] = sim_veh.state.vx
            
        return accel_sequence, state_sequence

    def get_human_acceleration_and_state_prediction(
        self, 
        dt: float, 
        Np: int, 
        vehicle: Vehicle,
        leader: Optional[Vehicle] = None
    ) -> np.ndarray:
        """Get single acceleration prediction (last step)."""
        acc, _ = self.get_human_acceleration_and_state_sequence(dt, Np, vehicle, leader)
        return acc[-1] if len(acc) > 0 else 0.0


__all__ = ['HumanDriver']