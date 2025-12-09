#!/usr/bin/env python3
"""
Human driver module containing the HumanDriver class.
Updated: Adds 'planning' mode for more tolerant predictions in Nash.
"""

import numpy as np
from typing import List, Tuple
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
        self.plan_time_headway = 0.8 # Shorter headway for planning
        self.plan_decel = 4.0 # Harder braking allowed in planning
        
        self.dt = dt
        self.merging = False
        self.lane_change_progress = 0.0
        
    def set_motion_model(self, use_kinematic: bool, use_state_space: bool = False):
        self.vehicle.set_motion_model(use_kinematic, use_state_space)
        
    def _get_leader(self, platoon_vehicles: List):
        if not platoon_vehicles: return None
        my_x = self.vehicle.state.x
        min_dist = float('inf')
        leader = None
        for v in platoon_vehicles:
            if v.vehicle_id == self.vehicle.vehicle_id: continue
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
            s_star = (self.min_spacing + v * T + (v * delta_v) / (2 * np.sqrt(self.max_acceleration * b)))
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

    def get_human_acceleration_and_state_sequence(self, dt: float, Np: int, vehicle: Vehicle) -> Tuple[np.ndarray, np.ndarray]:
        accel_sequence = np.zeros(Np)
        state_sequence = np.zeros((Np, 2))
        
        sim_veh = copy.deepcopy(vehicle)
        sim_driver = copy.deepcopy(self)
        sim_driver.vehicle = sim_veh
        
        dt_sim = 0.02
        steps_per_dt = max(1, int(dt / dt_sim))
        
        for i in range(Np):
            # Predict using PLANNING mode (more aggressive/tolerant)
            # We pass empty list for leader to simulate "optimistic" human intent (trying to maintain speed)
            # OR we could pass the leader if we had it, but with 'planning' parameters it won't brake as early.
            a_human = sim_driver.update(dt, platoon_vehicles=[], mode='planning') 
            
            sim_veh.a_desired = a_human
            for _ in range(steps_per_dt):
                sim_veh.update_dynamics(dt_sim)
            
            accel_sequence[i] = a_human
            state_sequence[i, 0] = sim_veh.state.x
            state_sequence[i, 1] = sim_veh.state.vx
            
        return accel_sequence, state_sequence

    def get_human_acceleration_and_state_prediction(self, dt: float, Np: int, vehicle: Vehicle) -> np.ndarray:
        acc, _ = self.get_human_acceleration_and_state_sequence(dt, Np, vehicle)
        return acc[-1] if len(acc) > 0 else 0.0

__all__ = ['HumanDriver']