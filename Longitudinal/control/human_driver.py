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
from config import SIMULATION_DT, NASH_NP, NASH_DRIVER_PARAMS, PLATOON_TARGET_VELOCITY
from vehicle.vehicle import Vehicle


class HumanDriver:
    def __init__(self, vehicle, target_speed: float = PLATOON_TARGET_VELOCITY, dt: float = SIMULATION_DT, driver_type='normal'):
        self.vehicle = vehicle
        
        # Standard Parameters (Execution)
        self.driver_type = driver_type
        profile = NASH_DRIVER_PARAMS.get(driver_type, NASH_DRIVER_PARAMS['normal'])
        self.max_acceleration = profile['max_acceleration']
        self.max_deceleration = profile['max_deceleration']
        self.desired_time_headway = profile['desired_time_headway']
        self.min_spacing = profile['min_spacing']
        self.delta_IDM = profile['delta_IDM']
        self.comfortable_deceleration = profile['comfortable_deceleration']
        self.target_speed = target_speed + profile['target_speed_offset'] / 3.6  # Convert km/h offset to m/s
        
        # Planning Parameters (More tolerant for Nash prediction)
        self.plan_time_headway = profile['plan_time_headway']
        self.plan_decel = profile['plan_decel']
        
        self.dt = dt
        self.merging = False
        self.lane_change_progress = 0.0
        self.ignore_platoon_before_merge = True

        # Startup ramp: gradually increase max_acceleration from 0 to full over ramp_duration
        # A real driver doesn't floor the gas from standstill
        self._startup_time = 0.0
        self._startup_ramp_duration = 3.0  # seconds

        
    def set_motion_model(self, use_kinematic: bool, use_state_space: bool = False,
                         use_hierarchical: bool = False):
        self.vehicle.set_motion_model(use_kinematic, use_state_space,
                                      use_hierarchical=use_hierarchical)
        
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

        # Startup ramp (execution only): gradually increase effective max_acceleration
        # from 0 to full over _startup_ramp_duration seconds — a real driver doesn't
        # instantly floor the gas from standstill.
        if mode == 'execution':
            self._startup_time += dt
            ramp = min(self._startup_time / self._startup_ramp_duration, 1.0)
            effective_max_accel = self.max_acceleration * ramp
        else:
            effective_max_accel = self.max_acceleration

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
                      (v * delta_v) / (2 * np.sqrt(effective_max_accel * b)))
            s = max(0.1, s)
            interaction_term = - (s_star / s) ** 2

        # Total Accel
        desired_accel = effective_max_accel * (free_road_term + interaction_term)

        # Constraints
        desired_accel = np.clip(desired_accel, self.max_deceleration, effective_max_accel)
        
        if mode == 'execution':
            # Apply to vehicle (Throttle/Brake logic)
            if desired_accel > 0.1:
                throttle = np.clip(desired_accel / self.max_acceleration, 0, 1)
                brake = 0.0
            elif desired_accel < -0.1:
                throttle = 0.0
                brake = np.clip(abs(desired_accel) / abs(self.max_deceleration), 0, 1)
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
        
        # Switch cloned vehicle to state-space (double integrator) mode.
        # The original may be hierarchical, but for prediction we use the
        # simple ZOH double integrator — not the full engine dynamics.
        sim_veh.use_hierarchical_model = False
        sim_veh.use_state_space_model = True
        sim_veh.use_kinematic_model = False
        
        sim_driver = copy.deepcopy(self)
        sim_driver.vehicle = sim_veh
        
        # === NEW: Create simulated leader if provided ===
        sim_leader = None
        if leader is not None:
            sim_leader = copy.deepcopy(leader)
        
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
            
            # Propagate state using vehicle's double integrator (planning model)
            # update_dynamics() routes to update_dynamics_state_space() which
            # uses ZOH discretization: x[k+1] = A_d @ x[k] + B_d @ u[k]
            sim_veh.a_desired = a_human
            sim_veh.update_dynamics(dt)
            
            accel_sequence[i] = a_human
            state_sequence[i, 0] = sim_veh.state.x
            state_sequence[i, 1] = sim_veh.state.vx
            
        return accel_sequence, state_sequence


__all__ = ['HumanDriver']