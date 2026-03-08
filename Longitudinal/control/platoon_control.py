"""
Platoon control module containing the PlatoonManager class and control algorithms.
Implements Rajamani controller and free road acceleration functions.
"""

import numpy as np
import time
from typing import List
from scipy.integrate import odeint
from vehicle.vehicle import Vehicle, VehicleParameters
from config import JERK_LIMIT,RAJAMANI_H,RAJAMANI_TAU

_vp = VehicleParameters()
def rajamani(Car_1, Car_2):
    """Rajamani controller (kept for desired-gap calculations in safety field)."""
    h = RAJAMANI_H   # [s] desired time headway
    tau = RAJAMANI_TAU # [s] time lag

    k1, k5 = -0.12, 0.1 # k1 < -tau/h, k5 > 0
    k2, k3, k4 = -k1-h*k1*k5, 1/h-k1*k5, k5/h

    e = Car_2.state.x - Car_1.state.x + Car_1.L  # [m] actual gap
    e_dot = Car_2.v - Car_1.v         # [m/s] relative velocity

    s_des = Car_1.L + h * Car_2.v # [m] desired gap
    a_des = -k1*Car_1.a - k2*Car_2.a - k3*e_dot - k4*e - k5*Car_2.v

    return a_des, s_des


def free_road_acc(v, t, v_target, a_max,delta=4):
    """Free road acceleration - identical to platoon_control.py"""

    if (v_target >= v):
        dv_dt = a_max * (1 - (v/v_target)**delta)
    else:
        dv_dt = -a_max * (1 - (v_target/v)**delta)
    return dv_dt


class PlatoonManager:
    """Manages autonomous platoon behavior - using exact platoon_control.py algorithms"""
    
    def __init__(self, vehicles: List[Vehicle], use_state_space: bool = False):
        self.vehicles = vehicles
        self.target_velocity = 120.0 / 3.6  # 120 km/h in m/s (matching platoon_control)
        self.max_velocity = _vp.max_velocity  # m/s (matching platoon_control.py)
        self.max_acceleration = _vp.max_acceleration  # m/s^2 (matching platoon_control.py)
        
        self.use_state_space = use_state_space  # Whether to use state-space model
        
        # Data storage like platoon_control.py
        self.actual_gaps_history = [[] for _ in range(len(vehicles)-1)]
        self.desired_gaps_history = [[] for _ in range(len(vehicles)-1)]

        # Jerk limiting: store previous a_desired per vehicle id
        self._a_prev = {}

        # Set all vehicles to autonomous mode
        for vehicle in vehicles:
            vehicle.autonomous_mode = True


    def _jerk_limit(self, vehicle_id, a_des, dt):
        """Apply jerk limiting: clip da/dt to JERK_LIMIT."""
        a_prev = self._a_prev.get(vehicle_id, a_des)
        da_max = JERK_LIMIT * dt
        a_limited = a_prev + np.clip(a_des - a_prev, -da_max, da_max)
        self._a_prev[vehicle_id] = a_limited
        return a_limited

    def update(self, dt: float, is_prediction_mode: bool = False):
        """Update platoon control - implementing exact algorithm from platoon_control.py"""
        if not self.vehicles:
            return
            
        # Leader vehicle - free road acceleration (exactly like platoon_control.py)
        Car_1 = self.vehicles[0]  # Lead vehicle
        
        # Check if leader has Nash acceleration override
        # if hasattr(Car_1, 'nash_acceleration') and Car_1.nash_acceleration is not None:
        if not is_prediction_mode  and not Car_1.nash_acceleration is None:
            lead_acc = Car_1.nash_acceleration
            Car_1.nash_acceleration = None  # Reset for next iteration
            new_velocity_leader = Car_1.v + lead_acc * dt
            new_velocity_leader = np.clip(new_velocity_leader, 0,  Car_1.max_velocity)
        else:
            # Normal free road acceleration (exactly like platoon_control.py)
            new_velocity_leader = odeint(free_road_acc, Car_1.v, [0, dt], args=(self.target_velocity, self.max_acceleration))[-1][0]
            new_velocity_leader = np.clip(new_velocity_leader, 0,  Car_1.max_velocity)  # velocity limits
            lead_acc = (new_velocity_leader - Car_1.v) / dt if dt > 0 else 0

        # Clamp to vehicle's physical capability
        lead_a_max = Car_1.get_dynamic_max_acceleration() if hasattr(Car_1, 'get_dynamic_max_acceleration') else self.max_acceleration
        lead_acc = np.clip(lead_acc, -self.max_acceleration, lead_a_max)

        # Jerk-limit leader acceleration
        lead_acc = self._jerk_limit(Car_1.vehicle_id, lead_acc, dt)

        Car_1.a_desired = lead_acc
        # Update leader using platoon kinematic model
        Car_1.update_dynamics(dt)
            # Car_1.update_dynamics_state_space(dt)
        # else:
        #     delta_f = Car_1.state.delta_f  # front wheel steering angle (straight line)
        #     delta_r = Car_1.state.delta_r  # rear wheel steering angle
        #     Car_1.state_equation_platoon(delta_f, delta_r, new_velocity_leader, dt)
        
        # Following vehicles - Rajamani controller (exactly like platoon_control.py)
        for car_num in range(1, len(self.vehicles)):
            leader = self.vehicles[car_num-1]
            follower = self.vehicles[car_num]
            
            # Calculate actual gap like platoon_control.py
            actual_gap = leader.state.x - follower.state.x 
            
            # Ensure we have enough history arrays
            while len(self.actual_gaps_history) < len(self.vehicles) - 1:
                self.actual_gaps_history.append([])
            while len(self.desired_gaps_history) < len(self.vehicles) - 1:
                self.desired_gaps_history.append([])
                
            self.actual_gaps_history[car_num-1].append(actual_gap)
            
            # Check if follower has Nash acceleration override
            # if hasattr(follower, 'nash_acceleration') and follower.nash_acceleration is not None:
            if not is_prediction_mode  and not follower.nash_acceleration is None:
                a_des = follower.nash_acceleration
                follower.nash_acceleration = None  # Reset for next iteration
                # Calculate desired gap for history (even when using Nash)
                s_des = leader.L + 1.5 * follower.v  # h=1.5s
                self.desired_gaps_history[car_num-1].append(s_des)
            else:
                # Apply Rajamani control law - exactly like platoon_control.py
                a_des, s_des = rajamani(leader, follower)
                self.desired_gaps_history[car_num-1].append(s_des)

            # Clamp to vehicle's physical capability (engine-dependent upper limit)
            follower_a_max = follower.get_dynamic_max_acceleration() if hasattr(follower, 'get_dynamic_max_acceleration') else self.max_acceleration
            a_des = np.clip(a_des, -self.max_acceleration, follower_a_max)

            # Jerk-limit follower acceleration
            a_des = self._jerk_limit(follower.vehicle_id, a_des, dt)
            
            follower.a_desired = a_des
            # Update follower vehicle
            follower.update_dynamics(dt)
            # if follower.use_state_space_model:
            #     follower.update_dynamics_state_space(dt)
            # else:
            #     # Update velocity with acceleration constraint like platoon_control.py
            #     delta_f = follower.state.delta_f  # front wheel steering angle (straight line)
            #     delta_r = follower.state.delta_r  # rear wheel steering angle
            #     new_velocity_follower = follower.v + a_des * dt
            #     new_velocity_follower = np.clip(new_velocity_follower, 0, follower.max_velocity)  # velocity limits
            #     # Update vehicle state using platoon kinematic model
            #     follower.state_equation_platoon(delta_f, delta_r, new_velocity_follower, dt)
    
    def add_vehicle(self, vehicle):
        """Add a vehicle to the platoon"""
        vehicle.autonomous_mode = True
        vehicle.joined_platoon = True
        vehicle.joined_time = time.time()
        new_index = self.get_new_vehicle_index(vehicle)
        vehicle.target_velocity = self.target_velocity
        vehicle.max_acceleration = self.max_acceleration
        self.vehicles.insert(new_index, vehicle)
        
        # Fix history arrays
        if new_index == 0:
            # Vehicle joined at the front - add new gap at beginning
            self.actual_gaps_history.insert(0, [])
            self.desired_gaps_history.insert(0, [])
        elif new_index == len(self.vehicles) - 1:
            # Vehicle joined at the end - add new gap at end  
            self.actual_gaps_history.append([])
            self.desired_gaps_history.append([])
        else:
            # Vehicle joined in middle - add new gap at the position
            self.actual_gaps_history.insert(new_index, [])
            self.desired_gaps_history.insert(new_index, [])

    def get_new_vehicle_index(self, vehicle) -> int:
        """Get position for a new vehicle joining the platoon"""
        if not self.vehicles:
            print("No vehicles in platoon, adding at index 0")
            return 0
        
        # Find the right position based on x coordinate
        for i in range(len(self.vehicles)):
            if vehicle.state.x > self.vehicles[i].state.x:
                # print(f"New vehicle joining platoon at index {i}")
                return i
        
            if vehicle.vehicle_id == self.vehicles[i].vehicle_id:
                # Vehicle already in platoon
                return i
        # If we get here, vehicle goes at the end
        # print(f"New vehicle joining platoon at the end (index {len(self.vehicles)})")
        return len(self.vehicles)


__all__ = [
    'PlatoonManager',
    'rajamani', 
    'free_road_acc'
]