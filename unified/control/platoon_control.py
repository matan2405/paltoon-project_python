"""
unified/control/platoon_control.py — Platoon control for unified module.

Copied from Longitudinal/control/platoon_control.py with all imports
redirected to unified/config.py and unified/vehicle/components.py.
No dependency on the split Longitudinal/ or lateral/ module trees.
"""

import numpy as np
import time
from typing import List
from scipy.integrate import odeint
from unified.vehicle.components import VehicleParameters
from unified.config import (
    JERK_LIMIT, RAJAMANI_H, RAJAMANI_TAU,
    RAJAMANI_K1, RAJAMANI_K2, RAJAMANI_K3, RAJAMANI_K4, RAJAMANI_K5,
    PLATOON_TARGET_VELOCITY, FREE_ROAD_DELTA, PLATOON_STANDSTILL_DISTANCE,
)

_vp = VehicleParameters()


def rajamani(Car_1, Car_2):
    """Compute Rajamani acceleration command and desired inter-vehicle gap.

    Gap error definition includes standstill distance d0 so that the
    equilibrium bumper-to-bumper gap equals d0 + h*v — consistent with
    the Nash reference generator (PLATOON_STANDSTILL_DISTANCE + h*vx).
    The gain ratio k5/k4 = h is unchanged, so string stability holds.
    """
    h  = RAJAMANI_H
    d0 = PLATOON_STANDSTILL_DISTANCE

    k1, k5 = RAJAMANI_K1, RAJAMANI_K5
    k2, k3, k4 = RAJAMANI_K2, RAJAMANI_K3, RAJAMANI_K4

    e = Car_2.state.x - Car_1.state.x + Car_1.L + d0   # equilibrium at -(h*v)
    e_dot = Car_2.v - Car_1.v

    s_des = Car_1.L + d0 + h * Car_2.v   # desired centre-to-centre spacing
    a_des = -k1*Car_1.a - k2*Car_2.a - k3*e_dot - k4*e - k5*Car_2.v

    return a_des, s_des


def free_road_acc(v, t, v_target, a_max, delta=FREE_ROAD_DELTA):
    """Compute IDM-like free-road acceleration with exponent delta."""
    if v_target >= v:
        dv_dt = a_max * (1 - (v / v_target) ** delta)
    else:
        dv_dt = -a_max * (1 - (v_target / v) ** delta)
    return dv_dt


class PlatoonManager:
    """Manages autonomous platoon behavior using Rajamani controller."""

    def __init__(self, vehicles: List, use_state_space: bool = False):
        self.vehicles = vehicles
        self.target_velocity = PLATOON_TARGET_VELOCITY
        self.max_velocity = _vp.max_velocity
        self.max_acceleration = _vp.max_acceleration

        self.use_state_space = use_state_space

        self.actual_gaps_history = [[] for _ in range(len(vehicles) - 1)]
        self.desired_gaps_history = [[] for _ in range(len(vehicles) - 1)]

        self._a_prev = {}

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
        """Update platoon control — Rajamani + Nash injection."""
        if not self.vehicles:
            return

        Car_1 = self.vehicles[0]

        if not is_prediction_mode and Car_1.nash_acceleration is not None:
            lead_acc = Car_1.nash_acceleration
            Car_1.nash_acceleration = None
            new_velocity_leader = Car_1.v + lead_acc * dt
            new_velocity_leader = np.clip(new_velocity_leader, 0, Car_1.max_velocity)
        else:
            new_velocity_leader = odeint(
                free_road_acc, Car_1.v, [0, dt],
                args=(self.target_velocity, self.max_acceleration))[-1][0]
            new_velocity_leader = np.clip(new_velocity_leader, 0, Car_1.max_velocity)
            lead_acc = (new_velocity_leader - Car_1.v) / dt if dt > 0 else 0

        lead_a_max = (Car_1.get_dynamic_max_acceleration()
                      if hasattr(Car_1, 'get_dynamic_max_acceleration')
                      else self.max_acceleration)
        lead_acc = np.clip(lead_acc, -self.max_acceleration, lead_a_max)
        lead_acc = self._jerk_limit(Car_1.vehicle_id, lead_acc, dt)

        Car_1.a_desired = lead_acc
        Car_1.update_dynamics(dt)

        for car_num in range(1, len(self.vehicles)):
            leader = self.vehicles[car_num - 1]
            follower = self.vehicles[car_num]

            actual_gap = leader.state.x - follower.state.x

            while len(self.actual_gaps_history) < len(self.vehicles) - 1:
                self.actual_gaps_history.append([])
            while len(self.desired_gaps_history) < len(self.vehicles) - 1:
                self.desired_gaps_history.append([])

            self.actual_gaps_history[car_num - 1].append(actual_gap)

            if not is_prediction_mode and follower.nash_acceleration is not None:
                a_des = follower.nash_acceleration
                follower.nash_acceleration = None
                s_des = leader.L + RAJAMANI_H * follower.v
                self.desired_gaps_history[car_num - 1].append(s_des)
            else:
                a_des, s_des = rajamani(leader, follower)
                self.desired_gaps_history[car_num - 1].append(s_des)

            follower_a_max = (follower.get_dynamic_max_acceleration()
                              if hasattr(follower, 'get_dynamic_max_acceleration')
                              else self.max_acceleration)
            a_des = np.clip(a_des, -self.max_acceleration, follower_a_max)
            a_des = self._jerk_limit(follower.vehicle_id, a_des, dt)

            follower.a_desired = a_des
            follower.update_dynamics(dt)

    def add_vehicle(self, vehicle):
        """Add a vehicle to the platoon."""
        vehicle.autonomous_mode = True
        vehicle.joined_platoon = True
        vehicle.joined_time = time.time()
        new_index = self.get_new_vehicle_index(vehicle)
        vehicle.target_velocity = self.target_velocity
        vehicle.max_acceleration = self.max_acceleration
        self.vehicles.insert(new_index, vehicle)

        if new_index == 0:
            self.actual_gaps_history.insert(0, [])
            self.desired_gaps_history.insert(0, [])
        elif new_index == len(self.vehicles) - 1:
            self.actual_gaps_history.append([])
            self.desired_gaps_history.append([])
        else:
            self.actual_gaps_history.insert(new_index, [])
            self.desired_gaps_history.insert(new_index, [])

    def get_new_vehicle_index(self, vehicle) -> int:
        """Get position for a new vehicle joining the platoon."""
        if not self.vehicles:
            return 0

        for i in range(len(self.vehicles)):
            if vehicle.state.x > self.vehicles[i].state.x:
                return i
            if vehicle.vehicle_id == self.vehicles[i].vehicle_id:
                return i
        return len(self.vehicles)


__all__ = ['PlatoonManager', 'rajamani', 'free_road_acc']
