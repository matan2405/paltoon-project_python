"""
Platoon Control Module.
VERSION 2.0
"""

import numpy as np
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class PlatoonParams:
    time_gap: float = 1.5
    standstill_distance: float = 5.0
    target_velocity: float = 20.0
    vehicle_length: float = 4.5
    min_merge_gap: float = 15.0
    platoon_lane_y: float = 0.0


class PlatoonVehicle:
    def __init__(self, vehicle_id: str, initial_x: float, initial_y: float = 0.0, 
                 initial_vx: float = 20.0):
        self.vehicle_id = vehicle_id
        self.x = initial_x
        self.y = initial_y
        self.vx = initial_vx
        self.ax = 0.0
        self.L = 4.5
        
        class State:
            pass
        self.state = State()
        self.state.x = self.x
        self.state.y = self.y
        self.state.vx = self.vx
    
    def update(self, dt: float, acceleration: float = 0.0):
        self.ax = acceleration
        self.vx += self.ax * dt
        self.vx = max(0.0, self.vx)
        self.x += self.vx * dt
        self.state.x = self.x
        self.state.vx = self.vx
    
    def to_dict(self) -> Dict:
        return {'x': self.x, 'y': self.y, 'vx': self.vx, 'id': self.vehicle_id}


class PlatoonManager:
    def __init__(self, params: PlatoonParams = None):
        self.params = params or PlatoonParams()
        self.vehicles: List[PlatoonVehicle] = []
        self.target_velocity = self.params.target_velocity
        self.human_vehicle = None
        self.human_merged = False
        
        print(f"🚛 Platoon Manager Initialized (v={self.target_velocity*3.6:.0f} km/h)")
    
    def add_platoon_vehicle(self, vehicle_id: str, initial_x: float) -> PlatoonVehicle:
        vehicle = PlatoonVehicle(vehicle_id, initial_x, self.params.platoon_lane_y, 
                                  self.target_velocity)
        self.vehicles.append(vehicle)
        self.vehicles.sort(key=lambda v: -v.x)
        return vehicle
    
    def create_platoon(self, num_vehicles: int, leader_x: float, gap: float = None):
        if gap is None:
            gap = self.params.time_gap * self.target_velocity + self.params.standstill_distance
        self.vehicles.clear()
        for i in range(num_vehicles):
            x_pos = leader_x - i * (gap + self.params.vehicle_length)
            self.add_platoon_vehicle(f"P{i+1}", x_pos)
        print(f"🚛 Created platoon: {num_vehicles} vehicles, gap={gap:.1f}m")
    
    def get_vehicles_as_obstacles(self) -> List[Dict]:
        return [v.to_dict() for v in self.vehicles]
    
    def update(self, dt: float):
        for i, vehicle in enumerate(self.vehicles):
            if i == 0:
                v_error = self.target_velocity - vehicle.vx
                a = 0.5 * v_error
            else:
                leader = self.vehicles[i-1]
                d_des = self.params.vehicle_length + self.params.time_gap * vehicle.vx + self.params.standstill_distance
                d_actual = leader.x - vehicle.x
                e_d = d_actual - d_des
                e_v = leader.vx - vehicle.vx
                a = 0.3 * e_d + 0.8 * e_v
            vehicle.update(dt, np.clip(a, -3.0, 2.0))
    
    def add_human_vehicle(self, human_vehicle):
        self.human_vehicle = human_vehicle
    
    def complete_merge(self):
        self.human_merged = True
        print(f"✅ Merge complete!")
    
    def reset(self):
        self.vehicles.clear()
        self.human_vehicle = None
        self.human_merged = False


__all__ = ['PlatoonManager', 'PlatoonVehicle', 'PlatoonParams']
