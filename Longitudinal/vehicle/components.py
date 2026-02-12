"""
Vehicle components module containing the basic building blocks for vehicle simulation.
Includes VehicleParameters, Engine, Transmission, and VehicleState classes.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class VehicleParameters:
    """Audi TT Coupé 2.0 TFSI (230PS) parameters - from official technical data"""
    # Vehicle dimensions
    mass: float = 1305.0  # kg (kerb weight without driver)
    length: float = 4.177  # m (overall length)
    width: float = 1.832  # m (overall width)
    height: float = 1.353  # m (overall height)
    wheelbase: float = 2.505  # m
    track_width: float = 1.572  # m (front track)
    track_width_rear: float = 1.555  # m (rear track)
    
    # Center of gravity distances
    lf: float = 1.2525  # distance from CG to front axle
    lr: float = 1.2525  # distance from CG to rear axle
    
    # Aerodynamics
    drag_coefficient: float = 0.32  # Cd value from technical data
    frontal_area: float = 2.09  # m^2
    
    # Tire parameters
    wheel_radius: float = 0.3175  # m (225/50 R17)
    tire_friction_coeff: float = 0.8
    
    # Cornering stiffness
    Caf: float = 15000.0  # Front tire cornering stiffness N/rad
    Car: float = 18000.0  # Rear tire cornering stiffness N/rad
    
    # Moment of inertia
    Iz: float = 2500.0  # kg⋅m^2
    
    # Performance
    max_velocity: float = 250.0 / 3.6  # m/s (250 km/h)
    max_acceleration: float = 2.5  # m/s^2
    
    # Steering
    max_steering_angle: float = np.radians(30.0)  # rad
    steering_ratio: float = 14.6
    
    # Constants
    gravity: float = 9.81


class Engine:
    """Simplified engine model based on Audi TT 2.0 TFSI"""
    
    def __init__(self):
        # Engine characteristics
        self.max_torque = 370.0  # Nm
        self.max_power_kw = 169.0  # kW
        self.idle_rpm = 800.0
        self.redline_rpm = 6700.0
        
        # Current state
        self.rpm = self.idle_rpm
        self.state = 2  # 0=off, 1=starting, 2=running
        
    def get_torque(self, rpm: float) -> float:
        """Calculate engine torque at given RPM"""
        if rpm < self.idle_rpm or rpm > self.redline_rpm:
            return 0.0
            
        # Simplified torque curve
        if rpm <= 1200:
            return np.interp(rpm, [self.idle_rpm, 1200], [180, 240])
        elif rpm <= 1600:
            return np.interp(rpm, [1200, 1600], [240, self.max_torque])
        elif rpm <= 4300:
            return self.max_torque
        elif rpm <= 6200:
            # Power limited region
            angular_velocity = (2 * np.pi * rpm) / 60
            return (self.max_power_kw * 1000) / angular_velocity
        else:
            # Power drops after peak
            power_mult = np.interp(rpm, [6200, self.redline_rpm], [1.0, 0.7])
            adjusted_power = self.max_power_kw * power_mult
            angular_velocity = (2 * np.pi * rpm) / 60
            return (adjusted_power * 1000) / angular_velocity
    
    def update_rpm(self, target_rpm: float, dt: float):
        """Update engine RPM with realistic dynamics"""
        rpm_rate = 2500.0 if target_rpm > self.rpm else 2000.0
        self.rpm = np.clip(
            self.rpm + np.sign(target_rpm - self.rpm) * rpm_rate * dt,
            self.idle_rpm, self.redline_rpm
        )

class Transmission:
    """Automatic transmission model"""
    
    def __init__(self):
        # Gear ratios from Unity code
        self.gear_ratios = [3.769, 2.087, 1.481, 1.152, 1.167, 0.970]
        self.final_drive_ratios = [3.238, 3.238, 3.238, 3.238, 2.615, 2.615]
        
        # Shift points
        self.upshift_rpm = [2800, 3200, 3600, 4000, 4400]
        self.downshift_rpm = [1400, 1600, 1800, 2000, 2200]
        
        self.current_gear = 0  # 0-based index
        self.is_shifting = False
        self.shift_timer = 0.0
        
    def get_total_ratio(self) -> float:
        """Get current total gear reduction"""
        return self.gear_ratios[self.current_gear] * self.final_drive_ratios[self.current_gear]
    
    def update(self, engine_rpm: float, vehicle_speed: float, dt: float):
        """Update transmission state and gear selection"""
        if self.is_shifting:
            self.shift_timer += dt
            if self.shift_timer >= 0.2:  # 200ms shift time
                self.is_shifting = False
                self.shift_timer = 0.0
            return
        
        # Check for upshift
        if (self.current_gear < len(self.gear_ratios) - 1 and 
            engine_rpm > self.upshift_rpm[self.current_gear]):
            self.current_gear += 1
            self.is_shifting = True
            
        # Check for downshift  
        elif (self.current_gear > 0 and 
              engine_rpm < self.downshift_rpm[self.current_gear - 1]):
            self.current_gear -= 1
            self.is_shifting = True


class VehicleState:
    """Vehicle state vector"""
    
    def __init__(self):
        # Position and orientation
        self.x = 0.0      # longitudinal position (m)
        self.y = 0.0      # lateral position (m)
        self.psi = 0.0    # yaw angle (rad)
        
        # Velocities
        self.vx = 0.0     # longitudinal velocity (m/s)
        self.vy = 0.0     # lateral velocity (m/s)
        self.psi_dot = 0.0  # yaw rate (rad/s)
        
        # Accelerations
        self.ax = 0.0     # longitudinal acceleration (m/s^2)
        self.ay = 0.0     # lateral acceleration (m/s^2)
        
        self.steering_angle = 0.0  # steering angle (rad)
        self.wheel_speeds = [0.0, 0.0, 0.0, 0.0]  # wheel speeds (m/s) [FL, FR, RL, RR]
        self.wheel_torques = [0.0, 0.0, 0.0, 0.0]  # wheel torques (Nm) [FL, FR, RL, RR]        
        self.delta_f = 0.0  # steering input (rad)
        self.delta_r = 0.0  # rear steering input (rad) for 4WS
__all__ = [
    'VehicleParameters',
    'Engine', 
    'Transmission',
    'VehicleState'
]