"""
Vehicle components module — abstract vehicle model data classes.

Contains VehicleParameters and VehicleState.
Engine and Transmission have moved to control/lower_level_controller.py
because they are hardware components belonging to the lower-level controller.
"""

import numpy as np
from dataclasses import dataclass


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
    drag_coefficient: float = 0.30  # Cd value from official Audi TT specs
    frontal_area: float = 2.09  # m^2

    # Tire parameters
    wheel_radius: float = 0.3175  # m (225/50 R17)
    wheel_inertia: float = 1.5    # [kg·m²] total all wheels — Belousov Eq. 3.8: m_eff = m + Iw/r²
    tire_friction_coeff: float = 0.8

    # Cornering stiffness
    Caf: float = 15000.0  # Front tire cornering stiffness N/rad
    Car: float = 18000.0  # Rear tire cornering stiffness N/rad

    # Longitudinal tire stiffness (FWD — front axle, two tires combined)
    # Used in Rajamani Eq. 13.45 parabolic model: Fxf = Cx·κ for small slip κ
    Cx: float = 80000.0   # [N] front axle longitudinal stiffness

    # Moment of inertia
    Iz: float = 2500.0  # kg⋅m^2

    # Performance
    max_velocity: float = 250.0 / 3.6  # m/s (250 km/h)
    max_acceleration: float = 2.5  # m/s² (longitudinal comfort limit)
    max_deceleration: float = -3.5  # m/s² (emergency braking capability)
    comfortable_deceleration: float = 2.0  # m/s² (magnitude, for IDM)

    # Lateral dynamics constraints
    max_lateral_acceleration: float = 3.0  # m/s²
    max_lateral_jerk: float = 2.5  # m/s³

    # Steering
    max_steering_angle: float = np.radians(30.0)  # rad
    steering_ratio: float = 14.6

    # Constants
    gravity: float = 9.81

    # Environment / road constants (used by both vehicle dynamics and lower-level controller)
    air_density: float = 1.225          # [kg/m³] ISA sea-level standard atmosphere
    rolling_resistance_coeff: float = 0.012  # [-] typical tarmac road/tire constant
    road_grade: float = 0.0             # [rad] road grade angle θ (positive = uphill); Rg = m·g·sin(θ)


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
        self.Fxf = 0.0      # front axle longitudinal force [N] (FWD: Fxr=0 → Fxf = F_drive/cos(δf))
__all__ = [
    'VehicleParameters',
    'VehicleState'
]
