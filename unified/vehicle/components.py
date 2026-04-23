"""
unified/vehicle/components.py — Single source of truth for vehicle physical parameters.

Merges Longitudinal/vehicle/components.py and lateral/vehicle/components.py into
one canonical dataclass. Attribute naming follows the Longitudinal convention
(more complete); lateral-style aliases are provided as properties for
forward-compatibility if needed.

Naming differences resolved:
  Longitudinal  →  Unified (canonical)  ←  Lateral
  ─────────────────────────────────────────────────
  Caf           →  Caf                 ←  Cf
  Car           →  Car                 ←  Cr
  tire_friction_coeff → tire_friction_coeff ← mu
  gravity       →  gravity             ←  g
  max_steering_angle=30° → 25° (lateral is more conservative)
  (absent)      →  max_steering_rate   ←  max_steering_rate

No imports from any config module — this class is standalone.
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class VehicleParameters:
    """Audi TT Coupé 2.0 TFSI (230PS) — unified physical parameters."""

    # ── Inertia ──────────────────────────────────────────────────────────────
    mass: float = 1305.0       # [kg] kerb weight without driver
    Iz:   float = 2500.0       # [kg·m²] yaw moment of inertia (estimated)

    # ── Geometry ─────────────────────────────────────────────────────────────
    length:           float = 4.177   # [m] overall length
    width:            float = 1.832   # [m] overall width
    height:           float = 1.353   # [m] overall height
    wheelbase:        float = 2.505   # [m]
    track_width:      float = 1.572   # [m] front track
    track_width_rear: float = 1.555   # [m] rear track

    # Center-of-gravity distances (50/50 distribution assumed)
    lf: float = 1.2525   # [m] CG → front axle
    lr: float = 1.2525   # [m] CG → rear axle

    # ── Tires ────────────────────────────────────────────────────────────────
    # Cornering stiffness (per axle = 2 × per-wheel value)
    Caf: float = 15000.0   # [N/rad] front axle cornering stiffness
    Car: float = 18000.0   # [N/rad] rear  axle cornering stiffness

    # Longitudinal stiffness — FWD front axle combined (Rajamani Eq. 13.45)
    Cx:  float = 80000.0   # [N] front axle longitudinal stiffness

    # Wheel geometry (for Belousov effective mass: m_eff = m + Iw/r²)
    wheel_radius:   float = 0.3175   # [m] 225/50 R17
    wheel_inertia:  float = 1.5      # [kg·m²] equivalent drivetrain inertia at wheel

    # Friction
    tire_friction_coeff: float = 0.8   # [-] tire-road friction coefficient μ

    # ── Aerodynamics ─────────────────────────────────────────────────────────
    drag_coefficient: float = 0.30    # [-] Cd (Audi TT official)
    frontal_area:     float = 2.09    # [m²]
    air_density:      float = 1.225   # [kg/m³] ISA sea-level

    # ── Resistive forces ─────────────────────────────────────────────────────
    rolling_resistance_coeff: float = 0.012   # [-] typical tarmac
    road_grade:               float = 0.0     # [rad] positive = uphill; Rg = m·g·sin(θ)

    # ── Steering ─────────────────────────────────────────────────────────────
    max_steering_angle: float = np.radians(25.0)   # [rad] 25° (lateral module limit)
    max_steering_rate:  float = np.radians(15.0)   # [rad/s] 15°/s (from lateral module)
    steering_ratio:     float = 14.6               # [-] from Audi TT technical data

    # ── Performance limits ───────────────────────────────────────────────────
    max_velocity:             float = 250.0 / 3.6   # [m/s] 250 km/h
    max_acceleration:         float = 2.5            # [m/s²] longitudinal comfort limit
    max_deceleration:         float = -3.5           # [m/s²] emergency braking
    comfortable_deceleration: float = 2.0            # [m/s²] magnitude, for IDM
    max_lateral_acceleration: float = 3.0            # [m/s²]
    max_lateral_jerk:         float = 2.5            # [m/s³]

    # ── Physical constants ───────────────────────────────────────────────────
    gravity: float = 9.81   # [m/s²]

    # ── Lateral-naming aliases (properties) ──────────────────────────────────
    # These allow code that uses the lateral convention (p.Cf, p.mu, p.g) to
    # work unmodified with UnifiedVehicleParameters.
    @property
    def Cf(self)  -> float: return self.Caf
    @property
    def Cr(self)  -> float: return self.Car
    @property
    def mu(self)  -> float: return self.tire_friction_coeff
    @property
    def g(self)   -> float: return self.gravity


class VehicleState:
    """Vehicle state vector (used by platoon Vehicle class)."""

    def __init__(self):
        # Position and orientation
        self.x = 0.0
        self.y = 0.0
        self.psi = 0.0

        # Velocities
        self.vx = 0.0
        self.vy = 0.0
        self.psi_dot = 0.0

        # Accelerations
        self.ax = 0.0
        self.ay = 0.0

        self.steering_angle = 0.0
        self.wheel_speeds = [0.0, 0.0, 0.0, 0.0]
        self.wheel_torques = [0.0, 0.0, 0.0, 0.0]
        self.delta_f = 0.0   # front steering angle [rad]
        self.delta_r = 0.0   # rear steering angle [rad]
        self.Fxf = 0.0       # front axle longitudinal force [N]


__all__ = ['VehicleParameters', 'VehicleState']
