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
    drag_coefficient: float = 0.30  # Cd value from official Audi TT specs
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
        """Calculate engine torque at given RPM.

        Based on the official Audi TT 2.0 TFSI (230 PS) torque curve
        (automobile-catalog.com / ProfessorsCar approximation):

        - 800-1000  RPM:  50 →  75 Nm  (idle, turbo not yet spooled)
        - 1000-1200 RPM:  75 → 150 Nm  (turbo starting to build boost)
        - 1200-1600 RPM: 150 → 370 Nm  (rapid turbo spool-up)
        - 1600-4300 RPM: 370 Nm        (full boost plateau)
        - 4300-4500 RPM: 370 → 351.5   (slight decline before power peak)
        - 4500-6200 RPM: T = P_max/ω   (power-limited, 169 kW constant)
        - 6200-6700 RPM: reduced power  (100% → 85%)
        """
        if rpm < self.idle_rpm or rpm > self.redline_rpm:
            return 0.0

        if rpm <= 1000:
            # Sub-boost region: very low torque at idle
            return np.interp(rpm, [self.idle_rpm, 1000], [50, 75])
        elif rpm <= 1200:
            # Turbo starting to spool
            return np.interp(rpm, [1000, 1200], [75, 150])
        elif rpm <= 1600:
            # Rapid boost build-up
            return np.interp(rpm, [1200, 1600], [150, self.max_torque])
        elif rpm <= 4300:
            return self.max_torque
        elif rpm <= 4500:
            return np.interp(rpm, [4300, 4500],
                             [self.max_torque, self.max_torque * 0.95])
        elif rpm <= 6200:
            angular_velocity = (2 * np.pi * rpm) / 60
            return (self.max_power_kw * 1000) / angular_velocity
        else:
            # Graph shows ~225 Nm at 6700 RPM, so power drops to ~85%
            power_mult = np.interp(rpm, [6200, self.redline_rpm], [1.0, 0.85])
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
    """6-speed manual transmission model — mirrors Unity Transmission.cs.

    Gear ratios and final drive match the Audi TT 2.0 TFSI 230PS FWD
    6-speed manual (same values used in the Unity simulator).

    Shift logic replicates Unity's CheckForGearChange():
    - Upshift:   RPM > threshold  OR  speed > threshold
    - Downshift: RPM < threshold  OR  speed < threshold
    - Cooldown timer (0.3 s) prevents gear hunting between consecutive shifts.
    - Shift transition duration: 0.2 s (200 ms), matching Unity.
    - Torque output follows a realistic three-phase clutch engagement profile
      during each shift (drop → partial re-engage → full engagement).
    """

    def __init__(self):
        # Gear ratios matching Unity Transmission.cs (Audi TT 2.0 TFSI FWD 6-speed manual)
        self.gear_ratios = [3.769, 2.087, 1.481, 1.152, 1.167, 0.970]
        self.final_drive_ratios = [3.238, 3.238, 3.238, 3.238, 2.615, 2.615]

        # RPM-based shift thresholds — identical to Unity
        self.upshift_rpm   = [2800, 3200, 3600, 4000, 4400]
        self.downshift_rpm = [1400, 1600, 1800, 2000, 2200]

        # Speed-based shift thresholds [km/h] — identical to Unity
        self.upshift_speed   = [25.0,  45.0, 70.0, 100.0, 130.0]
        self.downshift_speed = [12.0,  25.0, 45.0,  70.0, 100.0]

        self.current_gear = 0      # 0-based index (gear 1 = index 0)
        self.is_shifting  = False
        self.shift_timer  = 0.0
        self.shift_duration = 0.2  # [s] — 200 ms, matching Unity shiftDuration

        # Cooldown prevents rapid back-to-back shifts (Unity: minShiftInterval = 0.3 s)
        self._shift_cooldown     = 0.0
        self._min_shift_interval = 0.3  # [s]

        # Ratio snapshot for Hermite blending during a shift
        self._old_total_ratio = self.gear_ratios[0] * self.final_drive_ratios[0]
        self._new_total_ratio = self._old_total_ratio

    # ------------------------------------------------------------------
    # Ratio / multiplier accessors
    # ------------------------------------------------------------------

    def get_total_ratio(self) -> float:
        """Instantaneous gear reduction of the engaged gear (used for RPM calc)."""
        return self.gear_ratios[self.current_gear] * self.final_drive_ratios[self.current_gear]

    def get_effective_ratio(self) -> float:
        """Gear ratio with Hermite smoothstep blending during a shift.

        Returns the instantaneous ratio outside of shifts; during a shift
        interpolates between old and new ratios via a smooth S-curve so the
        traction force transitions gradually rather than jumping.
        """
        if not self.is_shifting:
            return self.gear_ratios[self.current_gear] * self.final_drive_ratios[self.current_gear]

        t = min(self.shift_timer / self.shift_duration, 1.0)
        smooth_t = t * t * (3.0 - 2.0 * t)  # Hermite smoothstep: t²(3 − 2t)
        return (1.0 - smooth_t) * self._old_total_ratio + smooth_t * self._new_total_ratio

    def get_torque_multiplier(self) -> float:
        """Torque scaling during a gear shift — simulates realistic clutch engagement.

        Three-phase profile (normalised shift progress t ∈ [0, 1]):
          Phase 1  t ∈ [0.0, 0.3]: Clutch 1 disengages  → torque 1.0 → 0.3
          Phase 2  t ∈ [0.3, 0.7]: Clutch 2 engages     → torque 0.3 → 0.7
          Phase 3  t ∈ [0.7, 1.0]: Full engagement       → torque 0.7 → 1.0

        Returns 1.0 when no shift is in progress.
        """
        if not self.is_shifting:
            return 1.0

        t = min(self.shift_timer / self.shift_duration, 1.0)

        if t < 0.3:
            phase_t = t / 0.3
            return 1.0 - 0.7 * phase_t          # 1.0 → 0.3
        elif t < 0.7:
            phase_t = (t - 0.3) / 0.4
            return 0.3 + 0.4 * phase_t           # 0.3 → 0.7
        else:
            phase_t = (t - 0.7) / 0.3
            return 0.7 + 0.3 * phase_t           # 0.7 → 1.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _begin_shift(self, new_gear: int):
        """Initiate a shift to *new_gear* (0-based index)."""
        self._old_total_ratio = (self.gear_ratios[self.current_gear]
                                 * self.final_drive_ratios[self.current_gear])
        self.current_gear     = new_gear
        self._new_total_ratio = (self.gear_ratios[self.current_gear]
                                 * self.final_drive_ratios[self.current_gear])
        self.is_shifting      = True
        self.shift_timer      = 0.0
        self._shift_cooldown  = self._min_shift_interval

    # ------------------------------------------------------------------
    # Main update — called every simulation step
    # ------------------------------------------------------------------

    def update(self, engine_rpm: float, vehicle_speed: float, dt: float):
        """Update transmission state and select gears.

        Mirrors Unity's Transmission.Update() / CheckForGearChange() logic:
        * While a shift is in progress, just advance the timer.
        * While cooling down after a shift, do nothing.
        * Otherwise evaluate both RPM and speed conditions (OR logic).

        Args:
            engine_rpm:    Current engine RPM.
            vehicle_speed: Vehicle longitudinal speed [m/s].
            dt:            Time step [s].
        """
        # --- advance shift-in-progress ---
        if self.is_shifting:
            self.shift_timer += dt
            if self.shift_timer >= self.shift_duration:
                self.is_shifting = False
                self.shift_timer = 0.0
            return

        # --- advance cooldown ---
        if self._shift_cooldown > 0.0:
            self._shift_cooldown -= dt
            return

        speed_kmh = vehicle_speed * 3.6

        # --- upshift: RPM OR speed exceeds threshold ---
        if self.current_gear < len(self.gear_ratios) - 1:
            rpm_cond = engine_rpm  > self.upshift_rpm[self.current_gear]
            spd_cond = speed_kmh  > self.upshift_speed[self.current_gear]
            if rpm_cond or spd_cond:
                self._begin_shift(self.current_gear + 1)
                return

        # --- downshift: RPM OR speed falls below threshold ---
        if self.current_gear > 0:
            rpm_cond = engine_rpm  < self.downshift_rpm[self.current_gear - 1]
            spd_cond = speed_kmh  < self.downshift_speed[self.current_gear - 1]
            if rpm_cond or spd_cond:
                self._begin_shift(self.current_gear - 1)


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
