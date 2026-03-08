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
<<<<<<< HEAD
    """Automatic transmission model — matches Unity Transmission.cs.

    Shift criteria use both RPM **and** speed thresholds (OR logic), with a
    cooldown between consecutive shifts to prevent gear hunting.
    """
=======
    """Automatic transmission model with smooth gear shifts (DCT-style)"""
>>>>>>> 17fa96e0c85c7c38ca8539df6443fe540d22dcaa

    def __init__(self):
        # Gear ratios from Unity code
        self.gear_ratios = [3.769, 2.087, 1.481, 1.152, 1.167, 0.970]
        self.final_drive_ratios = [3.238, 3.238, 3.238, 3.238, 2.615, 2.615]

<<<<<<< HEAD
        # RPM-based shift thresholds (same as Unity)
        self.upshift_rpm = [2800, 3200, 3600, 4000, 4400]
        self.downshift_rpm = [1400, 1600, 1800, 2000, 2200]

        # Speed-based shift thresholds [km/h] (from Unity Transmission.cs)
        self.upshift_speed = [25.0, 45.0, 70.0, 100.0, 130.0]
        self.downshift_speed = [12.0, 25.0, 45.0, 70.0, 100.0]

        self.current_gear = 0  # 0-based index
        self.is_shifting = False
        self.shift_timer = 0.0
        self.shift_duration = 0.4  # [s] total shift transition time

        # Shift cooldown to prevent gear hunting (Unity: minShiftInterval=0.3s)
        self._shift_cooldown = 0.0
        self._min_shift_interval = 0.3  # [s]

        # Smooth ratio blending: track old ratio to blend during shifts
        self._old_total_ratio = self.gear_ratios[0] * self.final_drive_ratios[0]
        self._new_total_ratio = self._old_total_ratio

    def get_total_ratio(self) -> float:
        """Get current total gear reduction (instantaneous, for RPM calculation).

        Returns the current gear's ratio directly — no blending.
        Use ``get_effective_ratio()`` for the force calculation path.
        """
        return self.gear_ratios[self.current_gear] * self.final_drive_ratios[self.current_gear]

    def get_effective_ratio(self) -> float:
        """Get effective gear ratio with smooth Hermite blending during shifts.

        During a gear shift, blends between old and new gear ratios using
        a Hermite smoothstep (S-curve) for a smoother force transition than
        linear interpolation.
        """
        if not self.is_shifting:
            return self.gear_ratios[self.current_gear] * self.final_drive_ratios[self.current_gear]

        # Smooth S-curve blend (Hermite smoothstep: t²(3 - 2t))
        t = min(self.shift_timer / self.shift_duration, 1.0)
        smooth_t = t * t * (3.0 - 2.0 * t)

        return (1.0 - smooth_t) * self._old_total_ratio + smooth_t * self._new_total_ratio
=======
        # Shift points
        self.upshift_rpm = [2800, 3200, 3600, 4000, 4400]
        self.downshift_rpm = [1400, 1600, 1800, 2000, 2200]

        self.current_gear = 0  # 0-based index
        self.is_shifting = False
        self.shift_timer = 0.0

        # Smooth gear shift parameters
        self.previous_gear = 0            # Gear before shift started
        self.shift_duration = 0.3         # Total shift duration [s] (300ms for DCT)
        self.torque_blend_progress = 1.0  # 0.0 = shift just started, 1.0 = fully engaged

    def get_total_ratio(self) -> float:
        """Get current total gear reduction (instantaneous, for RPM calculation)"""
        return self.gear_ratios[self.current_gear] * self.final_drive_ratios[self.current_gear]

    def get_effective_ratio(self) -> float:
        """Get effective gear ratio with smooth blending during shifts.

        During a gear shift, blends between old and new gear ratios
        based on torque_blend_progress for smooth force transition.
        """
        if not self.is_shifting or self.torque_blend_progress >= 1.0:
            # No shift in progress — use current gear ratio
            return self.gear_ratios[self.current_gear] * self.final_drive_ratios[self.current_gear]

        # Blend between previous and current gear ratios
        old_ratio = self.gear_ratios[self.previous_gear] * self.final_drive_ratios[self.previous_gear]
        new_ratio = self.gear_ratios[self.current_gear] * self.final_drive_ratios[self.current_gear]

        # Smooth S-curve blend (smoother than linear)
        t = self.torque_blend_progress
        smooth_t = t * t * (3.0 - 2.0 * t)  # Hermite smoothstep

        return old_ratio * (1.0 - smooth_t) + new_ratio * smooth_t
>>>>>>> 17fa96e0c85c7c38ca8539df6443fe540d22dcaa

    def get_torque_multiplier(self) -> float:
        """Get torque multiplier during gear shifts.

        Simulates realistic DCT shift behavior:
        - Torque dips during clutch transition
        - Gradually restores as new gear engages

        Returns 1.0 when no shift is in progress.
        """
<<<<<<< HEAD
        if not self.is_shifting:
            return 1.0

        t = min(self.shift_timer / self.shift_duration, 1.0)
=======
        if not self.is_shifting or self.torque_blend_progress >= 1.0:
            return 1.0

        t = self.torque_blend_progress
>>>>>>> 17fa96e0c85c7c38ca8539df6443fe540d22dcaa

        # Realistic DCT torque profile during shift:
        # Phase 1 (0-30%): Clutch 1 disengaging — torque drops to 0.3
        # Phase 2 (30-70%): Clutch 2 engaging — torque rises to 0.7
        # Phase 3 (70-100%): Full engagement — torque rises to 1.0
        if t < 0.3:
<<<<<<< HEAD
            phase_t = t / 0.3
            return 1.0 - 0.7 * phase_t
        elif t < 0.7:
            phase_t = (t - 0.3) / 0.4
            return 0.3 + 0.4 * phase_t
        else:
            phase_t = (t - 0.7) / 0.3
            return 0.7 + 0.3 * phase_t

    def _begin_shift(self, new_gear: int):
        """Start a gear shift to *new_gear* (0-based)."""
        self._old_total_ratio = (self.gear_ratios[self.current_gear]
                                 * self.final_drive_ratios[self.current_gear])
        self.current_gear = new_gear
        self._new_total_ratio = (self.gear_ratios[self.current_gear]
                                 * self.final_drive_ratios[self.current_gear])
        self.is_shifting = True
        self.shift_timer = 0.0
        self._shift_cooldown = self._min_shift_interval

    def update(self, engine_rpm: float, vehicle_speed: float, dt: float):
        """Update transmission state and gear selection.

        Shift criteria mirror Unity's CheckForGearChange():
        * Upshift:   RPM > threshold  **OR**  speed > threshold
        * Downshift: RPM < threshold  **OR**  speed < threshold
        A cooldown timer prevents rapid back-to-back shifts.
        """
        # --- tick shift-in-progress ---
        if self.is_shifting:
            self.shift_timer += dt
=======
            # Torque drops from 1.0 to 0.3 (clutch disengaging)
            phase_t = t / 0.3
            return 1.0 - 0.7 * phase_t
        elif t < 0.7:
            # Torque rises from 0.3 to 0.7 (new gear engaging)
            phase_t = (t - 0.3) / 0.4
            return 0.3 + 0.4 * phase_t
        else:
            # Torque rises from 0.7 to 1.0 (full engagement)
            phase_t = (t - 0.7) / 0.3
            return 0.7 + 0.3 * phase_t

    def update(self, engine_rpm: float, vehicle_speed: float, dt: float):
        """Update transmission state and gear selection with smooth shifts"""
        if self.is_shifting:
            self.shift_timer += dt
            # Advance torque blend progress
            self.torque_blend_progress = min(1.0, self.shift_timer / self.shift_duration)

>>>>>>> 17fa96e0c85c7c38ca8539df6443fe540d22dcaa
            if self.shift_timer >= self.shift_duration:
                self.is_shifting = False
                self.shift_timer = 0.0
                self.torque_blend_progress = 1.0
            return

<<<<<<< HEAD
        # --- tick cooldown ---
        if self._shift_cooldown > 0.0:
            self._shift_cooldown -= dt
            return

        speed_kmh = vehicle_speed * 3.6

        # --- upshift check (RPM OR speed) ---
        if self.current_gear < len(self.gear_ratios) - 1:
            rpm_cond = engine_rpm > self.upshift_rpm[self.current_gear]
            spd_cond = speed_kmh > self.upshift_speed[self.current_gear]
            if rpm_cond or spd_cond:
                self._begin_shift(self.current_gear + 1)
                return

        # --- downshift check (RPM OR speed) ---
        if self.current_gear > 0:
            rpm_cond = engine_rpm < self.downshift_rpm[self.current_gear - 1]
            spd_cond = speed_kmh < self.downshift_speed[self.current_gear - 1]
            if rpm_cond or spd_cond:
                self._begin_shift(self.current_gear - 1)
=======
        # Check for upshift
        if (self.current_gear < len(self.gear_ratios) - 1 and
            engine_rpm > self.upshift_rpm[self.current_gear]):
            self.previous_gear = self.current_gear
            self.current_gear += 1
            self.is_shifting = True
            self.shift_timer = 0.0
            self.torque_blend_progress = 0.0

        # Check for downshift
        elif (self.current_gear > 0 and
              engine_rpm < self.downshift_rpm[self.current_gear - 1]):
            self.previous_gear = self.current_gear
            self.current_gear -= 1
            self.is_shifting = True
            self.shift_timer = 0.0
            self.torque_blend_progress = 0.0
>>>>>>> 17fa96e0c85c7c38ca8539df6443fe540d22dcaa


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