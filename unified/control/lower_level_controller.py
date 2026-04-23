"""
unified/control/lower_level_controller.py

Lower-Level Controller for hierarchical vehicle control in the unified module.
Identical logic to Longitudinal/control/lower_level_controller.py but:
  - imports parameters from unified.config
  - adds Ackermann trapezoid front-wheel angle computation

Ackermann steering geometry (trapezoid model):
  Given bicycle-model steering angle δ (at the virtual front axle):
    R     = L / tan(|δ|)            turning radius (rear axle to turn centre)
    δ_in  = arctan(L / (R - t/2))  inner front wheel
    δ_out = arctan(L / (R + t/2))  outer front wheel
  where L = wheelbase, t = front track width.
  Results stored as llc.delta_left / llc.delta_right (and mirrored to
  vehicle.delta_left / vehicle.delta_right when those attributes exist).

This file is placed in unified/control/ so that when unified/ is on sys.path,
the lazy import inside Longitudinal/vehicle/vehicle.py:
    from control.lower_level_controller import LowerLevelController
resolves to THIS module rather than Longitudinal/control/.

Architecture (Rajamani Ch.6 / Belousov et al.):
    Upper Controller (Nash / Rajamani / IDM)
        |  u_accel [m/s²]
        v
    vehicle._compute_Fxf(u_accel)  [feedforward cancellation]
        |  Fxf [N]
        v
    LowerLevelController.compute_control(Fxf)
        |  throttle/brake (display), direct_force = Fxf
        v
    update_dynamics_complex(dt)  → state update + Ackermann angles
"""

import numpy as np

from unified.config import (
    LOWER_CTRL_KP,
    LOWER_CTRL_KI,
    LOWER_CTRL_INTEGRATOR_LIMIT,
    LOWER_CTRL_THROTTLE_SMOOTHING,
    LOWER_CTRL_BRAKE_SMOOTHING,
    LOWER_CTRL_DEADBAND,
    LOWER_CTRL_GEAR_SHIFT_HOLD,
    ENGINE_BRAKING_BASE_TORQUE,
    ENGINE_BRAKING_MAX_TORQUE,
)


# =============================================================================
# ENGINE MODEL
# =============================================================================
class Engine:
    """Simplified engine model based on Audi TT 2.0 TFSI."""

    def __init__(self):
        self.max_torque    = 370.0
        self.max_power_kw  = 169.0
        self.idle_rpm      = 800.0
        self.redline_rpm   = 6700.0
        self.rpm           = self.idle_rpm
        self.state         = 2   # 0=off, 1=starting, 2=running

    def get_torque(self, rpm: float) -> float:
        if rpm < self.idle_rpm or rpm > self.redline_rpm:
            return 0.0
        if rpm <= 1000:
            return np.interp(rpm, [self.idle_rpm, 1000], [50, 75])
        elif rpm <= 1200:
            return np.interp(rpm, [1000, 1200], [75, 150])
        elif rpm <= 1600:
            return np.interp(rpm, [1200, 1600], [150, self.max_torque])
        elif rpm <= 4300:
            return self.max_torque
        elif rpm <= 4500:
            return np.interp(rpm, [4300, 4500],
                             [self.max_torque, self.max_torque * 0.95])
        elif rpm <= 6200:
            return (self.max_power_kw * 1000) / ((2 * np.pi * rpm) / 60)
        else:
            power_mult = np.interp(rpm, [6200, self.redline_rpm], [1.0, 0.85])
            return (self.max_power_kw * power_mult * 1000) / ((2 * np.pi * rpm) / 60)

    def update_rpm(self, target_rpm: float, dt: float):
        rpm_rate = 2500.0 if target_rpm > self.rpm else 2000.0
        self.rpm = np.clip(
            self.rpm + np.sign(target_rpm - self.rpm) * rpm_rate * dt,
            self.idle_rpm, self.redline_rpm
        )


# =============================================================================
# TRANSMISSION MODEL
# =============================================================================
class Transmission:
    """6-speed manual transmission — Audi TT 2.0 TFSI 230PS FWD."""

    def __init__(self):
        self.gear_ratios         = [3.769, 2.087, 1.481, 1.152, 1.167, 0.970]
        self.final_drive_ratios  = [3.238, 3.238, 3.238, 3.238, 2.615, 2.615]
        self.upshift_rpm         = [2800, 3200, 3600, 4000, 4400]
        self.downshift_rpm       = [1400, 1600, 1800, 2000, 2200]
        self.upshift_speed       = [25.0,  45.0, 70.0, 100.0, 130.0]
        self.downshift_speed     = [12.0,  25.0, 45.0,  70.0, 100.0]
        self.current_gear        = 0
        self.is_shifting         = False
        self.shift_timer         = 0.0
        self.shift_duration      = 0.2
        self._shift_cooldown     = 0.0
        self._min_shift_interval = 0.3
        self._old_total_ratio    = self.gear_ratios[0] * self.final_drive_ratios[0]
        self._new_total_ratio    = self._old_total_ratio

    def get_total_ratio(self) -> float:
        return self.gear_ratios[self.current_gear] * self.final_drive_ratios[self.current_gear]

    def get_effective_ratio(self) -> float:
        if not self.is_shifting:
            return self.get_total_ratio()
        t        = min(self.shift_timer / self.shift_duration, 1.0)
        smooth_t = t * t * (3.0 - 2.0 * t)
        return (1.0 - smooth_t) * self._old_total_ratio + smooth_t * self._new_total_ratio

    def get_torque_multiplier(self) -> float:
        if not self.is_shifting:
            return 1.0
        t = min(self.shift_timer / self.shift_duration, 1.0)
        if t < 0.3:
            return 1.0 - 0.7 * (t / 0.3)
        elif t < 0.7:
            return 0.3 + 0.4 * ((t - 0.3) / 0.4)
        else:
            return 0.7 + 0.3 * ((t - 0.7) / 0.3)

    def _begin_shift(self, new_gear: int):
        self._old_total_ratio = self.get_total_ratio()
        self.current_gear     = new_gear
        self._new_total_ratio = self.get_total_ratio()
        self.is_shifting      = True
        self.shift_timer      = 0.0
        self._shift_cooldown  = self._min_shift_interval

    def update(self, engine_rpm: float, vehicle_speed: float, dt: float):
        if self.is_shifting:
            self.shift_timer += dt
            if self.shift_timer >= self.shift_duration:
                self.is_shifting = False
                self.shift_timer = 0.0
            return
        if self._shift_cooldown > 0.0:
            self._shift_cooldown -= dt
            return
        speed_kmh = vehicle_speed * 3.6
        if self.current_gear < len(self.gear_ratios) - 1:
            if (engine_rpm > self.upshift_rpm[self.current_gear] or
                    speed_kmh > self.upshift_speed[self.current_gear]):
                self._begin_shift(self.current_gear + 1)
                return
        if self.current_gear > 0:
            if (engine_rpm < self.downshift_rpm[self.current_gear - 1] or
                    speed_kmh < self.downshift_speed[self.current_gear - 1]):
                self._begin_shift(self.current_gear - 1)


# =============================================================================
# ACKERMANN TRAPEZOID GEOMETRY
# =============================================================================

def ackermann_wheel_angles(delta: float, wheelbase: float, track_width: float):
    """Compute per-wheel steer angles using perfect Ackermann trapezoid geometry.

    Parameters
    ----------
    delta       : float  bicycle-model front-axle steer angle [rad]  (+ = left turn)
    wheelbase   : float  L [m]
    track_width : float  t [m]  (front wheel contact-patch centreline spacing)

    Returns
    -------
    (delta_left, delta_right) : (float, float)  [rad]

    Model
    -----
    All four wheels must rotate about a common instantaneous centre (IC).
    Locating the IC on the rear-axle extension:

        R      = L / tan(|δ|)          turn radius at rear axle midpoint
        δ_inner = arctan(L / (R - t/2))
        δ_outer = arctan(L / (R + t/2))

    For a left turn (δ > 0):  left  = inner,  right = outer
    For a right turn (δ < 0): right = inner,  left  = outer
    At δ ≈ 0 both angles equal δ (degenerate case handled separately).
    """
    if abs(delta) < 1e-6:
        return delta, delta

    sign         = np.sign(delta)
    R            = wheelbase / np.tan(abs(delta))          # turn radius [m]
    half_track   = track_width / 2.0

    delta_inner  = np.arctan(wheelbase / max(R - half_track, 1e-3))
    delta_outer  = np.arctan(wheelbase / (R + half_track))

    if delta > 0:          # left turn: left = inner, right = outer
        delta_left  = sign * delta_inner
        delta_right = sign * delta_outer
    else:                  # right turn: right = inner, left = outer
        delta_left  = sign * delta_outer
        delta_right = sign * delta_inner

    return delta_left, delta_right


# =============================================================================
# LOWER-LEVEL CONTROLLER
# =============================================================================
class LowerLevelController:
    """
    Lower-level controller that owns all hardware-level vehicle dynamics:
    engine, transmission, pedal actuation, force-to-state integration, and
    Ackermann front-wheel angle computation.

    Responsibilities
    ----------------
    1. Hardware ownership: Engine and Transmission instances live here.
    2. Pedal state: throttle_input/brake_input and rate-limited
       actual_throttle/actual_brake are maintained here.
    3. compute_control(Fxf): converts desired acceleration to direct_force
       and display throttle/brake using pure feedforward.
    4. update_dynamics_complex(dt): integrates the full longitudinal dynamics
       (engine, transmission, Newton's 2nd law, ZOH integration) AND computes
       per-wheel steer angles via Ackermann trapezoid geometry.
    5. calculate_forces(): computes net longitudinal force.
    6. get_dynamic_max_acceleration(): physics-based engine force ceiling.
    """

    def __init__(self, vehicle):
        self.vehicle  = vehicle
        self.params   = vehicle.params
        self.engine       = Engine()
        self.transmission = Transmission()

        self.throttle_input  = 0.0
        self.brake_input     = 0.0
        self.actual_throttle = 0.0
        self.actual_brake    = 0.0

        self.throttle_opening_rate = 0.4
        self.throttle_closing_rate = 0.6
        self.brake_apply_rate      = 1.5
        self.brake_release_rate    = 0.8

        self.air_density   = self.params.air_density
        self.rolling_coeff = self.params.rolling_resistance_coeff

        self.Kp               = LOWER_CTRL_KP
        self.Ki               = LOWER_CTRL_KI
        self.error_integral   = 0.0
        self.integrator_limit = LOWER_CTRL_INTEGRATOR_LIMIT
        self.throttle_smoothing = LOWER_CTRL_THROTTLE_SMOOTHING
        self.brake_smoothing    = LOWER_CTRL_BRAKE_SMOOTHING
        self.last_throttle      = 0.0
        self.last_brake         = 0.0
        self.deadband           = LOWER_CTRL_DEADBAND

        # Ackermann per-wheel steer angles [rad]  (updated each dynamics step)
        self.delta_left  = 0.0
        self.delta_right = 0.0

        self._debug_counter = 0

    # =========================================================================
    # SYNC & RATE LIMITING
    # =========================================================================

    def _sync_engine_to_velocity(self):
        """Synchronize engine RPM and gear to current vehicle velocity."""
        vx = self.vehicle.state.vx
        if abs(vx) > 0.1:
            wheel_rpm = (vx * 60) / (2 * np.pi * self.params.wheel_radius)
            selected_gear = 0
            for gear_idx in range(len(self.transmission.gear_ratios)):
                ratio = (self.transmission.gear_ratios[gear_idx]
                         * self.transmission.final_drive_ratios[gear_idx])
                engine_rpm = wheel_rpm * ratio
                if engine_rpm > self.engine.redline_rpm:
                    continue
                if engine_rpm < self.engine.idle_rpm:
                    break
                selected_gear = gear_idx
                if (gear_idx < len(self.transmission.upshift_rpm) and
                        engine_rpm > self.transmission.upshift_rpm[gear_idx]):
                    continue
                else:
                    break
            self.transmission.current_gear = selected_gear
            target_rpm = wheel_rpm * self.transmission.get_total_ratio()
            target_rpm = np.clip(target_rpm, self.engine.idle_rpm, self.engine.redline_rpm)
            self.engine.rpm = target_rpm
        else:
            self.engine.rpm = self.engine.idle_rpm
            self.transmission.current_gear = 0

    def _rate_limit_pedals(self, dt: float):
        throttle_error = self.throttle_input - self.actual_throttle
        if throttle_error > 0:
            self.actual_throttle += min(throttle_error, self.throttle_opening_rate * dt)
        else:
            self.actual_throttle += max(throttle_error, -self.throttle_closing_rate * dt)
        self.actual_throttle = np.clip(self.actual_throttle, 0.0, 1.0)

        brake_error = self.brake_input - self.actual_brake
        if brake_error > 0:
            self.actual_brake += min(brake_error, self.brake_apply_rate * dt)
        else:
            self.actual_brake += max(brake_error, -self.brake_release_rate * dt)
        self.actual_brake = np.clip(self.actual_brake, 0.0, 1.0)

    # =========================================================================
    # FORCE CALCULATIONS
    # =========================================================================

    def compute_resistive_forces(self, velocity: float) -> float:
        """Compute total aerodynamic drag + rolling resistance + grade resistance [N]."""
        F_aero = (0.5 * self.air_density * self.params.drag_coefficient
                  * self.params.frontal_area * velocity * abs(velocity))
        f_v       = (np.tanh(10 * abs(velocity) - 8) + 1) / 2
        F_rolling = f_v * self.rolling_coeff * self.params.mass * self.params.gravity
        F_grade   = self.params.mass * self.params.gravity * np.sin(self.params.road_grade)
        return F_aero + F_rolling + F_grade

    def compute_max_engine_force(self) -> float:
        engine_torque = self.engine.get_torque(self.engine.rpm)
        total_ratio   = self.transmission.get_total_ratio()
        return max(engine_torque * total_ratio / self.params.wheel_radius, 1.0)

    def compute_max_brake_force(self) -> float:
        return self.params.mass * self.params.gravity * self.params.tire_friction_coeff

    def calculate_forces(self):
        """Compute net longitudinal force.  Returns (Fx_total, 0.0)."""
        v = self.vehicle.state.vx
        F_aero = (0.5 * self.air_density * self.params.drag_coefficient *
                  self.params.frontal_area * v * abs(v))
        f_v = (np.tanh(10 * abs(v) - 8) + 1) / 2
        rolling_resistance = f_v * self.rolling_coeff * self.params.mass * self.params.gravity

        if self.vehicle.autonomous_mode:
            Rg          = self.params.mass * self.params.gravity * np.sin(self.params.road_grade)
            total_force = self.vehicle.direct_force - F_aero - rolling_resistance - Rg
            return total_force, 0.0

        # Human-driven: engine torque map + brake
        if self.actual_throttle > 0.001:
            engine_torque     = self.engine.get_torque(self.engine.rpm)
            effective_ratio   = self.transmission.get_effective_ratio()
            torque_multiplier = self.transmission.get_torque_multiplier()
            wheel_torque  = engine_torque * effective_ratio * torque_multiplier
            engine_force  = (wheel_torque / self.params.wheel_radius) * self.actual_throttle
        else:
            if abs(v) > 1.0 and self.actual_brake < 0.001:
                braking_torque = ENGINE_BRAKING_BASE_TORQUE * (self.engine.rpm / 2000.0)
                braking_torque = min(braking_torque, ENGINE_BRAKING_MAX_TORQUE)
                total_ratio    = self.transmission.get_total_ratio()
                engine_force   = -(braking_torque * total_ratio / self.params.wheel_radius)
            else:
                engine_force = 0.0

        brake_force = (self.compute_max_brake_force() * self.actual_brake
                       if self.actual_brake > 0.001 else 0.0)
        Fx_total = engine_force - brake_force - F_aero - rolling_resistance
        return Fx_total, 0.0

    def get_dynamic_max_acceleration(self) -> float:
        """Physics-based max acceleration from engine at current state."""
        T        = self.engine.get_torque(self.engine.rpm)
        i_total  = self.transmission.get_total_ratio()
        F_engine = T * i_total / self.params.wheel_radius
        F_resist = self.compute_resistive_forces(self.vehicle.state.vx)
        return max((F_engine - F_resist) / self.params.mass, 0.0)

    # =========================================================================
    # HIERARCHICAL CONTROL LAW
    # =========================================================================

    def compute_control(self, Fxf: float) -> None:
        """Accept front axle drive force (FWD feedforward) and set display actuators.

        direct_force = Fxf is passed to calculate_forces() when autonomous_mode=True.
        Display throttle/brake are normalized — no physics role.
        """
        self.vehicle.direct_force = Fxf
        F_req_max = self.params.mass * self.params.max_acceleration
        F_req_min = -(self.params.mass * abs(self.params.max_deceleration))
        if Fxf >= 0:
            self.actual_throttle = np.clip(Fxf / max(F_req_max, 1.0), 0.0, 1.0)
            self.actual_brake    = 0.0
        else:
            self.actual_throttle = 0.0
            self.actual_brake    = np.clip(abs(Fxf) / max(abs(F_req_min), 1.0), 0.0, 1.0)
        self.throttle_input       = self.actual_throttle
        self.brake_input          = self.actual_brake
        self.vehicle.steering_input = 0.0

    # =========================================================================
    # COMPLEX DYNAMICS UPDATE  (+ Ackermann geometry)
    # =========================================================================

    def update_dynamics_complex(self, dt: float):
        """Integrate full longitudinal dynamics (ZOH) and compute Ackermann wheel angles.

        Integration order (ZOH-consistent):
            position update   ← uses OLD velocity
            velocity update   ← THEN increments

        Ackermann trapezoid:
            δ_left, δ_right = ackermann_wheel_angles(steering_angle, L, t)
        stored as self.delta_left / self.delta_right and mirrored to the vehicle.
        """
        self._rate_limit_pedals(dt)
        Fx, _ = self.calculate_forces()

        # Bicycle-model steering angle
        steering_angle = self.vehicle.steering_input * self.params.max_steering_angle

        # ── Ackermann trapezoid: individual front-wheel steer angles ──────────
        self.delta_left, self.delta_right = ackermann_wheel_angles(
            steering_angle,
            self.params.wheelbase,
            self.params.track_width,
        )
        # Mirror to vehicle for external logging / access
        if hasattr(self.vehicle, 'delta_left'):
            self.vehicle.delta_left  = self.delta_left
            self.vehicle.delta_right = self.delta_right

        state    = self.vehicle.state
        state.ax = Fx / self.vehicle.m_eff

        if abs(state.vx) > 0.1:
            beta          = np.arctan(self.params.lr * np.tan(steering_angle)
                                      / self.params.wheelbase)
            state.psi_dot = (state.vx * np.cos(beta) * np.tan(steering_angle)
                             / self.params.wheelbase)
            state.vy      = state.vx * np.sin(beta)
        else:
            state.psi_dot = 0.0
            state.vy      = 0.0

        cos_psi  = np.cos(state.psi)
        sin_psi  = np.sin(state.psi)

        # Position first (ZOH: uses velocity BEFORE increment)
        state.psi += state.psi_dot * dt
        state.x   += ((state.vx * cos_psi - state.vy * sin_psi) * dt
                      + 0.5 * state.ax * cos_psi * dt * dt)
        state.y   += ((state.vx * sin_psi + state.vy * cos_psi) * dt
                      + 0.5 * state.ax * sin_psi * dt * dt)

        # Velocity update
        state.vx  += state.ax * dt
        state.vx   = np.clip(state.vx, 0.0, self.params.max_velocity)

        self.vehicle.v = state.vx
        self.vehicle.a = state.ax

        if abs(state.vx) > 0.1:
            wheel_rpm         = (state.vx * 60) / (2 * np.pi * self.params.wheel_radius)
            target_engine_rpm = wheel_rpm * self.transmission.get_effective_ratio()
            target_engine_rpm = max(target_engine_rpm, self.engine.idle_rpm)
            self.engine.update_rpm(target_engine_rpm, dt)
        else:
            self.engine.update_rpm(self.engine.idle_rpm, dt)

        self.transmission.update(self.engine.rpm, state.vx, dt)


__all__ = ['Engine', 'Transmission', 'LowerLevelController', 'ackermann_wheel_angles']
