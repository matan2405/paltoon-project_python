"""
Lower-Level Controller for Hierarchical Vehicle Control Architecture.

Architecture (Rajamani Ch.6, Belousov et al.):
===============================================
    Upper Controller (Nash / Rajamani / IDM)
        │   a_desired  (double integrator planning model)
        ▼
    ┌─────────────────────────────┐
    │   Lower-Level Controller    │
    │  ┌───────────────────────┐  │
    │  │ Feedforward:          │  │
    │  │  F = m·a_des          │  │
    │  │    + F_drag(v)        │  │
    │  │    + F_roll           │  │
    │  └───────────────────────┘  │
    │  ┌───────────────────────┐  │
    │  │ Engine / Transmission │  │
    │  │ (for RPM/gear display)│  │
    │  └───────────────────────┘  │
    └─────────────────────────────┘
        │   throttle, brake (display), direct_force
        ▼
    Complex Vehicle Dynamics (aero, tires, Newton's 2nd law)
        │   a_actual, v, x
        ▼
    Updated Vehicle State

Engine and Transmission class definitions live here — they are hardware
components that belong with the controller, not with the abstract vehicle model.

References:
-----------
- Rajamani, R. "Vehicle Dynamics and Control", Ch.6 - Longitudinal Control
- Belousov et al. "Development of a vehicle simulation model consisting
  of low and high frequency dynamics"
- Li et al. (2019) - Shared control with Nash equilibrium
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
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
    """Simplified engine model based on Audi TT 2.0 TFSI"""

    def __init__(self):
        # Engine characteristics
        self.max_torque = 370.0    # Nm
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
            angular_velocity = (2 * np.pi * rpm) / 60
            return (self.max_power_kw * 1000) / angular_velocity
        else:
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


# =============================================================================
# TRANSMISSION MODEL
# =============================================================================
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
        self.gear_ratios = [3.769, 2.087, 1.481, 1.152, 1.167, 0.970]
        self.final_drive_ratios = [3.238, 3.238, 3.238, 3.238, 2.615, 2.615]

        self.upshift_rpm   = [2800, 3200, 3600, 4000, 4400]
        self.downshift_rpm = [1400, 1600, 1800, 2000, 2200]

        self.upshift_speed   = [25.0,  45.0, 70.0, 100.0, 130.0]
        self.downshift_speed = [12.0,  25.0, 45.0,  70.0, 100.0]

        self.current_gear = 0
        self.is_shifting  = False
        self.shift_timer  = 0.0
        self.shift_duration = 0.2

        self._shift_cooldown     = 0.0
        self._min_shift_interval = 0.3

        self._old_total_ratio = self.gear_ratios[0] * self.final_drive_ratios[0]
        self._new_total_ratio = self._old_total_ratio

    def get_total_ratio(self) -> float:
        """Instantaneous gear reduction of the engaged gear."""
        return self.gear_ratios[self.current_gear] * self.final_drive_ratios[self.current_gear]

    def get_effective_ratio(self) -> float:
        """Gear ratio with Hermite smoothstep blending during a shift."""
        if not self.is_shifting:
            return self.gear_ratios[self.current_gear] * self.final_drive_ratios[self.current_gear]
        t = min(self.shift_timer / self.shift_duration, 1.0)
        smooth_t = t * t * (3.0 - 2.0 * t)
        return (1.0 - smooth_t) * self._old_total_ratio + smooth_t * self._new_total_ratio

    def get_torque_multiplier(self) -> float:
        """Torque scaling during a gear shift (realistic clutch engagement)."""
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
        self._old_total_ratio = (self.gear_ratios[self.current_gear]
                                 * self.final_drive_ratios[self.current_gear])
        self.current_gear     = new_gear
        self._new_total_ratio = (self.gear_ratios[self.current_gear]
                                 * self.final_drive_ratios[self.current_gear])
        self.is_shifting      = True
        self.shift_timer      = 0.0
        self._shift_cooldown  = self._min_shift_interval

    def update(self, engine_rpm: float, vehicle_speed: float, dt: float):
        """Update transmission state and select gears (mirrors Unity logic)."""
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
# LOWER-LEVEL CONTROLLER
# =============================================================================
class LowerLevelController:
    """
    Lower-level controller that owns all hardware-level vehicle dynamics:
    engine, transmission, pedal actuation, and force-to-state integration.

    Responsibilities
    ----------------
    1. **Hardware ownership**: Engine and Transmission instances live here.
    2. **Pedal state**: throttle_input/brake_input and rate-limited
       actual_throttle/actual_brake are maintained here.
    3. **compute_control(a_desired)**: converts desired acceleration (from the
       upper controller) to direct_force and display throttle/brake, using a
       pure feedforward law (F = m·a + F_drag + F_roll).
    4. **update_dynamics_complex(dt)**: integrates the full longitudinal
       dynamics (engine, transmission, Newton's 2nd law, ZOH integration).
    5. **calculate_forces()**: computes the net longitudinal force from
       engine/brake/aero/rolling, used in complex and kinematic modes.
    6. **get_dynamic_max_acceleration()**: physics-based engine force ceiling.
    """

    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.params = vehicle.params

        # ── Hardware components (moved from Vehicle) ─────────────────────────
        self.engine = Engine()
        self.transmission = Transmission()

        # ── Pedal state (moved from Vehicle) ─────────────────────────────────
        # Commanded inputs (0-1)
        self.throttle_input = 0.0
        self.brake_input = 0.0
        # Physical pedal positions after rate limiting (0-1)
        self.actual_throttle = 0.0
        self.actual_brake = 0.0
        # Rate limits [fraction/s]  (human-pedal dynamics)
        self.throttle_opening_rate = 0.4   # full open in ~2.5 s
        self.throttle_closing_rate = 0.6   # full close in ~1.7 s
        self.brake_apply_rate = 1.5        # fast for safety
        self.brake_release_rate = 0.8

        # ── Physical constants (from VehicleParameters) ───────────────────────
        self.air_density = self.params.air_density
        self.rolling_coeff = self.params.rolling_resistance_coeff

        # ── (Kept for future PI use — currently unused in hierarchical mode) ──
        self.Kp = LOWER_CTRL_KP
        self.Ki = LOWER_CTRL_KI
        self.error_integral = 0.0
        self.integrator_limit = LOWER_CTRL_INTEGRATOR_LIMIT
        self.throttle_smoothing = LOWER_CTRL_THROTTLE_SMOOTHING
        self.brake_smoothing = LOWER_CTRL_BRAKE_SMOOTHING
        self.last_throttle = 0.0
        self.last_brake = 0.0
        self.deadband = LOWER_CTRL_DEADBAND

        self._debug_counter = 0

    # =========================================================================
    # SYNC & RATE LIMITING
    # =========================================================================

    def _sync_engine_to_velocity(self):
        """Synchronize engine RPM and gear to current vehicle velocity.

        Simulates sequential upshifting (same thresholds as Transmission.update)
        so the initial gear is consistent with what the transmission would
        naturally reach through gear progression at the current speed.
        """
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
        """Rate-limit throttle and brake pedal positions for realistic driver behavior."""
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
        """Compute total aerodynamic drag + rolling resistance [N]."""
        F_aero = (0.5 * self.air_density * self.params.drag_coefficient
                  * self.params.frontal_area * velocity * abs(velocity))
        F_rolling = (self.rolling_coeff * self.params.mass * self.params.gravity
                     if abs(velocity) > 0.01 else 0.0)
        return F_aero + F_rolling

    def compute_max_engine_force(self) -> float:
        """Maximum available engine drive force at current RPM [N]."""
        engine_torque = self.engine.get_torque(self.engine.rpm)
        total_ratio = self.transmission.get_total_ratio()
        return max(engine_torque * total_ratio / self.params.wheel_radius, 1.0)

    def compute_max_brake_force(self) -> float:
        """Maximum available braking force [N]."""
        return self.params.mass * self.params.gravity * self.params.tire_friction_coeff

    def calculate_forces(self):
        """Calculate net longitudinal and lateral forces.

        In autonomous mode (direct_force is set): applies direct_force minus
        resistive loads, giving exactly F_net = m·a_desired.
        In human mode: uses engine torque map, engine braking, and brake force.

        Returns
        -------
        (Fx_total, 0.0)
        """
        v = self.vehicle.state.vx
        F_aero = (0.5 * self.air_density * self.params.drag_coefficient *
                  self.params.frontal_area * v * abs(v))
        rolling_resistance = (self.rolling_coeff * self.params.mass * self.params.gravity
                              if abs(v) > 0.01 else 0.0)

        if self.vehicle.autonomous_mode:
            # Direct force control: vehicle.direct_force already accounts for
            # aero + rolling. Subtract them to get net Newton force.
            total_force = self.vehicle.direct_force - F_aero - rolling_resistance
            return total_force, 0.0

        # Human-driven: engine torque map + brake
        if self.actual_throttle > 0.001:
            engine_torque = self.engine.get_torque(self.engine.rpm)
            effective_ratio = self.transmission.get_effective_ratio()
            torque_multiplier = self.transmission.get_torque_multiplier()
            wheel_torque = engine_torque * effective_ratio * torque_multiplier
            engine_force = (wheel_torque / self.params.wheel_radius) * self.actual_throttle
        else:
            # Engine braking when throttle released and vehicle moving
            if abs(v) > 1.0 and self.actual_brake < 0.001:
                braking_torque = ENGINE_BRAKING_BASE_TORQUE * (self.engine.rpm / 2000.0)
                braking_torque = min(braking_torque, ENGINE_BRAKING_MAX_TORQUE)
                total_ratio = self.transmission.get_total_ratio()
                engine_force = -(braking_torque * total_ratio / self.params.wheel_radius)
            else:
                engine_force = 0.0

        brake_force = (self.compute_max_brake_force() * self.actual_brake
                       if self.actual_brake > 0.001 else 0.0)

        Fx_total = engine_force - brake_force - F_aero - rolling_resistance
        return Fx_total, 0.0

    def get_dynamic_max_acceleration(self) -> float:
        """Physics-based max acceleration from the engine at current state.

        a_max = (T_engine(rpm) * i_total / r_wheel - F_aero - F_roll) / m

        Falls back to params.max_acceleration for models without engine dynamics.
        """
        T = self.engine.get_torque(self.engine.rpm)
        i_total = self.transmission.get_total_ratio()
        F_engine_max = T * i_total / self.params.wheel_radius

        v = self.vehicle.state.vx
        F_resist = self.compute_resistive_forces(v)

        a_max = (F_engine_max - F_resist) / self.params.mass
        return max(a_max, 0.0)

    # =========================================================================
    # HIERARCHICAL CONTROL LAW
    # =========================================================================

    def compute_control(self, a_desired: float) -> None:
        """Convert desired acceleration to direct_force and display throttle/brake.

        Pure feedforward law (no PI feedback):
            F_required = m·a_desired + F_aero + F_roll

        The direct_force is set on the vehicle and consumed by
        calculate_forces() when autonomous_mode=True.

        Display throttle/brake are normalized so that:
        - a_desired = 0 (cruise)    → throttle ≈ 15%  (overcoming drag)
        - a_desired at drag coast   → throttle = 0, brake = 0
        - a_desired strongly negative → brake > 0

        Engine RPM and transmission still update (via update_dynamics_complex)
        for realistic display — but they do NOT affect the actual force.
        """
        v = self.vehicle.state.vx
        F_aero = (0.5 * self.air_density * self.params.drag_coefficient *
                  self.params.frontal_area * v * abs(v))
        F_roll = (self.rolling_coeff * self.params.mass * self.params.gravity
                  if abs(v) > 0.01 else 0.0)
        self.vehicle.direct_force = self.params.mass * a_desired + F_aero + F_roll

        F_req_max = self.params.mass * self.params.max_acceleration + F_aero + F_roll
        F_req_min = -(self.params.mass * abs(self.params.max_deceleration) + F_aero + F_roll)
        if self.vehicle.direct_force >= 0:
            self.actual_throttle = np.clip(
                self.vehicle.direct_force / max(F_req_max, 1.0), 0.0, 1.0)
            self.actual_brake = 0.0
        else:
            self.actual_throttle = 0.0
            self.actual_brake = np.clip(
                abs(self.vehicle.direct_force) / max(abs(F_req_min), 1.0), 0.0, 1.0)
        self.throttle_input = self.actual_throttle
        self.brake_input = self.actual_brake
        self.vehicle.steering_input = 0.0

    # =========================================================================
    # COMPLEX DYNAMICS UPDATE
    # =========================================================================

    def update_dynamics_complex(self, dt: float):
        """Integrate full longitudinal dynamics (engine, transmission, Newton's 2nd law).

        Integration order is ZOH-consistent (same as the double integrator used
        by the upper controller for planning):
            x[k+1] = x[k] + v[k]·dt + 0.5·a·dt²   (position using OLD velocity)
            v[k+1] = v[k] + a·dt

        In autonomous mode (direct_force set by compute_control) the engine
        force path is bypassed; engine/transmission still update for display.
        """
        # Rate-limit pedals (no-op in autonomous mode since throttle_input=actual_throttle)
        self._rate_limit_pedals(dt)

        Fx, _ = self.calculate_forces()

        # Steering angle (lateral dynamics)
        steering_angle = self.vehicle.steering_input * self.params.max_steering_angle
        state = self.vehicle.state

        state.ax = Fx / self.params.mass

        if abs(state.vx) > 0.1:
            beta = np.arctan(self.params.lr * np.tan(steering_angle) / self.params.wheelbase)
            state.psi_dot = (state.vx * np.cos(beta) * np.tan(steering_angle)) / self.params.wheelbase
            state.vy = state.vx * np.sin(beta)
        else:
            state.psi_dot = 0.0
            state.vy = 0.0

        # ZOH-consistent integration: POSITION first (using OLD velocity)
        cos_psi = np.cos(state.psi)
        sin_psi = np.sin(state.psi)

        state.psi += state.psi_dot * dt
        state.x += ((state.vx * cos_psi - state.vy * sin_psi) * dt
                    + 0.5 * state.ax * cos_psi * dt * dt)
        state.y += ((state.vx * sin_psi + state.vy * cos_psi) * dt
                    + 0.5 * state.ax * sin_psi * dt * dt)

        # VELOCITY update (after position)
        state.vx += state.ax * dt
        state.vx = np.clip(state.vx, 0, self.params.max_velocity)

        # Update platoon compatibility variables
        self.vehicle.v = state.vx
        self.vehicle.a = state.ax

        # Update engine and transmission (for realistic RPM/gear display)
        if abs(state.vx) > 0.1:
            wheel_rpm = (state.vx * 60) / (2 * np.pi * self.params.wheel_radius)
            target_engine_rpm = wheel_rpm * self.transmission.get_effective_ratio()
            target_engine_rpm = max(target_engine_rpm, self.engine.idle_rpm)
            self.engine.update_rpm(target_engine_rpm, dt)
        else:
            self.engine.update_rpm(self.engine.idle_rpm, dt)

        self.transmission.update(self.engine.rpm, state.vx, dt)


__all__ = ['Engine', 'Transmission', 'LowerLevelController']
