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
    │  │  F_ff = m·a_des       │  │
    │  │      + F_drag(v)      │  │
    │  │      + F_roll         │  │
    │  └───────────────────────┘  │
    │  ┌───────────────────────┐  │
    │  │ Feedback (PI):        │  │
    │  │  e = a_des - a_actual │  │
    │  │  F_fb = Kp·m·e       │  │
    │  │       + Ki·m·∫e dt    │  │
    │  └───────────────────────┘  │
    │  ┌───────────────────────┐  │
    │  │ Actuator Mapping:     │  │
    │  │  F_total > 0 → throttle│ │
    │  │  F_total < 0 → brake  │  │
    │  └───────────────────────┘  │
    └─────────────────────────────┘
        │   throttle, brake
        ▼
    Complex Vehicle Dynamics (engine, transmission, aero, tires)
        │   a_actual, v, x
        ▼
    Updated Vehicle State

The upper controller plans using a simplified double-integrator model
(x[k+1] = A·x[k] + B·u[k]), while the lower controller ensures the
actual vehicle with full longitudinal dynamics tracks the commanded
acceleration.

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
    LOWER_CTRL_AIR_DENSITY,
    LOWER_CTRL_ROLLING_COEFF,
    LOWER_CTRL_GEAR_SHIFT_HOLD,
)


class LowerLevelController:
    """
    Lower-level controller that converts desired acceleration (from upper controller)
    to throttle/brake commands, compensating for longitudinal vehicle dynamics.

    The controller has three stages:
    1. Feedforward: Computes the total force required to achieve a_desired,
       compensating for known disturbances (aerodynamic drag, rolling resistance).
    2. Feedback (PI): Corrects for model mismatch between the double-integrator
       planning model and the actual vehicle dynamics.
    3. Actuator mapping: Converts the required net force to throttle (engine)
       or brake input, respecting engine torque limits and brake capacity.
    """

    def __init__(self, vehicle):
        """
        Initialize lower-level controller.

        Parameters
        ----------
        vehicle : Vehicle
            The vehicle instance this controller manages.
        """
        self.vehicle = vehicle
        self.params = vehicle.params

        # PI controller gains
        self.Kp = LOWER_CTRL_KP
        self.Ki = LOWER_CTRL_KI

        # Integrator state and anti-windup limit
        self.error_integral = 0.0
        self.integrator_limit = LOWER_CTRL_INTEGRATOR_LIMIT  # [m/s²·s]

        # Actuator smoothing (first-order filter for throttle/brake)
        self.throttle_smoothing = LOWER_CTRL_THROTTLE_SMOOTHING
        self.brake_smoothing = LOWER_CTRL_BRAKE_SMOOTHING
        self.last_throttle = 0.0
        self.last_brake = 0.0

        # Dead-band for switching between throttle and brake
        self.deadband = LOWER_CTRL_DEADBAND  # [N]

        # Physical constants
        self.air_density = LOWER_CTRL_AIR_DENSITY  # [kg/m³]
        self.rolling_coeff = LOWER_CTRL_ROLLING_COEFF

        # Debug
        self._debug_counter = 0

    def compute_resistive_forces(self, velocity: float) -> float:
        """
        Compute the total resistive force acting on the vehicle at current velocity.

        F_resist = F_aero + F_rolling

        Parameters
        ----------
        velocity : float
            Current longitudinal velocity [m/s].

        Returns
        -------
        float
            Total resistive force [N] (always >= 0).
        """
        # Aerodynamic drag: F_aero = 0.5 * rho * Cd * A * v²
        F_aero = (0.5 * self.air_density * self.params.drag_coefficient
                  * self.params.frontal_area * velocity * abs(velocity))

        # Rolling resistance: F_roll = f_r * m * g  (only when moving)
        if abs(velocity) > 0.01:
            F_rolling = self.rolling_coeff * self.params.mass * self.params.gravity
        else:
            F_rolling = 0.0

        return F_aero + F_rolling

    def compute_max_engine_force(self) -> float:
        """
        Compute maximum available engine drive force at current RPM.

        F_max = T_engine(rpm) * i_total / r_wheel

        Returns
        -------
        float
            Maximum engine force at the wheels [N].
        """
        engine = self.vehicle.engine
        transmission = self.vehicle.transmission

        engine_torque = engine.get_torque(engine.rpm)
        total_ratio = transmission.get_total_ratio()
        max_force = engine_torque * total_ratio / self.params.wheel_radius

        return max(max_force, 1.0)  # Avoid division by zero

    def compute_max_brake_force(self) -> float:
        """
        Compute maximum available braking force.

        F_brake_max = m * g * mu

        Returns
        -------
        float
            Maximum braking force [N].
        """
        return self.params.mass * self.params.gravity * self.params.tire_friction_coeff

    def compute_control(self, a_desired: float, dt: float) -> tuple:
        """
        Main control law: convert desired acceleration to throttle/brake commands.

        Parameters
        ----------
        a_desired : float
            Desired acceleration from upper controller [m/s²].
        dt : float
            Time step [s].

        Returns
        -------
        tuple
            (throttle, brake) where throttle in [0, 1] and brake in [0, 1].
        """
        velocity = self.vehicle.state.vx
        a_actual = self.vehicle.state.ax

        # =================================================================
        # Gear-shift hold (from Unity: RPM holds during shifts)
        # During gear transitions the available engine force changes rapidly.
        # Instead of letting the PI chase this transient, hold the last
        # actuator output until the shift is complete.
        # =================================================================
        if (LOWER_CTRL_GEAR_SHIFT_HOLD and
                hasattr(self.vehicle, 'transmission') and
                self.vehicle.transmission.is_shifting):
            return self.last_throttle, self.last_brake

        # ====================================================================
        # Stage 1: Feedforward — compensate for known resistive forces
        # ====================================================================
        F_resist = self.compute_resistive_forces(velocity)
        F_feedforward = self.params.mass * a_desired + F_resist

        # ====================================================================
        # Stage 2: Feedback (PI) — correct for model mismatch
        # ====================================================================
        accel_error = a_desired - a_actual

        # Update integrator with anti-windup
        self.error_integral += accel_error * dt
        self.error_integral = np.clip(
            self.error_integral,
            -self.integrator_limit,
            self.integrator_limit
        )

        F_feedback = self.params.mass * (
            self.Kp * accel_error + self.Ki * self.error_integral
        )

        # ====================================================================
        # Stage 3: Total required force
        # ====================================================================
        F_total = F_feedforward + F_feedback

        # ====================================================================
        # Stage 4: Actuator mapping (force → throttle / brake)
        # ====================================================================
        if F_total > self.deadband:
            # Positive force → need engine drive
            max_engine_force = self.compute_max_engine_force()
            raw_throttle = np.clip(F_total / max_engine_force, 0.0, 1.0)
            raw_brake = 0.0

            # Reset integrator when switching from brake to throttle
            if self.last_brake > 0.01:
                self.error_integral = 0.0  # Full reset on mode switch

        elif F_total < -self.deadband:
            # Negative force → need braking
            max_brake_force = self.compute_max_brake_force()
            raw_throttle = 0.0
            raw_brake = np.clip(abs(F_total) / max_brake_force, 0.0, 1.0)

            # Reset integrator when switching from throttle to brake
            if self.last_throttle > 0.01:
                self.error_integral = 0.0  # Full reset on mode switch
        else:
            # In deadband — coast
            raw_throttle = 0.0
            raw_brake = 0.0

        # ====================================================================
        # Stage 5: Smooth actuator commands (avoid step changes)
        # ====================================================================
        alpha_t = self.throttle_smoothing
        alpha_b = self.brake_smoothing
        throttle = alpha_t * raw_throttle + (1 - alpha_t) * self.last_throttle
        brake = alpha_b * raw_brake + (1 - alpha_b) * self.last_brake

        # Mutual exclusion: can't throttle and brake simultaneously
        if throttle > 0.01 and brake > 0.01:
            if F_total > 0:
                brake = 0.0
            else:
                throttle = 0.0

        # Store for next step
        self.last_throttle = throttle
        self.last_brake = brake

        return throttle, brake

    def reset(self):
        """Reset controller state (e.g., on mode switch)."""
        self.error_integral = 0.0
        self.last_throttle = 0.0
        self.last_brake = 0.0


__all__ = ['LowerLevelController']
