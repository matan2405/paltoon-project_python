"""
unified/vehicle/platoon_vehicle.py — Longitudinal-only Vehicle model for platoon members.

Copied from Longitudinal/vehicle/vehicle.py with all imports redirected to
unified/config.py and unified/vehicle/components.py.  No dependency on the
split Longitudinal/ or lateral/ module trees.

PLANT:      Nonlinear longitudinal model (RK4) — Belousov Eq. 3.11a
CONTROLLER: Dynamic linearization (Jacobians) at every timestep.
"""

import numpy as np
from scipy.linalg import expm
from typing import Tuple
from unified.vehicle.components import VehicleParameters, VehicleState
from unified.config import (
    SIMULATION_DT,
    LOWER_CTRL_ACCEL_FILTER_ALPHA,
    LONG_NASH_U1_MIN as NASH_U1_MIN,
    LONG_NASH_U1_MAX as NASH_U1_MAX,
)


class Vehicle:
    """Longitudinal vehicle model for platoon members (1D dynamics).

    Motion models
    -------------
    state-space  : ZOH double integrator — required for Nash solver.
    kinematic    : Simple kinematic bicycle model.
    hierarchical : Upper controller plans with double integrator; lower-level
                   controller (LowerLevelController) converts a_desired to
                   throttle/brake and propagates full dynamics.
    complex      : Full engine/transmission dynamics (no upper controller).
    """

    def __init__(self, initial_x: float = 0.0, initial_y: float = 0.0,
                 initial_heading: float = 0.0, vehicle_id: str = "Vehicle",
                 initial_velocity: float = 0.0):
        self.params = VehicleParameters()
        self.state = VehicleState()
        self.vehicle_id = vehicle_id
        self.human_vehicle = (vehicle_id == "Human")
        self.max_velocity = self.params.max_velocity

        self.state.x = initial_x
        self.state.y = initial_y
        self.state.psi = initial_heading
        self.state.vx = initial_velocity

        self.direct_force = 0.0
        self.steering_input = 0.0
        self.a_desired = 0.0

        self.autonomous_mode = False
        self.use_kinematic_model = False
        self.use_state_space_model = False
        self.use_hierarchical_model = False

        self.L = self.params.length
        self.v = initial_velocity
        self.a = 0.0

        self.joined_platoon = False
        self.joined_time = None

        self.nash_acceleration = None

        self.x_long = np.array([self.state.x, self.state.vx])
        self.A_c_current = None
        self.B_c_current = None
        self.A = None; self.B1 = None; self.B2 = None
        self.A_d = None; self.B_d = None

        r = self.params.wheel_radius
        self.m_eff = self.params.mass + self.params.wheel_inertia / (r * r)

        self._a_filtered = 0.0
        self._a_filter_initialized = False

        self._lower_level_controller = None
        self.target_velocity = 0.0
        self.target_acceleration = 0.0

        self._debug_counter = 0

    # =========================================================================
    # LOWER-LEVEL CONTROLLER ACCESS
    # =========================================================================

    @property
    def lower_level_controller(self):
        if self._lower_level_controller is None:
            from unified.control.lower_level_controller import LowerLevelController
            self._lower_level_controller = LowerLevelController(self)
        return self._lower_level_controller

    @lower_level_controller.setter
    def lower_level_controller(self, value):
        self._lower_level_controller = value

    @property
    def engine(self):
        return self.lower_level_controller.engine

    @property
    def transmission(self):
        return self.lower_level_controller.transmission

    @property
    def throttle_input(self):
        return self.lower_level_controller.throttle_input

    @throttle_input.setter
    def throttle_input(self, val):
        self.lower_level_controller.throttle_input = val

    @property
    def brake_input(self):
        return self.lower_level_controller.brake_input

    @brake_input.setter
    def brake_input(self, val):
        self.lower_level_controller.brake_input = val

    @property
    def actual_throttle(self):
        return self.lower_level_controller.actual_throttle

    @actual_throttle.setter
    def actual_throttle(self, val):
        self.lower_level_controller.actual_throttle = val

    @property
    def actual_brake(self):
        return self.lower_level_controller.actual_brake

    @actual_brake.setter
    def actual_brake(self, val):
        self.lower_level_controller.actual_brake = val

    # =========================================================================
    # STATE PROPERTIES
    # =========================================================================

    @property
    def x(self) -> float:
        return self.state.x

    @property
    def y(self) -> float:
        return self.state.y

    @property
    def psi(self) -> float:
        return self.state.psi

    @property
    def vx(self) -> float:
        return self.state.vx

    @property
    def vy(self) -> float:
        return self.state.vy

    # =========================================================================
    # MODEL SELECTION
    # =========================================================================

    def set_motion_model(self, use_kinematic: bool, use_state_space: bool = False,
                         use_hierarchical: bool = False):
        self.use_kinematic_model = False
        self.use_state_space_model = False
        self.use_hierarchical_model = False

        if use_hierarchical:
            self.use_hierarchical_model = True
            self.use_state_space_model = True
            model_name = "hierarchical"
            self.lower_level_controller._sync_engine_to_velocity()
        elif use_state_space:
            self.use_state_space_model = True
            model_name = "state-space"
        elif use_kinematic:
            self.use_kinematic_model = True
            model_name = "kinematic"
        else:
            model_name = "complex dynamics"

        print(f"{self.vehicle_id}: Motion model set to {model_name}")

    # =========================================================================
    # DYNAMICS DISPATCH
    # =========================================================================

    def update_dynamics(self, dt: float):
        if self.use_hierarchical_model:
            self.update_dynamics_hierarchical(dt)
        elif self.use_state_space_model:
            self.update_dynamics_state_space(dt)
        elif self.use_kinematic_model:
            self.update_dynamics_kinematic(dt)
        else:
            self.lower_level_controller.update_dynamics_complex(dt)

    # =========================================================================
    # HIERARCHICAL MODEL
    # =========================================================================

    def update_dynamics_hierarchical(self, dt: float):
        a_desired = self.a_desired if hasattr(self, 'a_desired') else self.target_acceleration

        llc = self.lower_level_controller
        Fxf = self._compute_Fxf(a_desired)
        llc.compute_control(Fxf)

        was_autonomous = self.autonomous_mode
        self.autonomous_mode = True
        llc.update_dynamics_complex(dt)
        self.autonomous_mode = was_autonomous

        a_raw = self.a
        if not self._a_filter_initialized:
            self._a_filtered = a_raw
            self._a_filter_initialized = True
        else:
            self._a_filtered = (LOWER_CTRL_ACCEL_FILTER_ALPHA * a_raw
                                + (1.0 - LOWER_CTRL_ACCEL_FILTER_ALPHA) * self._a_filtered)

        self.a = self._a_filtered
        self.state.ax = self._a_filtered

        self.x_long = np.array([self.state.x, self.state.vx])
        a_des = self.a_desired if hasattr(self, 'a_desired') else self.target_acceleration
        self._update_continuous_jacobians(a_des)

    # =========================================================================
    # KINEMATIC MODEL
    # =========================================================================

    def update_dynamics_kinematic(self, dt: float):
        if not self.autonomous_mode:
            Fx, _ = self.lower_level_controller.calculate_forces()
            target_acceleration = Fx / self.params.mass
        else:
            target_acceleration = self.a_desired if hasattr(self, 'a_desired') else self.target_acceleration

        new_velocity = self.v + target_acceleration * dt
        new_velocity = np.clip(new_velocity, 0, self.max_velocity)

        delta_f = self.steering_input * self.params.max_steering_angle
        delta_r = 0.0
        self._state_equation_platoon(delta_f, delta_r, new_velocity, dt)

    def _state_equation_platoon(self, delta_f, delta_r, V, dt):
        l_f = self.params.lf
        l_r = self.params.lr
        L = self.params.wheelbase

        beta = np.arctan((l_f * np.tan(delta_r) + l_r * np.tan(delta_f)) / L)
        x_dot = V * np.cos(self.state.psi + beta)
        y_dot = V * np.sin(self.state.psi + beta)
        psi_dot = (np.tan(delta_f) - np.tan(delta_r)) * np.cos(beta) * V / L

        self.state.x += x_dot * dt
        self.state.y += y_dot * dt
        self.state.psi += psi_dot * dt

        self.a = (V - self.v) / dt
        self.v = V
        self.state.vx = V
        self.state.ax = self.a

    # =========================================================================
    # NONLINEAR PLANT — Belousov Eq. 3.11a
    # =========================================================================

    def _discretize_zoh(self, A_c: np.ndarray, B_c: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        n, m = A_c.shape[0], B_c.shape[1]
        aug = np.zeros((n + m, n + m))
        aug[:n, :n] = A_c * dt
        aug[:n, n:] = B_c * dt
        e = expm(aug)
        return e[:n, :n], e[:n, n:]

    def _tire_force(self, alpha: float, C: float, mu: float, Fz: float) -> float:
        theta = C / (3.0 * mu * Fz)
        S = np.tan(alpha)
        if abs(S) <= 1.0 / theta:
            return mu * Fz * (3 * theta * S - 3 * theta**2 * S**2 + theta**3 * S**3)
        return float(np.sign(S)) * mu * Fz

    def _compute_Fxf(self, u_accel: float) -> float:
        vx = self.state.vx
        p = self.params
        vy = self.state.vy
        Omega_z = self.state.psi_dot
        delta_f = self.state.delta_f

        f_vx = (np.tanh(10.0 * abs(vx) - 8.0) + 1.0) / 2.0
        Ra = 0.5 * p.air_density * p.drag_coefficient * p.frontal_area * vx * abs(vx)
        Rg = p.mass * p.gravity * np.sin(p.road_grade)
        vx_sign = np.sign(vx) if abs(vx) > 1e-6 else 0.0
        Rr = f_vx * p.rolling_resistance_coeff * p.mass * p.gravity * vx_sign

        vx_safe = max(abs(vx), 0.1)
        alpha_f = f_vx * (delta_f - np.arctan2(vy + p.lf * Omega_z, vx_safe))
        Fz_f = p.mass * p.gravity * p.lr / (p.lf + p.lr)
        Fyf = self._tire_force(alpha_f, p.Caf, p.tire_friction_coeff, Fz_f)

        cos_df = np.cos(delta_f)
        Fxf_desired = (self.m_eff * u_accel + Ra + Rg + Rr + Fyf * np.sin(delta_f)) / max(abs(cos_df), 1e-6)

        kappa = Fxf_desired / p.Cx
        Fxf = self._tire_force(np.arctan(kappa), p.Cx, p.tire_friction_coeff, Fz_f)

        self.state.Fxf = Fxf
        return Fxf

    def _longitudinal_derivatives(self, x_long: np.ndarray, Fxf: float) -> np.ndarray:
        _, vx = x_long
        p = self.params
        vy = self.state.vy
        Omega_z = self.state.psi_dot
        delta_f = self.state.delta_f

        f_vx = (np.tanh(10.0 * abs(vx) - 8.0) + 1.0) / 2.0
        Ra = 0.5 * p.air_density * p.drag_coefficient * p.frontal_area * vx * abs(vx)
        Rg = p.mass * p.gravity * np.sin(p.road_grade)
        vx_sign = np.sign(vx) if abs(vx) > 1e-6 else 0.0
        Rr = f_vx * p.rolling_resistance_coeff * p.mass * p.gravity * vx_sign

        vx_safe = max(abs(vx), 0.1)
        alpha_f = f_vx * (delta_f - np.arctan2(vy + p.lf * Omega_z, vx_safe))
        Fz_f = p.mass * p.gravity * p.lr / (p.lf + p.lr)
        Fyf = self._tire_force(alpha_f, p.Caf, p.tire_friction_coeff, Fz_f)

        cos_df = np.cos(delta_f)
        sin_df = np.sin(delta_f)

        vx_dot = vy * Omega_z + (Fxf * cos_df - Fyf * sin_df - Ra - Rg - Rr) / self.m_eff
        return np.array([vx, vx_dot])

    def _update_continuous_jacobians(self, u_accel: float):
        eps = 1e-5
        Fxf = self._compute_Fxf(u_accel)
        f0 = self._longitudinal_derivatives(self.x_long, Fxf)
        n = 2
        A_c = np.zeros((n, n))
        for j in range(n):
            xp = self.x_long.copy()
            xp[j] += eps
            A_c[:, j] = (self._longitudinal_derivatives(xp, Fxf) - f0) / eps
        Fxf_p = self._compute_Fxf(u_accel + eps)
        B_c = (self._longitudinal_derivatives(self.x_long, Fxf_p) - f0).reshape(n, 1) / eps
        self.state.Fxf = Fxf
        self.A_c_current = A_c
        self.B_c_current = B_c

    def _linearize_at_current_state(self, u_accel: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        self._update_continuous_jacobians(u_accel)
        A_d, B_d = self._discretize_zoh(self.A_c_current, self.B_c_current, dt)
        self.A = A_d; self.B1 = B_d; self.B2 = B_d.copy()
        self.A_d = A_d; self.B_d = B_d
        return A_d, B_d

    # =========================================================================
    # STATE-SPACE MODEL
    # =========================================================================

    def update_dynamics_state_space(self, dt: float):
        if self.A_c_current is None:
            self._update_continuous_jacobians(0.0)

        u_accel = np.clip(
            self.a_desired if hasattr(self, 'a_desired') else self.target_acceleration,
            self.params.max_deceleration, self.params.max_acceleration)

        Fxf = self._compute_Fxf(u_accel)

        f = self._longitudinal_derivatives
        k1 = f(self.x_long, Fxf)
        k2 = f(self.x_long + 0.5 * dt * k1, Fxf)
        k3 = f(self.x_long + 0.5 * dt * k2, Fxf)
        k4 = f(self.x_long + dt * k3,        Fxf)
        self.x_long = self.x_long + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.x_long[1] = np.clip(self.x_long[1], 0.0, self.params.max_velocity)

        self.state.x = float(self.x_long[0])
        self.state.vx = float(self.x_long[1])
        self.v = self.state.vx
        self.a = float(k1[1])
        self.state.ax = self.a
        self.state.vy = 0.0
        self.state.psi_dot = 0.0

        self._update_continuous_jacobians(u_accel)

        if not np.isfinite(self.x_long).all():
            print(f"WARNING {self.vehicle_id}: NaN detected, resetting vx")
            self.x_long[1] = 0.0

    # =========================================================================
    # STATE-SPACE MATRICES
    # =========================================================================

    def get_state_space_matrices(self, dt: float = SIMULATION_DT) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        C = np.eye(2)
        if self.use_hierarchical_model:
            A_d, B_d = self._discretize_zoh(np.array([[0., 1.], [0., 0.]]),
                                             np.array([[0.], [1.]]), dt)
            return A_d, B_d, C
        if self.A_c_current is not None:
            A_d, B_d = self._discretize_zoh(self.A_c_current, self.B_c_current, dt)
            return A_d, B_d, C
        A_d, B_d = self._discretize_zoh(np.array([[0., 1.], [0., 0.]]),
                                         np.array([[0.], [1.]]), dt)
        return A_d, B_d, C

    # =========================================================================
    # HELPERS
    # =========================================================================

    def get_dynamic_max_acceleration(self) -> float:
        if not (self.use_hierarchical_model or
                (not self.use_kinematic_model and not self.use_state_space_model)):
            return self.params.max_acceleration
        return self.lower_level_controller.get_dynamic_max_acceleration()

    def get_u_bounds(self) -> Tuple[float, float]:
        p = self.params
        vx = self.state.vx
        vy = self.state.vy
        Omega_z = self.state.psi_dot
        delta_f = self.state.delta_f

        f_vx = (np.tanh(10.0 * abs(vx) - 8.0) + 1.0) / 2.0
        Ra = 0.5 * p.air_density * p.drag_coefficient * p.frontal_area * vx * abs(vx)
        Rg = p.mass * p.gravity * np.sin(p.road_grade)
        vx_sign = np.sign(vx) if abs(vx) > 1e-6 else 0.0
        Rr = f_vx * p.rolling_resistance_coeff * p.mass * p.gravity * vx_sign
        F_resist = Ra + Rg + Rr

        vx_safe = max(abs(vx), 0.1)
        alpha_f = f_vx * (delta_f - np.arctan2(vy + p.lf * Omega_z, vx_safe))
        Fz_f = p.mass * p.gravity * p.lr / (p.lf + p.lr)
        Fyf = self._tire_force(alpha_f, p.Caf, p.tire_friction_coeff, Fz_f)
        cos_df = np.cos(delta_f)
        lateral_coupling = Fyf * np.sin(delta_f)

        llc = self.lower_level_controller
        engine_force = llc.compute_max_engine_force() * llc.transmission.get_torque_multiplier()
        tire_max = p.tire_friction_coeff * Fz_f
        Fxf_max = min(engine_force, tire_max)
        Fxf_min = -llc.compute_max_brake_force()

        cos_df_safe = max(abs(cos_df), 1e-6) * np.sign(cos_df) if abs(cos_df) > 1e-6 else 1.0
        u_max = (Fxf_max * cos_df_safe - F_resist - lateral_coupling) / self.m_eff
        u_min = (Fxf_min * cos_df_safe - F_resist - lateral_coupling) / self.m_eff

        u_max = float(np.clip(u_max, NASH_U1_MIN, NASH_U1_MAX))
        u_min = float(np.clip(u_min, NASH_U1_MIN, NASH_U1_MAX))
        return u_min, u_max

    def get_state_vector(self):
        return np.array([[self.state.x],
                         [self.state.vx]])

    def set_state_vector(self, x_state):
        self.state.x = float(x_state[0, 0])
        self.state.vx = float(x_state[1, 0])
        self.v = self.state.vx
        self.x_long = np.array([self.state.x, self.state.vx])

    def set_manual_inputs(self, throttle: float, brake: float, steering: float):
        if not self.autonomous_mode:
            self.lower_level_controller.throttle_input = throttle
            self.lower_level_controller.brake_input = brake
            self.steering_input = np.clip(steering, -1.0, 1.0)

            if throttle > 0.001:
                self.direct_force = throttle * self.params.mass * self.params.max_acceleration
            elif brake > 0.001:
                max_brake = self.params.mass * self.params.gravity * self.params.tire_friction_coeff
                self.direct_force = -brake * max_brake
            else:
                self.direct_force = 0.0

        self.steering_input = np.clip(steering, -1.0, 1.0)


__all__ = ['Vehicle']
