"""
Vehicle class — abstract longitudinal vehicle model.

This module focuses on the vehicle as a mathematical abstraction:
state integration, state-space matrices, and mode dispatching.

All hardware-level logic (engine, transmission, pedal dynamics, force
calculations) lives in control/lower_level_controller.py, which also
defines the Engine and Transmission classes.
"""

import numpy as np
from typing import Tuple
from .components import VehicleParameters, VehicleState
from config import SIMULATION_DT


class Vehicle:
    """Main vehicle class.

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
        self.max_velocity = self.params.max_velocity  # m/s

        # ── Initial state ───────────────────────────────────────────────────
        self.state.x = initial_x
        self.state.y = initial_y
        self.state.psi = initial_heading
        self.state.vx = initial_velocity

        # ── Control inputs (live on Vehicle — used by upper controller) ─────
        self.direct_force = 0.0      # Direct longitudinal force [N] (autonomous)
        self.steering_input = 0.0    # -1 to 1
        self.a_desired = 0.0         # Desired acceleration [m/s²]

        # ── Mode flags ──────────────────────────────────────────────────────
        self.autonomous_mode = False
        self.use_kinematic_model = False
        self.use_state_space_model = False
        self.use_hierarchical_model = False

        # ── Platoon compatibility ────────────────────────────────────────────
        self.L = self.params.length
        self.v = initial_velocity
        self.a = 0.0

        # ── Platoon membership ───────────────────────────────────────────────
        self.joined_platoon = False
        self.joined_time = None

        # ── Nash acceleration override ───────────────────────────────────────
        self.nash_acceleration = None

        # ── Acceleration output filter (hierarchical mode) ───────────────────
        self._a_filtered = 0.0
        self._a_filter_initialized = False

        # ── State-space matrix cache ─────────────────────────────────────────
        self.A = None
        self.B1 = None
        self.B2 = None
        self.C = None

        # ── Lower-level controller (always present — owns hardware) ──────────
        # Created lazily on first use to avoid import-order issues.
        self._lower_level_controller = None
        # Convenience attributes: target_velocity for PlatoonManager.add_vehicle
        self.target_velocity = 0.0
        self.target_acceleration = 0.0

        self._debug_counter = 0

    # =========================================================================
    # LOWER-LEVEL CONTROLLER ACCESS
    # =========================================================================

    @property
    def lower_level_controller(self):
        if self._lower_level_controller is None:
            from control.lower_level_controller import LowerLevelController
            self._lower_level_controller = LowerLevelController(self)
        return self._lower_level_controller

    @lower_level_controller.setter
    def lower_level_controller(self, value):
        self._lower_level_controller = value

    # ── Hardware forwarding properties (backward compatibility) ─────────────

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
        """Select the motion model for this vehicle.

        Parameters
        ----------
        use_kinematic : bool
            Simple kinematic model.
        use_state_space : bool
            ZOH double-integrator (required for Nash solver).
        use_hierarchical : bool
            Upper controller plans with double integrator; lower-level
            controller converts a_desired to throttle/brake, and the complex
            dynamics model propagates the actual vehicle state.
        """
        self.use_kinematic_model = False
        self.use_state_space_model = False
        self.use_hierarchical_model = False

        if use_hierarchical:
            self.use_hierarchical_model = True
            self.use_state_space_model = True  # Nash still uses double integrator for planning
            model_name = "hierarchical (upper: double integrator, lower: complex dynamics)"
            # Ensure LLC exists and engine/transmission start in a realistic state
            self.lower_level_controller._sync_engine_to_velocity()
        elif use_state_space:
            self.use_state_space_model = True
            model_name = "state-space (ZOH double integrator)"
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
        """Update vehicle state — routes to the active motion model."""
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
        """Hierarchical update: upper controller → lower-level controller → dynamics.

        1. Upper controller (Nash/Rajamani/IDM) has written a_desired.
        2. lower_level_controller.compute_control(a_desired) converts it to
           direct_force + display throttle/brake (pure feedforward, no PI).
        3. lower_level_controller.update_dynamics_complex(dt) propagates state.
        4. An EMA filter smooths the noisy engine-level acceleration before
           it is fed back to the upper controller via vehicle.a.
        """
        a_desired = self.a_desired if hasattr(self, 'a_desired') else self.target_acceleration

        llc = self.lower_level_controller
        llc.compute_control(a_desired)

        was_autonomous = self.autonomous_mode
        self.autonomous_mode = True
        llc.update_dynamics_complex(dt)
        self.autonomous_mode = was_autonomous

        # Acceleration output filter (EMA)
        # Smooths gear-shift noise before upper controller (Rajamani/Nash) sees it.
        from config import LOWER_CTRL_ACCEL_FILTER_ALPHA
        a_raw = self.a

        if not self._a_filter_initialized:
            self._a_filtered = a_raw
            self._a_filter_initialized = True
        else:
            self._a_filtered = (LOWER_CTRL_ACCEL_FILTER_ALPHA * a_raw
                                + (1.0 - LOWER_CTRL_ACCEL_FILTER_ALPHA) * self._a_filtered)

        self.a = self._a_filtered
        self.state.ax = self._a_filtered

    # =========================================================================
    # KINEMATIC MODEL
    # =========================================================================

    def update_dynamics_kinematic(self, dt: float):
        """Update vehicle dynamics using the kinematic model."""
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
        """Kinematic bicycle model state update."""
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
    # STATE-SPACE MODEL (ZOH double integrator)
    # =========================================================================

    def update_dynamics_state_space(self, dt: float):
        """ZOH double-integrator update.

        x[k+1] = A_d @ x[k] + B_d @ u[k]
        State: [position, velocity]
        Input: acceleration (a_desired)
        """
        A_d, B_d, C = self.get_state_space_matrices(dt=dt)

        x_current = np.array([[self.state.x],
                               [self.state.vx]])

        u_acceleration = self.a_desired if hasattr(self, 'a_desired') else self.target_acceleration
        u_acceleration = np.clip(u_acceleration, -2.5, 2.5)
        self.a = u_acceleration
        self.state.ax = u_acceleration

        u1 = np.array([[u_acceleration]])
        x_next = A_d @ x_current + B_d @ u1

        self.state.x = float(x_next[0, 0])
        self.state.vx = float(x_next[1, 0])
        self.state.vx = np.clip(self.state.vx, 0, self.params.max_velocity)

        self.state.ax = u_acceleration
        self.a = u_acceleration
        self.v = self.state.vx
        self.state.vy = 0.0
        self.state.psi_dot = 0.0

    # =========================================================================
    # STATE-SPACE MATRICES
    # =========================================================================

    def get_state_space_matrices(self, dt: float = SIMULATION_DT):
        """Return ZOH discrete-time matrices (A_d, B_d, C) for the double integrator.

        Continuous-time:
            A = [[0, 1],   B = [[0],   C = I₂
                 [0, 0]]        [1]]

        ZOH discretization (exact analytical solution):
            A_d = [[1, dt],   B_d = [[0.5·dt²],
                   [0, 1 ]]          [dt     ]]
        """
        C = np.array([[1.0, 0.0],
                      [0.0, 1.0]])

        if dt is None:
            # Continuous-time matrices
            A = np.array([[0.0, 1.0], [0.0, 0.0]])
            B = np.array([[0.0], [1.0]])
            return A, B, C

        from scipy.linalg import expm

        A_cont = np.array([[0.0, 1.0], [0.0, 0.0]])
        B_cont = np.array([[0.0], [1.0]])

        n, m = 2, 1
        aug = np.zeros((n + m, n + m))
        aug[0:n, 0:n] = A_cont * dt
        aug[0:n, n:n+m] = B_cont * dt
        exp_aug = expm(aug)

        A_d = exp_aug[0:n, 0:n]
        B_d = exp_aug[0:n, n:n+m]

        self.A = A_d
        self.B1 = B_d
        self.B2 = B_d if self.autonomous_mode and self.vehicle_id == "Human" else None
        self.C = C
        return A_d, B_d, C

    # =========================================================================
    # HELPERS
    # =========================================================================

    def get_dynamic_max_acceleration(self) -> float:
        """Engine-based max acceleration ceiling (delegates to lower-level controller)."""
        if not (self.use_hierarchical_model or
                (not self.use_kinematic_model and not self.use_state_space_model)):
            return self.params.max_acceleration
        return self.lower_level_controller.get_dynamic_max_acceleration()

    def get_state_vector(self):
        """Return current state as [x, vx]ᵀ."""
        return np.array([[self.state.x],
                         [self.state.vx]])

    def set_state_vector(self, x_state):
        """Set state from [x, vx]ᵀ."""
        self.state.x = float(x_state[0, 0])
        self.state.vx = float(x_state[1, 0])
        self.v = self.state.vx

    def set_manual_inputs(self, throttle: float, brake: float, steering: float):
        """Set manual inputs for non-autonomous (human-driven) vehicles."""
        if not self.autonomous_mode:
            self.lower_level_controller.throttle_input = throttle
            self.lower_level_controller.brake_input = brake
            self.steering_input = np.clip(steering, -1.0, 1.0)

            # direct_force for state-space / kinematic fallback paths
            if throttle > 0.001:
                self.direct_force = throttle * self.params.mass * self.params.max_acceleration
            elif brake > 0.001:
                max_brake = self.params.mass * self.params.gravity * self.params.tire_friction_coeff
                self.direct_force = -brake * max_brake
            else:
                self.direct_force = 0.0

        self.steering_input = np.clip(steering, -1.0, 1.0)


__all__ = ['Vehicle']
