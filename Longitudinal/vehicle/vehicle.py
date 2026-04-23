"""
Vehicle class — longitudinal vehicle model.

Research basis and adopted elements:
1) Belousov longitudinal vehicle model:
    - Nonlinear longitudinal equation of motion with resistive forces.
    - RK4 propagation for simulation-time plant updates.
2) Rajamani/Fernandez modeling ingredients used in this implementation:
    - Drag/rolling resistance structure and low-speed blending.
    - Coupling-friendly state representation for control-oriented prediction.
3) MPC/Nash prediction practice:
    - Successive linearization around current operating point.
    - ZOH discretization at control timestep for solver matrices.

PLANT:      Nonlinear longitudinal model (RK4) — Belousov Eq. 3.11a
CONTROLLER: Dynamic linearization (Jacobians) — at every timestep, A_t and B_t
            are recomputed at the current operating point x_long(t) and used
            by the Nash MPC as prediction matrices.

Two-timestep simulation loop:
  Every SIMULATION_DT (0.01 s):
    1. RK4: x_long(t+dt) from full nonlinear Belousov Eq. 3.11a
    2. Update continuous Jacobians A_c, B_c at new state (no ZOH)
  Every NASH_CONTROL_DT (0.1 s):
    3. Nash reads A_c, B_c via get_state_space_matrices(NASH_CONTROL_DT)
    4. ZOH discretization at NASH_CONTROL_DT: A_d, B_d = expm([A_c·T, B_c·T; 0, 0])
    5. Nash solves for u_accel; held constant for the next 10 sim steps

Nonlinear EOM — Belousov Eq. 3.11a (no small-angle assumption):
  m_eff*(v̇x - vy·Ωz) = Fxf·cos(δf) + Fxr - Fyf·sin(δf) - Ra - Rg - Rr

Feedforward cancellation (Nash stays in m/s², B_c = [[0],[1]] preserved):
  Fxf·cos(δf) + Fxr = m_eff·u_accel + Ra + Rg + Rr + Fyf·sin(δf)
  ⟹ v̇x = vy·Ωz + u_accel  (double integrator seen by Nash)

All hardware-level logic (engine, transmission, pedal dynamics, force
calculations) lives in control/lower_level_controller.py.
"""

import numpy as np
from scipy.linalg import expm
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
        """Initialize vehicle states, model flags, and lower-level control interfaces."""
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
        self.a_desired = 0.0         # Desired acceleration [m/s²] — set by Nash/platoon/IDM
        # Fxf [N] is stored in self.state.Fxf (computed by _compute_Fxf each step)

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

        # ── Nonlinear longitudinal plant (kept behaviorally aligned with lateral vehicle model) ──────
        self.x_long = np.array([self.state.x, self.state.vx])  # [x, vx]
        self.A_c_current = None   # continuous Jacobian (cached for re-discretization)
        self.B_c_current = None
        # These are read by Nash solver (same attr names as lateral vehicle):
        self.A = None; self.B1 = None; self.B2 = None
        self.A_d = None; self.B_d = None
        # Effective mass — Belousov Eq. 3.8: m_eff = m + Iw/r²
        r = self.params.wheel_radius
        self.m_eff = self.params.mass + self.params.wheel_inertia / (r * r)

        # ── Acceleration output filter (hierarchical mode) ───────────────────
        self._a_filtered = 0.0
        self._a_filter_initialized = False

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
        # Feedforward: convert Nash u_accel [m/s²] → Fxf [N] (stored in self.state.Fxf)
        Fxf = self._compute_Fxf(a_desired)
        llc.compute_control(Fxf)

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

        # Sync x_long with LLC-propagated state and update Jacobians (step 7)
        # ZOH is computed once per Nash step via get_state_space_matrices(NASH_CONTROL_DT)
        self.x_long = np.array([self.state.x, self.state.vx])
        a_des = self.a_desired if hasattr(self, 'a_desired') else self.target_acceleration
        self._update_continuous_jacobians(a_des)

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
    # NONLINEAR PLANT — Belousov Eq. 3.11a
    # =========================================================================

    def _discretize_zoh(self, A_c: np.ndarray, B_c: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """ZOH discretization via matrix exponential of the augmented system."""
        n, m = A_c.shape[0], B_c.shape[1]
        aug = np.zeros((n + m, n + m))
        aug[:n, :n] = A_c * dt
        aug[:n, n:] = B_c * dt
        e = expm(aug)
        return e[:n, :n], e[:n, n:]

    def _tire_force(self, alpha: float, C: float, mu: float, Fz: float) -> float:
        """Nonlinear lateral tire force — Rajamani Eq. 13.45 (parabolic pressure distribution).

        Recovers the linear model Fy = C·α for small slip angles.
        Saturates at μ·Fz for full sliding.
        """
        theta = C / (3.0 * mu * Fz)
        S = np.tan(alpha)
        if abs(S) <= 1.0 / theta:
            return mu * Fz * (3 * theta * S - 3 * theta**2 * S**2 + theta**3 * S**3)
        return float(np.sign(S)) * mu * Fz

    def _compute_Fxf(self, u_accel: float) -> float:
        """Feedforward cancellation + parabolic saturation: Nash u_accel [m/s²] → Fxf [N].

        Three-step process:
          1. Feedforward: isolate Fxf_desired from Belousov Eq. 3.11a (FWD, Fxr=0):
               Fxf_desired = (m_eff·u_accel + Ra + Rg + Rr + Fyf·sin(δf)) / cos(δf)
          2. Longitudinal slip ratio (linear approximation — Rajamani Eq. 13.45):
               κ = Fxf_desired / Cx
          3. Parabolic saturation (same Rajamani Eq. 13.45 model as lateral, S←κ):
               Fxf = _tire_force(arctan(κ), Cx, μ, Fz_f)
             Recovers Fxf ≈ Cx·κ = Fxf_desired for small slip; saturates at μ·Fz_f.

        Fyf (lateral cross-coupling) also uses Rajamani Eq. 13.45.
        Result stored in self.state.Fxf for lateral coupling (Fxf·sin(δf) terms in Eq. 3.11b/c).
        """
        vx = self.state.vx
        p = self.params
        vy = self.state.vy
        Omega_z = self.state.psi_dot
        delta_f = self.state.delta_f

        # Low-speed blending (Fernández Eq. 4.4)
        f_vx = (np.tanh(10.0 * abs(vx) - 8.0) + 1.0) / 2.0

        # Aerodynamic drag — Fernández Eq. 3.6 / Belousov Eq. 3.5
        Ra = 0.5 * p.air_density * p.drag_coefficient * p.frontal_area * vx * abs(vx)
        # Grade resistance — Belousov Eq. 3.11a
        Rg = p.mass * p.gravity * np.sin(p.road_grade)
        # Rolling resistance — Fernández Eq. 3.7 + 4.4
        vx_sign = np.sign(vx) if abs(vx) > 1e-6 else 0.0
        Rr = f_vx * p.rolling_resistance_coeff * p.mass * p.gravity * vx_sign

        # Front slip angle — Rajamani Eq. 2.27
        vx_safe = max(abs(vx), 0.1)
        alpha_f = f_vx * (delta_f - np.arctan2(vy + p.lf * Omega_z, vx_safe))
        # Static normal load on front axle (50/50 CG assumed)
        Fz_f = p.mass * p.gravity * p.lr / (p.lf + p.lr)
        # Nonlinear front lateral tire force — Rajamani Eq. 13.45
        Fyf = self._tire_force(alpha_f, p.Caf, p.tire_friction_coeff, Fz_f)

        # Step 1 — Feedforward: isolate desired Fxf from Belousov Eq. 3.11a
        cos_df = np.cos(delta_f)
        Fxf_desired = (self.m_eff * u_accel + Ra + Rg + Rr + Fyf * np.sin(delta_f)) / max(abs(cos_df), 1e-6)

        # Step 2 — Longitudinal slip ratio κ (Rajamani Ch. 13: κ ≈ Fxf / Cx for small slip)
        kappa = Fxf_desired / p.Cx

        # Step 3 — Parabolic saturation (Rajamani Eq. 13.45, Rajamani Table 13.4 params):
        #   _tire_force uses S = tan(alpha), so passing arctan(κ) maps S ← κ correctly.
        #   Saturates at μ·Fz_f; recovers Fxf ≈ Cx·κ = Fxf_desired for small κ.
        Fxf = self._tire_force(np.arctan(kappa), p.Cx, p.tire_friction_coeff, Fz_f)

        # Store for lateral coupling (Fxf·sin(δf) in Belousov Eq. 3.11b/c)
        self.state.Fxf = Fxf
        return Fxf

    def _longitudinal_derivatives(self, x_long: np.ndarray, Fxf: float) -> np.ndarray:
        """Full nonlinear longitudinal EOM — Belousov Eq. 3.11a (no small-angle assumption).

        State: [x, vx],  Input: Fxf [N] = front axle drive force (FWD: Fxr = 0)

        m_eff*(v̇x - vy·Ωz) = Fxf·cos(δf) - Fyf·sin(δf) - Ra - Rg - Rr  [Belousov Eq. 3.11a]

        Rearranged:
          v̇x = vy·Ωz + (Fxf·cos(δf) - Fyf·sin(δf) - Ra - Rg - Rr) / m_eff

        Due to feedforward cancellation (Fxf = (m_eff·u_accel + Ra + Rg + Rr + Fyf·sin(δf))/cos(δf)):
          v̇x = vy·Ωz + u_accel  (double integrator seen by Nash)

        External params read from self.state (zero for straight-line platooning):
          vy    = self.state.vy        [lateral velocity — Coriolis coupling]
          Ωz    = self.state.psi_dot  [yaw rate — Coriolis coupling]
          δf    = self.state.delta_f  [front steering angle — exact sin(δf)]

        Ra    = 0.5*ρ*Cd*Af*vx*|vx|              [Fernández Eq. 3.6, Belousov Eq. 3.5]
        Rg    = m·g·sin(θ_road)                   [Belousov Eq. 3.11a grade resistance]
        Rr    = f_vx * fr * m * g * sign(vx)     [Fernández Eq. 3.7]
        f_vx  = (tanh(10*|vx| - 8) + 1) / 2     [Fernández Eq. 4.4]
        Fyf   = Rajamani Eq. 13.45 nonlinear parabolic tire model
        α_f   = f_vx * (δf - arctan2(vy + lf·Ωz, max(|vx|,0.1)))  [Rajamani Eq. 2.27]
        m_eff = m + wheel_inertia / r²            [Belousov Eq. 3.8]
        """
        _, vx = x_long
        p = self.params
        # External state (Coriolis and steering coupling — zero for straight-line)
        vy = self.state.vy
        Omega_z = self.state.psi_dot
        delta_f = self.state.delta_f

        # Low-speed blending (Fernández Eq. 4.4)
        f_vx = (np.tanh(10.0 * abs(vx) - 8.0) + 1.0) / 2.0

        # Aerodynamic drag — Fernández Eq. 3.6 / Belousov Eq. 3.5 (direction-correct)
        Ra = 0.5 * p.air_density * p.drag_coefficient * p.frontal_area * vx * abs(vx)
        # Grade resistance — Belousov Eq. 3.11a
        Rg = p.mass * p.gravity * np.sin(p.road_grade)
        # Rolling resistance — Fernández Eq. 3.7 + 4.4
        vx_sign = np.sign(vx) if abs(vx) > 1e-6 else 0.0
        Rr = f_vx * p.rolling_resistance_coeff * p.mass * p.gravity * vx_sign

        # Front lateral tire force — slip angle per Rajamani Eq. 2.27, nonlinear Rajamani 13.45
        # (cross-coupling: Fyf·sin(δf) term in Belousov Eq. 3.11a)
        vx_safe = max(abs(vx), 0.1)
        alpha_f = f_vx * (delta_f - np.arctan2(vy + p.lf * Omega_z, vx_safe))
        Fz_f = p.mass * p.gravity * p.lr / (p.lf + p.lr)
        Fyf = self._tire_force(alpha_f, p.Caf, p.tire_friction_coeff, Fz_f)

        # Exact steering geometry — no small-angle approximation
        cos_df = np.cos(delta_f)
        sin_df = np.sin(delta_f)

        # Belousov Eq. 3.11a — full form (FWD: Fxr=0, so only Fxf·cos(δf) in drive term)
        vx_dot = vy * Omega_z + (Fxf * cos_df - Fyf * sin_df - Ra - Rg - Rr) / self.m_eff
        return np.array([vx, vx_dot])

    def _update_continuous_jacobians(self, u_accel: float):
        """Update continuous Jacobians A_c, B_c at the current operating point.

        Called every simulation step (SIMULATION_DT = 0.01 s) — no ZOH,
        only numerical forward differences.  Nash re-discretizes via
        get_state_space_matrices(dt=NASH_CONTROL_DT) once per Nash step (0.1 s),
        so ZOH is always computed at the correct Nash timestep.
        """
        eps = 1e-5
        # Feedforward: convert Nash u_accel [m/s²] → Fxf [N] at current state
        Fxf = self._compute_Fxf(u_accel)
        f0 = self._longitudinal_derivatives(self.x_long, Fxf)
        n = 2
        # A_c: ∂f/∂x — state Jacobian (Fxf held constant at nominal)
        A_c = np.zeros((n, n))
        for j in range(n):
            xp = self.x_long.copy()
            xp[j] += eps
            A_c[:, j] = (self._longitudinal_derivatives(xp, Fxf) - f0) / eps
        # B_c: ∂f/∂u_accel — perturb u_accel, recompute Fxf, take numerical derivative
        # Chain rule: ∂f/∂u = (∂f/∂Fxf)·(∂Fxf/∂u) = (cos(δf)/m_eff)·(m_eff/cos(δf)) = 1 → B_c=[[0],[1]]
        Fxf_p = self._compute_Fxf(u_accel + eps)
        B_c = (self._longitudinal_derivatives(self.x_long, Fxf_p) - f0).reshape(n, 1) / eps
        # Restore nominal Fxf — _compute_Fxf(u+eps) above overwrites self.state.Fxf with the
        # perturbed value; restore it so lateral coupling (Fxf·sin(δf)) sees the correct force.
        self.state.Fxf = Fxf
        self.A_c_current = A_c
        self.B_c_current = B_c

    def _linearize_at_current_state(self, u_accel: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Numerical Jacobians at x_long, then ZOH discretize at the given dt.

        Used for explicit one-shot initialisation (dt = NASH_CONTROL_DT).
        Normal steady-state updates use _update_continuous_jacobians() each sim
        step and let Nash re-discretize at NASH_CONTROL_DT via
        get_state_space_matrices().
        """
        self._update_continuous_jacobians(u_accel)
        A_d, B_d = self._discretize_zoh(self.A_c_current, self.B_c_current, dt)
        self.A = A_d; self.B1 = B_d; self.B2 = B_d.copy()
        self.A_d = A_d; self.B_d = B_d
        return A_d, B_d

    # =========================================================================
    # STATE-SPACE MODEL (nonlinear plant + RK4)
    # =========================================================================

    def update_dynamics_state_space(self, dt: float):
        """Nonlinear plant with RK4 + dynamic linearization.

        Simulation loop (two-timestep architecture):
          - Every SIMULATION_DT (0.01 s): RK4 propagation + update A_c, B_c
          - Every NASH_CONTROL_DT (0.1 s): Nash reads A_c, B_c via
            get_state_space_matrices(NASH_CONTROL_DT) and runs ZOH once.
        ZOH is therefore always computed at NASH_CONTROL_DT, not at sim dt.
        """
        if self.A_c_current is None:
            self._update_continuous_jacobians(0.0)

        u_accel = np.clip(
            self.a_desired if hasattr(self, 'a_desired') else self.target_acceleration,
            self.params.max_deceleration, self.params.max_acceleration)

        # Feedforward: convert u_accel [m/s²] → Fxf [N] (stored in self.state.Fxf)
        Fxf = self._compute_Fxf(u_accel)

        # RK4 propagation — step 6 in simulation loop (plant takes Fxf [N], not u_accel)
        f = self._longitudinal_derivatives
        k1 = f(self.x_long, Fxf)
        k2 = f(self.x_long + 0.5 * dt * k1, Fxf)
        k3 = f(self.x_long + 0.5 * dt * k2, Fxf)
        k4 = f(self.x_long + dt * k3,        Fxf)
        self.x_long = self.x_long + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.x_long[1] = np.clip(self.x_long[1], 0.0, self.params.max_velocity)

        # Sync to VehicleState
        self.state.x = float(self.x_long[0])
        self.state.vx = float(self.x_long[1])
        self.v = self.state.vx
        self.a = float(k1[1])
        self.state.ax = self.a
        self.state.vy = 0.0
        self.state.psi_dot = 0.0

        # Step 7: Update continuous Jacobians at new state (no ZOH — Nash does ZOH at 0.1s)
        self._update_continuous_jacobians(u_accel)

        if not np.isfinite(self.x_long).all():
            print(f"WARNING {self.vehicle_id}: NaN detected, resetting vx")
            self.x_long[1] = 0.0

    # =========================================================================
    # STATE-SPACE MATRICES
    # =========================================================================

    def get_state_space_matrices(self, dt: float = SIMULATION_DT) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (A_d, B_d, C) re-discretized at requested dt.

        In hierarchical mode the lower-level controller compensates for all
        resistive forces (drag, rolling), so the effective plant seen by the
        upper Nash controller is a pure double integrator:
            x[k+1] = x[k] + v[k]·dt + 0.5·u·dt²
            v[k+1] = v[k] + u·dt
        Using the nonlinear (drag-including) Jacobians here would cause z_free
        to show natural drag deceleration that never actually occurs in the
        closed-loop hierarchical system, making the Nash solver output ≈ 0
        when the vehicle is above target speed.

        Behaviorally aligned with lateral vehicle get_state_space_matrices().
        """
        C = np.eye(2)
        # Pure double integrator for hierarchical mode (LLC cancels all drag)
        if self.use_hierarchical_model:
            A_d, B_d = self._discretize_zoh(np.array([[0., 1.], [0., 0.]]),
                                             np.array([[0.], [1.]]), dt)
            return A_d, B_d, C
        if self.A_c_current is not None:
            A_d, B_d = self._discretize_zoh(self.A_c_current, self.B_c_current, dt)
            return A_d, B_d, C
        # Fallback: double integrator (before first linearization)
        A_d, B_d = self._discretize_zoh(np.array([[0., 1.], [0., 0.]]),
                                         np.array([[0.], [1.]]), dt)
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

    def get_u_bounds(self) -> Tuple[float, float]:
        """Compute dynamic u_accel bounds from physics at the current operating point.

        Inverts the feedforward law to find the u_accel range that the plant
        can actually deliver, accounting for:
          - Engine torque at current RPM and gear (including gear-shift torque drop)
          - Longitudinal tire saturation at μ·Fz_f (from _compute_Fxf parabolic model)
          - Braking capacity: μ·m·g (full tire friction)
          - Resistive forces (Ra, Rg, Rr, Fyf·sin(δf)) at current state

        Inverse feedforward (Belousov Eq. 3.11a, FWD: Fxr=0):
            Fxf·cos(δf) = m_eff·u + Ra + Rg + Rr + Fyf·sin(δf)
            u = (Fxf·cos(δf) - Ra - Rg - Rr - Fyf·sin(δf)) / m_eff

        Returns
        -------
        (u_min, u_max) : float
            Clipped to config hard limits [NASH_U1_MIN, NASH_U1_MAX].
        """
        p = self.params
        vx = self.state.vx
        vy = self.state.vy
        Omega_z = self.state.psi_dot
        delta_f = self.state.delta_f

        # Resistive forces at current state (same as _compute_Fxf)
        f_vx = (np.tanh(10.0 * abs(vx) - 8.0) + 1.0) / 2.0
        Ra = 0.5 * p.air_density * p.drag_coefficient * p.frontal_area * vx * abs(vx)
        Rg = p.mass * p.gravity * np.sin(p.road_grade)
        vx_sign = np.sign(vx) if abs(vx) > 1e-6 else 0.0
        Rr = f_vx * p.rolling_resistance_coeff * p.mass * p.gravity * vx_sign
        F_resist = Ra + Rg + Rr

        # Front lateral force coupling term
        vx_safe = max(abs(vx), 0.1)
        alpha_f = f_vx * (delta_f - np.arctan2(vy + p.lf * Omega_z, vx_safe))
        Fz_f = p.mass * p.gravity * p.lr / (p.lf + p.lr)
        Fyf = self._tire_force(alpha_f, p.Caf, p.tire_friction_coeff, Fz_f)
        cos_df = np.cos(delta_f)
        lateral_coupling = Fyf * np.sin(delta_f)

        # Fxf_max: minimum of engine force (with gear-shift torque drop) and tire limit
        llc = self.lower_level_controller
        engine_force = llc.compute_max_engine_force() * llc.transmission.get_torque_multiplier()
        tire_max = p.tire_friction_coeff * Fz_f   # parabolic saturation ceiling
        Fxf_max = min(engine_force, tire_max)

        # Fxf_min: full braking (both axles — compute_max_brake_force uses total μ·m·g)
        Fxf_min = -llc.compute_max_brake_force()

        # Invert feedforward: u = (Fxf·cos(δf) - F_resist - Fyf·sin(δf)) / m_eff
        cos_df_safe = max(abs(cos_df), 1e-6) * np.sign(cos_df) if abs(cos_df) > 1e-6 else 1.0
        u_max = (Fxf_max * cos_df_safe - F_resist - lateral_coupling) / self.m_eff
        u_min = (Fxf_min * cos_df_safe - F_resist - lateral_coupling) / self.m_eff

        # Clip to config hard limits (safety backstop)
        from config import NASH_U1_MIN, NASH_U1_MAX
        u_max = float(np.clip(u_max, NASH_U1_MIN, NASH_U1_MAX))
        u_min = float(np.clip(u_min, NASH_U1_MIN, NASH_U1_MAX))
        return u_min, u_max

    def get_state_vector(self):
        """Return current state as [x, vx]ᵀ."""
        return np.array([[self.state.x],
                         [self.state.vx]])

    def set_state_vector(self, x_state):
        """Set state from [x, vx]ᵀ."""
        self.state.x = float(x_state[0, 0])
        self.state.vx = float(x_state[1, 0])
        self.v = self.state.vx
        self.x_long = np.array([self.state.x, self.state.vx])

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
