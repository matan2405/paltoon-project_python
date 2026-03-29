"""
Vehicle class implementing 2-DOF bicycle model for lateral dynamics.

Research basis and adopted elements:
1) Rajamani (Vehicle Dynamics and Control), Chapter 2:
    - Bicycle-model state structure and lateral/yaw dynamic terms.
    - World-frame kinematics used for (X_world, Y_world) propagation.
2) Belousov (vehicle simulation model):
    - Nonlinear plant integration pattern for simulation-time updates.
    - Practical RK4 propagation for the body-frame states.
3) Linearized prediction model for MPC/Nash:
    - Successive linearization around current operating point each step.
    - ZOH discretization of Jacobians for prediction matrices A, B1.

What this file contributes in code:
- Nonlinear lateral plant propagation (simulation truth model).
- Online linearization and discretization for Nash solver prediction.
- Consistent bridge between body-frame dynamics and world-frame trajectory.

PLANT:      Nonlinear bicycle model (RK4) — Rajamani Ch.2, Belousov Eq. 3.11
CONTROLLER: Dynamic linearization (Jacobians) — at every timestep, A_t and B_t
            are recomputed at the current operating point x_body(t) and used
            by the Nash MPC as prediction matrices.

Simulation loop:
  1. x_body(t) available from previous RK4 step
  2. Linearize at x_body(t): A_t = ∂f/∂x, B_t = ∂f/∂u  (numerical Jacobians)
  3. Nash solver reads vehicle.A = A_t, vehicle.B1 = B_t
  4. Nash solver computes delta_commanded
  5. ZOH: delta_actual frozen constant during RK4
  6. RK4: x_body(t+dt) from full nonlinear equations
  7. Update A_t+dt for next iteration → back to step 1

World-frame (Rajamani Eq. 2.10-2.11):
  Ẋ_world = vx·cos(ψ) - vy·sin(ψ)
  Ẏ_world = vx·sin(ψ) + vy·cos(ψ)
"""

import numpy as np
from scipy.linalg import expm
from typing import Tuple
from .components import VehicleParameters, VehicleState
from config import NOMINAL_VELOCITY, SIMULATION_DT

class Vehicle:
    """2-DOF Bicycle model — nonlinear plant with dynamic linearization."""

    def __init__(self,
                 initial_y: float = 0.0,
                 initial_psi: float = 0.0,
                 initial_x: float = 0.0,
                 vehicle_id: str = "Vehicle",
                 longitudinal_velocity: float = NOMINAL_VELOCITY):
        """Initialize lateral vehicle states and matrices used by simulation and Nash MPC."""

        self.params = VehicleParameters()
        self.state = VehicleState()
        self.vehicle_id = vehicle_id

        self.state.y = initial_y
        self.state.psi = initial_psi
        self.state.x = initial_x
        self.state.vx = longitudinal_velocity
        self.state.X_world = initial_x   # inertial frame: align with body-frame start
        self.state.Y_world = initial_y   # inertial frame: align with body-frame start
        self.vx = longitudinal_velocity
        self.L = self.params.length

        # Body-frame state: [y_body, vy, ψ, ψ̇]
        self.x_body = np.array([initial_y, 0.0, initial_psi, 0.0])

        # Steering history
        self.delta_prev = 0.0
        self.ay_prev = 0.0

        # Fixed linear matrices (body-frame Eq. 2.31)
        self.A_body_c = None; self.A_body_d = None; self.B_body_d = None
        self.B_c = None; self.C = None

        # Dynamic linearization matrices — updated every timestep
        # Nash solver reads these. Initialized to body-frame linear model.
        self.A = None; self.B1 = None; self.B2 = None
        self.A_d = None; self.B_d = None

        # Continuous-time Jacobians — stored so get_state_space_matrices can
        # re-discretize at any requested dt (Nash uses NASH_CONTROL_DT ≠ sim dt)
        self.A_c_current = None; self.B_c_current = None

        self._build_all_matrices()

        print(f"Vehicle '{vehicle_id}' (Nonlinear + Dynamic Linearization) at y={initial_y:.1f}m")

    # ------------------------------------------------------------------
    # Matrix construction
    # ------------------------------------------------------------------

    def _build_all_matrices(self, dt: float = SIMULATION_DT):
        p = self.params
        vx = max(self.vx, 0.1)   # consistent with _body_derivatives
        mu = p.mu  # road friction coefficient (Swain & Rath 2023, Eq. 1)

        # Low-speed blending (Fernández Eq. 4.4, pp. 36-37):
        #   f_vx → 0  for vx < 0.5 m/s,  f_vx → 1  for vx > 1.2 m/s
        # All tire-force terms (derived from alpha = f_vx·(δ - slip/vx)) are scaled:
        #   b2/b4  : steering-input term from f_vx·Cf·δ inside alpha_f
        #   a22/a24/a42/a44 : state-feedback terms from f_vx·Cf·(state/vx) inside alpha
        # Exceptions (not scaled):
        #   centripetal -vx in a24   — kinematic, not from tire force
        #   a13, a33                  — no 1/vx, ∂Fy/∂ψ_road terms
        f_vx = (np.tanh(10 * self.vx - 8) + 1) / 2

        # Input matrix B
        b2 = f_vx * 2 * mu * p.Cf / p.mass
        b4 = f_vx * 2 * mu * (p.lf * p.Cf) / p.Iz
        self.B_c = np.array([[0.0], [b2], [0.0], [b4]])
        self.C = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

        # Body-frame state matrix
        a22 = f_vx * (-2 * mu * (p.Cf + p.Cr)) / (p.mass * vx)
        a24 = -vx + f_vx * (-(2 * mu * p.Cf * p.lf - 2 * mu * p.Cr * p.lr)) / (p.mass * vx)
        a42 = f_vx * (-(2 * mu * p.lf * p.Cf - 2 * mu * p.lr * p.Cr)) / (p.Iz * vx)
        a44 = f_vx * (-(2 * mu * p.lf ** 2 * p.Cf + 2 * mu * p.lr ** 2 * p.Cr)) / (p.Iz * vx)

        # Row 0: ẏ = vy + vx·ψ (linearization of vx·sin(ψ) + vy·cos(ψ) at ψ=0)
        # x2 = vy_body (Belousov Eq. 3.11b, Fernández Eq. 3.10, Swain & Rath Eq. 1)
        # Swain & Rath show: z1=y, z2=vy+vx·ψ → ż1=z2 → row 0=[0,1,vx,0]
        self.A_body_c = np.array([
            [0, 1, vx, 0], [0, a22, 0, a24],
            [0, 0, 0,  1], [0, a42, 0, a44]
        ])

        self.A_body_d, self.B_body_d = self._discretize_zoh(self.A_body_c, self.B_c, dt)

        # Initialize dynamic linearization to body-frame linear model
        self.A_d = self.A_body_d; self.B_d = self.B_body_d
        self.A = self.A_body_d; self.B1 = self.B_body_d; self.B2 = self.B_body_d.copy()
        self.A_c_current = self.A_body_c; self.B_c_current = self.B_c

    def _discretize_zoh(self, A_c, B_c, dt):
        n, m = A_c.shape[0], B_c.shape[1]
        aug = np.zeros((n + m, n + m))
        aug[:n, :n] = A_c * dt; aug[:n, n:] = B_c * dt
        e = expm(aug)
        return e[:n, :n], e[:n, n:]

    # ------------------------------------------------------------------
    # Nonlinear bicycle model
    # ------------------------------------------------------------------

    def _tire_force(self, alpha: float, C: float, mu: float, Fz: float) -> float:
        """Nonlinear lateral tire force (Rajamani Eq. 13.45, parabolic pressure).

        Recovers the linear model Fy = C·α for small slip angles.
        Saturates at μ·Fz for full sliding.
        """
        theta = C / (3.0 * mu * Fz)
        S = np.tan(alpha)
        if abs(S) <= 1.0 / theta:
            return mu * Fz * (3 * theta * S - 3 * theta ** 2 * S ** 2 + theta ** 3 * S ** 3)
        return float(np.sign(S)) * mu * Fz

    def _body_derivatives(self, state_vec: np.ndarray, delta: float) -> np.ndarray:
        """Continuous-time nonlinear bicycle model f(x, u).

        Equations:
          - Slip angles: Rajamani Eq. 2.27-2.28 / Belousov Eq. 3.14-3.15
          - EOM:         Belousov Eq. 3.11b/c (full cos(δ) steering geometry)
          - Tire forces: Rajamani Eq. 13.45 (nonlinear, parabolic pressure)

        Args:
            state_vec: [y, vy, psi, psi_dot]
            delta:     front wheel steering angle (rad)

        Returns:
            dx/dt = [y_dot, vy_dot, psi_dot, psi_ddot]
            y_dot = vx·sin(ψ) + vy·cos(ψ)  (exact body→road, no small-angle)
        """
        _, vy, psi, psi_dot = state_vec
        vx = max(self.vx, 0.1)   # denominator safety for arctan2
        p = self.params

        # Low-speed blending (Fernández Eq. 4.4, pp. 36-37):
        # Multiplies entire slip angle → suppresses all tire forces for vx < 0.5 m/s
        f_vx = (np.tanh(10 * self.vx - 8) + 1) / 2

        # Nonlinear slip angles (arctan, Rajamani Eq. 2.27-2.28)
        alpha_f = f_vx * (delta - np.arctan2(vy + p.lf * psi_dot, vx))
        alpha_r = f_vx * np.arctan2(p.lr * psi_dot - vy, vx)

        # Static normal loads (50/50 CG assumed)
        Fz_f = p.mass * p.g * p.lr / (p.lf + p.lr)
        Fz_r = p.mass * p.g * p.lf / (p.lf + p.lr)

        # Nonlinear tire forces (Rajamani Eq. 13.45)
        Fyf = self._tire_force(alpha_f, p.Cf, p.mu, Fz_f)
        Fyr = self._tire_force(alpha_r, p.Cr, p.mu, Fz_r)

        # Equations of motion (Belousov Eq. 3.11b/c — full steering geometry)
        # Fxf·sin(δf) terms: longitudinal front axle force projected onto lateral/yaw axes
        Fxf = self.state.Fxf   # set by longitudinal module; zero if not coupled
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        vy_dot = (Fyf * cos_d + Fyr + Fxf * sin_d) / p.mass - vx * psi_dot
        psi_ddot = (p.lf * Fyf * cos_d - p.lr * Fyr + p.lf * Fxf * sin_d) / p.Iz

        # Exact body→road kinematic transformation (no small-angle assumption):
        # ẏ = vx·sin(ψ) + vy·cos(ψ)
        # Jacobian at ψ=0: ∂ẏ/∂vy=1, ∂ẏ/∂ψ=vx  →  consistent with A_body_c row [0,1,vx,0]
        y_dot = vx * np.sin(psi) + vy * np.cos(psi)
        return np.array([y_dot, vy_dot, psi_dot, psi_ddot])

    # ------------------------------------------------------------------
    # Dynamic (successive) linearization
    # ------------------------------------------------------------------

    def _linearize_at_current_state(self, delta: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Jacobians of f(x,u) at x_body and discretize (ZOH).

        Numerical Jacobians (forward differences):
          A_c[i,j] = ∂fi/∂xj   (4×4)
          B_c[i]   = ∂fi/∂δ    (4×1)

        Returns: (A_d, B_d) — discrete-time Jacobian matrices for Nash MPC.
        """
        eps = 1e-5
        f0 = self._body_derivatives(self.x_body, delta)
        n = 4
        A_c = np.zeros((n, n))
        for j in range(n):
            xp = self.x_body.copy()
            xp[j] += eps
            A_c[:, j] = (self._body_derivatives(xp, delta) - f0) / eps
        B_c = (self._body_derivatives(self.x_body, delta + eps) - f0).reshape(n, 1) / eps
        # Cache continuous-time Jacobians for re-discretization at other dt values
        self.A_c_current = A_c
        self.B_c_current = B_c
        return self._discretize_zoh(A_c, B_c, dt)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_state_space_matrices(self, dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return linearized matrices discretized at the requested dt.

        The Nash solver uses NASH_CONTROL_DT (0.05s) while the simulation runs
        at 0.01s. Returning matrices discretized at the wrong dt causes HQ1H to
        be ~25× too small, making the MPC compute near-zero steering.
        """
        if abs(dt - 0.01) < 1e-9 or self.A_c_current is None:
            return self.A, self.B1, self.C
        A_d, B_d = self._discretize_zoh(self.A_c_current, self.B_c_current, dt)
        return A_d, B_d, self.C

    def get_state_vector(self) -> np.ndarray:
        """Physical state [y, vy, ψ, ψ̇] from body model — x0 for Nash MPC."""
        return self.x_body.copy()

    def update_dynamics(self, dt: float, delta_commanded: float):
        """Propagate nonlinear plant (RK4) and update dynamic linearization."""
        if self.A_body_d is None:
            self._build_all_matrices(dt)

        # 1. Rate-limited steering (ZOH: freeze delta during RK4)
        delta_commanded = np.clip(delta_commanded,
                                  -self.params.max_steering_angle,
                                  self.params.max_steering_angle)
        max_dd = self.params.max_steering_rate * dt
        delta_actual = self.delta_prev + np.clip(
            delta_commanded - self.delta_prev, -max_dd, max_dd)
        self.delta_prev = delta_actual
        self.state.delta = delta_actual

        # 2. Nonlinear body-frame propagation — RK4 (Belousov Eq. 3.11)
        f = self._body_derivatives
        k1 = f(self.x_body, delta_actual)
        k2 = f(self.x_body + 0.5 * dt * k1, delta_actual)
        k3 = f(self.x_body + 0.5 * dt * k2, delta_actual)
        k4 = f(self.x_body + dt * k3, delta_actual)
        x_b_next = (self.x_body + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)).reshape(-1, 1)

        # Clip body state for numerical safety
        x_b_next[1, 0] = np.clip(x_b_next[1, 0], -2.0, 2.0)
        x_b_next[2, 0] = np.clip(x_b_next[2, 0], -np.radians(15), np.radians(15))
        x_b_next[3, 0] = np.clip(x_b_next[3, 0], -0.3, 0.3)
        self.x_body = x_b_next.flatten()

        # 3. Write body-frame state to VehicleState (physical truth)
        self.state.y = self.x_body[0]
        self.state.y_dot = self.x_body[1]
        self.state.psi = self.x_body[2]
        self.state.psi_dot = self.x_body[3]

        # 5. Lateral acceleration
        v_total = np.sqrt(self.vx ** 2 + self.state.y_dot ** 2)
        ay_new = v_total * self.state.psi_dot
        max_ay_change = self.params.max_lateral_jerk * dt
        self.state.ay = self.ay_prev + np.clip(
            ay_new - self.ay_prev, -max_ay_change, max_ay_change)
        self.ay_prev = self.state.ay

        # 6. Body-frame longitudinal position (along vehicle axis)
        self.state.x += v_total * np.cos(self.state.psi) * dt

        # 7. World-frame transformation (Rajamani Eq. 2.10-2.11)
        #    Ẋ_world = vx·cos(ψ) - vy·sin(ψ)
        #    Ẏ_world = vx·sin(ψ) + vy·cos(ψ)
        psi_now = self.x_body[2]
        vy_now = self.x_body[1]
        self.state.X_world += (self.vx * np.cos(psi_now) - vy_now * np.sin(psi_now)) * dt
        self.state.Y_world += (self.vx * np.sin(psi_now) + vy_now * np.cos(psi_now)) * dt

        # 8. Dynamic linearization at new x_body — ready for next Nash solver call
        A_d_t, B_d_t = self._linearize_at_current_state(delta_actual, dt)
        self.A = A_d_t
        self.A_d = A_d_t
        self.B1 = B_d_t
        self.B2 = B_d_t.copy()
        self.B_d = B_d_t

        # 9. NaN guard
        if not np.isfinite(self.x_body).all():
            print(f"WARNING {self.vehicle_id}: NaN detected, resetting velocities")
            self.x_body[1] = 0.0; self.x_body[3] = 0.0

    def reset_steering(self):
        self.delta_prev = 0.0
        self.ay_prev = 0.0
        self.state.delta = 0.0


__all__ = ['Vehicle']
