"""
unified/vehicle/vehicle_6d.py — Unified 6D vehicle model.

Propagates Belousov Eq. 3.11a/b/c simultaneously in a single RK4 step.

State:   x_6d = [x, vx, y, vy, ψ, ψ̇]
Inputs:  u_long [m/s²]  — desired longitudinal acceleration (from long Nash)
         δ      [rad]   — front steering angle (from lateral Nash)

Coupling terms included (neglected in the separated modules):
  Fxf·sin(δ)  → feeds longitudinal drive force into lateral/yaw EOM  (Eq. 3.11b/c)
  Fyf·sin(δ)  → feeds front lateral force into longitudinal EOM       (Eq. 3.11a)
  vy·ψ̇        → Coriolis coupling lateral↔longitudinal                (Eq. 3.11a)

Separate Jacobian sets are maintained per axis so both Nash solvers can
call get_state_space_matrices() with the correct dimensions:
  LongVehicleProxy.get_state_space_matrices(dt) → (2×2 A_d, 2×1 B_d, 2×2 C)
  LatVehicleProxy.get_state_space_matrices(dt)  → (4×4 A_d, 4×1 B_d, 2×4 C)

Both proxies are read-through wrappers — all state lives on Vehicle6D.
"""

import numpy as np
from scipy.linalg import expm
from typing import Tuple

from unified.vehicle.components import VehicleParameters
from unified.control.lower_level_controller import LowerLevelController


# =============================================================================
# Unified Vehicle State
# =============================================================================

class UnifiedVehicleState:
    """Mutable state struct holding the full 6D vehicle state."""

    def __init__(self):
        # Longitudinal (Eq. 3.11a)
        self.x   = 0.0   # world-frame longitudinal position [m]
        self.vx  = 0.0   # body-frame longitudinal velocity [m/s]
        self.ax  = 0.0   # longitudinal acceleration [m/s²]

        # Lateral (Eq. 3.11b/c)
        self.y       = 0.0   # world-frame lateral position [m]
        self.vy      = 0.0   # body-frame lateral velocity [m/s] (y_dot in body frame)
        self.y_dot   = 0.0   # alias for vy (lateral vehicle compat)
        self.psi     = 0.0   # yaw angle [rad]
        self.psi_dot = 0.0   # yaw rate [rad/s]
        self.ay      = 0.0   # lateral acceleration [m/s²]

        # Steering (set by update step)
        self.delta   = 0.0   # front steering angle [rad] (ZOH)
        self.delta_f = 0.0   # alias (longitudinal vehicle compat)

        # Coupling term: longitudinal front axle force [N]
        # Set each step by _compute_Fxf; read by _body_derivatives for Fxf·sin(δ) terms
        self.Fxf = 0.0

        # World-frame position (Rajamani Eq. 2.10-2.11)
        self.X_world = 0.0
        self.Y_world = 0.0


# =============================================================================
# Vehicle6D
# =============================================================================

class Vehicle6D:
    """
    Unified 6D vehicle model integrating Belousov Eq. 3.11a/b/c.

    Physical plant propagated via RK4 at SIMULATION_DT (100 Hz).
    Jacobians updated every sim step for Nash solver re-discretization.

    The two proxy objects (long_proxy, lat_proxy) expose the interface
    expected by the unmodified Nash solvers from each module.
    """

    def __init__(self,
                 initial_x:   float = 0.0,
                 initial_vx:  float = 33.3,
                 initial_y:   float = 0.0,
                 initial_psi: float = 0.0,
                 vehicle_id:  str   = "ego"):

        self.params     = VehicleParameters()
        self.state      = UnifiedVehicleState()
        self.vehicle_id = vehicle_id
        self.L          = self.params.length   # for PlatoonManager gap computation

        # ── Initial state ────────────────────────────────────────────────────
        self.state.x   = initial_x
        self.state.vx  = initial_vx
        self.state.y   = initial_y
        self.state.psi = initial_psi
        self.state.X_world = initial_x
        self.state.Y_world = initial_y
        self.vx            = initial_vx   # class-level alias (lateral vehicle compat)

        # ── 6D state vector (mirrors x_long and x_lateral in existing modules) ──
        self.x_long = np.array([initial_x, initial_vx])            # [x, vx]
        self.x_lateral = np.array([initial_y, 0.0, initial_psi, 0.0]) # [y, vy, ψ, ψ̇]

        # ── Effective mass — Belousov Eq. 3.8 ────────────────────────────────
        r = self.params.wheel_radius
        self.m_eff = self.params.mass + self.params.wheel_inertia / (r * r)

        # ── Steering state (rate limiter) ─────────────────────────────────────
        self.delta_prev = 0.0

        # ── Acceleration output filter ────────────────────────────────────────
        self._a_filtered           = 0.0
        self._a_filter_initialized = False
        self.a_desired             = 0.0   # written by upper controller
        self.a                     = 0.0   # actual acceleration (filtered)
        self.v                     = initial_vx   # speed alias

        # ── Nash interface ────────────────────────────────────────────────────
        self.nash_acceleration     = None  # MUST reset to None after reading

        # ── Jacobian caches ───────────────────────────────────────────────────
        # Longitudinal (2×2 A_c, 2×1 B_c)
        self._long_A_c: np.ndarray = None
        self._long_B_c: np.ndarray = None
        # Lateral (4×4 A_c, 4×1 B_c)
        self._lat_A_c:  np.ndarray = None
        self._lat_B_c:  np.ndarray = None
        # C matrices
        self._C_long   = np.eye(2)
        self._C_lat    = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

        # Compute initial Jacobians
        self._update_jacobians(u_long=0.0, delta=0.0)

        # ── Lower-level controller ────────────────────────────────────────────
        self.llc = LowerLevelController(self)
        self.llc._sync_engine_to_velocity()

        # ── Proxy objects (passed to Nash solvers at construction) ────────────
        self.long_proxy = LongVehicleProxy(self)
        self.lat_proxy  = LatVehicleProxy(self)

        print(f"Vehicle6D '{vehicle_id}': x={initial_x:.1f}m, vx={initial_vx:.1f}m/s, "
              f"y={initial_y:.1f}m, psi={np.degrees(initial_psi):.1f}deg")

    # =========================================================================
    # TIRE FORCE MODEL (Rajamani Eq. 13.45 — parabolic pressure distribution)
    # =========================================================================

    def _tire_force(self, alpha: float, C: float, mu: float, Fz: float) -> float:
        """Nonlinear lateral tire force per wheel.
        Returns C·α for small slip; saturates at μ·Fz."""
        theta = C / (3.0 * mu * Fz)
        S = np.tan(alpha)
        if abs(S) <= 1.0 / theta:
            return mu * Fz * (3 * theta * S - 3 * theta**2 * S**2 + theta**3 * S**3)
        return float(np.sign(S)) * mu * Fz

    # =========================================================================
    # FEEDFORWARD CANCELLATION (Longitudinal vehicle, Belousov Eq. 3.11a)
    # =========================================================================

    def _compute_Fxf(self, u_accel: float) -> float:
        """Convert Nash u_accel [m/s²] → front axle drive force Fxf [N].

        Inverts Belousov Eq. 3.11a (FWD: Fxr=0):
            Fxf·cos(δ) - Fyf·sin(δ) = m_eff·u_accel + Ra + Rg + Rr
        Applies parabolic longitudinal tire saturation (Rajamani Eq. 13.45).
        Stores result in self.state.Fxf for lateral coupling terms.
        """
        p = self.params
        vx      = self.state.vx
        vy      = self.state.vy
        Omega_z = self.state.psi_dot
        delta_f = self.state.delta_f

        f_vx = (np.tanh(10.0 * abs(vx) - 8.0) + 1.0) / 2.0

        # Resistive forces
        Ra       = 0.5 * p.air_density * p.drag_coefficient * p.frontal_area * vx * abs(vx)
        Rg       = p.mass * p.gravity * np.sin(p.road_grade)
        vx_sign  = np.sign(vx) if abs(vx) > 1e-6 else 0.0
        Rr       = f_vx * p.rolling_resistance_coeff * p.mass * p.gravity * vx_sign

        # Front lateral tire force (cross-coupling term in Eq. 3.11a)
        # Fz_f is per-wheel (axle load / 2); call _tire_force per-wheel × 2 → correct θ = C/(3μFz_per_wheel)
        vx_safe  = max(abs(vx), 0.1)
        alpha_f  = f_vx * (delta_f - np.arctan2(vy + p.lf * Omega_z, vx_safe))
        Fz_f_axle   = p.mass * p.gravity * p.lr / (p.lf + p.lr)   # total front axle load [N]
        Fz_f_wheel  = Fz_f_axle / 2.0                              # per-wheel normal load [N]
        Fyf      = 2.0 * self._tire_force(alpha_f, p.Caf, p.tire_friction_coeff, Fz_f_wheel)

        # Feedforward inversion
        cos_df     = np.cos(delta_f)
        Fxf_des    = (self.m_eff * u_accel + Ra + Rg + Rr + Fyf * np.sin(delta_f)) / max(abs(cos_df), 1e-6)

        # Longitudinal tire saturation (same per-wheel convention: Cx per-wheel × 2)
        kappa = Fxf_des / p.Cx
        Fxf   = 2.0 * self._tire_force(np.arctan(kappa), p.Cx / 2.0, p.tire_friction_coeff, Fz_f_wheel)

        self.state.Fxf  = Fxf
        self.state.delta_f = delta_f   # keep in sync
        return Fxf

    # =========================================================================
    # DERIVATIVES — Belousov Eq. 3.11a/b/c (6D, fully coupled)
    # =========================================================================

    def _derivatives_6d(self, state: np.ndarray, Fxf: float, delta: float) -> np.ndarray:
        """Full coupled EOM for state = [x, vx, y, vy, ψ, ψ̇].

        Eq. 3.11a (longitudinal):
            ẋ  = vx·cos(ψ) - vy·sin(ψ)       [world-frame kinematics]
            v̇x = vy·ψ̇ + (Fxf·cos(δ) - Fyf·sin(δ) - Ra - Rg - Rr) / m_eff

        Eq. 3.11b (lateral):
            ẏ  = vx·sin(ψ) + vy·cos(ψ)       [world-frame kinematics]
            v̇y = (Fyf·cos(δ) + Fyr + Fxf·sin(δ)) / m - vx·ψ̇

        Eq. 3.11c (yaw):
            ψ̇  = ψ̇                            [definition]
            ψ̈  = (lf·Fyf·cos(δ) - lr·Fyr + lf·Fxf·sin(δ)) / Iz

        Coupling terms (new in unified model):
          - Fxf·sin(δ): longitudinal force into lateral/yaw      (Eq. 3.11b/c)
          - Fyf·sin(δ): front lateral force into longitudinal     (Eq. 3.11a)
          - vy·ψ̇:       Coriolis coupling                         (Eq. 3.11a)
        """
        _, vx, _, vy, psi, psi_dot = state
        p = self.params

        vx_safe = max(abs(vx), 0.1)
        f_vx    = (np.tanh(10.0 * abs(vx) - 8.0) + 1.0) / 2.0

        # ── Resistive forces (Belousov Eq. 3.5–3.7) ──────────────────────────
        Ra      = 0.5 * p.air_density * p.drag_coefficient * p.frontal_area * vx * abs(vx)
        Rg      = p.mass * p.gravity * np.sin(p.road_grade)
        vx_sign = np.sign(vx) if abs(vx) > 1e-6 else 0.0
        Rr      = f_vx * p.rolling_resistance_coeff * p.mass * p.gravity * vx_sign

        # ── Slip angles (Rajamani Eq. 2.27-2.28) ──────────────────────────────
        alpha_f = f_vx * (delta - np.arctan2(vy + p.lf * psi_dot, vx_safe))
        alpha_r = f_vx * np.arctan2(p.lr * psi_dot - vy, vx_safe)

        # ── Normal loads (50/50 CG, per wheel) ───────────────────────────────
        Fz_f    = p.mass * p.gravity * p.lr / (p.lf + p.lr) / 2.0
        Fz_r    = p.mass * p.gravity * p.lf / (p.lf + p.lr) / 2.0
        mu      = p.tire_friction_coeff

        # ── Lateral tire forces — total axle (2 × per-wheel) ─────────────────
        Fyf = 2.0 * self._tire_force(alpha_f, p.Caf, mu, Fz_f)
        Fyr = 2.0 * self._tire_force(alpha_r, p.Car, mu, Fz_r)

        cos_d = np.cos(delta)
        sin_d = np.sin(delta)

        # ── World-frame kinematics ────────────────────────────────────────────
        x_dot   = vx * np.cos(psi) - vy * np.sin(psi)   # Rajamani Eq. 2.10
        y_dot   = vx * np.sin(psi) + vy * np.cos(psi)   # Rajamani Eq. 2.11

        # ── Eq. 3.11a — longitudinal EOM ─────────────────────────────────────
        vx_dot  = vy * psi_dot + (Fxf * cos_d - Fyf * sin_d - Ra - Rg - Rr) / self.m_eff

        # ── Eq. 3.11b — lateral EOM ──────────────────────────────────────────
        vy_dot  = (Fyf * cos_d + Fyr + Fxf * sin_d) / p.mass - vx * psi_dot

        # ── Eq. 3.11c — yaw EOM ──────────────────────────────────────────────
        psi_ddot = (p.lf * Fyf * cos_d - p.lr * Fyr + p.lf * Fxf * sin_d) / p.Iz

        return np.array([x_dot, vx_dot, y_dot, vy_dot, psi_dot, psi_ddot])

    # =========================================================================
    # RK4 INTEGRATION
    # =========================================================================

    def update_dynamics_6d(self, u_long: float, delta: float, dt: float):
        """Propagate full 6D state one timestep using RK4.

        Steps:
          1. Rate-limit and apply steering input
          2. Compute Fxf via feedforward cancellation
          3. RK4 integration of Belousov Eq. 3.11a/b/c
          4. Clip state for numerical safety
          5. Sync all state aliases
          6. Update Jacobians for next Nash solve
          7. World-frame position integration
        """
        p = self.params

        # 1. Rate-limited steering (Rajamani lateral model)
        delta_clipped = np.clip(delta, -p.max_steering_angle, p.max_steering_angle)
        max_dd        = self.params.max_steering_rate * dt
        delta_actual  = self.delta_prev + np.clip(
            delta_clipped - self.delta_prev, -max_dd, max_dd)
        self.delta_prev        = delta_actual
        self.state.delta       = delta_actual
        self.state.delta_f     = delta_actual   # longitudinal compat

        # 2. Feedforward: u_long [m/s²] → Fxf [N] (tire saturation)
        Fxf_desired = self._compute_Fxf(u_long)

        # 3. Lower-level controller: engine map → actual throttle/brake + Ackermann
        self.llc.compute_control(Fxf_desired, delta_actual)
        Fxf = self.llc.compute_drive_force()   # engine-limited drive force

        # 4. RK4 over full 6D state
        s = np.array([self.state.x, self.state.vx,
                      self.state.y, self.state.vy,
                      self.state.psi, self.state.psi_dot])
        f  = self._derivatives_6d
        k1 = f(s,                    Fxf, delta_actual)
        k2 = f(s + 0.5 * dt * k1,   Fxf, delta_actual)
        k3 = f(s + 0.5 * dt * k2,   Fxf, delta_actual)
        k4 = f(s + dt * k3,          Fxf, delta_actual)
        s_new = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # 4. Numerical safety clipping (lateral vehicle pattern)
        s_new[1] = np.clip(s_new[1], 0.0,           p.max_velocity)  # vx >= 0
        s_new[3] = np.clip(s_new[3], -2.0,   2.0)                    # vy
        s_new[4] = np.clip(s_new[4], -np.radians(15), np.radians(15)) # ψ
        s_new[5] = np.clip(s_new[5], -0.3,   0.3)                    # ψ̇

        # 5. Sync to state struct and class-level aliases
        self.state.x       = float(s_new[0])
        self.state.vx      = float(s_new[1])
        self.state.y       = float(s_new[2])
        self.state.vy      = float(s_new[3])
        self.state.y_dot   = float(s_new[3])  # lateral compat
        self.state.psi     = float(s_new[4])
        self.state.psi_dot = float(s_new[5])
        self.state.ax      = float(k1[1])     # acceleration at start of step
        self.state.ay      = float(k1[3])
        self.vx            = self.state.vx
        self.v             = self.state.vx
        self.a             = self.state.ax
        self.x_long        = np.array([self.state.x, self.state.vx])
        self.x_lateral        = np.array([self.state.y, self.state.vy,
                                       self.state.psi, self.state.psi_dot])

        # 6. Update engine RPM and transmission
        vx_new = self.state.vx
        if abs(vx_new) > 0.1:
            wheel_rpm         = (vx_new * 60) / (2 * np.pi * p.wheel_radius)
            target_engine_rpm = wheel_rpm * self.llc.transmission.get_effective_ratio()
            target_engine_rpm = max(target_engine_rpm, self.llc.engine.idle_rpm)
            self.llc.engine.update_rpm(target_engine_rpm, dt)
        else:
            self.llc.engine.update_rpm(self.llc.engine.idle_rpm, dt)
        self.llc.transmission.update(self.llc.engine.rpm, vx_new, dt)

        # 7. Update Jacobians
        self._update_jacobians(u_long, delta_actual)

        # 7. World-frame update (Rajamani Eq. 2.10-2.11)
        psi = self.state.psi
        vy  = self.state.vy
        vx  = self.state.vx
        self.state.X_world += (vx * np.cos(psi) - vy * np.sin(psi)) * dt
        self.state.Y_world += (vx * np.sin(psi) + vy * np.cos(psi)) * dt

        # NaN guard
        if not (np.isfinite(s_new).all()):
            print(f"WARNING {self.vehicle_id}: NaN in 6D state — resetting lateral velocities")
            self.state.vy      = 0.0
            self.state.psi_dot = 0.0
            self.x_lateral[1]     = 0.0
            self.x_lateral[3]     = 0.0

    # =========================================================================
    # JACOBIANS (numerical, forward differences)
    # =========================================================================

    def _update_jacobians(self, u_long: float, delta: float):
        """Update continuous Jacobians for both axes at the current operating point."""
        Fxf = self._compute_Fxf(u_long)
        s   = np.array([self.state.x, self.state.vx,
                        self.state.y, self.state.vy,
                        self.state.psi, self.state.psi_dot])
        f0  = self._derivatives_6d(s, Fxf, delta)
        eps = 1e-5

        # ── Longitudinal Jacobian (2×2 A, 2×1 B w.r.t. u_long) ──────────────
        # Indices of [x, vx] in 6D state: 0, 1
        A_long = np.zeros((2, 2))
        for j_loc, j_full in enumerate([0, 1]):
            sp = s.copy()
            sp[j_full] += eps
            df = self._derivatives_6d(sp, Fxf, delta)
            A_long[:, j_loc] = (df[[0, 1]] - f0[[0, 1]]) / eps
        Fxf_p  = self._compute_Fxf(u_long + eps)
        df_u   = self._derivatives_6d(s, Fxf_p, delta)
        B_long = ((df_u[[0, 1]] - f0[[0, 1]]) / eps).reshape(2, 1)
        self.state.Fxf = Fxf   # restore after Fxf_p perturbation
        self._long_A_c = A_long
        self._long_B_c = B_long

        # ── Lateral Jacobian (4×4 A, 4×1 B w.r.t. δ) ─────────────────────────
        # Indices of [y, vy, ψ, ψ̇] in 6D state: 2, 3, 4, 5
        A_lat = np.zeros((4, 4))
        for j_loc, j_full in enumerate([2, 3, 4, 5]):
            sp = s.copy()
            sp[j_full] += eps
            df = self._derivatives_6d(sp, Fxf, delta)
            A_lat[:, j_loc] = (df[[2, 3, 4, 5]] - f0[[2, 3, 4, 5]]) / eps
        df_d   = self._derivatives_6d(s, Fxf, delta + eps)
        B_lat  = ((df_d[[2, 3, 4, 5]] - f0[[2, 3, 4, 5]]) / eps).reshape(4, 1)
        self._lat_A_c = A_lat
        self._lat_B_c = B_lat

    # =========================================================================
    # ZOH DISCRETIZATION
    # =========================================================================

    @staticmethod
    def _zoh(A_c: np.ndarray, B_c: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        n, m = A_c.shape[0], B_c.shape[1]
        aug = np.zeros((n + m, n + m))
        aug[:n, :n] = A_c * dt
        aug[:n, n:] = B_c * dt
        e = expm(aug)
        return e[:n, :n], e[:n, n:]

    # =========================================================================
    # STATE SPACE MATRICES (axis-specific)
    # =========================================================================

    def get_state_space_matrices_long(self, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (A_d 2×2, B_d 2×1, C 2×2) for the longitudinal Nash solver."""
        if self._long_A_c is None:
            # Fallback to pure double integrator
            A_d, B_d = self._zoh(np.array([[0., 1.], [0., 0.]]),
                                  np.array([[0.], [1.]]), dt)
            return A_d, B_d, self._C_long
        A_d, B_d = self._zoh(self._long_A_c, self._long_B_c, dt)
        return A_d, B_d, self._C_long

    def get_state_space_matrices_lat(self, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (A_d 4×4, B_d 4×1, C 2×4) for the lateral Nash solver."""
        if self._lat_A_c is None:
            vx  = max(self.state.vx, 0.1)
            p   = self.params
            mu  = p.tire_friction_coeff
            f_v = (np.tanh(10 * vx - 8) + 1) / 2
            a22 = f_v * (-2 * mu * (p.Caf + p.Car)) / (p.mass * vx)
            a24 = -vx + f_v * (-(2 * mu * p.Caf * p.lf - 2 * mu * p.Car * p.lr)) / (p.mass * vx)
            a42 = f_v * (-(2 * mu * p.lf * p.Caf - 2 * mu * p.lr * p.Car)) / (p.Iz * vx)
            a44 = f_v * (-(2 * mu * p.lf**2 * p.Caf + 2 * mu * p.lr**2 * p.Car)) / (p.Iz * vx)
            b2  = f_v * 2 * mu * p.Caf / p.mass
            b4  = f_v * 2 * mu * (p.lf * p.Caf) / p.Iz
            A_c = np.array([[0, 1, vx, 0], [0, a22, 0, a24], [0, 0, 0, 1], [0, a42, 0, a44]])
            B_c = np.array([[0.0], [b2], [0.0], [b4]])
            A_d, B_d = self._zoh(A_c, B_c, dt)
            return A_d, B_d, self._C_lat
        A_d, B_d = self._zoh(self._lat_A_c, self._lat_B_c, dt)
        return A_d, B_d, self._C_lat

    # =========================================================================
    # STATE ACCESSORS (compatibility with existing modules)
    # =========================================================================

    def get_long_state(self) -> np.ndarray:
        """[x, vx] for longitudinal Nash solver (x0 argument)."""
        return np.array([self.state.x, self.state.vx])

    def get_lat_state_vector(self) -> np.ndarray:
        """[y, vy, ψ, ψ̇] for lateral Nash solver (x0 via get_state_vector)."""
        return self.x_lateral.copy()

    # =========================================================================
    # SCALAR PROPERTIES
    # =========================================================================

    @property
    def x(self)       -> float: return self.state.x
    @property
    def y(self)       -> float: return self.state.y
    @property
    def psi(self)     -> float: return self.state.psi


# =============================================================================
# PROXY CLASSES — present Vehicle6D with the interface each Nash solver expects
# =============================================================================

class LongVehicleProxy:
    """Wraps Vehicle6D to expose the longitudinal Nash solver interface (2D).

    The longitudinal Nash solver calls:
      vehicle.get_state_space_matrices(dt)  → 2×2 / 2×1 / 2×2
      vehicle.state.x, vehicle.state.vx
      vehicle.x_long
      vehicle.nash_acceleration  (write)
      vehicle.a_desired          (read for reference generator)
      vehicle.L                  (platoon gap)
    """

    def __init__(self, v6d: Vehicle6D):
        self._v = v6d

    def __getattr__(self, name: str):
        return getattr(self._v, name)

    def __setattr__(self, name: str, value):
        if name == '_v':
            super().__setattr__(name, value)
        else:
            setattr(self._v, name, value)

    def get_state_space_matrices(self, dt: float):
        return self._v.get_state_space_matrices_long(dt)

    def get_state_vector(self):
        return self._v.get_long_state()

    @property
    def max_velocity(self) -> float:
        """Physical velocity ceiling — mirrors Longitudinal Vehicle.max_velocity."""
        return self._v.params.max_velocity

    def update_dynamics(self, dt: float):
        """No-op: ego 6D dynamics are handled by simulator via update_dynamics_6d().
        Called by PlatoonManager.update() — skipped here to avoid double-integration."""
        pass

    def get_u_bounds(self) -> Tuple[float, float]:
        """Compute dynamic u_accel bounds from physics at the current operating point.

        Mirrors Longitudinal/vehicle/vehicle.py get_u_bounds().
        Inverts the feedforward law (Belousov Eq. 3.11a, FWD):
            Fxf·cos(δf) = m_eff·u + Ra + Rg + Rr + Fyf·sin(δf)
            u = (Fxf·cos(δf) - F_resist - Fyf·sin(δf)) / m_eff

        Fxf_max: tire traction limit  μ·Fz_f  (no engine model in Vehicle6D)
        Fxf_min: braking limit        -μ·m·g  (full tire friction both axles)

        Returns
        -------
        (u_min, u_max) : float
            Clipped to [MAX_DECELERATION, MAX_ACCELERATION] from unified config.
        """
        from unified.config import MAX_DECELERATION, MAX_ACCELERATION

        v = self._v
        p = v.params
        vx      = v.state.vx
        vy      = v.state.vy
        Omega_z = v.state.psi_dot
        delta_f = v.state.delta_f

        # Resistive forces at current state (same as _compute_Fxf)
        f_vx    = (np.tanh(10.0 * abs(vx) - 8.0) + 1.0) / 2.0
        Ra      = 0.5 * p.air_density * p.drag_coefficient * p.frontal_area * vx * abs(vx)
        Rg      = p.mass * p.gravity * np.sin(p.road_grade)
        vx_sign = np.sign(vx) if abs(vx) > 1e-6 else 0.0
        Rr      = f_vx * p.rolling_resistance_coeff * p.mass * p.gravity * vx_sign
        F_resist = Ra + Rg + Rr

        # Front lateral force coupling term  (Fyf·sin(δf))
        vx_safe = max(abs(vx), 0.1)
        alpha_f = f_vx * (delta_f - np.arctan2(vy + p.lf * Omega_z, vx_safe))
        # Per-wheel normal load (front axle / 2 wheels)
        Fz_f_wheel = p.mass * p.gravity * p.lr / (p.lf + p.lr) / 2.0
        Fyf = 2.0 * v._tire_force(alpha_f, p.Caf, p.tire_friction_coeff, Fz_f_wheel)
        cos_df          = np.cos(delta_f)
        lateral_coupling = Fyf * np.sin(delta_f)

        # Fxf_max: longitudinal tire traction limit (parabolic saturation ceiling)
        #   θ = Cx/(3μFz_f_wheel) but we use simpler μ·Fz_f_axle as ceiling
        Fz_f_axle = p.mass * p.gravity * p.lr / (p.lf + p.lr)
        Fxf_max   = p.tire_friction_coeff * Fz_f_axle   # traction limit [N]

        # Fxf_min: full braking capacity on both axles — μ·m·g
        Fxf_min = -(p.tire_friction_coeff * p.mass * p.gravity)

        # Invert feedforward: u = (Fxf·cos(δf) - F_resist - Fyf·sin(δf)) / m_eff
        cos_df_safe = max(abs(cos_df), 1e-6)
        u_max = (Fxf_max * cos_df_safe - F_resist - lateral_coupling) / v.m_eff
        u_min = (Fxf_min * cos_df_safe - F_resist - lateral_coupling) / v.m_eff

        # Clip to config hard limits
        u_max = float(np.clip(u_max, MAX_DECELERATION, MAX_ACCELERATION))
        u_min = float(np.clip(u_min, MAX_DECELERATION, MAX_ACCELERATION))
        return u_min, u_max


class LatVehicleProxy:
    """Wraps Vehicle6D to expose the lateral Nash solver interface (4D).

    The lateral Nash solver calls:
      vehicle.get_state_space_matrices(dt)  → 4×4 / 4×1 / 2×4
      vehicle.get_state_vector()            → [y, vy, ψ, ψ̇]
      vehicle.state.y, vehicle.state.psi, vehicle.state.psi_dot
      vehicle.x_lateral
      vehicle.vx                            (scheduling parameter)
    """

    def __init__(self, v6d: Vehicle6D):
        self._v = v6d

    def __getattr__(self, name: str):
        return getattr(self._v, name)

    def __setattr__(self, name: str, value):
        if name == '_v':
            super().__setattr__(name, value)
        else:
            setattr(self._v, name, value)

    def get_state_space_matrices(self, dt: float):
        return self._v.get_state_space_matrices_lat(dt)

    def get_state_vector(self) -> np.ndarray:
        return self._v.get_lat_state_vector()


__all__ = ['Vehicle6D', 'LongVehicleProxy', 'LatVehicleProxy', 'UnifiedVehicleState']
