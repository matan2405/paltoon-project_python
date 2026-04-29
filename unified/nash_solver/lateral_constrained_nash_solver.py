#!/usr/bin/env python3
"""
Lateral Nash MPC Solver for Autonomous Platooning Emergency Intervention.

=============================================================================
ARCHITECTURE: DMPC with Iterative Best Response (Li et al. 2019)
=============================================================================
The solver architecture strictly follows Li et al. (2019), "Shared Control
with a Novel Dynamic Authority Allocation Strategy Based on Game Theory and
Driving Safety Field." The two-player Nash Equilibrium is computed via
Distributed MPC (DMPC) using Iterative Best Response (IBR / convex iteration):

  1. Fix u2, solve Player 1's QP  →  u1_new
  2. Fix u1_new, solve Player 2's QP  →  u2_new
  3. Repeat until ||Δu|| < tol  (converges in ~3–5 iterations)

This produces a true Nash Equilibrium, where each player's control is a
best response to the other's. The asymmetric coupling (M12 = H'Q1H ≠ α·H'Q1H
= M21) makes a single joint QP impossible; two separate QPs are required.

  True Nash KKT conditions (Li et al. Eq. 4–9):
    Player 1: (H'Q1H + R1·I)·u1 + H'Q1H·u2       = H'Q1(r1 − U·x0)
    Player 2: (α·H'Q1H + R2·I)·u2 + α·H'Q1H·u1   = α·H'Q1(r2 − U·x0)

  Player 1 Hessian: P1 = H'Q1H + R1·I        (λ-independent, built once)
  Player 2 Hessian: P2 = α·H'Q1H + R2·I      (one per λ level)

  Combined output: u_shared = u1 + u2  (no external blending)

  State:  x = [y, y_dot, psi, psi_dot]
  Input:  u = delta (steering angle [rad])
  Output: z = [y, psi]

=============================================================================
AUTHORITY ALLOCATION: Intentional Inversion for Autonomous Platooning
=============================================================================
Li et al. designed their authority allocation for a human-driven vehicle
where a human driver is the primary agent. In their context, high risk (large
driving safety field force) grants MORE authority to the human, because the
human must steer away from an emergency obstacle.

This solver intentionally reverses that policy for Autonomous Platooning:

  - In platooning, the AUTONOMOUS SYSTEM is the primary safety agent.
  - A high-risk lateral scenario (e.g., unintended lane departure that could
    cause a sideswipe within the platoon) requires the SYSTEM to override the
    human and apply corrective steering, not the other way around.
  - Lateral string stability and safe inter-vehicle clearance demand that the
    automated lateral controller dominate precisely when the situation is most
    critical.

=============================================================================
MATHEMATICAL MECHANISM: Pustilnik & Borrelli (2025) α Scaling
=============================================================================
The authority inversion is achieved by adopting the scaling factor α from
Pustilnik & Borrelli (2025), "Non-Normalized Solutions of Generalized Nash
Equilibria in Dynamic Games With Shared Constraints":

    α(λ) = 1 / (1 + λ)

where λ is the authority ratio derived from the Driving Safety Field.

  - As risk increases, λ increases  →  α → 0
  - α appears as the weight on the Human player's tracking cost: Q2 = α·Q1
  - When α → 0, the Human's tracking objective vanishes; the Human minimizes
    effort only, producing a small u2 → the System's u1 dominates the output.
  - When α → 1 (λ → 0, safe regime), the Human tracks aggressively and
    contributes meaningful authority to the shared steering control.

  Concretely (platooning semantics):
    λ = 0.1  →  α = 0.91  →  Human dominant  (safe lateral position)
    λ = 1.0  →  α = 0.50  →  Equal authority  (moderate lateral risk)
    λ = 10   →  α = 0.09  →  System dominant  (emergency lane correction)

Pustilnik & Borrelli's Corollary 2.2 guarantees that this scaling yields a
valid Generalized Nash Equilibrium for all α ∈ (0, 1), providing theoretical
soundness for the authority inversion across the full operating range.
"""

import os
import sys
import numpy as np
import cvxpy as cp
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

from unified.config import (
    LAT_NASH_CONTROL_DT              as NASH_CONTROL_DT,
    NASH_RISK_GAMMA,
    NASH_NP, NASH_NU,
    LAT_NASH_Q_Y                     as NASH_Q_Y,
    LAT_NASH_Q_PSI                   as NASH_Q_PSI,
    LAT_NASH_Q_Y_TERMINAL_FAC        as NASH_Q_Y_TERMINAL_FACTOR,
    LAT_NASH_Q_PSI_TERMINAL_FAC      as NASH_Q_PSI_TERMINAL_FACTOR,
    LAT_NASH_R1                      as NASH_R1,
    LAT_NASH_R2                      as NASH_R2,
    LAT_NASH_S1                      as NASH_S1,
    LAT_NASH_S2                      as NASH_S2,
    LAT_NASH_Y_MAX                   as NASH_Y_MAX,
    LAT_NASH_PSI_MAX                 as NASH_PSI_MAX,
    LAT_NASH_LAMBDA_LEVELS           as NASH_LAMBDA_LEVELS,
    NASH_SOLVER_BACKEND, NASH_WARM_START, NASH_VERBOSE,
    LAT_NASH_MAX_ITER                as NASH_MAX_ITER,
    LAT_NASH_EPS_ABS                 as NASH_EPS_ABS,
    LAT_NASH_EPS_REL                 as NASH_EPS_REL,
    NASH_POLISH,
    NASH_REGULARIZATION,
    LAT_NASH_REBUILD_DVX,
    MAX_STEERING_ANGLE,
    MAX_STEERING_RATE,
)


@dataclass
class ConstrainedLateralNashParams:
    """
    Parameters for DMPC Nash lateral solver (IBR formulation).

    Weight Design Philosophy (following Li et al. 2019):
    ===================================================
    Q_y  : Position tracking weight. Controls how aggressively the controller
           corrects lateral offset. Too low -> slow correction -> instability.
    Q_psi: Heading tracking weight. Must be >> Q_y to stabilize heading first.
    R1   : System control effort. Higher -> gentler steering.
    R2   : Human control effort.
    S1   : (unused in IBR — cross-coupling does not appear in Nash KKT)
    S2   : (unused in IBR — cross-coupling does not appear in Nash KKT)

    Critical Ratios:
    - Q_y / R1 determines correction speed. Need Q_y/R1 >= 0.01 for stability.
    - Q_psi / Q_y determines heading vs position priority.

    Pustilnik alpha: alpha = 1/(1+lambda)
    - lambda=0.1: alpha=0.91 (human dominant — safe)
    - lambda=1.0: alpha=0.50 (equal authority)
    - lambda=10 : alpha=0.09 (system dominant — dangerous)
    """

    # Prediction horizons
    Np: int = NASH_NP               # Prediction horizon
    Nu: int = NASH_NU               # Control horizon
    dt: float = NASH_CONTROL_DT     # Time step [s]

    Q_y: float = NASH_Q_Y                                    # Lateral position weight
    Q_psi: float = NASH_Q_PSI                               # Heading angle weight

    # Terminal weights
    Q_y_terminal: float = NASH_Q_Y_TERMINAL_FACTOR * NASH_Q_Y
    Q_psi_terminal: float = NASH_Q_PSI_TERMINAL_FACTOR * NASH_Q_PSI

    # Control effort weights
    R1: float = NASH_R1     # System control effort
    R2: float = NASH_R2     # Human control effort

    # Cross-coupling weights — kept for API/config compatibility; not used in IBR cost
    # (S terms do not appear in the per-player Nash KKT conditions)
    S1: float = NASH_S1
    S2: float = NASH_S2

    # Input constraints [rad]
    delta_min: float = -MAX_STEERING_ANGLE
    delta_max: float = MAX_STEERING_ANGLE

    # Rate constraints [rad/s]
    ddelta_max: float = MAX_STEERING_RATE

    # State constraints
    y_max: float = NASH_Y_MAX
    psi_max: float = NASH_PSI_MAX

    # Lambda levels for pre-computation
    lambda_levels: tuple = NASH_LAMBDA_LEVELS

    # IBR convergence settings (Li et al. Section 2.2.3)
    ibr_max_iter: int = 15        # Max IBR iterations (converges in ~3-5)
    ibr_tol: float = 1e-4         # Convergence tolerance

    # Solver settings
    solver: str = NASH_SOLVER_BACKEND
    warm_start: bool = NASH_WARM_START
    verbose: bool = NASH_VERBOSE
    max_iter: int = NASH_MAX_ITER
    eps_abs: float = NASH_EPS_ABS
    eps_rel: float = NASH_EPS_REL
    polish: bool = NASH_POLISH

    # Regularization
    regularization: float = NASH_REGULARIZATION

    # Physics-based linearization rebuild threshold [m/s]
    # Rebuild when vx has drifted more than this from the last rebuild point.
    # Derived from bicycle model Coriolis sensitivity — see unified/config.py.
    rebuild_dvx: float = LAT_NASH_REBUILD_DVX

    # ── Risk-aware planning: E[J] + γ·Var[J] (MA-IDM, Zhang & Sun 2024) ─────
    # SE-kernel GP hyperparameters for the human player's lateral residual.
    # sigma_k: stationary std [rad];  ell: lengthscale [s]  (MA_IDM_LAT_PARAMS)
    # gamma_risk: risk sensitivity γ;  0 = disabled.
    sigma_k:    float = 0.0
    ell:        float = 1.0
    gamma_risk: float = 0.0


class ConstrainedLateralNashSolver:
    """
    DMPC Nash Equilibrium Solver for Lateral Control — IBR (Li et al. 2019).

    Implements true Nash Equilibrium via IBR (convex iteration):
      Each player minimizes their own cost with the other player's control fixed.
      Alternate until convergence -> Nash Equilibrium.

    Nash Equilibrium (per-player KKT, Li et al. Eq. 4-9):
    ======================================================
    Player 1 (System): min_{u1} ||H(u1+u2)+Ux0 - r1||^2_Q1 + R1*||u1||^2
    Player 2 (Human):  min_{u2} alpha*||H(u1+u2)+Ux0 - r2||^2_Q1 + R2*||u2||^2

    where alpha = 1/(1+lambda): high lambda -> small alpha -> system dominates.

    Player 1 Hessian: P1 = H'Q1H + R1*I  (lambda-independent, built once)
    Player 2 Hessian: P2 = alpha*H'Q1H + R2*I  (one per lambda level)
    """

    def __init__(self, vehicle, params: ConstrainedLateralNashParams = None):
        """Initialize solver matrices, cached lambda problems, and runtime statistics."""
        if vehicle is None:
            raise ValueError("Vehicle must be provided")

        self.params = params or ConstrainedLateralNashParams()
        self.vehicle = vehicle

        self._verbose_dmpc = True   # print only during __init__; silenced after
        self._build_state_space()
        self._build_prediction_matrices()

        self.R2_eff = self.params.R2
        self.Q_y_eff = self.params.Q_y

        self._build_base_cost_matrices()
        self._build_dmpc_problems()
        self._verbose_dmpc = False  # silence runtime rebuilds

        self.u1_prev = 0.0
        self.u2_prev = 0.0
        self._vx_at_last_rebuild: float = 0.0

        self.current_lambda_idx = 3  # Start at lambda=1.0
        self.last_lambda_used = 1.0

        # Full horizon solution storage
        self.u1_full = np.zeros(self.params.Nu)
        self.u2_full = np.zeros(self.params.Nu)

        # Statistics
        self.last_solve_time = 0.0
        self.last_ibr_iters = 0
        self.last_status = 'initialized'
        self.last_costs = [0.0, 0.0]

        self.constraint_stats = {
            'total_solves': 0,
            'u1_at_min': 0, 'u1_at_max': 0,
            'u2_at_min': 0, 'u2_at_max': 0,
            'u1_rate_active': 0, 'u2_rate_active': 0,
            'lambda_level_usage': {lam: 0 for lam in self.params.lambda_levels}
        }

        q_r_ratio = self.params.Q_y / self.params.R1 if self.params.R1 > 0 else 0
        print(f"DMPC Lateral Nash Solver Initialized (IBR - Li et al. 2019)")
        print(f"   Horizons: Np={self.params.Np}, Nu={self.params.Nu}")
        print(f"   Steering bounds: delta in [{np.degrees(self.params.delta_min):.1f}deg, {np.degrees(self.params.delta_max):.1f}deg]")
        print(f"   R1={self.params.R1:.0f}, R2={self.params.R2:.0f}")
        print(f"   Q_y/R1 ratio: {q_r_ratio:.4f}")
        print(f"   IBR: max_iter={self.params.ibr_max_iter}, tol={self.params.ibr_tol}")
        print(f"   Lambda levels: {self.params.lambda_levels}")

    def _build_state_space(self):
        """Get discrete-time bicycle model state-space from vehicle."""
        p = self.params
        self.A, self.B, self.C = self.vehicle.get_state_space_matrices(p.dt)
        self.nx = self.A.shape[0]
        self.nz = self.C.shape[0]
        print(f"   Using vehicle's state-space matrices (dt={p.dt}s)")

    def _build_prediction_matrices(self):
        """Build prediction matrices U, H."""
        A, B, C = self.A, self.B, self.C
        Np, Nu = self.params.Np, self.params.Nu

        A_pows = [np.eye(self.nx)]
        for _ in range(Np):
            A_pows.append(A_pows[-1] @ A)

        self.U = np.zeros((Np * self.nz, self.nx))
        for i in range(Np):
            self.U[i*self.nz:(i+1)*self.nz, :] = C @ A_pows[i+1]

        CAB = [C @ A_pows[k] @ B for k in range(Np)]
        self.H = np.zeros((Np * self.nz, Nu))
        for i in range(Np):
            for j in range(min(i+1, Nu)):
                self.H[i*self.nz:(i+1)*self.nz, j:j+1] = CAB[i-j]

    def _build_base_cost_matrices(self):
        """Pre-compute base cost matrices with terminal weights."""
        p = self.params
        Np = p.Np

        Q_blocks = []
        for k in range(Np):
            if k == Np - 1:
                Q_blocks.append(np.diag([p.Q_y_terminal, p.Q_psi_terminal]))
            else:
                Q_blocks.append(np.diag([self.Q_y_eff, p.Q_psi]))

        self.Q1_full = np.block([
            [Q_blocks[i] if i == j else np.zeros((self.nz, self.nz))
             for j in range(Np)]
            for i in range(Np)
        ])

        self.HQ1H = self.H.T @ self.Q1_full @ self.H
        self.HQ1H += self.params.regularization * np.eye(self.params.Nu)

        # ── Risk-aware Hessian: HQ1H_risk = (I + 4γ̃·HQ1H·Σ_ε) @ HQ1H ─────────
        # Σ_ε[i,j] = σ_k² · exp(−(i−j)²·dt² / (2ℓ²))  — SE kernel (Eq. 9)
        #
        # Dimensional normalisation: γ̃ = γ / trace(HQ1H) so that γ=0.1 means
        # "add 10% risk weight relative to the nominal cost scale."
        Nu = self.params.Nu
        if self.params.gamma_risk > 0.0 and self.params.sigma_k > 0.0:
            idx = np.arange(Nu)
            dt_mat    = np.abs(idx[:, None] - idx[None, :]) * self.params.dt
            Sigma_eps = self.params.sigma_k ** 2 * np.exp(
                -dt_mat ** 2 / (2.0 * self.params.ell ** 2)
            )
            norm_factor       = max(float(np.trace(self.HQ1H)), 1.0)
            gamma_scaled      = self.params.gamma_risk / norm_factor
            M_risk            = 4.0 * gamma_scaled * self.HQ1H @ Sigma_eps
            self.I_plus_Mrisk = np.eye(Nu) + M_risk
            self.HQ1H_risk    = self.I_plus_Mrisk @ self.HQ1H
        else:
            self.I_plus_Mrisk = np.eye(Nu)
            self.HQ1H_risk    = self.HQ1H.copy()

    def _pustilnik_alpha(self, lambda_k: float) -> float:
        """
        Map authority ratio lambda to Pustilnik scaling factor alpha.

        alpha(lambda) = 1 / (1 + lambda)

        - lambda=0.1 -> alpha=0.909 (human dominant — safe)
        - lambda=1.0 -> alpha=0.500 (normalized — equal)
        - lambda=10  -> alpha=0.091 (system dominant — dangerous)
        """
        return 1.0 / (1.0 + lambda_k)

    def _build_dmpc_problems(self):
        """
        Build separate CVXPY problems for each player (IBR formulation).

        Player 1 (System): P1 = H'Q1H + R1*I  (lambda-independent, built once)
        Player 2 (Human):  P2 = alpha*H'Q1H + R2*I  (one per lambda level)

        Only the linear terms q1, q2 change each IBR iteration — DPP compliant.
        """
        p = self.params
        Nu = p.Nu
        ddelta_lim = p.ddelta_max * p.dt

        # ---- Player 1: lambda-independent ----
        P1 = self.HQ1H_risk + p.R1 * np.eye(Nu)
        P1 = 0.5 * (P1 + P1.T)

        u1_var = cp.Variable(Nu, name='u1')
        q1_param = cp.Parameter(Nu, name='q1')
        u1_prev_param = cp.Parameter(name='u1_prev')

        cost1 = cp.quad_form(u1_var, P1) + q1_param @ u1_var
        c1 = [u1_var >= p.delta_min, u1_var <= p.delta_max,
              u1_var[0] - u1_prev_param <= ddelta_lim,
              u1_var[0] - u1_prev_param >= -ddelta_lim]
        if Nu > 1:
            for k in range(1, Nu):
                c1 += [u1_var[k] - u1_var[k-1] <= ddelta_lim,
                       u1_var[k] - u1_var[k-1] >= -ddelta_lim]

        prob1 = cp.Problem(cp.Minimize(cost1), c1)

        self.prob1 = {
            'problem': prob1,
            'u1_var': u1_var,
            'q1_param': q1_param,
            'u1_prev_param': u1_prev_param,
            'P1': P1,
            'warm_start': np.zeros(Nu),
        }

        # ---- Player 2: one per lambda level ----
        self.prob2 = {}
        self.lambda_levels = list(p.lambda_levels)

        if self._verbose_dmpc:
            print(f"   Building DMPC problems (IBR): 1 system + {len(self.lambda_levels)} human...")

        for lam in self.lambda_levels:
            alpha = self._pustilnik_alpha(lam)
            P2 = alpha * self.HQ1H_risk + self.R2_eff * np.eye(Nu)
            P2 = 0.5 * (P2 + P2.T)

            u2_var = cp.Variable(Nu, name=f'u2_lam{lam}')
            q2_param = cp.Parameter(Nu, name=f'q2_lam{lam}')
            u2_prev_param = cp.Parameter(name=f'u2_prev_lam{lam}')

            cost2 = cp.quad_form(u2_var, P2) + q2_param @ u2_var
            c2 = [u2_var >= p.delta_min, u2_var <= p.delta_max,
                  u2_var[0] - u2_prev_param <= ddelta_lim,
                  u2_var[0] - u2_prev_param >= -ddelta_lim]
            if Nu > 1:
                for k in range(1, Nu):
                    c2 += [u2_var[k] - u2_var[k-1] <= ddelta_lim,
                           u2_var[k] - u2_var[k-1] >= -ddelta_lim]

            prob2_problem = cp.Problem(cp.Minimize(cost2), c2)
            is_dpp = prob2_problem.is_dcp(dpp=True)

            self.prob2[lam] = {
                'problem': prob2_problem,
                'u2_var': u2_var,
                'q2_param': q2_param,
                'u2_prev_param': u2_prev_param,
                'P2': P2,
                'alpha': alpha,
                'warm_start': np.zeros(Nu),
            }
            status = "ok" if is_dpp else "WARN"
            if self._verbose_dmpc:
                print(f"      lam={lam:5.2f} (alpha={alpha:.3f}): DPP {status}")

    def _find_closest_lambda(self, lambda_k: float) -> float:
        """Find the closest pre-computed lambda level (log-space)."""
        lambda_clipped = np.clip(lambda_k, self.lambda_levels[0], self.lambda_levels[-1])
        log_lambda = np.log(lambda_clipped)
        log_levels = np.log(self.lambda_levels)
        closest_idx = np.argmin(np.abs(log_levels - log_lambda))
        return self.lambda_levels[closest_idx]

    def update_linearization(self, vx: float) -> bool:
        """Rebuild all prediction matrices if vx has changed beyond the physics threshold.

        Must rebuild the full chain (_build_state_space → _build_prediction_matrices →
        _build_base_cost_matrices → _build_dmpc_problems) to keep H, U, HQ1H, and
        the CVXPY P1/P2 matrices consistent. Partial rebuilds cause H/HQ1H mismatch.

        Parameters
        ----------
        vx : float
            Current longitudinal speed [m/s] of the ego vehicle.

        Returns
        -------
        bool
            True if matrices are up to date (rebuilt or still valid).
        """
        if abs(vx - self._vx_at_last_rebuild) < self.params.rebuild_dvx:
            return True
        self._build_state_space()
        self._build_prediction_matrices()
        self._build_base_cost_matrices()
        self._build_dmpc_problems()
        self._vx_at_last_rebuild = vx
        return True

    def solve_nash_equilibrium(self, x0: np.ndarray,
                                R1_ref: np.ndarray,
                                R2_ref: np.ndarray,
                                lambda_k: float,
                                field_force: float = 0.0) -> Tuple[float, float]:
        """
        Solve Nash Equilibrium via Iterative Best Response (IBR).

        Li et al. (2019), Section 2.2.3 - Convex Iteration:
          1. Fix u2, solve Player 1's QP -> u1_new
          2. Fix u1_new, solve Player 2's QP -> u2_new
          3. Check convergence; repeat until |delta_u| < tol

        IBR linear terms at each iteration:
          q1(u2_k)   = -2   * H'Q1 * (r1 - z_free - H*u2_k)
          q2(u1_new) = -2*alpha * H'Q1 * (r2 - z_free - H*u1_new)

        The simulator should use: u_shared = u1 + u2

        Args:
            x0: Current state [y, y_dot, psi, psi_dot]
            R1_ref: System reference trajectory (Np, 2)
            R2_ref: Human reference trajectory (Np, 2)
            lambda_k: Authority ratio from safety field
            field_force: Safety field force (for logging)

        Returns:
            (delta1_opt, delta2_opt): Optimal steering angles [rad]
        """
        import time
        start_time = time.perf_counter()

        p = self.params
        Nu = p.Nu

        lambda_used = self._find_closest_lambda(lambda_k)
        alpha = self._pustilnik_alpha(lambda_used)

        self.last_lambda_used = lambda_used
        self.constraint_stats['lambda_level_usage'][lambda_used] += 1

        r1 = R1_ref.flatten()
        r2 = R2_ref.flatten()
        z_free = self.U @ x0

        # Pre-compute constant parts of q terms
        # Apply risk modification: HQ1_r_risk = (I + 4γ·HQ1H·Σ_ε) @ HQ1_r
        HQ1_r1 = self.I_plus_Mrisk @ (self.H.T @ self.Q1_full @ (r1 - z_free))
        HQ1_r2 = self.I_plus_Mrisk @ (self.H.T @ self.Q1_full @ (r2 - z_free))
        HQ1H = self.HQ1H_risk  # risk-aware Hessian used in IBR linear terms

        prob1 = self.prob1
        prob2 = self.prob2[lambda_used]

        # Set u_prev parameters (constant throughout IBR loop)
        prob1['u1_prev_param'].value = self.u1_prev
        prob2['u2_prev_param'].value = self.u2_prev

        # Warm start from previous solution
        u2_k = prob2['warm_start'].copy()
        u1_k = prob1['warm_start'].copy()

        u1_new = u1_k.copy()
        u2_new = u2_k.copy()

        solver_kwargs = dict(
            solver=cp.OSQP,
            warm_start=p.warm_start,
            verbose=p.verbose,
            eps_abs=p.eps_abs,
            eps_rel=p.eps_rel,
            max_iter=p.max_iter,
            polish=p.polish,
        )

        ibr_converged = False
        try:
            for ibr_iter in range(p.ibr_max_iter):
                # ---- Player 1: fix u2_k ----
                prob1['q1_param'].value = -2.0 * (HQ1_r1 - HQ1H @ u2_k)
                if p.warm_start:
                    prob1['u1_var'].value = prob1['warm_start']
                prob1['problem'].solve(**solver_kwargs)

                if prob1['problem'].status not in ['optimal', 'optimal_inaccurate']:
                    print(f"Player 1 solver: {prob1['problem'].status} (lam={lambda_used}, IBR iter {ibr_iter})")
                    break
                u1_new = prob1['u1_var'].value.copy()

                # ---- Player 2: fix u1_new ----
                prob2['q2_param'].value = -2.0 * alpha * (HQ1_r2 - HQ1H @ u1_new)
                if p.warm_start:
                    prob2['u2_var'].value = prob2['warm_start']
                prob2['problem'].solve(**solver_kwargs)

                if prob2['problem'].status not in ['optimal', 'optimal_inaccurate']:
                    print(f"Player 2 solver: {prob2['problem'].status} (lam={lambda_used}, IBR iter {ibr_iter})")
                    break
                u2_new = prob2['u2_var'].value.copy()

                # ---- Convergence check ----
                delta = max(np.max(np.abs(u1_new - u1_k)), np.max(np.abs(u2_new - u2_k)))
                u1_k = u1_new
                u2_k = u2_new

                if delta < p.ibr_tol:
                    ibr_converged = True
                    break

            self.last_ibr_iters = ibr_iter + 1
            self.last_status = 'optimal' if ibr_converged else 'optimal_inaccurate'
            self.last_solve_time = time.perf_counter() - start_time

            u1_opt = float(u1_new[0])
            u2_opt = float(u2_new[0])

            # Store full trajectories
            self.u1_full = u1_new.copy()
            self.u2_full = u2_new.copy()

            # Update warm starts (shift horizon)
            prob1['warm_start'] = np.roll(u1_new, -1)
            prob1['warm_start'][-1] = u1_new[-1]
            prob2['warm_start'] = np.roll(u2_new, -1)
            prob2['warm_start'][-1] = u2_new[-1]

            self._update_constraint_stats(u1_opt, u2_opt)

            self.u1_prev = u1_opt
            self.u2_prev = u2_opt

            self.last_costs = [
                float(prob1['problem'].value) if prob1['problem'].value is not None else 0.0,
                float(prob2['problem'].value) if prob2['problem'].value is not None else 0.0,
            ]

            return u1_opt, u2_opt

        except Exception as e:
            print(f"IBR solver error: {e}")
            self.last_status = 'error'
            return self.u1_prev, self.u2_prev

    def _update_constraint_stats(self, u1_opt: float, u2_opt: float):
        """Update constraint activity statistics."""
        p = self.params
        tol = 1e-5

        self.constraint_stats['total_solves'] += 1

        if abs(u1_opt - p.delta_min) < tol:
            self.constraint_stats['u1_at_min'] += 1
        if abs(u1_opt - p.delta_max) < tol:
            self.constraint_stats['u1_at_max'] += 1
        if abs(u2_opt - p.delta_min) < tol:
            self.constraint_stats['u2_at_min'] += 1
        if abs(u2_opt - p.delta_max) < tol:
            self.constraint_stats['u2_at_max'] += 1

        ddelta_max_step = p.ddelta_max * p.dt
        if abs(abs(u1_opt - self.u1_prev) - ddelta_max_step) < tol:
            self.constraint_stats['u1_rate_active'] += 1
        if abs(abs(u2_opt - self.u2_prev) - ddelta_max_step) < tol:
            self.constraint_stats['u2_rate_active'] += 1

    def get_full_trajectories(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get full optimal control sequences."""
        return self.u1_full.copy(), self.u2_full.copy()

    def get_predicted_states(self, x0: np.ndarray) -> Optional[np.ndarray]:
        """Get predicted output trajectory."""
        u_shared = self.u1_full + self.u2_full
        z_pred = self.U @ x0 + self.H @ u_shared
        return z_pred.reshape(self.params.Np, 2)

    def get_solver_stats(self) -> Dict:
        """Get solver statistics."""
        return {
            'status': self.last_status,
            'solve_time_ms': self.last_solve_time * 1000,
            'ibr_iters': self.last_ibr_iters,
            'costs': self.last_costs,
            'u1_prev': self.u1_prev,
            'u2_prev': self.u2_prev,
            'lambda_used': self.last_lambda_used,
            'alpha_pustilnik': self._pustilnik_alpha(self.last_lambda_used),
            'lambda_levels': self.lambda_levels,
            'formulation': 'DMPC IBR (Li et al. 2019)',
        }

    def get_constraint_summary(self) -> str:
        """Get formatted summary of constraint activity."""
        stats = self.constraint_stats
        total = stats['total_solves']
        p = self.params

        if total == 0:
            return "No solves performed yet."

        lambda_usage = "\n".join([
            f"    lam={lam:5.2f} (alpha={self._pustilnik_alpha(lam):.3f}): "
            f"{stats['lambda_level_usage'][lam]:>6} times "
            f"({100*stats['lambda_level_usage'][lam]/total:>5.1f}%)"
            for lam in self.params.lambda_levels
        ])

        summary = f"""
LATERAL DMPC NASH (IBR) - CONSTRAINT SUMMARY
  Total solves: {total}
  LAMBDA LEVEL USAGE (Pustilnik alpha):
{lambda_usage}
  INPUT BOUNDS:
    delta1 at min ({np.degrees(p.delta_min):>6.1f}deg): {stats['u1_at_min']:>6} times
    delta1 at max ({np.degrees(p.delta_max):>6.1f}deg): {stats['u1_at_max']:>6} times
    delta2 at min ({np.degrees(p.delta_min):>6.1f}deg): {stats['u2_at_min']:>6} times
    delta2 at max ({np.degrees(p.delta_max):>6.1f}deg): {stats['u2_at_max']:>6} times
  RATE CONSTRAINTS:
    delta1 rate active: {stats['u1_rate_active']:>6} times
    delta2 rate active: {stats['u2_rate_active']:>6} times
"""
        return summary

    def reset(self):
        """Reset solver state."""
        self.u1_prev = 0.0
        self.u2_prev = 0.0
        self.last_status = 'reset'
        self.u1_full = np.zeros(self.params.Nu)
        self.u2_full = np.zeros(self.params.Nu)

        self.prob1['warm_start'] = np.zeros(self.params.Nu)
        for lam in self.lambda_levels:
            self.prob2[lam]['warm_start'] = np.zeros(self.params.Nu)

        for key in self.constraint_stats:
            if isinstance(self.constraint_stats[key], dict):
                for subkey in self.constraint_stats[key]:
                    self.constraint_stats[key][subkey] = 0
            else:
                self.constraint_stats[key] = 0

    def print_constraint_summary(self):
        """Print constraint summary."""
        print(self.get_constraint_summary())

    def set_driver_type(self, driver_type: str):
        """
        Driver type no longer modifies Nash weights.
        Driver personality is expressed through behavior parameters
        (T_lc, k_e, k_psi) in reference generators, NOT in the Nash game.
        Kept for API compatibility.
        """
        pass
