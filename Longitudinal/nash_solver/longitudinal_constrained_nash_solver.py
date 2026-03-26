#!/usr/bin/env python3
"""
Longitudinal Nash MPC Solver for Autonomous Platooning Emergency Intervention.

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

=============================================================================
AUTHORITY ALLOCATION: Intentional Inversion for Autonomous Platooning
=============================================================================
Li et al. designed their authority allocation for a human-driven vehicle
where a human driver is the primary agent. In their context, high risk (large
driving safety field force) grants MORE authority to the human, because the
human must react to an emergency obstacle.

This solver intentionally reverses that policy for Autonomous Platooning:

  - In platooning, the AUTONOMOUS SYSTEM is the primary safety agent.
  - A high-risk scenario (e.g., sudden lead-vehicle braking, dangerously
    small gap) requires the SYSTEM to override the human and apply emergency
    braking, not the other way around.
  - String stability of the platoon demands that the automated longitudinal
    controller dominate precisely when the situation is most critical.

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
    contributes meaningful authority to the shared control.

  Concretely (platooning semantics):
    λ = 0.1  →  α = 0.91  →  Human dominant  (safe following distance)
    λ = 1.0  →  α = 0.50  →  Equal authority  (moderate risk)
    λ = 10   →  α = 0.09  →  System dominant  (emergency intervention)

Pustilnik & Borrelli's Corollary 2.2 guarantees that this scaling yields a
valid Generalized Nash Equilibrium for all α ∈ (0, 1), providing theoretical
soundness for the authority inversion across the full operating range.
"""

import os
import sys
import numpy as np
import cvxpy as cp
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import time
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    NASH_CONTROL_DT, NASH_NP, NASH_NU,
    NASH_Q_POS, NASH_Q_VEL,
    NASH_R1, NASH_R2, NASH_S1, NASH_S2,
    NASH_U1_MIN, NASH_U1_MAX, NASH_U2_MIN, NASH_U2_MAX,
    NASH_DU1_MAX, NASH_DU2_MAX,
    NASH_V_MIN, NASH_V_MAX, NASH_GAP_MIN,
    NASH_LAMBDA_LEVELS,
    NASH_SOLVER_BACKEND, NASH_WARM_START, NASH_VERBOSE,
    NASH_MAX_ITER, NASH_EPS_ABS, NASH_EPS_REL, NASH_REGULARIZATION,
)


@dataclass
class ConstrainedNashParams:
    """
    Parameters for DMPC Nash MPC solver (IBR formulation).

    Pustilnik α: α = 1/(1+λ)
    True Nash via Iterative Best Response (Li et al. 2019)
    """

    # Prediction horizons
    Np: int = NASH_NP              # Prediction horizon
    Nu: int = NASH_NU              # Control horizon
    dt: float = NASH_CONTROL_DT    # Time step [s]

    # Cost weights (base values — config is single source of truth)
    Q_pos: float = NASH_Q_POS     # Position tracking weight
    Q_vel: float = NASH_Q_VEL     # Velocity tracking weight

    # Terminal weights (higher at final prediction step for stability)
    Q_pos_terminal: float = 10.0 * NASH_Q_POS
    Q_vel_terminal: float = 4.0 * NASH_Q_VEL

    # Control effort weights
    R1: float = NASH_R1           # System control effort
    R2: float = NASH_R2           # Human control effort

    # Cross-coupling weights — kept for API/config compatibility; not used in IBR cost
    # (S terms do not appear in the per-player Nash KKT conditions)
    S1: float = NASH_S1
    S2: float = NASH_S2

    # Input constraints [m/s²]
    u1_min: float = NASH_U1_MIN
    u1_max: float = NASH_U1_MAX
    u2_min: float = NASH_U2_MIN
    u2_max: float = NASH_U2_MAX

    # Rate constraints [m/s³]
    du1_max: float = NASH_DU1_MAX
    du2_max: float = NASH_DU2_MAX

    # State constraints
    v_min: float = NASH_V_MIN
    v_max: float = NASH_V_MAX
    gap_min: float = NASH_GAP_MIN

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

    # Regularization
    regularization: float = NASH_REGULARIZATION


class ConstrainedLongitudinalNashSolver:
    """
    DMPC Nash Equilibrium Solver — Iterative Best Response (Li et al. 2019).

    Implements true Nash Equilibrium via IBR (convex iteration):
      Each player minimizes their own cost with the other player's control fixed.
      Alternate until convergence → Nash Equilibrium.

    Nash Equilibrium (per-player KKT, Li et al. Eq. 4-9):
    ======================================================
    Player 1 (System): min_{u1} ||H(u1+u2)+Ux0 - r1||²_Q1 + R1·||u1||²
    Player 2 (Human):  min_{u2} α·||H(u1+u2)+Ux0 - r2||²_Q1 + R2·||u2||²

    where α = 1/(1+λ): high λ → small α → system dominates (platooning semantics)
    """

    def __init__(self, vehicle=None, platoon_manager=None, human_driver=None,
                 params: ConstrainedNashParams = None,
                 Np: int = None, Nu: int = None, dt: float = None):
        """Initialize the DMPC Nash solver."""
        self.vehicle = vehicle
        self.platoon_manager = platoon_manager
        self.human_driver = human_driver

        self.params = params or ConstrainedNashParams()

        if Np is not None:
            self.params.Np = Np
        if Nu is not None:
            self.params.Nu = Nu
        if dt is not None:
            self.params.dt = dt

        self.R2_eff = self.params.R2
        self.Q_pos_eff = self.params.Q_pos

        self._build_state_space()
        self._build_prediction_matrices()
        self._build_base_cost_matrices()
        self._build_dmpc_problems()

        self.u1_prev = 0.0
        self.u2_prev = 0.0
        self.current_lambda_idx = 3  # Start at λ=1.0

        # Full horizon solution storage
        self.u1_full = np.zeros(self.params.Nu)
        self.u2_full = np.zeros(self.params.Nu)

        # Statistics
        self.last_solve_time = 0.0
        self.last_ibr_iters = 0
        self.last_status = 'initialized'
        self.last_costs = [0.0, 0.0]
        self.last_lambda_used = 1.0
        self.last_alpha = 0.5

        self.constraint_stats = {
            'total_solves': 0,
            'u1_min_active': 0,
            'u1_max_active': 0,
            'u2_min_active': 0,
            'u2_max_active': 0,
            'u1_rate_active': 0,
            'u2_rate_active': 0,
            'lambda_level_usage': {lam: 0 for lam in self.params.lambda_levels}
        }
        self.last_constraint_info = {}

        print(f"DMPC Nash Solver Initialized (IBR - Li et al. 2019)")
        print(f"   Horizons: Np={self.params.Np}, Nu={self.params.Nu}")
        print(f"   Weights: R1={self.params.R1}, R2={self.params.R2}")
        print(f"   Terminal: Q_pos_T={self.params.Q_pos_terminal:.0f}, Q_vel_T={self.params.Q_vel_terminal:.0f}")
        print(f"   IBR: max_iter={self.params.ibr_max_iter}, tol={self.params.ibr_tol}")
        print(f"   Lambda levels: {self.params.lambda_levels}")

    # ==========================================
    # Pustilnik α Mapping
    # ==========================================

    def _pustilnik_alpha(self, lambda_k: float) -> float:
        """
        Map authority ratio λ to Pustilnik scaling factor α.

        α(λ) = 1 / (1 + λ)

        - λ=0.1 → α=0.909 (human dominant — safe, human tracks aggressively)
        - λ=0.5 → α=0.667 (human has more authority)
        - λ=1.0 → α=0.500 (normalized solution — equal)
        - λ=2.0 → α=0.333 (system has more authority)
        - λ=10  → α=0.091 (system dominant — dangerous, human barely acts)

        Corollary 2.2 guarantees: reducing α for the system
        cannot worsen the system's cost.
        """
        return 1.0 / (1.0 + lambda_k)

    # ==========================================
    # State Space & Prediction
    # ==========================================

    def _build_state_space(self):
        """Build discrete-time state-space matrices.

        Reads from vehicle's dynamic linearization (nonlinear plant Jacobians).
        Falls back to double integrator if vehicle not available.
        """
        if self.vehicle is not None:
            self.A, self.B, self.C = self.vehicle.get_state_space_matrices(self.params.dt)
        else:
            dt = self.params.dt
            self.A = np.array([[1.0, dt], [0.0, 1.0]])
            self.B = np.array([[0.5 * dt**2], [dt]])
            self.C = np.eye(2)

        self.nx = 2  # states
        self.nz = 2  # outputs

    def update_linearization(self, dt=None):
        """Re-linearize at vehicle's current operating point.

        Called at each Nash control step (step 3 in simulation loop).
        Only rebuilds OSQP problems when A or B has changed significantly —
        at steady cruising the matrices are nearly constant so rebuilding is skipped.
        """
        if self.vehicle is None:
            return False
        dt = dt or self.params.dt
        A_new, B_new, C_new = self.vehicle.get_state_space_matrices(dt)
        if (self.A is not None
                and np.allclose(A_new, self.A, rtol=1e-4, atol=1e-6)
                and np.allclose(B_new, self.B, rtol=1e-4, atol=1e-6)):
            return True
        self.A = A_new
        self.B = B_new
        self.C = C_new
        self._build_prediction_matrices()
        self._build_base_cost_matrices()
        self._build_dmpc_problems()
        return True

    def _build_prediction_matrices(self):
        """Build prediction matrices U, H (lambda-independent)."""
        A, B, C = self.A, self.B, self.C
        Np, Nu = self.params.Np, self.params.Nu
        nx, nz = self.nx, self.nz

        A_pows = [np.eye(nx)]
        for _ in range(Np):
            A_pows.append(A_pows[-1] @ A)

        self.U = np.zeros((Np * nz, nx))
        for i in range(Np):
            self.U[i*nz:(i+1)*nz, :] = C @ A_pows[i+1]

        CAB = [C @ A_pows[k] @ B for k in range(Np)]
        self.H = np.zeros((Np * nz, Nu))
        for i in range(Np):
            for j in range(min(i+1, Nu)):
                self.H[i*nz:(i+1)*nz, j:j+1] = CAB[i-j]

    # ==========================================
    # Terminal weights + driver type
    # ==========================================

    def _build_base_cost_matrices(self):
        """Build base cost matrices with terminal weights.

        Block diagonal Q1 with higher terminal weights at k=Np-1
        for improved stability.
        """
        p = self.params
        Np = p.Np

        Q_blocks = []
        for k in range(Np):
            if k == Np - 1:
                Q_blocks.append(np.diag([p.Q_pos_terminal, p.Q_vel_terminal]))
            else:
                Q_blocks.append(np.diag([self.Q_pos_eff, p.Q_vel]))

        self.Q1_full = np.block([
            [Q_blocks[i] if i == j else np.zeros((self.nz, self.nz))
             for j in range(Np)]
            for i in range(Np)
        ])

        # Pre-compute H'Q1H (used in P1 and P2)
        self.HQ1H = self.H.T @ self.Q1_full @ self.H
        self.HQ1H += self.params.regularization * np.eye(self.params.Nu)

    # ==========================================
    # DMPC Problem Construction (IBR)
    # ==========================================

    def _build_dmpc_problems(self):
        """
        Build separate CVXPY problems for each player (IBR formulation).

        Player 1 (System): P1 = H'Q1H + R1·I  (lambda-independent, built once)
        Player 2 (Human):  P2 = alpha·H'Q1H + R2·I  (one per lambda level)

        Only the linear terms q1, q2 change each IBR iteration — DPP compliant.

        IBR algorithm (Li et al. Section 2.2.3):
          for k = 1..ibr_max_iter:
            u1_new = argmin_{u1} u1'P1u1 + q1(u2_k)'u1   s.t. constraints
            u2_new = argmin_{u2} u2'P2u2 + q2(u1_new)'u2  s.t. constraints
            if max(|u1_new - u1_k|, |u2_new - u2_k|) < tol: break
        """
        p = self.params
        Nu = p.Nu

        du1_lim = p.du1_max * p.dt
        du2_lim = p.du2_max * p.dt

        # ---- Player 1: lambda-independent ----
        P1 = self.HQ1H + p.R1 * np.eye(Nu)
        P1 = 0.5 * (P1 + P1.T)  # ensure symmetric

        u1_var = cp.Variable(Nu, name='u1')
        q1_param = cp.Parameter(Nu, name='q1')
        u1_prev_param = cp.Parameter(name='u1_prev')

        cost1 = cp.quad_form(u1_var, P1) + q1_param @ u1_var
        c1 = [u1_var >= p.u1_min, u1_var <= p.u1_max,
              u1_var[0] - u1_prev_param <= du1_lim,
              u1_var[0] - u1_prev_param >= -du1_lim]
        if Nu > 1:
            for k in range(1, Nu):
                c1 += [u1_var[k] - u1_var[k-1] <= du1_lim,
                       u1_var[k] - u1_var[k-1] >= -du1_lim]

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

        print(f"   Building DMPC problems (IBR): 1 system + {len(self.lambda_levels)} human...")

        for lam in self.lambda_levels:
            alpha = self._pustilnik_alpha(lam)
            P2 = alpha * self.HQ1H + self.R2_eff * np.eye(Nu)
            P2 = 0.5 * (P2 + P2.T)

            u2_var = cp.Variable(Nu, name=f'u2_lam{lam}')
            q2_param = cp.Parameter(Nu, name=f'q2_lam{lam}')
            u2_prev_param = cp.Parameter(name=f'u2_prev_lam{lam}')

            cost2 = cp.quad_form(u2_var, P2) + q2_param @ u2_var
            c2 = [u2_var >= p.u2_min, u2_var <= p.u2_max,
                  u2_var[0] - u2_prev_param <= du2_lim,
                  u2_var[0] - u2_prev_param >= -du2_lim]
            if Nu > 1:
                for k in range(1, Nu):
                    c2 += [u2_var[k] - u2_var[k-1] <= du2_lim,
                           u2_var[k] - u2_var[k-1] >= -du2_lim]

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
            print(f"      lam={lam:5.2f} (alpha={alpha:.3f}): DPP {status}")

    def _find_closest_lambda(self, lambda_k: float) -> float:
        """Find the closest pre-computed lambda level (in log space)."""
        lambda_clipped = np.clip(lambda_k, self.lambda_levels[0], self.lambda_levels[-1])
        log_lambda = np.log(lambda_clipped)
        log_levels = np.log(self.lambda_levels)
        closest_idx = np.argmin(np.abs(log_levels - log_lambda))
        return self.lambda_levels[closest_idx]

    # ==========================================
    # IBR Solve
    # ==========================================

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
          q1(u2_k)  = -2 * H'Q1 * (r1 - z_free - H*u2_k)
          q2(u1_new) = -2*alpha * H'Q1 * (r2 - z_free - H*u1_new)

        The simulator should use: u_shared = u1 + u2

        Args:
            x0: Current state [position, velocity]
            R1_ref: System reference trajectory (Np, 2)
            R2_ref: Human reference trajectory (Np, 2)
            lambda_k: Authority ratio from safety field
            field_force: Safety field force (for logging)

        Returns:
            (u1_opt, u2_opt): Optimal control inputs [m/s^2]
        """
        start_time = time.perf_counter()

        p = self.params
        Nu = p.Nu

        lambda_used = self._find_closest_lambda(lambda_k)
        alpha = self._pustilnik_alpha(lambda_used)

        self.last_lambda_used = lambda_used
        self.last_alpha = alpha
        self.constraint_stats['lambda_level_usage'][lambda_used] += 1

        r1 = R1_ref.flatten()
        r2 = R2_ref.flatten()
        z_free = self.U @ x0

        # Pre-compute constant parts of q terms (updated each IBR step with u_other)
        HQ1_r1 = self.H.T @ self.Q1_full @ (r1 - z_free)
        HQ1_r2 = self.H.T @ self.Q1_full @ (r2 - z_free)
        HQ1H = self.HQ1H

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
            polish=False,
        )

        ibr_converged = False
        try:
            for ibr_iter in range(p.ibr_max_iter):
                # ---- Player 1: fix u2_k ----
                # q1 = -2 * H'Q1 * (r1 - z_free - H*u2_k) = -2*(HQ1_r1 - HQ1H*u2_k)
                prob1['q1_param'].value = -2.0 * (HQ1_r1 - HQ1H @ u2_k)
                if p.warm_start:
                    prob1['u1_var'].value = prob1['warm_start']
                prob1['problem'].solve(**solver_kwargs)

                if prob1['problem'].status not in ['optimal', 'optimal_inaccurate']:
                    print(f"Player 1 solver: {prob1['problem'].status} (lam={lambda_used}, IBR iter {ibr_iter})")
                    break
                u1_new = prob1['u1_var'].value.copy()

                # ---- Player 2: fix u1_new ----
                # q2 = -2*alpha * H'Q1 * (r2 - z_free - H*u1_new) = -2*alpha*(HQ1_r2 - HQ1H*u1_new)
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

            self._check_constraint_activity(u1_opt, u2_opt)

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

    # ==========================================
    # Diagnostics (unchanged API)
    # ==========================================

    def _check_constraint_activity(self, u1_opt: float, u2_opt: float):
        """Check which constraints are active and update statistics."""
        p = self.params
        tol = 1e-3

        self.constraint_stats['total_solves'] += 1

        if abs(u1_opt - p.u1_min) < tol:
            self.constraint_stats['u1_min_active'] += 1
        if abs(u1_opt - p.u1_max) < tol:
            self.constraint_stats['u1_max_active'] += 1
        if abs(u2_opt - p.u2_min) < tol:
            self.constraint_stats['u2_min_active'] += 1
        if abs(u2_opt - p.u2_max) < tol:
            self.constraint_stats['u2_max_active'] += 1

        du1_max_step = p.du1_max * p.dt
        du2_max_step = p.du2_max * p.dt

        if abs(abs(u1_opt - self.u1_prev) - du1_max_step) < tol:
            self.constraint_stats['u1_rate_active'] += 1
        if abs(abs(u2_opt - self.u2_prev) - du2_max_step) < tol:
            self.constraint_stats['u2_rate_active'] += 1

        self.last_constraint_info = {
            'u1_opt': u1_opt,
            'u2_opt': u2_opt,
            'lambda_used': self.last_lambda_used,
            'alpha': self.last_alpha,
        }

    def get_constraint_summary(self) -> str:
        """Get formatted summary of constraint activity including lambda usage."""
        stats = self.constraint_stats
        total = stats['total_solves']
        p = self.params

        if total == 0:
            return "No solves performed yet."

        lambda_usage = "\n".join([
            f"    lam={lam:5.2f} (alpha={1/(1+lam):.3f}): {stats['lambda_level_usage'][lam]:>6} times ({100*stats['lambda_level_usage'][lam]/total:>5.1f}%)"
            for lam in self.params.lambda_levels
        ])

        summary = f"""
DMPC NASH SOLVER (IBR) - CONSTRAINT SUMMARY
  Total solves: {total}
  LAMBDA LEVEL USAGE:
{lambda_usage}
  INPUT BOUNDS:
    u1 at min ({p.u1_min:>5.1f}): {stats['u1_min_active']:>6} times ({100*stats['u1_min_active']/total:>5.1f}%)
    u1 at max ({p.u1_max:>5.1f}): {stats['u1_max_active']:>6} times ({100*stats['u1_max_active']/total:>5.1f}%)
    u2 at min ({p.u2_min:>5.1f}): {stats['u2_min_active']:>6} times ({100*stats['u2_min_active']/total:>5.1f}%)
    u2 at max ({p.u2_max:>5.1f}): {stats['u2_max_active']:>6} times ({100*stats['u2_max_active']/total:>5.1f}%)
  RATE CONSTRAINTS (Jerk limits):
    u1 rate limited: {stats['u1_rate_active']:>6} times ({100*stats['u1_rate_active']/total:>5.1f}%)
    u2 rate limited: {stats['u2_rate_active']:>6} times ({100*stats['u2_rate_active']/total:>5.1f}%)
"""
        return summary

    def get_full_trajectories(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the full optimal control sequences (not just first step)."""
        return self.u1_full.copy(), self.u2_full.copy()

    def get_solver_stats(self) -> Dict:
        """Get solver statistics for analysis."""
        return {
            'status': self.last_status,
            'solve_time_ms': self.last_solve_time * 1000,
            'ibr_iters': self.last_ibr_iters,
            'costs': self.last_costs,
            'u1_prev': self.u1_prev,
            'u2_prev': self.u2_prev,
            'lambda_used': self.last_lambda_used,
            'alpha': self.last_alpha,
            'lambda_levels': self.lambda_levels,
            'formulation': 'DMPC IBR (Li et al. 2019)',
        }

    def reset(self):
        """Reset solver state."""
        self.u1_prev = 0.0
        self.u2_prev = 0.0
        self.last_status = 'reset'
        self.last_alpha = 0.5
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


# ============================================================================
# UNIT TESTS
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("DMPC Nash MPC Solver (IBR - Li et al. 2019) - Integration Test")
    print("="*70)

    import time

    # Create solver
    params = ConstrainedNashParams(Np=20, Nu=10, dt=0.1)
    solver = ConstrainedLongitudinalNashSolver(params=params)

    # Test 1: Basic solve with different lambda values
    print("\nTest 1: Nash solve with varying lambda")
    x0 = np.array([0.0, 20.0])

    R1_ref = np.zeros((params.Np, 2))
    R2_ref = np.zeros((params.Np, 2))

    for k in range(params.Np):
        t = k * params.dt
        R1_ref[k] = [20.0 * t, 22.0]  # System wants 22 m/s
        R2_ref[k] = [20.0 * t, 25.0]  # Human wants 25 m/s

    test_lambdas = [0.1, 1.0, 5.0, 50.0]
    print(f"\n{'lambda':>8} | {'alpha':>8} | {'u1':>8} | {'u2':>8} | {'IBR':>5} | {'Time':>8}")
    print("-" * 65)

    for lam in test_lambdas:
        solver.reset()
        u1_opt, u2_opt = solver.solve_nash_equilibrium(
            x0=x0, R1_ref=R1_ref, R2_ref=R2_ref, lambda_k=lam
        )
        stats = solver.get_solver_stats()
        alpha = 1.0 / (1.0 + lam)
        print(f"{lam:>8.2f} | {alpha:>8.3f} | {u1_opt:>8.3f} | {u2_opt:>8.3f} | {stats['ibr_iters']:>5} | {stats['solve_time_ms']:>6.2f}ms")

    # Test 2: Verify lambda effect on authority allocation
    print("\nTest 2: Verify lambda effect on authority allocation")
    print("  High lambda (dangerous) -> system dominates (|u1| >> |u2|)")
    print("  Low  lambda (safe)      -> human dominates  (|u2| >= |u1|)")

    solver.reset()
    results = []

    for lam in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
        solver.u1_prev = 0.0
        solver.u2_prev = 0.0
        u1, u2 = solver.solve_nash_equilibrium(x0, R1_ref, R2_ref, lam)
        alpha = 1.0 / (1.0 + lam)
        results.append((lam, alpha, u1, u2, abs(u1 - u2)))

    print(f"\n{'lambda':>8} | {'alpha':>8} | {'u1':>8} | {'u2':>8} | {'|u1-u2|':>8}")
    print("-" * 50)
    for lam, alpha, u1, u2, diff in results:
        print(f"{lam:>8.2f} | {alpha:>8.3f} | {u1:>8.3f} | {u2:>8.3f} | {diff:>8.3f}")

    # Test 3: Performance
    print("\nTest 3: Performance (500 solves with random lambda)")
    solver.reset()

    times = []
    np.random.seed(42)

    for i in range(500):
        x0_test = np.array([i * 0.5, 20.0 + np.random.randn() * 0.2])
        lam_test = np.random.uniform(0.1, 50.0)

        start = time.perf_counter()
        solver.solve_nash_equilibrium(x0_test, R1_ref, R2_ref, lam_test)
        times.append((time.perf_counter() - start) * 1000)

    times = np.array(times)
    print(f"   Mean: {times.mean():.2f} ms")
    print(f"   Max:  {times.max():.2f} ms")
    print(f"   P99:  {np.percentile(times, 99):.2f} ms")

    print(solver.get_constraint_summary())

    print("\n" + "="*70)
    print("DMPC Nash Solver (IBR) Test Complete!")
    print("="*70)
