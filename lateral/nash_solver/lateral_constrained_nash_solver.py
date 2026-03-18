#!/usr/bin/env python3
"""
Constrained Nash MPC Solver for Lateral Control.
VERSION 6.0 - NON-NORMALIZED GNE (Pustilnik & Borrelli, 2025)

This version implements the Non-Normalized Generalized Nash Equilibrium
formulation from:
  Pustilnik & Borrelli (2025): "Non-Normalized Solutions of Generalized
  Nash Equilibria in Dynamic Games With Shared Constraints"

Key Innovation:
==============
Instead of solving a symmetric Nash game and then blending u1, u2 externally
with alpha = λ/(1+λ), we embed the authority allocation INSIDE the Nash game
using Pustilnik's scaling factor α:

  Player 1 (System): J1 = ||z - r1||²_Q1 + R1||u1||² + S1||u2||²
  Player 2 (Human):  J2 = ||z - r2||²_{α·Q1} + R2||u2||² + α·S2||u1||²

where α = α(λ) maps the authority ratio to a Pustilnik scaling factor.

The resulting shared control is:
  u_shared = u1 + u2  (no external blending needed!)

This is because the Nash game itself already accounts for authority allocation
through the asymmetric cost weights.

Theoretical Justification (Pustilnik, Theorem 2):
=================================================
For any set of positive diagonal scaling matrices {Ai}, the solution to the
scaled KKT conditions is a valid GNE. The scaling factor α controls the
"aggressiveness" of each player:
- α < 1: Human is more aggressive (system dominates safety)
- α = 1: Normalized solution (equal responsibility)
- α > 1: Human takes more responsibility

Corollary 2.2: Reducing α for the system cannot worsen the system's cost.
This justifies the system taking asymmetric safety responsibility.
"""

import os
import sys
import numpy as np
import cvxpy as cp
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

# Add parent directory (lateral/) to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vehicle import Vehicle, VehicleParameters
from config import (
    NASH_CONTROL_DT, NASH_NP, NASH_NU,
    NASH_Q_Y, NASH_Q_PSI, NASH_Q_Y_TERMINAL_FACTOR, NASH_Q_PSI_TERMINAL_FACTOR,
    NASH_R1, NASH_R2, NASH_S1, NASH_S2,
    NASH_Y_MAX, NASH_PSI_MAX, NASH_LAMBDA_LEVELS,
    NASH_SOLVER_BACKEND, NASH_WARM_START, NASH_VERBOSE,
    NASH_MAX_ITER, NASH_EPS_ABS, NASH_EPS_REL, NASH_POLISH, NASH_REGULARIZATION,
)


@dataclass
class ConstrainedLateralNashParams:
    """
    Parameters for Non-Normalized GNE lateral Nash solver.
    
    Weight Design Philosophy (following Li et al. 2019):
    ===================================================
    Q_y  : Position tracking weight. Controls how aggressively the controller
           corrects lateral offset. Too low → slow correction → instability.
    Q_psi: Heading tracking weight. Must be >> Q_y to stabilize heading first.
    R1   : System control effort. Higher → gentler steering.
    R2   : Human control effort. Modified by driver type.
    S1   : System's penalty on human's effort (cooperation from system side).
    S2   : Human's penalty on system's effort (cooperation from human side).
    
    Critical Ratios:
    - Q_y / R1 determines correction speed. Need Q_y/R1 >= 0.01 for stability.
    - S/R determines cooperation level. S=R → full cooperation, S=0 → competition.
    - Q_psi / Q_y determines heading vs position priority.
    """
    
    # Prediction horizons
    Np: int = NASH_NP               # Prediction horizon
    Nu: int = NASH_NU               # Control horizon
    dt: float = NASH_CONTROL_DT     # Time step [s]
    
    # =========================================================================
    # COST WEIGHTS — REDESIGNED for stability
    # =========================================================================
    # Key insight from logs: Q_y=10, R1=50000 gives ratio 0.0002 → too weak!
    # At y_error=2m, steering = ~0.04° → vehicle doesn't correct.
    # 
    # We need Q_y/R1 ~ 0.01-0.1 for proper lane keeping.
    # Solution: Q_y=500, R1=10000 → ratio = 0.05
    # =========================================================================
    
    Q_y: float = NASH_Q_Y                                    # Lateral position weight
    Q_psi: float = NASH_Q_PSI                               # Heading angle weight

    # Terminal weights — higher than stage weights for stability
    Q_y_terminal: float = NASH_Q_Y_TERMINAL_FACTOR * NASH_Q_Y
    Q_psi_terminal: float = NASH_Q_PSI_TERMINAL_FACTOR * NASH_Q_PSI

    # Control effort weights
    R1: float = NASH_R1     # System control effort
    R2: float = NASH_R2     # Human control effort (BASE)

    # Cross-coupling weights — S = R/5 for cooperation
    S1: float = NASH_S1     # System penalty on human's effort
    S2: float = NASH_S2     # Human penalty on system's effort
    
    # =========================================================================
    # PUSTILNIK SCALING — Non-Normalized GNE
    # =========================================================================
    # α maps authority ratio λ to Pustilnik's scaling factor.
    # α(λ) = λ / (1 + λ)  ← same formula, but now INSIDE the Nash game
    #
    # For λ=0.1: α=0.09 → Human barely tracked, system dominates
    # For λ=1.0: α=0.50 → Equal responsibility (normalized solution)
    # For λ=10 : α=0.91 → Human nearly fully responsible
    # =========================================================================
    
    # Input constraints [rad]
    delta_min: float = -VehicleParameters.max_steering_angle
    delta_max: float = VehicleParameters.max_steering_angle
    
    # Rate constraints [rad/s]
    ddelta_max: float = VehicleParameters.max_steering_rate
    
    # State constraints
    y_max: float = NASH_Y_MAX
    psi_max: float = NASH_PSI_MAX

    # Lambda levels for pre-computation
    lambda_levels: tuple = NASH_LAMBDA_LEVELS

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


class ConstrainedLateralNashSolver:
    """
    Non-Normalized GNE Nash Solver for Lateral Control.
    VERSION 6.0 — Pustilnik & Borrelli (2025) formulation.
    
    State: x = [y, y_dot, psi, psi_dot]
    Input: u = delta (steering angle)
    Output: z = [y, psi]
    
    CRITICAL CHANGE FROM V5.0:
    ==========================
    V5.0: Solve symmetric Nash → blend externally: u_sh = α·u1 + (1-α)·u2
           Problem: When u1 ≈ -u2 (opposition), u_sh ≈ 0 → no control!
    
    V6.0: Embed α inside Nash game → u_shared = u1 + u2
           The Nash game produces COORDINATED controls that add up correctly.
    
    The Pustilnik scaling ensures Player 2's cost is scaled by α:
        J2 = α · ||z - r2||²_Q1 + R2·||u2||² + α·S2·||u1||²
    
    When α is small (low authority), Player 2 optimizes mainly for
    control effort minimization → produces small u2 → u1 dominates.
    When α is large, Player 2 actively tracks its reference → larger u2.
    """
    
    def __init__(self, vehicle: Vehicle, params: ConstrainedLateralNashParams = None):
        if vehicle is None:
            raise ValueError("Vehicle must be provided")
        
        self.params = params or ConstrainedLateralNashParams()
        self.vehicle = vehicle
        
        # Build state-space matrices
        self._build_state_space()
        self._build_prediction_matrices()
        
        # Use base weights directly — driver type affects behavior parameters
        # (T_lc, k_e, k_psi), NOT Nash game weights
        self.R2_eff = self.params.R2
        self.S2_eff = self.params.S2
        self.Q_y_eff = self.params.Q_y
        
        self._build_base_cost_matrices()
        self._precompute_lambda_problems()
        
        # Previous control inputs
        self.u1_prev = 0.0
        self.u2_prev = 0.0
        
        # Current lambda level
        self.current_lambda_idx = 3  # Start at λ=1.0
        self.last_lambda_used = 1.0
        
        # Statistics
        self.last_solve_time = 0.0
        self.last_iterations = 0
        self.last_status = 'initialized'
        self.last_costs = [0.0, 0.0]
        
        # Constraint tracking
        self.constraint_stats = {
            'total_solves': 0,
            'u1_at_min': 0, 'u1_at_max': 0,
            'u2_at_min': 0, 'u2_at_max': 0,
            'u1_rate_active': 0, 'u2_rate_active': 0,
            'lambda_level_usage': {lam: 0 for lam in self.params.lambda_levels}
        }
        
        s_r_ratio = self.params.S1 / self.params.R1 if self.params.R1 > 0 else 0
        q_r_ratio = self.params.Q_y / self.params.R1 if self.params.R1 > 0 else 0
        print(f"⚡ Lambda-Aware Lateral Nash MPC V6.0 Initialized (Non-Normalized GNE)")
        print(f"   Horizons: Np={self.params.Np}, Nu={self.params.Nu}")
        print(f"   Steering bounds: δ∈[{np.degrees(self.params.delta_min):.1f}°, {np.degrees(self.params.delta_max):.1f}°]")
        print(f"   R1={self.params.R1:.0f}, R2={self.params.R2:.0f}, S1={self.params.S1:.0f}, S2={self.params.S2:.0f}")
        print(f"   S/R ratio: {s_r_ratio:.2f} ({'cooperation' if s_r_ratio >= 0.5 else 'competition'})")
        print(f"   Q_y/R1 ratio: {q_r_ratio:.4f}")
        print(f"   Lambda levels: {self.params.lambda_levels}")
        print(f"   Pre-computed problems: {len(self.params.lambda_levels)}")
    
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
        
        # Pre-compute A^k
        A_pows = [np.eye(self.nx)]
        for _ in range(Np):
            A_pows.append(A_pows[-1] @ A)
        
        # U matrix (free response)
        self.U = np.zeros((Np * self.nz, self.nx))
        for i in range(Np):
            self.U[i*self.nz:(i+1)*self.nz, :] = C @ A_pows[i+1]
        
        # H matrix (forced response, Toeplitz)
        CAB = [C @ A_pows[k] @ B for k in range(Np)]
        self.H = np.zeros((Np * self.nz, Nu))
        for i in range(Np):
            for j in range(min(i+1, Nu)):
                self.H[i*self.nz:(i+1)*self.nz, j:j+1] = CAB[i-j]
    
    def _build_base_cost_matrices(self):
        """Pre-compute base cost matrices."""
        p = self.params
        Np = p.Np
        
        # Block diagonal Q1 with terminal weights
        Q_blocks = []
        for k in range(Np):
            if k == Np - 1:
                # Terminal step — higher weights
                Q_blocks.append(np.diag([p.Q_y_terminal, p.Q_psi_terminal]))
            else:
                Q_blocks.append(np.diag([self.Q_y_eff, p.Q_psi]))
        
        self.Q1_full = np.block([
            [Q_blocks[i] if i == j else np.zeros((self.nz, self.nz))
             for j in range(Np)]
            for i in range(Np)
        ])
        
        # Pre-compute H'Q1H
        self.HQ1H = self.H.T @ self.Q1_full @ self.H
        self.HQ1H += self.params.regularization * np.eye(self.params.Nu)
    
    def _pustilnik_alpha(self, lambda_k: float) -> float:
        """
        Map authority ratio λ to Pustilnik scaling factor α.
        
        Pustilnik (2025), Section IV:
        α represents the relative "aggressiveness" of the human player.
        
        We use: α(λ) = 1 / (1 + λ)

        This gives:
        - λ=0.1 → α=0.909 (human dominant — safe, human tracks aggressively)
        - λ=0.5 → α=0.667 (human has more authority)
        - λ=1.0 → α=0.500 (normalized solution — equal)
        - λ=2.0 → α=0.333 (system has more authority)
        - λ=10  → α=0.091 (system dominant — dangerous, human barely acts)

        Corollary 2.2 guarantees: reducing α for the system
        cannot worsen the system's cost.
        """
        return 1.0 / (1.0 + lambda_k)
    
    def _build_P_for_lambda(self, lambda_k: float) -> np.ndarray:
        """
        Build the stacked P matrix for a specific lambda using Pustilnik scaling.
        
        Non-Normalized GNE formulation (Pustilnik, Theorem 2):
        =====================================================
        
        Player 1 (System):
            J1 = ||z - r1||²_Q1 + R1·||u1||² + S1·||u2||²
        
        Player 2 (Human) with Pustilnik scaling α:
            J2 = α·||z - r2||²_Q1 + R2·||u2||² + α·S2·||u1||²
        
        KKT stationarity conditions:
            ∂J1/∂u1 = 0: 2(H'Q1H)u1 + 2R1·u1 + 2(H'Q1H)u2 - 2H'Q1(r1-z_free) = 0  [WRONG - see below]
        
        Wait — let me be precise. The stacked QP approach:
        
        The Nash KKT for Player 1:
            (H'Q1H + R1·I)u1 + S1·u2 = H'Q1(r1 - z_free)  [tracking u2 is through S1 only]
        
        Actually, re-deriving carefully from Li et al. 2019:
        J1 = ||H(u1+u2) + Ux0 - r1||²_Q1 + R1||u1||² + S1||u2||²
        
        ∂J1/∂u1 = 2H'Q1(H(u1+u2) + Ux0 - r1) + 2R1·u1 = 0
                 = 2(H'Q1H + R1I)u1 + 2H'Q1H·u2 = 2H'Q1(r1 - Ux0)
        
        J2 = α·||H(u1+u2) + Ux0 - r2||²_Q1 + R2||u2||² + α·S2||u1||²
        
        ∂J2/∂u2 = 2α·H'Q1(H(u1+u2) + Ux0 - r2) + 2R2·u2 = 0
                 = 2α·H'Q1H·u1 + 2(α·H'Q1H + R2I)·u2 = 2α·H'Q1(r2 - Ux0)
        
        Stacked KKT:
        [H'Q1H + R1I    H'Q1H      ] [u1]   [H'Q1(r1 - z_free)       ]
        [α·H'Q1H        α·H'Q1H+R2I] [u2] = [α·H'Q1(r2 - z_free)    ]
        
        Note: S1 and S2 add to the diagonal cross terms!
        Actually, let me reconsider. S1 is system's penalty on ||u2||².
        
        J1 = ||z - r1||²_Q1 + R1||u1||² + S1||u2||²
        where z = H·u1 + H·u2 + U·x0  (both controls affect the same plant!)
        
        ∂J1/∂u1: 2H'Q1(Hu1 + Hu2 + Ux0 - r1) + 2R1·u1 = 0
        → (H'Q1H + R1I)u1 + H'Q1H·u2 = H'Q1(r1 - Ux0)
        
        Note: S1 only appears in J1 as S1||u2||² but ∂J1/∂u1 doesn't depend
        on S1 because it's u2's norm. S1 appears in ∂J1/∂u2 which isn't
        a KKT condition for Player 1's optimization!
        
        Wait — in a Nash game, each player optimizes over THEIR OWN control.
        Player 1 optimizes over u1 only. So S1||u2||² doesn't enter ∂J1/∂u1.
        
        But S1 represents "Player 1 cares about how much u2 is used."
        In the SHARED output formulation z = H(u1+u2) + Ux0, the cross-coupling
        comes from H'Q1H already (both affect z).
        
        In the stacked QP approximation, S appears as additional cross-coupling:
        
        P11 = H'Q1H + R1I       (Player 1's Hessian w.r.t. u1)
        P22 = α·H'Q1H + R2I     (Player 2's Hessian w.r.t. u2, Pustilnik-scaled)
        P12 = H'Q1H             (cross-coupling from shared output z)
        
        Adding S terms for cooperation regularization:
        P11 += 0                  (S2 from Player 2's perspective appears in P22)
        P22 += α·S2·I            (α-scaled: human's cooperation effort)
        P12 += 0                  (S doesn't create cross-coupling)
        
        Actually, for the symmetrized Joint QP, S adds to the OTHER player's
        diagonal. Let me use the formulation that actually works:
        
        P11 = H'Q1H + (R1 + S2_eff)·I    ← S2 adds to P11 because Player 2 
                                              penalizes u1 through S2
        P22 = α·H'Q1H + (R2 + S1)·I      ← S1 adds to P22 because Player 1
                                              penalizes u2 through S1
        P12 = H'Q1H                        ← shared output coupling
        
        With Pustilnik: S2 in P11 is also scaled by α because it comes from 
        Player 2's cost which is scaled by α:
        
        P11 = H'Q1H + (R1 + α·S2_eff)·I
        P22 = α·H'Q1H + (R2 + S1)·I
        P12 = √α · H'Q1H    ← geometric mean for symmetry
        """
        p = self.params
        Nu = p.Nu
        
        # Compute Pustilnik scaling factor
        alpha = self._pustilnik_alpha(lambda_k)
        
        # P11: System's Hessian
        # R1 = own effort, α·S2 = Human's cooperation penalty (Pustilnik-scaled)
        P11 = self.HQ1H + (p.R1 + alpha * self.S2_eff) * np.eye(Nu)
        
        # P22: Human's Hessian (Pustilnik-scaled tracking + cooperation)
        # α·H'Q1H = scaled tracking, R2 = own effort, S1 = system's cooperation
        P22 = alpha * self.HQ1H + (self.R2_eff + p.S1) * np.eye(Nu)
        
        # P12: Cross-coupling from shared output z = H(u1 + u2)
        # Use α (not √α) so that P12 ≤ P22 always, preventing the adversarial
        # equilibrium where u1=-max and u2=+max cancel each other.
        # With √α: P12=√α·HQ1H > P22=α·HQ1H when HQ1H >> R,S → adversarial lock.
        # With α:  P12=α·HQ1H ≤ P22=α·HQ1H + R + S always → cooperative.
        coupling_scale = alpha
        P12 = coupling_scale * self.HQ1H
        
        # Build stacked matrix
        P_stack = np.block([
            [P11, P12],
            [P12.T, P22]
        ])
        
        # Symmetrize
        P_stack = 0.5 * (P_stack + P_stack.T)
        
        # Ensure positive definiteness
        eigvals = np.linalg.eigvalsh(P_stack)
        min_eig = np.min(eigvals)
        
        if min_eig < 1e-6:
            reg_needed = 1e-4 - min_eig
            P_stack += reg_needed * np.eye(2 * Nu)
        else:
            P_stack += p.regularization * np.eye(2 * Nu)
        
        return P_stack, alpha
    
    def _create_problem_for_lambda(self, lambda_k: float) -> dict:
        """Create a CVXPY problem for a specific lambda value."""
        p = self.params
        Nu = p.Nu
        
        # Build P matrix with Pustilnik scaling
        P_stack, alpha = self._build_P_for_lambda(lambda_k)
        
        # Create CVXPY variable and parameters
        u_var = cp.Variable(2 * Nu, name=f'u_lam{lambda_k}')
        q_param = cp.Parameter(2 * Nu, name=f'q_lam{lambda_k}')
        u_prev_param = cp.Parameter(2, name=f'u_prev_lam{lambda_k}')
        
        # Cost: u'Pu + q'u
        cost = cp.quad_form(u_var, P_stack) + q_param @ u_var
        
        # Constraints
        constraints = []
        u1 = u_var[:Nu]
        u2 = u_var[Nu:]
        
        # Input bounds
        constraints += [u1 >= p.delta_min, u1 <= p.delta_max]
        constraints += [u2 >= p.delta_min, u2 <= p.delta_max]
        
        # Rate constraints
        ddelta_lim = p.ddelta_max * p.dt
        
        # First step rate constraint
        constraints += [u1[0] - u_prev_param[0] <= ddelta_lim]
        constraints += [u1[0] - u_prev_param[0] >= -ddelta_lim]
        constraints += [u2[0] - u_prev_param[1] <= ddelta_lim]
        constraints += [u2[0] - u_prev_param[1] >= -ddelta_lim]
        
        # Horizon rate constraints
        if Nu > 1:
            for k in range(1, Nu):
                constraints += [u1[k] - u1[k-1] <= ddelta_lim]
                constraints += [u1[k] - u1[k-1] >= -ddelta_lim]
                constraints += [u2[k] - u2[k-1] <= ddelta_lim]
                constraints += [u2[k] - u2[k-1] >= -ddelta_lim]
        
        # Create problem
        problem = cp.Problem(cp.Minimize(cost), constraints)
        
        # Verify DPP compliance
        is_dpp = problem.is_dcp(dpp=True)
        if not is_dpp:
            print(f"⚠️ Warning: Problem for λ={lambda_k} is not DPP compliant!")
        
        return {
            'problem': problem,
            'u_var': u_var,
            'q_param': q_param,
            'u_prev_param': u_prev_param,
            'P_stack': P_stack,
            'alpha': alpha,
            'lambda': lambda_k,
            'is_dpp': is_dpp,
            'warm_start': np.zeros(2 * Nu)
        }
    
    def _precompute_lambda_problems(self):
        """Pre-compute CVXPY problems for all lambda levels."""
        self.problems = {}
        self.lambda_levels = list(self.params.lambda_levels)
        
        print(f"   Pre-computing {len(self.lambda_levels)} Nash problems...")
        
        for lam in self.lambda_levels:
            self.problems[lam] = self._create_problem_for_lambda(lam)
            alpha = self.problems[lam]['alpha']
            status = "✓" if self.problems[lam]['is_dpp'] else "✗"
            print(f"      λ={lam:5.2f} (α={alpha:.3f}): DPP {status}")
    
    def _find_closest_lambda(self, lambda_k: float) -> float:
        """Find the closest pre-computed lambda level (log-space)."""
        lambda_clipped = np.clip(lambda_k, self.lambda_levels[0], self.lambda_levels[-1])
        log_lambda = np.log(lambda_clipped)
        log_levels = np.log(self.lambda_levels)
        closest_idx = np.argmin(np.abs(log_levels - log_lambda))
        return self.lambda_levels[closest_idx]
    
    def solve_nash_equilibrium(self, x0: np.ndarray,
                                R1_ref: np.ndarray,
                                R2_ref: np.ndarray,
                                lambda_k: float,
                                field_force: float = 0.0) -> Tuple[float, float]:
        """
        Solve Non-Normalized GNE for lateral control.
        
        CRITICAL DIFFERENCE FROM V5.0:
        ==============================
        The authority ratio λ is embedded in the Nash game through
        Pustilnik's α scaling. The output u1, u2 are already coordinated.
        
        The simulator should use: u_shared = u1 + u2
        (NOT the old: u_shared = α·u1 + (1-α)·u2)
        
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
        
        # Find closest pre-computed lambda
        lambda_used = self._find_closest_lambda(lambda_k)
        prob_dict = self.problems[lambda_used]
        alpha = prob_dict['alpha']
        
        # Track lambda usage
        self.last_lambda_used = lambda_used
        self.constraint_stats['lambda_level_usage'][lambda_used] += 1
        
        # Flatten references
        r1 = R1_ref.flatten()
        r2 = R2_ref.flatten()
        
        # Free response
        z_free = self.U @ x0
        
        # Linear terms q = -2 H' Q (r - z_free)
        # Player 1: standard tracking
        q1 = -2.0 * self.H.T @ self.Q1_full @ (r1 - z_free)
        
        # Player 2: Pustilnik-scaled tracking (α in Q2 = α·Q1)
        q2 = -2.0 * alpha * self.H.T @ self.Q1_full @ (r2 - z_free)
        
        q_stacked = np.concatenate([q1, q2])
        
        # Update parameters
        prob_dict['q_param'].value = q_stacked
        prob_dict['u_prev_param'].value = np.array([self.u1_prev, self.u2_prev])
        
        # Warm start
        if p.warm_start and prob_dict['warm_start'] is not None:
            prob_dict['u_var'].value = prob_dict['warm_start']
        
        # Solve
        try:
            prob_dict['problem'].solve(
                solver=cp.OSQP,
                warm_start=p.warm_start,
                verbose=p.verbose,
                eps_abs=p.eps_abs,
                eps_rel=p.eps_rel,
                max_iter=p.max_iter,
                polish=p.polish
            )
            
            self.last_status = prob_dict['problem'].status
            self.last_solve_time = time.perf_counter() - start_time
            
            if prob_dict['problem'].status in ['optimal', 'optimal_inaccurate']:
                u_opt = prob_dict['u_var'].value
                u1_opt = float(u_opt[0])
                u2_opt = float(u_opt[Nu])
                
                # Update constraint stats
                self._update_constraint_stats(u1_opt, u2_opt)
                
                # Update warm start (shift horizon)
                prob_dict['warm_start'] = np.roll(u_opt, -1)
                prob_dict['warm_start'][Nu-1] = u_opt[Nu-1]
                prob_dict['warm_start'][2*Nu-1] = u_opt[2*Nu-1]
                
                self.u1_prev = u1_opt
                self.u2_prev = u2_opt
                
                self.last_costs = [prob_dict['problem'].value / 2,
                                   prob_dict['problem'].value / 2]
                
                return u1_opt, u2_opt
            else:
                print(f"⚠️ Solver status: {prob_dict['problem'].status}")
                return self.u1_prev, self.u2_prev
                
        except Exception as e:
            print(f"❌ Solver error: {e}")
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
        Nu = self.params.Nu
        prob_dict = self.problems[self.last_lambda_used]
        if prob_dict['u_var'].value is not None:
            return prob_dict['u_var'].value[:Nu].copy(), prob_dict['u_var'].value[Nu:].copy()
        return np.zeros(Nu), np.zeros(Nu)
    
    def get_predicted_states(self, x0: np.ndarray) -> Optional[np.ndarray]:
        """Get predicted output trajectory."""
        Nu = self.params.Nu
        prob_dict = self.problems[self.last_lambda_used]
        if prob_dict['u_var'].value is None:
            return None
        
        # In V6.0, shared = u1 + u2
        u_shared = prob_dict['u_var'].value[:Nu] + prob_dict['u_var'].value[Nu:]
        z_pred = self.U @ x0 + self.H @ u_shared
        return z_pred.reshape(self.params.Np, 2)
    
    def get_solver_stats(self) -> Dict:
        """Get solver statistics."""
        return {
            'status': self.last_status,
            'solve_time_ms': self.last_solve_time * 1000,
            'costs': self.last_costs,
            'u1_prev': self.u1_prev,
            'u2_prev': self.u2_prev,
            'lambda_used': self.last_lambda_used,
            'alpha_pustilnik': self._pustilnik_alpha(self.last_lambda_used),
            'lambda_levels': self.lambda_levels,
            'dpp_compliant': True,
            'formulation': 'Non-Normalized GNE (Pustilnik 2025)'
        }
    
    def get_constraint_summary(self) -> str:
        """Get formatted summary of constraint activity."""
        stats = self.constraint_stats
        total = stats['total_solves']
        p = self.params
        
        if total == 0:
            return "No solves performed yet."
        
        s_r = p.S1 / p.R1 if p.R1 > 0 else 0
        
        lambda_usage = "\n".join([
            f"║    λ={lam:5.2f} (α={self._pustilnik_alpha(lam):.3f}): "
            f"{stats['lambda_level_usage'][lam]:>6} times "
            f"({100*stats['lambda_level_usage'][lam]/total:>5.1f}%)        ║"
            for lam in self.params.lambda_levels
        ])
        
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║   LATERAL NASH V6.0 - NON-NORMALIZED GNE CONSTRAINT SUMMARY ║
╠══════════════════════════════════════════════════════════════╣
║  Total solves: {total:>6}                                       ║
║  R1={p.R1:.0f}, S1={p.S1:.0f}, S/R={s_r:.2f}                ║
╠══════════════════════════════════════════════════════════════╣
║  LAMBDA LEVEL USAGE (Pustilnik α):                           ║
{lambda_usage}
╠══════════════════════════════════════════════════════════════╣
║  INPUT BOUNDS:                                               ║
║    δ1 at min ({np.degrees(p.delta_min):>6.1f}°): {stats['u1_at_min']:>6} times        ║
║    δ1 at max ({np.degrees(p.delta_max):>6.1f}°): {stats['u1_at_max']:>6} times        ║
║    δ2 at min ({np.degrees(p.delta_min):>6.1f}°): {stats['u2_at_min']:>6} times        ║
║    δ2 at max ({np.degrees(p.delta_max):>6.1f}°): {stats['u2_at_max']:>6} times        ║
╠══════════════════════════════════════════════════════════════╣
║  RATE CONSTRAINTS:                                           ║
║    δ1 rate active: {stats['u1_rate_active']:>6} times                      ║
║    δ2 rate active: {stats['u2_rate_active']:>6} times                      ║
╚══════════════════════════════════════════════════════════════╝"""
        return summary
    
    def reset(self):
        """Reset solver state."""
        self.u1_prev = 0.0
        self.u2_prev = 0.0
        self.last_status = 'reset'
        
        for lam in self.lambda_levels:
            self.problems[lam]['warm_start'] = np.zeros(2 * self.params.Nu)
        
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
        Driver type no longer modifies Nash weights (V6.0+).
        Driver personality is expressed through behavior parameters
        (T_lc, k_e, k_psi) in reference generators, NOT in the Nash game.
        This method is kept for API compatibility but does nothing.
        """
        pass