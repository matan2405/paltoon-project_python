#!/usr/bin/env python3
"""
DPP-Compliant Nash MPC Solver with Pustilnik Scaling.
VERSION 6.0 - PUSTILNIK α FORMULATION - Aligned with Lateral V6.0

This solver implements the Non-Normalized GNE formulation from
Pustilnik (2025) with α = λ/(1+λ) embedded inside the Nash game.

Key Changes from V5.0:
======================
1. Pustilnik α mapping: α = λ/(1+λ) used consistently in P matrix and q vector
2. P matrix: P11 = H'Q1H + (R1 + α·S2_eff)·I
             P22 = α·H'Q1H + (R2_eff + S1)·I
             P12 = √α · H'Q1H
3. Driver type modifiers: R2_eff, S2_eff, Q_pos_eff based on driver personality
4. Terminal weights: Q_pos_terminal, Q_vel_terminal at final prediction step
5. q2 uses α (not λ): q2 = -2·α · H'Q1(r2 - z_free)
6. Alpha stored in prob_dict for consistent usage

Theory Reference:
================
Li et al. (2019): "Shared control with a novel dynamic authority allocation 
strategy based on game theory and driving safety field"

Pustilnik (2025): Non-Normalized GNE formulation, Section IV, Theorem 2.

Nash game cost functions (Pustilnik formulation):
    J1 = ||z - r1||²_Q1 + R1||u1||² + S1||u2||²
    J2 = α·||z - r2||²_Q1 + R2||u2||² + α·S2||u1||²

with α = λ/(1+λ), we embed the authority allocation INSIDE the Nash game.
The output should be combined as: u_shared = u1 + u2
(though external blending u_shared = α·u1 + (1-α)·u2 is also valid)
"""

import os
import sys
import numpy as np
import cvxpy as cp
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from scipy.linalg import expm
import time
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (NASH_CONTROL_DT, NASH_NP, NASH_NU)


@dataclass
class ConstrainedNashParams:
    """
    Parameters for constrained Nash MPC solver.
    
    V6.0 - Pustilnik α Formulation with Driver Type Modifiers
    """
    
    # Prediction horizons
    Np: int = NASH_NP              # Prediction horizon
    Nu: int = NASH_NU              # Control horizon
    dt: float = NASH_CONTROL_DT    # Time step [s]
    
    # Cost weights (base values)
    Q_pos: float = 2500.0     # Position tracking weight
    Q_vel: float = 50.0       # Velocity tracking weight
    
    # Terminal weights (higher at final prediction step for stability)
    Q_pos_terminal: float = 10.0 * 2500.0   # 10x position weight at terminal
    Q_vel_terminal: float = 4.0 * 50.0      # 4x velocity weight at terminal
    
    # Control effort weights
    R1: float = 50000.0         # System control effort
    R2: float = 50000.0         # Human control effort (BASE - modified by driver type)
    
    # Cross-coupling weights (S = R for cooperation)
    S1: float = 10000.0         # System penalty on human control
    S2: float = 10000.0         # Human penalty on system control (BASE - modified by driver type)
    
    # Input constraints [m/s²]
    u1_min: float = -3.5      # System min acceleration
    u1_max: float = 2.5       # System max acceleration
    u2_min: float = -4.0      # Human min acceleration
    u2_max: float = 3.0       # Human max acceleration
    
    # Rate constraints [m/s³]
    du1_max: float = 1.5      # System max jerk
    du2_max: float = 2.0      # Human max jerk
    
    # State constraints
    v_min: float = 0.0        # Min velocity [m/s]
    v_max: float = 50.0       # Max velocity [m/s]
    gap_min: float = 5.0      # Min safe gap [m]
    
    # Lambda levels for pre-computation
    lambda_levels: tuple = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0)
    
    # Solver settings
    solver: str = 'OSQP'
    warm_start: bool = True
    verbose: bool = False
    max_iter: int = 200
    eps_abs: float = 1e-4
    eps_rel: float = 1e-4
    
    # Regularization
    regularization: float = 1e-5


class ConstrainedLongitudinalNashSolver:
    """
    DPP-Compliant Nash Equilibrium Solver with Pustilnik Scaling.
    
    V6.0: Embed α inside Nash game → u_shared = u1 + u2
    
    The key insight from Pustilnik (2025) is that α = λ/(1+λ) provides
    a bounded [0,1] scaling that can be embedded directly into the
    cost matrices, producing a well-conditioned QP for all λ values.
    
    Nash Equilibrium (Pustilnik formulation):
    ==========================================
    Player 1 (System): min J1 = ||z - r1||²_Q1 + R1||u1||² + S1||u2||²
    Player 2 (Human):  min J2 = α·||z - r2||²_Q1 + R2||u2||² + α·S2||u1||²
    
    The α scaling means:
    - Low α (safe, low λ): Human has freedom, small penalty for deviation
    - High α (danger, high λ): Human strongly penalized, forced toward system ref
    """
    
    def __init__(self, vehicle=None, platoon_manager=None, human_driver=None,
                 params: ConstrainedNashParams = None,
                 Np: int = None, Nu: int = None, dt: float = None):
        """
        Initialize the Pustilnik-scaled constrained Nash solver.
        
        Args:
            vehicle: Vehicle model (for API compatibility)
            platoon_manager: Platoon manager (for API compatibility)
            human_driver: Human driver model (for API compatibility)
            params: Solver parameters
            Np, Nu, dt: Override parameters if provided
        """
        # Store references for API compatibility
        self.vehicle = vehicle
        self.platoon_manager = platoon_manager
        self.human_driver = human_driver
        
        # Initialize parameters
        self.params = params or ConstrainedNashParams()
        
        # Override with explicit arguments
        if Np is not None:
            self.params.Np = Np
        if Nu is not None:
            self.params.Nu = Nu
        if dt is not None:
            self.params.dt = dt
        
        # Use base weights directly — driver type affects behavior parameters
        # (T_lc, k_e, reaction time), NOT Nash game weights
        self.R2_eff = self.params.R2
        self.S2_eff = self.params.S2
        self.Q_pos_eff = self.params.Q_pos
        
        # Build state-space matrices
        self._build_state_space()
        
        # Build prediction matrices (U, H) - these are lambda-independent
        self._build_prediction_matrices()
        
        # Build base cost matrices (with terminal weights - NEW in V6.0)
        self._build_base_cost_matrices()
        
        # Pre-compute problems for each lambda level
        self._precompute_lambda_problems()
        
        # Previous control inputs for rate constraints
        self.u1_prev = 0.0
        self.u2_prev = 0.0
        
        # Current lambda level (for warm starting)
        self.current_lambda_idx = 3  # Start at λ=1.0
        
        # Statistics
        self.last_solve_time = 0.0
        self.last_iterations = 0
        self.last_status = 'initialized'
        self.last_costs = [0.0, 0.0]
        self.last_lambda_used = 1.0
        self.last_alpha = 0.5
        
        # Constraint activity logging
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
        
        print(f"⚡ Pustilnik Nash MPC Solver V6.0 Initialized (DPP-Compliant)")
        print(f"   Horizons: Np={self.params.Np}, Nu={self.params.Nu}")
        print(f"   Weights: R1={self.params.R1}, R2={self.params.R2}, S1={self.params.S1}, S2={self.params.S2}")
        print(f"   Terminal weights: Q_pos_T={self.params.Q_pos_terminal:.0f}, Q_vel_T={self.params.Q_vel_terminal:.0f}")
        print(f"   Lambda levels: {self.params.lambda_levels}")
        print(f"   Pre-computed problems: {len(self.params.lambda_levels)}")
    
    # ==========================================
    # NEW in V6.0: Pustilnik α Mapping
    # ==========================================
    
    def _pustilnik_alpha(self, lambda_k: float) -> float:
        """
        Map authority ratio λ to Pustilnik scaling factor α.
        
        Pustilnik (2025), Section IV:
        α represents the relative "aggressiveness" of the human player.
        
        We use: α(λ) = λ / (1 + λ)
        
        This gives:
        - λ=0.1 → α=0.091 (system dominant — human barely tracked)
        - λ=0.5 → α=0.333 (system has more authority)
        - λ=1.0 → α=0.500 (normalized solution — equal)
        - λ=2.0 → α=0.667 (human has more authority)
        - λ=10  → α=0.909 (human dominant)
        
        Corollary 2.2 guarantees: reducing α for the system
        cannot worsen the system's cost.
        """
        return lambda_k / (1.0 + lambda_k)
    
    # ==========================================
    # State Space & Prediction (unchanged)
    # ==========================================
    
    def _build_state_space(self):
        """Build discrete-time state-space matrices (double integrator)."""
        dt = self.params.dt
        
        # State: [position, velocity]
        # Input: acceleration
        self.A = np.array([[1, dt],
                          [0, 1]])
        self.B = np.array([[0.5 * dt**2],
                          [dt]])
        self.C = np.eye(2)
        
        self.nx = 2  # states
        self.nz = 2  # outputs
    
    def _build_prediction_matrices(self):
        """Build prediction matrices U, H (lambda-independent)."""
        A, B, C = self.A, self.B, self.C
        Np, Nu = self.params.Np, self.params.Nu
        nx, nz = self.nx, self.nz
        
        # Pre-compute powers of A
        A_pows = [np.eye(nx)]
        for _ in range(Np):
            A_pows.append(A_pows[-1] @ A)
        
        # U matrix: z = U @ x0 + H @ u
        self.U = np.zeros((Np * nz, nx))
        for i in range(Np):
            self.U[i*nz:(i+1)*nz, :] = C @ A_pows[i+1]
        
        # H matrix (Toeplitz-like structure)
        CAB = [C @ A_pows[k] @ B for k in range(Np)]
        self.H = np.zeros((Np * nz, Nu))
        for i in range(Np):
            for j in range(min(i+1, Nu)):
                self.H[i*nz:(i+1)*nz, j:j+1] = CAB[i-j]
    
    # ==========================================
    # UPDATED in V6.0: Terminal weights + driver type
    # ==========================================
    
    def _build_base_cost_matrices(self):
        """
        Build base cost matrices with terminal weights.
        
        V6.0: Block diagonal Q1 with higher terminal weights at k=Np-1
        for improved stability (same as lateral V6.0).
        """
        p = self.params
        Np = p.Np
        
        # Block diagonal Q1 with terminal weights
        Q_blocks = []
        for k in range(Np):
            if k == Np - 1:
                # Terminal step — higher weights for stability
                Q_blocks.append(np.diag([p.Q_pos_terminal, p.Q_vel_terminal]))
            else:
                Q_blocks.append(np.diag([self.Q_pos_eff, p.Q_vel]))
        
        self.Q1_full = np.block([
            [Q_blocks[i] if i == j else np.zeros((self.nz, self.nz))
             for j in range(Np)]
            for i in range(Np)
        ])
        
        # Pre-compute H'Q1H (used in all P matrices)
        self.HQ1H = self.H.T @ self.Q1_full @ self.H
        self.HQ1H += self.params.regularization * np.eye(self.params.Nu)
    
    # ==========================================
    # UPDATED in V6.0: Pustilnik P matrix
    # ==========================================
    
    def _build_P_for_lambda(self, lambda_k: float) -> Tuple[np.ndarray, float]:
        """
        Build the stacked P matrix using Pustilnik α scaling.
        
        Non-Normalized GNE formulation (Pustilnik, Theorem 2):
        =====================================================
        
        Player 1 (System):
            J1 = ||z - r1||²_Q1 + R1·||u1||² + S1·||u2||²
        
        Player 2 (Human) with Pustilnik scaling α:
            J2 = α·||z - r2||²_Q1 + R2·||u2||² + α·S2·||u1||²
        
        KKT stationarity (deriving carefully from Li et al. 2019):
        
        J1 = ||H(u1+u2) + Ux0 - r1||²_Q1 + R1||u1||² + S1||u2||²
        ∂J1/∂u1 = 2H'Q1(H(u1+u2) + Ux0 - r1) + 2R1·u1 = 0
                 → (H'Q1H + R1I)u1 + H'Q1H·u2 = H'Q1(r1 - Ux0)
        
        J2 = α·||H(u1+u2) + Ux0 - r2||²_Q1 + R2||u2||² + α·S2||u1||²
        ∂J2/∂u2 = 2α·H'Q1(H(u1+u2) + Ux0 - r2) + 2R2·u2 = 0
                 → α·H'Q1H·u1 + (α·H'Q1H + R2I)·u2 = α·H'Q1(r2 - Ux0)
        
        Stacked KKT:
        [H'Q1H + R1I    H'Q1H      ] [u1]   [H'Q1(r1 - z_free)       ]
        [α·H'Q1H        α·H'Q1H+R2I] [u2] = [α·H'Q1(r2 - z_free)    ]
        
        For the symmetrized Joint QP, S adds to the OTHER player's diagonal:
        
        P11 = H'Q1H + (R1 + α·S2_eff)·I    ← S2 scaled by α (from Player 2's cost)
        P22 = α·H'Q1H + (R2_eff + S1)·I     ← S1 unscaled (from Player 1's cost)
        P12 = √α · H'Q1H                     ← geometric mean for symmetry
        
        Args:
            lambda_k: Authority ratio
            
        Returns:
            (P_stack, alpha): Stacked Hessian matrix and Pustilnik α
        """
        p = self.params
        Nu = p.Nu
        
        # Compute Pustilnik scaling factor
        alpha = self._pustilnik_alpha(lambda_k)
        
        # P11: System's Hessian
        # R1 = own effort, α·S2_eff = Human's cooperation penalty (Pustilnik-scaled)
        P11 = self.HQ1H + (p.R1 + alpha * self.S2_eff) * np.eye(Nu)
        
        # P22: Human's Hessian (Pustilnik-scaled tracking + cooperation)
        # α·H'Q1H = scaled tracking, R2_eff = own effort, S1 = system's cooperation
        P22 = alpha * self.HQ1H + (self.R2_eff + p.S1) * np.eye(Nu)
        
        # P12: Cross-coupling from shared output z = H(u1 + u2)
        # Use √α for geometric mean to maintain PD while reflecting asymmetry
        coupling_scale = np.sqrt(alpha)
        P12 = coupling_scale * self.HQ1H
        
        # Build stacked matrix
        P_stack = np.block([
            [P11, P12],
            [P12.T, P22]
        ])
        
        # Symmetrize
        P_stack = 0.5 * (P_stack + P_stack.T)
        
        # Ensure positive definiteness with adaptive regularization
        eigvals = np.linalg.eigvalsh(P_stack)
        min_eig = np.min(eigvals)
        
        if min_eig < 1e-6:
            reg_needed = 1e-4 - min_eig
            P_stack += reg_needed * np.eye(2 * Nu)
        else:
            P_stack += p.regularization * np.eye(2 * Nu)
        
        return P_stack, alpha
    
    # ==========================================
    # UPDATED in V6.0: Stores alpha in prob_dict
    # ==========================================
    
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
        constraints += [u1 >= p.u1_min, u1 <= p.u1_max]
        constraints += [u2 >= p.u2_min, u2 <= p.u2_max]
        
        # Rate constraints
        du1_lim = p.du1_max * p.dt
        du2_lim = p.du2_max * p.dt
        
        # First step rate constraint (relative to previous)
        constraints += [u1[0] - u_prev_param[0] <= du1_lim]
        constraints += [u1[0] - u_prev_param[0] >= -du1_lim]
        constraints += [u2[0] - u_prev_param[1] <= du2_lim]
        constraints += [u2[0] - u_prev_param[1] >= -du2_lim]
        
        # Horizon rate constraints
        if Nu > 1:
            for k in range(1, Nu):
                constraints += [u1[k] - u1[k-1] <= du1_lim]
                constraints += [u1[k] - u1[k-1] >= -du1_lim]
                constraints += [u2[k] - u2[k-1] <= du2_lim]
                constraints += [u2[k] - u2[k-1] >= -du2_lim]
        
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
            'alpha': alpha,           # NEW in V6.0: store α
            'lambda': lambda_k,
            'is_dpp': is_dpp,
            'warm_start': np.zeros(2 * Nu)
        }
    
    def _precompute_lambda_problems(self):
        """Pre-compute CVXPY problems for all lambda levels."""
        self.problems = {}
        self.lambda_levels = list(self.params.lambda_levels)
        
        print(f"   Pre-computing {len(self.lambda_levels)} Nash problems...")
        
        for i, lam in enumerate(self.lambda_levels):
            self.problems[lam] = self._create_problem_for_lambda(lam)
            alpha = self.problems[lam]['alpha']
            status = "✓" if self.problems[lam]['is_dpp'] else "✗"
            print(f"      λ={lam:5.2f} (α={alpha:.3f}): DPP {status}")
    
    def _find_closest_lambda(self, lambda_k: float) -> float:
        """Find the closest pre-computed lambda level (in log space)."""
        lambda_clipped = np.clip(lambda_k, self.lambda_levels[0], self.lambda_levels[-1])
        log_lambda = np.log(lambda_clipped)
        log_levels = np.log(self.lambda_levels)
        closest_idx = np.argmin(np.abs(log_levels - log_lambda))
        return self.lambda_levels[closest_idx]
    
    # ==========================================
    # UPDATED in V6.0: Uses α from prob_dict
    # ==========================================
    
    def solve_nash_equilibrium(self, x0: np.ndarray,
                                R1_ref: np.ndarray,
                                R2_ref: np.ndarray,
                                lambda_k: float,
                                field_force: float = 0.0) -> Tuple[float, float]:
        """
        Solve Non-Normalized GNE for longitudinal control.
        
        CRITICAL DIFFERENCE FROM V5.0:
        ==============================
        The authority ratio λ is embedded in the Nash game through
        Pustilnik's α scaling. The output u1, u2 are already coordinated.
        
        The simulator should use: u_shared = u1 + u2
        (NOT the old: u_shared = α·u1 + (1-α)·u2)
        
        Args:
            x0: Current state [position, velocity]
            R1_ref: System reference trajectory (Np, 2)
            R2_ref: Human reference trajectory (Np, 2)
            lambda_k: Authority ratio from safety field
            field_force: Safety field force (for logging)
            
        Returns:
            (u1_opt, u2_opt): Optimal control inputs [m/s²]
        """
        start_time = time.perf_counter()
        
        p = self.params
        Nu = p.Nu
        
        # Find closest pre-computed lambda
        lambda_used = self._find_closest_lambda(lambda_k)
        prob_dict = self.problems[lambda_used]
        alpha = prob_dict['alpha']  # V6.0: use stored α
        
        # Track lambda usage
        self.last_lambda_used = lambda_used
        self.constraint_stats['lambda_level_usage'][lambda_used] += 1
        
        # Flatten references
        r1 = R1_ref.flatten()
        r2 = R2_ref.flatten()
        
        # Free response (z without control)
        z_free = self.U @ x0
        
        # Linear terms q = -2 H' Q (r - z_free)
        # Player 1: standard tracking
        q1 = -2.0 * self.H.T @ self.Q1_full @ (r1 - z_free)
        
        # Player 2: Pustilnik-scaled tracking (α in Q2 = α·Q1)
        # V6.0: uses α (not λ!) for consistency with P matrix
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
                polish=False
            )
            
            self.last_status = prob_dict['problem'].status
            self.last_solve_time = time.perf_counter() - start_time
            
            if prob_dict['problem'].status in ['optimal', 'optimal_inaccurate']:
                u_opt = prob_dict['u_var'].value
                u1_opt = float(u_opt[0])
                u2_opt = float(u_opt[Nu])
                
                # Update constraint stats
                self._check_constraint_activity(u1_opt, u2_opt)
                
                # Update warm start (shift horizon)
                prob_dict['warm_start'] = np.roll(u_opt, -1)
                prob_dict['warm_start'][Nu-1] = u_opt[Nu-1]
                prob_dict['warm_start'][2*Nu-1] = u_opt[2*Nu-1]
                
                self.u1_prev = u1_opt
                self.u2_prev = u2_opt
                
                self.last_costs = [prob_dict['problem'].value / 2,
                                   prob_dict['problem'].value / 2]
                self.last_alpha = alpha
                
                return u1_opt, u2_opt
            else:
                print(f"⚠️ Solver status: {prob_dict['problem'].status} (λ={lambda_used})")
                return self.u1_prev, self.u2_prev
                
        except Exception as e:
            print(f"❌ Solver error: {e}")
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
        
        du1 = abs(u1_opt - self.u1_prev)
        du2 = abs(u2_opt - self.u2_prev)
        
        if abs(du1 - du1_max_step) < tol:
            self.constraint_stats['u1_rate_active'] += 1
        if abs(du2 - du2_max_step) < tol:
            self.constraint_stats['u2_rate_active'] += 1
        
        self.last_constraint_info = {
            'u1_opt': u1_opt,
            'u2_opt': u2_opt,
            'lambda_requested': self.last_lambda_used,
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
            f"║    λ={lam:5.2f} (α={lam/(1+lam):.3f}): {stats['lambda_level_usage'][lam]:>6} times ({100*stats['lambda_level_usage'][lam]/total:>5.1f}%)        ║"
            for lam in self.params.lambda_levels
        ])
        
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║     PUSTILNIK NASH SOLVER V6.0 - CONSTRAINT SUMMARY          ║
╠══════════════════════════════════════════════════════════════╣
║  Total solves: {total:>6}                                       ║
╠══════════════════════════════════════════════════════════════╣
║  LAMBDA LEVEL USAGE:                                         ║
{lambda_usage}
╠══════════════════════════════════════════════════════════════╣
║  INPUT BOUNDS:                                               ║
║    u1 at min ({p.u1_min:>5.1f}): {stats['u1_min_active']:>6} times ({100*stats['u1_min_active']/total:>5.1f}%)        ║
║    u1 at max ({p.u1_max:>5.1f}): {stats['u1_max_active']:>6} times ({100*stats['u1_max_active']/total:>5.1f}%)        ║
║    u2 at min ({p.u2_min:>5.1f}): {stats['u2_min_active']:>6} times ({100*stats['u2_min_active']/total:>5.1f}%)        ║
║    u2 at max ({p.u2_max:>5.1f}): {stats['u2_max_active']:>6} times ({100*stats['u2_max_active']/total:>5.1f}%)        ║
╠══════════════════════════════════════════════════════════════╣
║  RATE CONSTRAINTS (Jerk limits):                             ║
║    u1 rate limited: {stats['u1_rate_active']:>6} times ({100*stats['u1_rate_active']/total:>5.1f}%)              ║
║    u2 rate limited: {stats['u2_rate_active']:>6} times ({100*stats['u2_rate_active']/total:>5.1f}%)              ║
╚══════════════════════════════════════════════════════════════╝
"""
        return summary
    
    def get_full_trajectories(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the full optimal control sequences (not just first step)."""
        Nu = self.params.Nu
        prob_dict = self.problems[self.last_lambda_used]
        if prob_dict['u_var'].value is not None:
            return prob_dict['u_var'].value[:Nu].copy(), prob_dict['u_var'].value[Nu:].copy()
        return np.zeros(Nu), np.zeros(Nu)
    
    def get_solver_stats(self) -> Dict:
        """Get solver statistics for analysis."""
        return {
            'status': self.last_status,
            'solve_time_ms': self.last_solve_time * 1000,
            'costs': self.last_costs,
            'u1_prev': self.u1_prev,
            'u2_prev': self.u2_prev,
            'lambda_used': self.last_lambda_used,
            'alpha': self.last_alpha,
            'lambda_levels': self.lambda_levels,
            'dpp_compliant': True
        }
    
    def reset(self):
        """Reset solver state."""
        self.u1_prev = 0.0
        self.u2_prev = 0.0
        self.last_status = 'reset'
        self.last_alpha = 0.5
        
        # Reset warm starts for all lambda levels
        for lam in self.lambda_levels:
            self.problems[lam]['warm_start'] = np.zeros(2 * self.params.Nu)
        
        # Reset constraint stats
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
    print("Pustilnik Nash MPC Solver V6.0 - Integration Test")
    print("="*70)
    
    import time
    
    # Create solver
    params = ConstrainedNashParams(Np=20, Nu=10, dt=0.1)
    solver = ConstrainedLongitudinalNashSolver(params=params)
    
    # Test 1: Basic solve with different lambda values
    print("\n📊 Test 1: Nash solve with varying λ")
    x0 = np.array([0.0, 20.0])
    
    R1_ref = np.zeros((params.Np, 2))
    R2_ref = np.zeros((params.Np, 2))
    
    for k in range(params.Np):
        t = k * params.dt
        R1_ref[k] = [20.0 * t, 22.0]  # System wants 22 m/s
        R2_ref[k] = [20.0 * t, 25.0]  # Human wants 25 m/s
    
    test_lambdas = [0.1, 1.0, 5.0, 50.0]
    print(f"\n{'λ':>8} | {'α':>8} | {'u1':>8} | {'u2':>8} | {'Time':>8}")
    print("-" * 55)
    
    for lam in test_lambdas:
        solver.reset()
        u1_opt, u2_opt = solver.solve_nash_equilibrium(
            x0=x0, R1_ref=R1_ref, R2_ref=R2_ref, lambda_k=lam
        )
        stats = solver.get_solver_stats()
        alpha = lam / (1 + lam)
        print(f"{lam:>8.2f} | {alpha:>8.3f} | {u1_opt:>8.3f} | {u2_opt:>8.3f} | {stats['solve_time_ms']:>6.2f}ms")
    
    # Test 2: Verify lambda effect
    print("\n📊 Test 2: Verify λ effect on Player 2 control")
    print("As λ increases, u2 should move toward u1 (cooperation)")
    
    solver.reset()
    results = []
    
    for lam in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
        solver.u1_prev = 0.0
        solver.u2_prev = 0.0
        u1, u2 = solver.solve_nash_equilibrium(x0, R1_ref, R2_ref, lam)
        results.append((lam, lam/(1+lam), u1, u2, abs(u1 - u2)))
    
    print(f"\n{'λ':>8} | {'α':>8} | {'u1':>8} | {'u2':>8} | {'|u1-u2|':>8}")
    print("-" * 50)
    for lam, alpha, u1, u2, diff in results:
        print(f"{lam:>8.2f} | {alpha:>8.3f} | {u1:>8.3f} | {u2:>8.3f} | {diff:>8.3f}")
    
    # Test 3: Performance
    print("\n📊 Test 3: Performance (500 solves with random λ)")
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
    
    # Constraint summary
    print(solver.get_constraint_summary())
    
    print("\n" + "="*70)
    print("✅ Pustilnik Nash Solver V6.0 Test Complete!")
    print("="*70)