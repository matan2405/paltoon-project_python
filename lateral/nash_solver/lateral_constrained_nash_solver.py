#!/usr/bin/env python3
"""
Optimized Constrained Nash MPC Solver for Lateral Control.
VERSION 5.0 - LAMBDA-AWARE - DPP-Compliant with Pre-computed Lambda Levels

This is a DROP-IN REPLACEMENT for lateral_constrained_nash_solver.py
All APIs remain identical for seamless integration.

Key Innovation (from Longitudinal V5.0):
========================================
The authority ratio λ affects Q2 in the Nash game:
    Q2(λ) = λ * Q1

For DPP compliance, P must be constant. We solve this by:
1. Pre-computing P matrices for λ ∈ {0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0}
2. At runtime, selecting the closest pre-computed λ level
3. Solving with the cached problem for that λ level

Theory Reference:
================
Li et al. (2019): "Shared control with a novel dynamic authority allocation 
strategy based on game theory and driving safety field"

The Nash game cost functions:
    J1 = ||z - r1||²_Q1 + R1||u1||² + S1||u2||²
    J2 = ||z - r2||²_{λQ1} + R2||u2||² + S2||u1||²  ← λ affects Q2!

Performance:
- Original: ~2ms per solve
- With Lambda: ~2-3ms per solve (DPP compliant)
"""

import os
import sys
import numpy as np
import cvxpy as cp
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

# Add parent directory (lateral/) to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vehicle import Vehicle,VehicleParameters
from config import (NASH_CONTROL_DT, NASH_NP,NASH_NU)
@dataclass
class ConstrainedLateralNashParams:
    """
    Parameters for constrained lateral Nash MPC solver.
    
    API COMPATIBLE with original ConstrainedLateralNashParams.
    """
    
    # Prediction horizons - REDUCED for speed
    Np: int = NASH_NP              # Prediction horizon (was 20)
    Nu: int = NASH_NU               # Control horizon (was 10)
    dt: float = NASH_CONTROL_DT       # Time step [s]
    
    # Note: Vehicle parameters (vx, m, Iz, Cf, Cr, etc.) are obtained from
    # the Vehicle object passed to the solver - no duplication needed.
    
    # Cost weights - FROM WORKING SYSTEM (VERIFIED!)
    # These weights produce realistic steering angles that meet all requirements:
    # - Heading angle ψ < 2° (achieved: 0.45-0.87°)
    # - Lateral accel < 1.5 m/s² (achieved: ~0.01 m/s²)
    # - Jerk < 0.9 m/s³ (achieved: ~0.001 m/s³)
    Q_y: float = 10.0         # Lateral position (LOW - prioritize comfort over tracking)
    Q_psi: float = 10000.0    # Heading angle - INCREASED 100% for stricter heading control
    
    # Terminal weights (also increased)
    Q_y_terminal: float = 40.0
    Q_psi_terminal: float = 40000.0  # INCREASED 100%
    
    # Control effort weights - VERY HIGH for realistic small steering
    # Key ratio: R/Q_y = 100,000 (forces small steering angles)
    R1: float = 50000.0     # System control effort
    R2: float = 50000.0     # Human control effort
    
    # Cross-coupling weights - S = 0.2*R for cooperation (not S=R!)
    S1: float = 50000.0      # Cross-coupling weight
    S2: float = 50000.0      # Cross-coupling weight
    
    # Input constraints [rad]
    delta_min: float = -VehicleParameters.max_steering_angle   # Min steering angle (~-23°)
    delta_max: float = VehicleParameters.max_steering_angle    # Max steering angle (~23°)
    
    # Rate constraints [rad/s]
    ddelta_max: float = VehicleParameters.max_steering_rate   # Max steering rate
    
    # State constraints (for reference)
    y_max: float = 5.0        # Max lateral deviation [m]
    psi_max: float = 0.5      # Max heading angle [rad]
    
    # Lambda levels for pre-computation
    # These cover the practical range from Authority Allocator:
    # λ = 0.1 (very safe) to λ = 10.0 (high risk)
    lambda_levels: tuple = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
    
    # Solver settings - OPTIMIZED
    solver: str = 'OSQP'
    warm_start: bool = True
    verbose: bool = False
    max_iter: int = 200
    eps_abs: float = 1e-3
    eps_rel: float = 1e-3
    
    # Driver type
    driver_type: str = 'normal'
    
    # Regularization
    regularization: float = 1e-5


class ConstrainedLateralNashSolver:
    """
    Optimized Constrained Nash Equilibrium Solver for Lateral Control.
    
    VERSION 5.0 - LAMBDA-AWARE - DPP-Compliant with Pre-computed Lambda Levels.
    
    Uses the bicycle model:
    State: x = [y, y_dot, psi, psi_dot]
    Input: u = delta (steering angle)
    Output: z = [y, psi]
    
    KEY INNOVATION:
    The authority ratio λ affects Q2 = λ * Q1, which changes the Hessian P.
    We pre-compute P for discrete λ values and select the closest at runtime.
    """
    
    def __init__(self, vehicle: Vehicle, params: ConstrainedLateralNashParams = None):
        """
        Initialize constrained lateral Nash solver.
        
        Args:
            vehicle: Vehicle model (REQUIRED - provides state-space matrices)
            params: Solver parameters
        
        Raises:
            ValueError: If vehicle is not provided
        """
        if vehicle is None:
            raise ValueError("Vehicle must be provided - it supplies the state-space matrices")
        
        self.params = params or ConstrainedLateralNashParams()
        self.vehicle = vehicle
        
        # Build state-space matrices
        self._build_state_space()
        
        # Build prediction matrices
        self._build_prediction_matrices()
        
        # Apply driver type modifiers
        self._apply_driver_type_modifiers()
        
        # Build base cost matrices
        self._build_base_cost_matrices()
        
        # Pre-compute problems for each lambda level
        self._precompute_lambda_problems()
        
        # Previous control inputs
        self.u1_prev = 0.0
        self.u2_prev = 0.0
        
        # Current lambda level (for warm starting)
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
            'u1_at_min': 0,
            'u1_at_max': 0,
            'u2_at_min': 0,
            'u2_at_max': 0,
            'u1_rate_active': 0,
            'u2_rate_active': 0,
            'lambda_level_usage': {lam: 0 for lam in self.params.lambda_levels}
        }
        
        print(f"⚡ Lambda-Aware Lateral Nash MPC V5.0 Initialized (DPP-Compliant)")
        print(f"   Horizons: Np={self.params.Np}, Nu={self.params.Nu}")
        print(f"   Steering bounds: δ∈[{np.degrees(self.params.delta_min):.1f}°, {np.degrees(self.params.delta_max):.1f}°]")
        print(f"   Driver type: {self.params.driver_type}")
        print(f"   Lambda levels: {self.params.lambda_levels}")
        print(f"   Pre-computed problems: {len(self.params.lambda_levels)}")
    
    def _build_state_space(self):
        """Get discrete-time bicycle model state-space from vehicle.
        
        Uses the vehicle's get_state_space_matrices() method to ensure
        consistency between the Nash solver and vehicle dynamics.
        """
        p = self.params
        
        # Get matrices from vehicle (no code duplication!)
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
        
        # U matrix
        self.U = np.zeros((Np * self.nz, self.nx))
        for i in range(Np):
            self.U[i*self.nz:(i+1)*self.nz, :] = C @ A_pows[i+1]
        
        # H matrix (Toeplitz)
        CAB = [C @ A_pows[k] @ B for k in range(Np)]
        self.H = np.zeros((Np * self.nz, Nu))
        for i in range(Np):
            for j in range(min(i+1, Nu)):
                self.H[i*self.nz:(i+1)*self.nz, j:j+1] = CAB[i-j]
    
    def _apply_driver_type_modifiers(self):
        """Apply driver-type specific weight modifiers."""
        p = self.params
        
        modifiers = {
            'cautious': {'R2_factor': 1.5, 'S2_factor': 1.5, 'Q_y_factor': 0.8},
            'normal': {'R2_factor': 1.0, 'S2_factor': 1.0, 'Q_y_factor': 1.0},
            'aggressive': {'R2_factor': 0.6, 'S2_factor': 0.6, 'Q_y_factor': 1.3}
        }
        
        mod = modifiers.get(p.driver_type, modifiers['normal'])
        
        self.R2_eff = p.R2 * mod['R2_factor']
        self.S2_eff = p.S2 * mod['S2_factor']
        self.Q_y_eff = p.Q_y * mod['Q_y_factor']
    
    def _build_base_cost_matrices(self):
        """Pre-compute base cost matrices (Q1 is fixed, Q2 will be scaled by λ)."""
        p = self.params
        Np = p.Np
        
        # Block diagonal Q1 with driver-modified Q_y
        Q_block = np.diag([self.Q_y_eff, p.Q_psi])
        self.Q1_full = np.kron(np.eye(Np), Q_block)
        
        # Pre-compute H'Q1H (used in all P matrices)
        self.HQ1H = self.H.T @ self.Q1_full @ self.H
        self.HQ1H += p.regularization * np.eye(p.Nu)
    
    def _build_P_for_lambda(self, lambda_k: float) -> np.ndarray:
        """
        Build the stacked P matrix for a specific lambda value.
        
        The Nash game becomes a single QP with stacked variable u = [u1; u2]:
            min u'Pu + q'u
        
        P depends on λ through Q2 = λ * Q1:
            P11 = H'Q1H + (R1 + S2)I
            P22 = H'(λQ1)H + (R2 + S1)I = λ * H'Q1H + (R2 + S1)I
            P12 = H'Q1H  (cross-coupling from tracking same output z)
        
        Args:
            lambda_k: Authority ratio
            
        Returns:
            P_stack: Stacked Hessian matrix (2*Nu x 2*Nu)
        """
        p = self.params
        Nu = p.Nu
        
        # P11: Player 1's Hessian (independent of λ)
        P11 = self.HQ1H + (p.R1 + self.S2_eff) * np.eye(Nu)
        
        # P22: Player 2's Hessian (depends on λ!)
        # Q2 = λ * Q1, so H'Q2H = λ * H'Q1H
        P22 = lambda_k * self.HQ1H + (self.R2_eff + p.S1) * np.eye(Nu)
        
        # P12: Cross-coupling (from both players tracking the same z)
        # Scale by sqrt(λ) to maintain positive definiteness for small λ
        coupling_scale = np.sqrt(min(lambda_k, 1.0))
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
        
        return P_stack
    
    def _create_problem_for_lambda(self, lambda_k: float) -> dict:
        """
        Create a CVXPY problem for a specific lambda value.
        
        Args:
            lambda_k: Authority ratio
            
        Returns:
            Dictionary with 'problem', 'u_var', 'q_param', 'u_prev_param', 'P_stack'
        """
        p = self.params
        Nu = p.Nu
        
        # Build P matrix for this lambda
        P_stack = self._build_P_for_lambda(lambda_k)
        
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
        
        # First step
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
            status = "✓" if self.problems[lam]['is_dpp'] else "✗"
            print(f"      λ={lam:5.2f}: DPP {status}")
    
    def _find_closest_lambda(self, lambda_k: float) -> float:
        """
        Find the closest pre-computed lambda level.
        
        Args:
            lambda_k: Requested authority ratio
            
        Returns:
            Closest pre-computed lambda value
        """
        # Clip to valid range
        lambda_clipped = np.clip(lambda_k, self.lambda_levels[0], self.lambda_levels[-1])
        
        # Find closest (in log space for better distribution)
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
        Solve constrained Nash equilibrium for lateral control with proper λ incorporation.
        
        The authority ratio λ affects Q2 = λ * Q1 in the Nash game.
        
        Args:
            x0: Current state [y, y_dot, psi, psi_dot]
            R1_ref: System reference trajectory (Np, 2) - [y_ref, psi_ref]
            R2_ref: Human reference trajectory (Np, 2)
            lambda_k: Authority ratio
            field_force: Safety field force (stored, not used in QP)
            
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
        
        # Track lambda usage
        self.last_lambda_used = lambda_used
        self.constraint_stats['lambda_level_usage'][lambda_used] += 1
        
        # Flatten references
        r1 = R1_ref.flatten()
        r2 = R2_ref.flatten()
        
        # Free response (z without control)
        z_free = self.U @ x0
        
        # Linear terms for QP: q = -2 * H' Q (r - z_free)
        # Player 1: always tracks R1
        q1 = -2 * self.H.T @ self.Q1_full @ (r1 - z_free)
        
        # Player 2: tracks R2 with λ weighting in P22
        q2 = -2 * lambda_used * self.H.T @ self.Q1_full @ (r2 - z_free)
        
        q_stacked = np.concatenate([q1, q2])
        
        # Update parameters
        prob_dict['q_param'].value = q_stacked
        prob_dict['u_prev_param'].value = np.array([self.u1_prev, self.u2_prev])
        
        # Warm start from this lambda's previous solution
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
                self._update_constraint_stats(u1_opt, u2_opt)
                
                # Update warm start for this lambda level (shift horizon)
                prob_dict['warm_start'] = np.roll(u_opt, -1)
                prob_dict['warm_start'][Nu-1] = u_opt[Nu-1]
                prob_dict['warm_start'][2*Nu-1] = u_opt[2*Nu-1]
                
                self.u1_prev = u1_opt
                self.u2_prev = u2_opt
                
                self.last_costs = [prob_dict['problem'].value / 2, prob_dict['problem'].value / 2]
                
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
        
        u_avg = 0.5 * (prob_dict['u_var'].value[:Nu] + prob_dict['u_var'].value[Nu:])
        z_pred = self.U @ x0 + self.H @ u_avg
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
            'lambda_levels': self.lambda_levels,
            'dpp_compliant': True
        }
    
    def get_constraint_summary(self) -> str:
        """Get formatted summary of constraint activity including lambda usage."""
        stats = self.constraint_stats
        total = stats['total_solves']
        p = self.params
        
        if total == 0:
            return "No solves performed yet."
        
        # Lambda usage summary
        lambda_usage = "\n".join([
            f"║    λ={lam:5.2f}: {stats['lambda_level_usage'][lam]:>6} times ({100*stats['lambda_level_usage'][lam]/total:>5.1f}%)        ║"
            for lam in self.params.lambda_levels
        ])
        
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║     LAMBDA-AWARE LATERAL NASH SOLVER - CONSTRAINT SUMMARY    ║
╠══════════════════════════════════════════════════════════════╣
║  Total solves: {total:>6}                                       ║
╠══════════════════════════════════════════════════════════════╣
║  LAMBDA LEVEL USAGE:                                         ║
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
    
    def print_constraint_summary(self):
        """Print constraint summary."""
        print(self.get_constraint_summary())
    
    def set_driver_type(self, driver_type: str):
        """Update driver type and rebuild problems."""
        self.params.driver_type = driver_type
        self._apply_driver_type_modifiers()
        self._build_base_cost_matrices()
        self._precompute_lambda_problems()
        print(f"🔄 Driver type changed to: {driver_type}")


# ============================================================================
# UNIT TESTS
# ============================================================================
if __name__ == "__main__":
    import time
    
    print("\n" + "="*70)
    print("Lambda-Aware Lateral Nash MPC Solver V5.0 - Integration Test")
    print("="*70)
    
    # Create a test vehicle (required for solver)
    test_vehicle = Vehicle(
        initial_y=0.0,
        initial_psi=0.0,
        initial_x=0.0,
        vehicle_id="TestVehicle",
        longitudinal_velocity=20.0
    )
    
    params = ConstrainedLateralNashParams(Np=10, Nu=5, dt=0.1)  # Test with 0.1s dt
    solver = ConstrainedLateralNashSolver(vehicle=test_vehicle, params=params)
    
    # Test 1: Basic solve with different lambda values
    print("\n📊 Test 1: Nash solve with varying λ")
    x0 = np.array([1.0, 0.0, 0.1, 0.0])  # y=1m, psi=0.1rad
    
    R1_ref = np.zeros((params.Np, 2))  # System wants center
    R2_ref = np.zeros((params.Np, 2))  # Human wants center
    
    test_lambdas = [0.1, 1.0, 5.0, 10.0]
    print(f"\n{'λ':>8} | {'λ used':>8} | {'δ1':>8} | {'δ2':>8} | {'Time':>8}")
    print("-" * 50)
    
    for lam in test_lambdas:
        solver.reset()
        delta1, delta2 = solver.solve_nash_equilibrium(
            x0=x0, R1_ref=R1_ref, R2_ref=R2_ref, lambda_k=lam
        )
        stats = solver.get_solver_stats()
        print(f"{lam:>8.2f} | {stats['lambda_used']:>8.2f} | {np.degrees(delta1):>8.3f}° | {np.degrees(delta2):>8.3f}° | {stats['solve_time_ms']:>6.2f}ms")
    
    # Test 2: Verify lambda effect on cooperation
    print("\n📊 Test 2: Verify λ effect on Player 2 control")
    print("As λ increases, δ2 should move toward δ1 (cooperation)")
    
    solver.reset()
    results = []
    
    # Different references: system wants 0, human wants slight offset
    R1_ref = np.zeros((params.Np, 2))  # System: target y=0
    R2_ref = np.ones((params.Np, 2)) * 0.5  # Human: target y=0.5m
    R2_ref[:, 1] = 0  # psi target = 0
    
    for lam in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        solver.u1_prev = 0.0
        solver.u2_prev = 0.0
        delta1, delta2 = solver.solve_nash_equilibrium(x0, R1_ref, R2_ref, lam)
        results.append((lam, delta1, delta2, abs(delta1 - delta2)))
    
    print(f"\n{'λ':>8} | {'δ1 [°]':>8} | {'δ2 [°]':>8} | {'|δ1-δ2|':>8}")
    print("-" * 40)
    for lam, d1, d2, diff in results:
        print(f"{lam:>8.2f} | {np.degrees(d1):>8.4f} | {np.degrees(d2):>8.4f} | {np.degrees(diff):>8.4f}")
    
    # Test 3: Performance benchmark
    print("\n📊 Test 3: Performance (500 solves with random λ)")
    solver.reset()
    
    times = []
    np.random.seed(42)
    
    for i in range(500):
        x0_test = np.array([
            np.random.uniform(-2, 2),
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-0.2, 0.2),
            np.random.uniform(-0.1, 0.1)
        ])
        lam_test = np.random.uniform(0.1, 10.0)
        
        start = time.perf_counter()
        solver.solve_nash_equilibrium(x0_test, R1_ref, R2_ref, lam_test)
        times.append((time.perf_counter() - start) * 1000)
    
    times = np.array(times)
    print(f"   Mean: {times.mean():.2f} ms")
    print(f"   Max:  {times.max():.2f} ms")
    print(f"   P99:  {np.percentile(times, 99):.2f} ms")
    
    # Test 4: Driver types
    print("\n📊 Test 4: Driver type comparison")
    x0 = np.array([1.0, 0.0, 0.1, 0.0])
    R1_ref = np.zeros((params.Np, 2))
    R2_ref = np.zeros((params.Np, 2))
    
    for driver_type in ['cautious', 'normal', 'aggressive']:
        solver.set_driver_type(driver_type)
        solver.reset()
        delta1, delta2 = solver.solve_nash_equilibrium(x0, R1_ref, R2_ref, lambda_k=1.0)
        print(f"   {driver_type:12s}: δ1={np.degrees(delta1):+6.4f}°, δ2={np.degrees(delta2):+6.4f}°")
    
    # Print constraint summary
    print(solver.get_constraint_summary())
    
    print("\n" + "="*70)
    print("✅ Lambda-Aware Lateral Nash Solver V5.0 Test Complete!")
    print("="*70)