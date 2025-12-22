#!/usr/bin/env python3
"""
File: enhanced_nash_solver.py
Description: Coupled Nash equilibrium solver based on Li et al. 2019.

COUPLED NASH GAME FORMULATION:
==============================
In the coupled game, each player's cost function includes the other player's 
control input, creating interdependency:

Player 1 (System): J1 = (z - r1)^T Q1 (z - r1) + u1^T R1 u1 + u2^T S1 u2
Player 2 (Human):  J2 = (z - r2)^T Q2 (z - r2) + u1^T S2 u1 + u2^T R2 u2

Where S1, S2 are the "cross-coupling" weights representing how each player
penalizes the other's control effort.

The coupled Nash equilibrium satisfies:
∂J1/∂u1 = 0  and  ∂J2/∂u2 = 0

This leads to a coupled linear system that must be solved simultaneously.
"""
import numpy as np
from scipy.linalg import solve
from typing import Tuple
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class EnhancedNashSolver:
    """
    Coupled Nash Equilibrium Solver for Shared Control.
    
    Based on Li et al. 2019: "Shared control with a novel dynamic authority 
    allocation strategy based on game theory and driving safety field"
    """
    
    def __init__(self, vehicle, platoon_manager, human_driver, Np: int = 20, Nu: int = 10, dt: float = 0.1):
        self.model = vehicle
        self.platoon_manager = platoon_manager
        self.human_driver = human_driver
        self.Np = Np
        self.Nu = Nu
        self.dt_nash = dt
        
        self.A, self.B1, self.C = self.model.get_state_space_matrices(self.dt_nash)
        self.B2 = self.B1.copy() 
        
        # === OUTPUT COST WEIGHTS (Q) ===
        self.Q_output_base = np.diag([2500, 50])  # [position, velocity]
        self.Q_output = self.Q_output_base.copy()
        
        # === OWN CONTROL EFFORT WEIGHTS (R) ===
        self.R1_base = 800.0  # System's cost on its own control
        self.R2_base = 800.0  # Human's cost on its own control
        self.R1 = self.R1_base
        self.R2 = self.R2_base
        
        # === CROSS-COUPLING WEIGHTS (S) - THE KEY ADDITION FOR COUPLED GAME ===
        # S1: How much the SYSTEM penalizes the HUMAN's control effort in J1
        # S2: How much the HUMAN penalizes the SYSTEM's control effort in J2
        # 
        # Higher S values → More cooperation (players don't want the other to work hard)
        # S = 0 → Decoupled (non-cooperative) Nash game
        self.S1_base = 200.0  # System cares about human effort
        self.S2_base = 200.0  # Human cares about system effort
        self.S1 = self.S1_base
        self.S2 = self.S2_base
        
        self.u1_min, self.u1_max = -2.5, 2.0
        self.u2_min, self.u2_max = -3.5, 2.5
        
        # Store last costs for analysis
        self.last_costs = [0.0, 0.0]
        
        self._build_prediction_matrices()
        print("🛡️ Coupled Nash Solver initialized (Li et al. 2019 formulation).")

    def _build_prediction_matrices(self):
        nx = self.A.shape[0]
        nu = self.B1.shape[0]
        nz = self.C.shape[0]
        self.U = np.zeros((self.Np * nz, nx))
        self.H1 = np.zeros((self.Np * nz, self.Nu * nu))
        self.H2 = np.zeros((self.Np * nz, self.Nu * nu))
        
        for i in range(self.Np):
            self.U[i*nz:(i+1)*nz, :] = self.C @ np.linalg.matrix_power(self.A, i+1)
            for j in range(min(i+1, self.Nu)):
                term1 = self.C @ np.linalg.matrix_power(self.A, i-j) @ self.B1.reshape(-1, 1)
                term2 = self.C @ np.linalg.matrix_power(self.A, i-j) @ self.B2.reshape(-1, 1)
                self.H1[i*nz:(i+1)*nz, j*nu:(j+1)*nu] = term1
                self.H2[i*nz:(i+1)*nz, j*nu:(j+1)*nu] = term2

    def adapt_weights(self, field_force: float, velocity_error: float, lambda_k: float = 1.0):
        """
        Adapt weight matrices based on the current driving situation.
        
        The coupling weights S1, S2 are also adjusted based on λ(k):
        - Higher λ (system dominant) → Lower S1 (system less affected by human)
        - Lower λ (human dominant) → Lower S2 (human less affected by system)
        """
        force_mag = abs(field_force)
        scaling = 1.0
        if force_mag > 300:
            scaling = 1.5
        self.Q_output = self.Q_output_base * scaling
        self.R1 = self.R1_base
        self.R2 = self.R2_base
        
        # === ADAPTIVE COUPLING BASED ON AUTHORITY ===
        # When λ is high (system dominant), the system is less influenced by human
        # When λ is low (human dominant), the human is less influenced by system
        alpha = lambda_k / (1.0 + lambda_k)  # α ∈ [0, 1], higher = more system authority
        
        # S1 (how much system cares about human's effort): decreases when system is dominant
        # S2 (how much human cares about system's effort): increases when system is dominant
        self.S1 = self.S1_base * (1.0 - 0.5 * alpha)  # Range: [0.5, 1.0] * base
        self.S2 = self.S2_base * (0.5 + 0.5 * alpha)  # Range: [0.5, 1.0] * base

    def solve_nash_equilibrium(self, x0: np.ndarray, R1_ref: np.ndarray, 
                              R2_ref: np.ndarray, lambda_k: float,
                              field_force: float = 0.0) -> Tuple[float, float]:
        """
        Solve the COUPLED Nash equilibrium for shared control.
        
        COUPLED COST FUNCTIONS (Li et al. 2019):
        =========================================
        J1 = ||z - r1||²_Q1 + ||u1||²_R1 + ||u2||²_S1
        J2 = ||z - r2||²_Q2 + ||u1||²_S2 + ||u2||²_R2
        
        Where:
        - z = U*x0 + H1*u1 + H2*u2 (predicted output)
        - Q1, Q2: output tracking weights
        - R1, R2: own control effort weights  
        - S1, S2: cross-coupling weights (other player's control)
        
        OPTIMALITY CONDITIONS:
        ======================
        ∂J1/∂u1 = 0: H1'Q1(H1*u1 + H2*u2 + U*x0 - r1) + R1*u1 = 0
        ∂J2/∂u2 = 0: H2'Q2(H1*u1 + H2*u2 + U*x0 - r2) + R2*u2 = 0
        
        Note: The coupling appears through the shared prediction z that
        depends on BOTH u1 and u2. The S terms add direct penalty on
        the other player's control.
        
        Rearranging into matrix form:
        [H1'Q1H1 + R1,  H1'Q1H2      ] [u1]   [H1'Q1(r1 - U*x0)]
        [H2'Q2H1,       H2'Q2H2 + R2 ] [u2] = [H2'Q2(r2 - U*x0)]
        
        Adding cross-coupling S terms to the diagonal blocks doesn't change
        the structure since S penalizes the OTHER player's control. But it
        affects the effective cost and the equilibrium point.
        
        Args:
            x0: Current state [position, velocity]
            R1_ref: System reference trajectory (Np x 2)
            R2_ref: Human reference trajectory (Np x 2)
            lambda_k: Authority ratio (higher = more system authority)
            field_force: Safety field force for weight adaptation
            
        Returns:
            (u1_opt, u2_opt): Optimal control inputs for both players
        """
        self._build_prediction_matrices()
        
        # Calculate velocity error for weight adaptation
        velocity_error = 0.0
        if len(R1_ref) > 0:
            velocity_error = R1_ref[0, 1] - x0[1]
        
        # Adapt weights including coupling terms
        self.adapt_weights(field_force, velocity_error, lambda_k)
        
        # Flatten reference trajectories
        r1 = R1_ref.flatten()
        r2 = R2_ref.flatten()
        
        # Calculate free response (system evolution without control)
        z_free = self.U @ x0
        
        # Tracking errors
        e1 = r1 - z_free
        e2 = r2 - z_free
        
        # === BUILD BLOCK DIAGONAL WEIGHT MATRICES ===
        # For λ(k) > 1: Scale Q2 to give more weight to system's objectives
        Q1_bar = np.kron(np.eye(self.Np), self.Q_output)
        Q2_bar = np.kron(np.eye(self.Np), self.Q_output * lambda_k)
        
        # Control effort weights (diagonal)
        nu = self.B1.shape[0]
        R1_bar = self.R1 * np.eye(self.Nu * nu)
        R2_bar = self.R2 * np.eye(self.Nu * nu)
        
        # Cross-coupling weights (for the coupled game)
        S1_bar = self.S1 * np.eye(self.Nu * nu)  # System's penalty on human control
        S2_bar = self.S2 * np.eye(self.Nu * nu)  # Human's penalty on system control
        
        try:
            # === COUPLED NASH EQUILIBRIUM EQUATIONS ===
            # 
            # For Player 1 (System):
            # J1 = (z-r1)'Q1(z-r1) + u1'R1*u1 + u2'S1*u2
            # 
            # Taking derivative ∂J1/∂u1 = 0:
            # H1'Q1(H1*u1 + H2*u2 - e1) + R1*u1 = 0
            # (H1'Q1H1 + R1)*u1 + H1'Q1H2*u2 = H1'Q1*e1
            #
            # For Player 2 (Human):
            # J2 = (z-r2)'Q2(z-r2) + u1'S2*u1 + u2'R2*u2
            #
            # Taking derivative ∂J2/∂u2 = 0:
            # H2'Q2(H1*u1 + H2*u2 - e2) + R2*u2 = 0
            # H2'Q2H1*u1 + (H2'Q2H2 + R2)*u2 = H2'Q2*e2
            #
            # The COUPLING comes from:
            # 1. The shared prediction z = U*x0 + H1*u1 + H2*u2 (both controls affect output)
            # 2. The cross-terms H1'Q1H2 and H2'Q2H1 in the equations
            # 3. (Optional) The S terms which add direct penalty on other's control
            
            # Block (1,1): H1'Q1H1 + R1 + S2 (S2 = human penalizes system's control)
            H11 = self.H1.T @ Q1_bar @ self.H1 + R1_bar + S2_bar
            
            # Block (1,2): H1'Q1H2 - THE COUPLING TERM
            H12 = self.H1.T @ Q1_bar @ self.H2
            
            # Block (2,1): H2'Q2H1 - THE COUPLING TERM  
            H21 = self.H2.T @ Q2_bar @ self.H1
            
            # Block (2,2): H2'Q2H2 + R2 + S1 (S1 = system penalizes human's control)
            H22 = self.H2.T @ Q2_bar @ self.H2 + R2_bar + S1_bar
            
            # Right-hand side
            g1 = self.H1.T @ Q1_bar @ e1
            g2 = self.H2.T @ Q2_bar @ e2
            
            # Add regularization for numerical stability
            reg = 1e-2 * np.eye(H11.shape[0])
            H11 += reg
            H22 += reg
            
            # === SOLVE THE COUPLED SYSTEM ===
            # [H11  H12] [u1]   [g1]
            # [H21  H22] [u2] = [g2]
            #
            # This is the KEY difference from uncoupled Nash:
            # In uncoupled Nash, H12 and H21 would be zero (or ignored)
            # In coupled Nash, we solve the full system simultaneously
            
            A_coupled = np.block([
                [H11, H12],
                [H21, H22]
            ])
            b_coupled = np.concatenate([g1, g2])
            
            # Solve for both control sequences simultaneously
            d_coupled = np.linalg.solve(A_coupled, b_coupled)
            
            # Extract first control input for each player
            u1 = float(np.clip(d_coupled[0], self.u1_min, self.u1_max))
            u2 = float(np.clip(d_coupled[H11.shape[0]], self.u2_min, self.u2_max))
            
            # Calculate costs for analysis (extract scalar from matrix result)
            z_pred = z_free.reshape(-1, 1) + self.H1[:, 0:1] * u1 + self.H2[:, 0:1] * u2
            e1_pred = z_pred - r1.reshape(-1, 1)
            e2_pred = z_pred - r2.reshape(-1, 1)
            
            # Quadratic form returns a scalar, but in matrix form it's (1,1)
            J1_tracking = e1_pred.T @ Q1_bar @ e1_pred
            J2_tracking = e2_pred.T @ Q2_bar @ e2_pred
            
            # Extract scalar values properly
            J1 = float(np.squeeze(J1_tracking)) + self.R1 * u1**2 + self.S1 * u2**2
            J2 = float(np.squeeze(J2_tracking)) + self.S2 * u1**2 + self.R2 * u2**2
            self.last_costs = [J1, J2]
            
            return u1, u2
            
        except Exception as e:
            print(f"   ⚠️ Coupled Nash solver error: {e}")
            return 0.0, 0.0

    def get_coupling_parameters(self) -> dict:
        """
        Get the current coupling parameters for analysis.
        
        Returns:
            Dictionary with current S1, S2 values and their base values.
        """
        return {
            'S1': self.S1,
            'S2': self.S2,
            'S1_base': self.S1_base,
            'S2_base': self.S2_base,
            'R1': self.R1,
            'R2': self.R2,
            'coupling_ratio_S1': self.S1 / self.R1 if self.R1 > 0 else 0,
            'coupling_ratio_S2': self.S2 / self.R2 if self.R2 > 0 else 0
        }
    
    def set_coupling_weights(self, S1: float, S2: float):
        """
        Manually set the coupling weights.
        
        Use this to experiment with different coupling strengths:
        - S1, S2 = 0: Fully decoupled (non-cooperative) Nash
        - S1, S2 = R1, R2: Equal penalty on own and other's control
        - S1, S2 > R1, R2: Strongly cooperative behavior
        
        Args:
            S1: System's penalty on human control effort
            S2: Human's penalty on system control effort
        """
        self.S1_base = S1
        self.S2_base = S2
        self.S1 = S1
        self.S2 = S2
        print(f"🔧 Coupling weights updated: S1={S1:.1f}, S2={S2:.1f}")