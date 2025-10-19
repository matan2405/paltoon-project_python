#!/usr/bin/env python3
"""
File: nash_solver.py
Description: Contains the DMPC-based Nash equilibrium solver.
"""

import numpy as np
from scipy.linalg import solve
from typing import Tuple

# To make this file runnable independently, we need the VehicleModel class definition.
# In a real project structure, this would be `from vehicle_model import VehicleModel`.
# For this self-contained file, we add a placeholder or the actual class.
try:
    from vehicle_model import VehicleModel
except ImportError:
    # If vehicle_model.py is not in the path, use a placeholder.
    # This allows the class to be defined without runtime errors,
    # but the full system needs the actual VehicleModel.
    print("Warning: VehicleModel not found. Using a placeholder for NashSolver.")
    class VehicleModel:
        def __init__(self):
            self.A = np.eye(4)
            self.B1 = np.zeros(4)
            self.B2 = np.zeros(4)
            self.C = np.zeros((2, 4))


class EnhancedNashSolver:
    """Improved Nash equilibrium solver with better convergence"""
    
    def __init__(self, vehicle_model: VehicleModel, Np: int = 15, Nu: int = 10):
        self.model = vehicle_model
        self.Np = Np  # Prediction horizon
        self.Nu = Nu  # Control horizon
        
        # Cost function weights - TUNED for comfort
        self.Q_output = np.diag([150.0, 300.0])  # [y, φ] weights
        self.R1 = 50.0  # Controller effort weight - INCREASED for smoother control
        self.R2 = 60.0  # Human effort weight - INCREASED for smoother control
        
        self._build_prediction_matrices()
        
    def _build_prediction_matrices(self):
        """Build prediction matrices U, H1, H2"""
        A, B1, B2, C = self.model.A, self.model.B1, self.model.B2, self.model.C
        nx, nu, nz = A.shape[0], 1, C.shape[0]
        
        self.U = np.zeros((nz * self.Np, nx))
        self.H1 = np.zeros((nz * self.Np, nu * self.Nu))
        self.H2 = np.zeros((nz * self.Np, nu * self.Nu))
        
        for i in range(self.Np):
            self.U[i*nz:(i+1)*nz, :] = C @ np.linalg.matrix_power(A, i+1)
            
            for j in range(min(i+1, self.Nu)):
                H1_block = C @ np.linalg.matrix_power(A, i-j) @ B1.reshape(-1,1)
                H2_block = C @ np.linalg.matrix_power(A, i-j) @ B2.reshape(-1,1)
                
                self.H1[i*nz:(i+1)*nz, j*nu:(j+1)*nu] = H1_block
                self.H2[i*nz:(i+1)*nz, j*nu:(j+1)*nu] = H2_block
    
    def solve_nash_equilibrium(self, x0: np.ndarray, R1_ref: np.ndarray, 
                              R2_ref: np.ndarray, lambda_k: float) -> Tuple[float, float]:
        """Solve Nash equilibrium with improved convergence"""
        
        r1, r2 = R1_ref.flatten(), R2_ref.flatten()
        z_free = self.U @ x0
        e1, e2 = r1 - z_free, r2 - z_free
        
        Q1_bar = np.kron(np.eye(self.Np), self.Q_output)
        Q2_bar = np.kron(np.eye(self.Np), self.Q_output * lambda_k)
        
        R1_bar, R2_bar = self.R1 * np.eye(self.Nu), self.R2 * np.eye(self.Nu)
        
        try:
            H11 = self.H1.T @ Q1_bar @ self.H1 + R1_bar
            H12 = self.H1.T @ Q1_bar @ self.H2
            g1 = self.H1.T @ Q1_bar @ e1
            
            H21 = self.H2.T @ Q2_bar @ self.H1
            H22 = self.H2.T @ Q2_bar @ self.H2 + R2_bar
            g2 = self.H2.T @ Q2_bar @ e2
            
            reg_factor = 1e-4
            H11 += reg_factor * np.eye(H11.shape[0])
            H22 += reg_factor * np.eye(H22.shape[0])
            
            d1_seq, d2_seq = np.zeros(self.Nu), np.zeros(self.Nu)
            
            for _ in range(15):
                d1_prev, d2_prev = d1_seq.copy(), d2_seq.copy()
                
                d1_seq = solve(H11, g1 - H12 @ d2_prev)
                d2_seq = solve(H22, g2 - H21 @ d1_seq)
                
                d1_seq, d2_seq = np.clip(d1_seq, -0.3, 0.3), np.clip(d2_seq, -0.3, 0.3)
                
                if np.linalg.norm(d1_seq - d1_prev) < 1e-4 and np.linalg.norm(d2_seq - d2_prev) < 1e-4:
                    break
                    
            u1_optimal = d1_seq[0] if len(d1_seq) > 0 else 0.0
            u2_optimal = d2_seq[0] if len(d2_seq) > 0 else 0.0
            
            return u1_optimal, u2_optimal
            
        except np.linalg.LinAlgError:
            print("⚠️ Nash solver: Singular matrix, using fallback")
            return 0.0, 0.0
