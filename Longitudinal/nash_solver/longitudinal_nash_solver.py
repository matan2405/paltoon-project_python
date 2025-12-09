#!/usr/bin/env python3
"""
File: enhanced_nash_solver.py
Description: Enhanced Nash equilibrium solver.
"""
import numpy as np
from scipy.linalg import solve
from typing import Tuple
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class EnhancedNashSolver:
    def __init__(self, vehicle, platoon_manager, human_driver, Np: int = 20, Nu: int = 10, dt: float = 0.1):
        self.model = vehicle
        self.platoon_manager = platoon_manager
        self.human_driver = human_driver
        self.Np = Np
        self.Nu = Nu
        self.dt_nash = dt
        
        self.A, self.B1, self.C = self.model.get_state_space_matrices(self.dt_nash)
        self.B2 = self.B1.copy() 
        
        self.Q_output_base = np.diag([2500,50])#np.diag([800.0, 1000.0]) 
        self.R1_base = 800.0#200.0
        self.R2_base = 800.0#240.0
        self.Q_output = self.Q_output_base.copy()
        self.R1 = self.R1_base
        self.R2 = self.R2_base
        
        self.u1_min, self.u1_max = -2.5, 2.0
        self.u2_min, self.u2_max = -3.5, 2.5
        
        self._build_prediction_matrices()
        print("🛡️ Enhanced Nash Solver initialized.")

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

    def adapt_weights(self, field_force: float, velocity_error: float):
        force_mag = abs(field_force)
        scaling = 1.0
        if force_mag > 300: scaling = 1.5
        self.Q_output = self.Q_output_base * scaling
        self.R1 = self.R1_base
        self.R2 = self.R2_base

    def solve_nash_equilibrium(self, x0: np.ndarray, R1_ref: np.ndarray, 
                              R2_ref: np.ndarray, lambda_k: float,
                              field_force: float = 0.0) -> Tuple[float, float]:
        self._build_prediction_matrices()
        
        velocity_error = 0.0
        if len(R1_ref) > 0: velocity_error = R1_ref[0,1] - x0[1]
        self.adapt_weights(field_force, velocity_error)
        
        r1 = R1_ref.flatten()
        r2 = R2_ref.flatten()
        z_free = self.U @ x0
        e1 = r1 - z_free
        e2 = r2 - z_free
        
        Q1_bar = np.kron(np.eye(self.Np), self.Q_output)
        Q2_bar = np.kron(np.eye(self.Np), self.Q_output * lambda_k)
        R1_bar = self.R1 * np.eye(self.Nu * self.B1.shape[0])
        R2_bar = self.R2 * np.eye(self.Nu * self.B2.shape[0])
        
        try:
            H11 = self.H1.T @ Q1_bar @ self.H1 + R1_bar
            H12 = self.H1.T @ Q1_bar @ self.H2
            g1 = self.H1.T @ Q1_bar @ e1
            
            H21 = self.H2.T @ Q2_bar @ self.H1
            H22 = self.H2.T @ Q2_bar @ self.H2 + R2_bar
            g2 = self.H2.T @ Q2_bar @ e2
            
            reg = 1e-2 * np.eye(H11.shape[0])
            H11 += reg
            H22 += reg
            
            A_global = np.block([[H11, H12], [H21, H22]])
            b_global = np.concatenate([g1, g2])
            
            d_global = np.linalg.solve(A_global, b_global)
            
            u1 = float(np.clip(d_global[0], self.u1_min, self.u1_max))
            u2 = float(np.clip(d_global[H11.shape[0]], self.u2_min, self.u2_max))
            
            return u1, u2
            
        except Exception as e:
            print(f"   ⚠️ Nash solver error: {e}")
            return 0.0, 0.0