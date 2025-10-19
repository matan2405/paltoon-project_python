#!/usr/bin/env python3
"""
File: vehicle_model.py
Description: Contains the vehicle parameters and the 2-DOF bicycle model.
"""

import numpy as np
from dataclasses import dataclass

@dataclass
class VehicleParams:
    """Vehicle parameters based on Li et al. 2019 Table 3"""
    m: float = 1650.0      # Vehicle mass [kg]
    la: float = 1.40       # Distance from CoG to front axle [m]  
    lb: float = 1.65       # Distance from CoG to rear axle [m]
    length: float = 4.50   # Vehicle length [m]
    width: float = 1.80    # Vehicle width [m]
    Iz: float = 3234.0     # Yaw moment of inertia [kg⋅m²]
    Cf: float = 120000.0   # Front tire stiffness [N/rad]
    Cr: float = 120000.0   # Rear tire stiffness [N/rad]
    dt: float = 0.1        # Time step [s]
    max_steer: float = 0.5 # Maximum steering angle [rad] (~28.6°)
    max_lat_acc: float = 2.5  # Comfort limit [m/s²]
    

class VehicleModel:
    """2-DOF bicycle model from Li et al. 2019"""
    
    def __init__(self, params: VehicleParams, velocity: float = 20.0):
        self.params = params
        self.v = velocity
        self.lambda_k = 0.169  # Default value
        self._build_matrices()
    
    def _build_matrices(self):
        """Build discrete-time state-space matrices"""
        p = self.params
        v = self.v
        dt = p.dt
        
        # Continuous-time matrices - Equation (1) from Li et al.
        A_c = np.array([
            [-(p.Cf + p.Cr)/(p.m * v), -v - (p.Cf*p.la - p.Cr*p.lb)/(p.m*v), 0, 0],
            [-(p.Cf*p.la - p.Cr*p.lb)/p.Iz, -(p.Cf*p.la**2 + p.Cr*p.lb**2)/(p.Iz*v), 0, 0],
            [1, 0, 0, v],  # Fixed: lateral position integration [y_dot -> y] and [phi*v -> y]
            [0, 1, 0, 0]   # yaw angle integration [phi_dot -> phi]
        ])
        
        B_c = np.array([
            [p.Cf/p.m],
            [p.Cf*p.la/p.Iz], 
            [0],
            [0]
        ])
        
        # Discretization using matrix exponential for stability
        I = np.eye(4)
        self.A = I + A_c * dt + 0.5 * (A_c @ A_c) * dt**2  # Second-order approximation
        self.B1 = (B_c * dt).flatten()  # Controller input matrix - flatten to (4,)
        self.B2 = (B_c * dt * self.lambda_k).flatten()  # Human input matrix - flatten to (4,)

        # Output matrix - only measured variables [y, φ]
        self.C = np.array([
            [0, 0, 1, 0],  # y (lateral position)
            [0, 0, 0, 1]   # φ (yaw angle)
        ])
        
    def step(self, x: np.ndarray, u1: float, u2: float) -> np.ndarray:
        """Simulate one time step"""
        # Input saturation
        u1 = np.clip(u1, -self.params.max_steer, self.params.max_steer)
        u2 = np.clip(u2, -self.params.max_steer, self.params.max_steer)
        
        # State update
        x_next = self.A @ x + self.B1 * u1 + self.B2 * u2
        
        # Comfort constraint: limit lateral velocity change (directly limits acceleration)
        max_y_dot_change = 0.25  # m/s per time step (0.25/0.1 = 2.5 m/s² max acceleration)
        if len(x) > 0 and len(x_next) > 0:
            y_dot_change = x_next[0] - x[0]
            if abs(y_dot_change) > max_y_dot_change:
                # Scale back the change to maintain comfort
                scale_factor = max_y_dot_change / abs(y_dot_change)
                x_next[0] = x[0] + y_dot_change * scale_factor
        
        # Stability check
        if not np.isfinite(x_next).all():
            print("⚠️ Unstable state detected, using damped update")
            x_next = 0.9 * x + 0.1 * x_next
            
        return x_next
    
    def output(self, x: np.ndarray) -> np.ndarray:
        """Get measured outputs [y, φ]"""
        return self.C @ x
