"""
Vehicle class implementing 2-DOF bicycle model for lateral dynamics.
VERSION 3.2 - DUAL MODEL WITH FULL ERROR MODEL PROPAGATION

PLANT:      Body-frame (Eq. 2.31) — x_b[k+1] = A_body · x_b[k] + B · δ
CONTROLLER: Error model (Eq. 2.45) — x_e[k+1] = A_error · x_e[k] + B · δ

Both run every timestep with the same δ.
Nash solver reads x_e directly from the error model propagation.
"""

import numpy as np
from scipy.linalg import expm
from typing import Tuple
from .components import VehicleParameters, VehicleState


class Vehicle:
    """2-DOF Bicycle model — dual propagation architecture."""
    
    def __init__(self, 
                 initial_y: float = 0.0,
                 initial_psi: float = 0.0,
                 initial_x: float = 0.0,
                 vehicle_id: str = "Vehicle",
                 longitudinal_velocity: float = 20.0):
        
        self.params = VehicleParameters()
        self.state = VehicleState()
        self.vehicle_id = vehicle_id
        
        self.state.y = initial_y
        self.state.psi = initial_psi
        self.state.x = initial_x
        self.state.vx = longitudinal_velocity
        self.vx = longitudinal_velocity
        self.L = self.params.length
        
        # Body-frame state: [y_body, ẏ_body, ψ, ψ̇]
        self.x_body = np.array([initial_y, 0.0, initial_psi, 0.0])
        
        # Error model state: [e1, ė1, e2, ė2]
        self.x_error = np.array([initial_y, 0.0, initial_psi, 0.0])
        
        # Steering history
        self.delta_prev = 0.0
        self.ay_prev = 0.0
        
        # Matrices
        self.A_body_c = None; self.A_body_d = None; self.B_body_d = None
        self.A_error_c = None; self.A_error_d = None; self.B_error_d = None
        self.B_c = None; self.C = None
        
        # Legacy aliases for Nash solver
        self.A = None; self.B1 = None; self.B2 = None
        self.A_d = None; self.B_d = None
        
        self.joined_platoon = False
        self._build_all_matrices()
        
        print(f"🚗 Vehicle '{vehicle_id}' V3.2 (Dual Propagation) at y={initial_y:.1f}m")
    
    def _build_all_matrices(self, dt: float = 0.01):
        p = self.params
        vx = max(self.vx, 0.001)
        
        b2 = 2 * p.Cf / p.mass
        b4 = 2 * (p.lf * p.Cf) / p.Iz
        self.B_c = np.array([[0.0], [b2], [0.0], [b4]])
        self.C = np.array([[1,0,0,0],[0,0,1,0]])
        
        # Body-frame (Eq. 2.31)
        a22 = -2*(p.Cf+p.Cr)/(p.mass*vx)
        a24 = -vx - (2*p.Cf*p.lf - 2*p.Cr*p.lr)/(p.mass*vx)
        a42 = -(2*p.lf*p.Cf - 2*p.lr*p.Cr)/(p.Iz*vx)
        a44 = -(2*p.lf**2*p.Cf + 2*p.lr**2*p.Cr)/(p.Iz*vx)
        
        self.A_body_c = np.array([
            [0, 1, 0, 0], [0, a22, 0, a24],
            [0, 0, 0, 1], [0, a42, 0, a44]
        ])
        
        # Error model (Eq. 2.45, straight road)
        a12 = -2*(p.Cf+p.Cr)/(p.mass*vx)
        a13 = 2*(p.Cf+p.Cr)/p.mass
        a14 = -2*(p.Cf*p.lf - p.Cr*p.lr)/(p.mass*vx)
        a32 = -2*(p.lf*p.Cf - p.lr*p.Cr)/(p.Iz*vx)
        a33 = 2*(p.lf*p.Cf - p.lr*p.Cr)/p.Iz
        a34 = -2*(p.lf**2*p.Cf + p.lr**2*p.Cr)/(p.Iz*vx)
        
        self.A_error_c = np.array([
            [0, 1, 0, 0], [0, a12, a13, a14],
            [0, 0, 0, 1], [0, a32, a33, a34]
        ])
        
        self.A_body_d, self.B_body_d = self._discretize_zoh(self.A_body_c, self.B_c, dt)
        self.A_error_d, self.B_error_d = self._discretize_zoh(self.A_error_c, self.B_c, dt)
        
        self.A_d = self.A_error_d; self.B_d = self.B_error_d
        self.A = self.A_error_d; self.B1 = self.B_error_d; self.B2 = self.B_error_d.copy()
    
    def _discretize_zoh(self, A_c, B_c, dt):
        n, m = A_c.shape[0], B_c.shape[1]
        aug = np.zeros((n+m, n+m))
        aug[:n,:n] = A_c*dt; aug[:n,n:] = B_c*dt
        e = expm(aug)
        return e[:n,:n], e[:n,n:]
    
    def get_state_space_matrices(self, dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Error model matrices for Nash solver (Eq. 2.45)."""
        self.A_error_d, self.B_error_d = self._discretize_zoh(self.A_error_c, self.B_c, dt)
        self.A_d = self.A_error_d; self.B_d = self.B_error_d
        self.A = self.A_error_d; self.B1 = self.B_error_d; self.B2 = self.B_error_d.copy()
        return self.A_error_d, self.B_error_d, self.C
    
    def get_state_vector(self) -> np.ndarray:
        """Error model state [e1, ė1, e2, ė2] — from Eq. 2.45 propagation."""
        return self.x_error.copy()
    
    def update_dynamics(self, dt: float, delta_commanded: float):
        """Propagate both models with the same δ."""
        if self.A_body_d is None:
            self._build_all_matrices(dt)
        
        # 1. Rate-limited steering
        delta_commanded = np.clip(delta_commanded,
                                  -self.params.max_steering_angle,
                                  self.params.max_steering_angle)
        max_dd = self.params.max_steering_rate * dt
        delta_actual = self.delta_prev + np.clip(
            delta_commanded - self.delta_prev, -max_dd, max_dd)
        self.delta_prev = delta_actual
        self.state.delta = delta_actual
        u = np.array([[delta_actual]])
        
        # 2. Body-frame propagation (Eq. 2.31)
        x_b_next = self.A_body_d @ self.x_body.reshape(-1,1) + self.B_body_d @ u
        x_b_next[1,0] = np.clip(x_b_next[1,0], -2.0, 2.0)
        x_b_next[2,0] = np.clip(x_b_next[2,0], -np.radians(15), np.radians(15))
        x_b_next[3,0] = np.clip(x_b_next[3,0], -0.3, 0.3)
        self.x_body = x_b_next.flatten()
        
        # 3. Error model propagation (Eq. 2.45)
        x_e_next = self.A_error_d @ self.x_error.reshape(-1,1) + self.B_error_d @ u
        x_e_next[1,0] = np.clip(x_e_next[1,0], -5.0, 5.0)
        x_e_next[2,0] = np.clip(x_e_next[2,0], -np.radians(15), np.radians(15))
        x_e_next[3,0] = np.clip(x_e_next[3,0], -0.3, 0.3)
        self.x_error = x_e_next.flatten()
        
        # 4. Write to state (for simulator, safety field, plotting)
        self.state.y = self.x_error[0]          # e1
        self.state.y_dot = self.x_body[1]       # ẏ_body
        self.state.psi = self.x_error[2]        # e2 = ψ
        self.state.psi_dot = self.x_error[3]    # ė2 = ψ̇
        
        # 5. Lateral acceleration
        ay_new = self.vx * self.state.psi_dot
        max_ay_change = self.params.max_lateral_jerk * dt
        self.state.ay = self.ay_prev + np.clip(
            ay_new - self.ay_prev, -max_ay_change, max_ay_change)
        self.ay_prev = self.state.ay
        
        # 6. Longitudinal position
        self.state.x += self.vx * dt
        
        # 7. NaN check
        if not (np.isfinite(self.x_body).all() and np.isfinite(self.x_error).all()):
            print(f"⚠️ {self.vehicle_id}: NaN detected, resetting")
            self.x_body[1] = 0.0; self.x_body[3] = 0.0
            self.x_error[1] = 0.0; self.x_error[3] = 0.0
    
    def reset_steering(self):
        self.delta_prev = 0.0
        self.ay_prev = 0.0
        self.state.delta = 0.0


__all__ = ['Vehicle']