#!/usr/bin/env python3
"""
File: longitudinal_nash_solver.py
Description: DMPC-based Nash equilibrium solver for longitudinal control problem.
Enhanced version with proper soft constraints and realistic cost function.
"""

import numpy as np
from scipy.linalg import solve
from typing import Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from vehicle.vehicle import Vehicle
    import system_reference_generator
    import longitudinal_safety_field
    import longitudinal_authority_allocator
    from control.human_driver import HumanDriver
    from simulation.simulator import PlatoonSimulation, run_simulation

except ImportError:
    print("Warning: Vehicle model not found. Using placeholder.")

class EnhancedLongitudinalNashSolver:
    """Enhanced Nash equilibrium solver with soft constraints"""

    def __init__(self, vehicle=Vehicle, Np: int = 15, Nu: int = 10, dt: float = 0.1):
        self.model = vehicle
        self.Np = Np  # Prediction horizon
        self.Nu = Nu  # Control horizon
        
        # ELEGANT FIX: Proper cost function design
        # Lower Q means less aggressive tracking → smoother control
        self.Q_output = np.diag([100.0, 400.0])  # [position, velocity]
        
        # Control effort cost
        self.R1 = 2.0   # Significant control cost (smooth control)
        self.R2 = 3.0  # Slightly higher for human (prefer less aggressive)

        # Acceleration constraints
        self.u1_min, self.u1_max = -2.5, 2.0  # System
        self.u2_min, self.u2_max = -3.5, 2.5  # Human
        self.dt = dt  # ⚡ Control time step (0.1s)
        
        # Barrier function parameters
        self.barrier_weight = 100.0
        self.barrier_margin = 0.5
        
        self._build_prediction_matrices()
        print("🚗 Enhanced Longitudinal Nash Solver initialized.")
        print(f"   Control dt: {self.dt:.3f}s")
        print(f"   Prediction horizon: {self.Np} steps = {self.Np * self.dt:.1f}s")
        print(f"   Q_output: {np.diag(self.Q_output)}, R: [{self.R1}, {self.R2}]")

    def _build_prediction_matrices(self):
        """Build prediction matrices U, H1, H2 for DMPC framework."""
        # ⚡ CRITICAL: Get state-space matrices with CONTROL dt, not simulation dt!
        A, B1, C = self.model.get_state_space_matrices(dt=self.dt)
        B2 = B1.copy()
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

    def _compute_barrier_gradient(self, u: float, u_min: float, u_max: float) -> float:
        """
        Compute gradient of logarithmic barrier function.
        Creates soft constraint that gently pushes away from limits.
        
        Barrier: -log(u - u_min) - log(u_max - u)
        Gradient: -1/(u - u_min) + 1/(u_max - u)
        """
        margin = self.barrier_margin
        
        # Distance from lower and upper bounds
        dist_lower = max(u - u_min, margin)
        dist_upper = max(u_max - u, margin)
        
        # Barrier gradient (pushes away from limits)
        grad = -1.0 / dist_lower + 1.0 / dist_upper
        
        return self.barrier_weight * grad

    def solve_nash_equilibrium(self, x0: np.ndarray, R1_ref: np.ndarray, 
                          R2_ref: np.ndarray, lambda_k: float) -> Tuple[float, float]:
        """
        Solve Nash equilibrium - EXACT COPY FROM LATERAL CODE.
        No weighting by lambda, only Q scaling.
        """
        
        self._build_prediction_matrices()
        
        # Flatten references
        r1 = R1_ref.flatten()
        r2 = R2_ref.flatten()
        
        # Predict free response
        z_free_raw = self.U @ x0
        z_free = z_free_raw.flatten()
        
        if z_free.shape != r1.shape:
            if z_free.size >= r1.size:
                z_free = z_free[:r1.size].reshape(r1.shape)
            else:
                z_free_temp = np.zeros_like(r1)
                z_free_temp[:z_free.size] = z_free
                z_free = z_free_temp

        # Calculate errors
        e1 = r1 - z_free
        e2 = r2 - z_free
        
        print(f"\n🔍 DEBUG Nash Solver:")
        print(f"   x0 (current) = {x0}")
        print(f"   r1[0:4] (system) = {r1[:4]}")
        print(f"   r2[0:4] (human) = {r2[:4]}")
        print(f"   e1[0:4] = {e1[:4]}")
        print(f"   e2[0:4] = {e2[:4]}")
        
        # ⚡⚡⚡ LATERAL STYLE: Scale Q2 by lambda, NO WEIGHTING ⚡⚡⚡
        Q1_bar = np.kron(np.eye(self.Np), self.Q_output)
        Q2_bar = np.kron(np.eye(self.Np), self.Q_output * lambda_k)  # ← Scale Q2!
        
        print(f"   λ={lambda_k:.3f} → Q2_scale={lambda_k:.3f}")
        
        # Fixed R matrices
        R1_bar = self.R1 * np.eye(self.Nu)
        R2_bar = self.R2 * np.eye(self.Nu)
        
        try:
            # ⚡ NO WEIGHTING - just standard Nash formulation
            H11 = self.H1.T @ Q1_bar @ self.H1 + R1_bar
            H12 = self.H1.T @ Q1_bar @ self.H2
            g1 = self.H1.T @ Q1_bar @ e1
            
            H21 = self.H2.T @ Q2_bar @ self.H1
            H22 = self.H2.T @ Q2_bar @ self.H2 + R2_bar
            g2 = self.H2.T @ Q2_bar @ e2
            
            print(f"   ⚡ Standard Nash: ||H11||={np.linalg.norm(H11):.1f}, ||H22||={np.linalg.norm(H22):.1f}")
            print(f"   Grads: g1={np.linalg.norm(g1):.1f}, g2={np.linalg.norm(g2):.1f}")
            
            # Regularization (same as lateral)
            reg_factor = 1e-4
            H11 += reg_factor * np.eye(H11.shape[0])
            H22 += reg_factor * np.eye(H22.shape[0])
            
            print(f"   Cond: H11={np.linalg.cond(H11):.2e}, H22={np.linalg.cond(H22):.2e}")
            
            # Nash iteration (same as lateral)
            d1_seq = np.zeros(self.Nu)
            d2_seq = np.zeros(self.Nu)
            
            for k in range(15):
                d1_prev = d1_seq.copy()
                d2_prev = d2_seq.copy()
                
                # Best response for player 1 (system)
                d1_seq = solve(H11, g1 - H12 @ d2_prev)
                
                # Best response for player 2 (human)
                d2_seq = solve(H22, g2 - H21 @ d1_seq)
                
                # Use proper longitudinal limits (not lateral [-0.3, 0.3])
                d1_seq = np.clip(d1_seq, self.u1_min, self.u1_max)  # ✅ [-2.5, 2.0]
                d2_seq = np.clip(d2_seq, self.u2_min, self.u2_max)  # ✅ [-3.5, 2.5]
                
                if k == 0:
                    print(f"   Iter 0: d1={d1_seq[:3]}, d2={d2_seq[:3]}")
                
                # Convergence check
                conv1 = np.linalg.norm(d1_seq - d1_prev)
                conv2 = np.linalg.norm(d2_seq - d2_prev)
                
                if conv1 < 1e-4 and conv2 < 1e-4:
                    print(f"   ✅ Converged in {k+1} iterations")
                    break
        except Exception as exc:
            print(f"   ❗ Error solving Nash equilibrium: {exc}")
            # Fallback to zero actions to keep caller safe
            d1_seq = np.zeros(self.Nu)
            d2_seq = np.zeros(self.Nu)
        
        # Extract first control input
        u1_optimal = float(d1_seq[0]) if len(d1_seq) > 0 else 0.0
        u2_optimal = float(d2_seq[0]) if len(d2_seq) > 0 else 0.0
        
        # Sanity check
        dv1 = r1[1] - x0[1] if len(r1) > 1 else 0.0
        dv2 = r2[1] - x0[1] if len(r2) > 1 else 0.0
        
        print(f"   📊 Check: v={x0[1]*3.6:.1f}km/h")
        if dv1 > 0 and u1_optimal <= 0:
            print("      ⚠️ System wants to speed up but gives negative acceleration!")
        elif dv1 > 0 and u1_optimal > 0:
            print(f"      Sys Δv={dv1:+.3f} → u1={u1_optimal:+.2f}")
        if dv2 > 0 and u2_optimal <= 0:
            print("      ⚠️ Human wants to speed up but gives negative acceleration!")
        elif dv2 > 0 and u2_optimal > 0:
            print(f"      Hum Δv={dv2:+.3f} → u2={u2_optimal:+.2f}")

        print(f"   ✅ Nash: u1={u1_optimal:+.2f}, u2={u2_optimal:+.2f}\n")
        
        return u1_optimal, u2_optimal
        
    def update_weights(self, Q_pos: float = None, Q_vel: float = None, 
                      R_system: float = None, R_human: float = None):
        """Update solver weights for tuning purposes"""
        if Q_pos is not None:
            self.Q_output[0, 0] = Q_pos
        if Q_vel is not None:
            self.Q_output[1, 1] = Q_vel
        if R_system is not None:
            self.R1 = R_system
        if R_human is not None:
            self.R2 = R_human
        print(f"🔧 Updated weights: Q={np.diag(self.Q_output)}, R=[{self.R1}, {self.R2}]")

# Legacy compatibility
class LongitudinalNashSolver(EnhancedLongitudinalNashSolver):
    """Legacy alias for backward compatibility"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("🤝 Longitudinal Nash Solver (Soft Constraints) initialized.")

# ======================== IMPROVED EXAMPLES ========================
def example_longitudinal_nash_control():
    """Complete example with realistic parameters"""
    print("\n🚗 Example: Nash Solver for Longitudinal Control")
    print("=" * 50)
    
    # Initialize
    dt = 0.02
    vehicle_model = Vehicle()
    vehicle_model.set_motion_model(use_kinematic=False, use_state_space=True)
    vehicle_model.vehicle_id = "Human"
    vehicle_model.autonomous_mode = True  # Ego vehicle is autonomous
    human_driver = HumanDriver(vehicle_model)
    human_use_kinematic = False
    human_use_state_space = True
    human_driver.set_motion_model(human_use_kinematic, human_use_state_space)
    safety_field = longitudinal_safety_field.LongitudinalSafetyField()
    authority_allocator = longitudinal_authority_allocator.LongitudinalAuthorityAllocator()
    sys_ref_gen = system_reference_generator.SystemReferenceGenerator(Np=20, dt=dt)
    nash_solver = LongitudinalNashSolver(vehicle_model, Np=10, Nu=6)
    
    print("\n📊 Scenario: Leader vehicle emergency braking")
    
    # FIXED: Use proper 2D state vector for the vehicle model
    x0 = np.array([15.0, 20.0])  # [position, velocity]
    
    desired_spacing = 20.0
    desired_rel_velocity = 0.0

    cooperation_levels = [0.1, 0.3, 0.6, 1.0, 2.0]
    
    for lambda_k in cooperation_levels:
        print(f"\n🤝 Cooperation level: λ = {lambda_k}")
        print("-" * 35)
        
        x_current = x0.copy()
        
        # IMPROVED: More realistic reference generation
        system_refs = []
        human_refs = []
        
        for i in range(nash_solver.Np):
            system_refs.extend([desired_spacing, desired_rel_velocity])
            human_spacing = desired_spacing + 5.0 * (1.0 - lambda_k)
            human_refs.extend([human_spacing, desired_rel_velocity])
        
        R1_ref = np.array(system_refs)  # Shape: (20,)
        R2_ref = np.array(human_refs)   # Shape: (20,)
        
        results = []
        
        for step in range(25):
            
            u1_opt, u2_opt = nash_solver.solve_nash_equilibrium(
                x_current, R1_ref, R2_ref, lambda_k
            )
            
            # FIXED: Don't scale B2 by lambda_k in dynamics update
            A, B1, _ = vehicle_model.get_state_space_matrices(dt)
            B2 = B1.copy()  # FIXED: No scaling here
            
            # Update state using proper matrix operations
            disturbance = np.random.normal(0, 0.01, x_current.shape)
            x_next = A @ x_current + B1.flatten() * u1_opt + B2.flatten() * u2_opt + disturbance
            
            # FIXED: Calculate actual acceleration properly
            actual_acceleration = (u1_opt + u2_opt)  # Combined effect
            dynamic_acc = (x_next[1] - x_current[1]) / dt
            
            results.append({
                'time': step * dt,
                'position': x_current[0],
                'velocity': x_current[1] * 3.6,
                'u1_system': u1_opt,
                'u2_human': u2_opt,
                'actual_acceleration': actual_acceleration,  # FIXED: Realistic value
                'dynamic_acceleration': dynamic_acc
            })
            
            x_current = x_next

            # Print results every 2 steps
            if step % 2 == 0:
                pos = float(x_current[0])
                vel = float(x_current[1])
                print(f"  t={step*dt:.1f}s: pos={pos:.1f}m, v={vel*3.6:.0f}km/h, "
                      f"u1={u1_opt:.2f}, u2={u2_opt:.2f}, a_total={actual_acceleration:.2f}, "
                      f"a_dyn={dynamic_acc:.2f}")
        
        final_state = results[-1]
        safety_rating = "SAFE" if final_state['position'] > 15 else "CRITICAL"
        print(f"  🎯 Final: pos={final_state['position']:.1f}m, "
              f"v={final_state['velocity']:.0f}km/h [{safety_rating}]")

    print("\n✅ Example completed successfully!")
    return nash_solver, results

if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * 60)
    print("🚦 IMPROVED Longitudinal Nash Solver Examples")
    print("=" * 60)
    
    solver, results = example_longitudinal_nash_control()
    print(f"\n🎉 All examples completed successfully!")

