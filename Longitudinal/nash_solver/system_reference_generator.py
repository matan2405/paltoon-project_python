#!/usr/bin/env python3
"""
File: system_reference_generator.py  
Description: Predicts human vehicle acceleration over Np steps using convoy simulation.
Multi-rate: Uses control dt (0.1s), not simulation dt (0.02s)
"""
from typing import Optional, Tuple
import numpy as np
import sys
import os
import copy

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.simulator import PlatoonSimulation, run_simulation

class SystemReferenceGenerator:
    """Predicts human vehicle acceleration by running convoy simulation."""
    
    def __init__(self, Np: int = 20, dt: float = 0.1):  # ✅ dt=0.1 for control!
        self.Np = Np
        self.dt = dt  # Control time step
        print(f"📊 System Reference Generator: dt={self.dt:.3f}s, Np={self.Np}")

    def create_prediction_simulation(self, current_sim: PlatoonSimulation) -> PlatoonSimulation:
        """Create a deep copy of current simulation for prediction."""
        try:
            # Deep copy preserves everything
            pred_sim = copy.deepcopy(current_sim)
            
            # ⚡ CRITICAL: Keep original simulation dt (0.02)
            # We'll run multiple sim steps per control step
            pred_sim.dt = current_sim.dt  # ✅ Keep 0.02 for accurate physics
            
            return pred_sim
        except Exception as e:
            print(f"Deep copy failed: {e}")
            return self._manual_copy(current_sim)
    
    def _manual_copy(self, current_sim: PlatoonSimulation) -> PlatoonSimulation:
        """Manual copy as fallback."""
        platoon_positions = [v.state.x for v in current_sim.platoon_vehicles]
        platoon_velocities = [v.state.vx for v in current_sim.platoon_vehicles]
        
        pred_sim = PlatoonSimulation(
            Np=self.Np,
            initial_x_platoon=platoon_positions.copy(),
            initial_velocity_platoon=platoon_velocities.copy(),
            num_cars_platoon=len(current_sim.platoon_vehicles),
            initial_x_human=current_sim.human_vehicle.state.x,
            initial_velocity_human=current_sim.human_vehicle.state.vx,
            use_state_space=getattr(current_sim, 'use_state_space', False)
        )
        
        pred_sim.time = current_sim.time
        pred_sim.human_vehicle.joined_platoon = current_sim.human_vehicle.joined_platoon
        
        if current_sim.human_vehicle.joined_platoon:
            pred_sim.human_driver.merging = True
            pred_sim.platoon_manager.add_vehicle(pred_sim.human_vehicle)
        
        return pred_sim

    def predict_system_acceleration_and_state_sequence(self, current_simulation: PlatoonSimulation) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Predict system acceleration sequence for Np steps.

        Multi-rate architecture:
        - Each prediction step = dt_control (0.1s)
        - Each step runs multiple simulation substeps (dt_sim = 0.02s)
        """
        if not current_simulation.human_vehicle.joined_platoon:
            return np.zeros(self.Np), np.zeros((self.Np, 2)), False
            
        # 1. Copy convoy  
        pred_sim = self.create_prediction_simulation(current_simulation)
        pred_sim.is_prediction_mode = True
        
        # ⚡ Calculate how many simulation steps per control step
        dt_sim = pred_sim.dt  # 0.02s
        dt_control = self.dt  # 0.1s
        substeps_per_control = int(np.round(dt_control / dt_sim))  # 5 substeps
        
        print(f"   🔄 Multi-rate: {substeps_per_control} sim steps per control step")
        
        # 2. Predict Np steps with multi-rate
        acceleration_sequence = np.zeros(self.Np)
        state_sequence = np.zeros((self.Np, 2))  # Store position and velocity
        
        for step in range(self.Np):
            # Run substeps to advance by dt_control
            for substep in range(substeps_per_control):
                pred_sim.update()  # Uses dt_sim internally
            
            # Save state after dt_control
            acceleration_sequence[step] = pred_sim.human_vehicle.a_desired
            state_sequence[step, :] = [
                pred_sim.human_vehicle.state.x, 
                pred_sim.human_vehicle.state.vx
            ]
        
        return acceleration_sequence, state_sequence, True

    def get_system_acceleration_and_state_prediction(self, simulation: PlatoonSimulation) -> Optional[float]:
        """Get final predicted acceleration."""
        accel_sequence, state_sequence, is_in_platoon = self.predict_system_acceleration_and_state_sequence(simulation)
        if not is_in_platoon:
            return None
        return accel_sequence[-1]

    def get_system_acceleration_and_state_sequence(self, simulation: PlatoonSimulation) -> Tuple[np.ndarray, np.ndarray]:
        """Get complete acceleration sequence."""
        accel_sequence, state_sequence, _ = self.predict_system_acceleration_and_state_sequence(simulation)
        return accel_sequence, state_sequence

__all__ = ['SystemReferenceGenerator']
# ================================
# Example usage in platoon control
# ================================

# Test functionality
if __name__ == "__main__":
    print("🧪 Testing Human Acceleration Prediction")
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Create simulation
    test_sim = PlatoonSimulation()
    generator = SystemReferenceGenerator(Np=10, dt=test_sim.dt)
    
    total_time = 60.0
    print_interval = 20.0  # Print every 20 seconds
    next_print_time = 0.0
    time_steps = int(total_time / test_sim.dt)
    prediction_interval = test_sim.dt  # Predict every dt seconds
    prediction_steps = int(prediction_interval / test_sim.dt)
    
    prediction_errors = []
    
    # Main simulation loop
    for step in range(time_steps):
        current_time = test_sim.time
        
        # Run simulation step
        human_acc = test_sim.update()
        
        # Join platoon at 20s
        if current_time >= 20 and not test_sim.human_vehicle.joined_platoon:
            test_sim.human_driver.merging = True
            test_sim.human_vehicle.joined_platoon = True
            test_sim.platoon_manager.add_vehicle(test_sim.human_vehicle)
            print(f"🔄 Time {current_time:.1f}s: Human joined platoon")
        
        # Run prediction every second (only if in platoon)
        if step % prediction_steps == 0 and test_sim.human_vehicle.joined_platoon:
            actual_accel = test_sim.human_vehicle.state.ax
            predicted_accel, predicted_state = generator.get_human_acceleration_and_state_prediction(test_sim)
            
            if predicted_accel is not None:
                error = abs(predicted_accel - actual_accel)
                prediction_errors.append(error)
               
         # Print status periodically
        if test_sim.time >= next_print_time:
            test_sim.print_status()
            next_print_time += print_interval
            if test_sim.human_vehicle.joined_platoon:
                print(f"human acceleration at {test_sim.time:.4f}s: {human_acc}")
                print(f"Predicted acceleration={predicted_accel:.4f}, "
                      f"Actual acceleration={actual_accel:.4f}, Error={error:.4f}")

    # Final results
    if prediction_errors:
        avg_error = np.mean(prediction_errors)
        print(f"\n📊 Average prediction error: {avg_error:.4f} m/s²")
        print(f"📈 Total predictions: {len(prediction_errors)}")
    
    print("✅ Test completed!")