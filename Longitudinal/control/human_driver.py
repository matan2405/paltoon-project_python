"""
Human driver module containing the HumanDriver class.
Simulates human driving behavior with motion model selection.
"""

import numpy as np
from typing import List, Tuple
import sys
import copy
import os
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Example usage
from vehicle.vehicle import Vehicle  # Assuming Vehicle class is defined in vehicle/vehicle.py
class HumanDriver:
    """Human driver model with motion model selection"""
    
    def __init__(self, vehicle,target_speed: float = 120.0 / 3.6, dt: float = 0.02):
        """Initialize human driver with vehicle and target speed."""
        self.vehicle = vehicle
        self.target_speed = target_speed  # target speed in m/s  
        self.max_acceleration = 2.0  # m/s²
        self.max_velocity = 250.0  # m/s
        self.lane_change_progress = 0.0
        self.dt=dt
        self.delta_IDM = 2.0  # IDM acceleration exponent
        self.merging = False
        # print(f"Human driver target speed set to: {self.target_speed * 3.6:.1f} km/h")
        
    def set_motion_model(self, use_kinematic: bool, use_state_space: bool = False):
        """Set motion model for the human vehicle"""
        self.vehicle.set_motion_model(use_kinematic, use_state_space)
        
    def update(self, dt: float, platoon_vehicles: List):
        """Update human driver inputs"""
        # Improved speed control with smoother transitions
        speed_error = self.target_speed - self.vehicle.state.vx
        speed_error_rel = speed_error / self.target_speed
        
        # Calculate desired acceleration based on speed error
        # Limit acceleration to realistic values
        max_accel = 5.0  # m/s²
        max_decel = -3.0  # m/s²
        delta = 2.5  # exponent for smooth approach to target speed
        if (self.target_speed >= self.vehicle.state.vx):
            desired_accel = max_accel * (1 - (self.vehicle.state.vx/self.target_speed)**self.delta_IDM)
        else:
            desired_accel = max_decel * (1 - (self.target_speed/self.vehicle.state.vx)**self.delta_IDM)
        
        # Convert acceleration to throttle/brake smoothly
        if desired_accel > 0.1:
            # Accelerate smoothly
            throttle = min(0.8, desired_accel / max_accel * 0.8)
            brake = 0.0
        elif desired_accel < -0.1:
            # Brake smoothly
            throttle = 0.0
            brake = min(0.6, abs(desired_accel) / abs(max_decel) * 0.6)
        else:
            # Maintain speed with minimal input
            throttle = 0.1  # Small throttle to maintain speed
            brake = 0.0
        
        # Simple lane changing logic
        steering = 0.0
        if self.merging and self.lane_change_progress < 1.0:
            # More gradual lane change for smooth merging
            # steering = 0.15 * np.sin(self.lane_change_progress * np.pi)
            self.lane_change_progress += dt * 0.25  # Take 4 seconds to change lanes
        
        self.vehicle.set_manual_inputs(throttle, brake, steering)
        
        return desired_accel
        

    def get_human_acceleration_and_state_sequence(self, dt: float, Np: int, vehicle: Vehicle) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get complete acceleration sequence.
        
        Multi-rate: dt is control time step (0.1s), but we simulate with finer dt_sim
        """
        accel_sequence = np.zeros(Np)
        state_sequence = np.zeros((Np, 2))  # Store [position, velocity] for each step
        
        vehicle_copy = copy.deepcopy(vehicle)
        human_driver_copy = copy.deepcopy(self)
        human_driver_copy.vehicle = vehicle_copy
        
        # ⚡ Multi-rate: calculate substeps
        dt_sim = 0.02  # Simulation time step (from vehicle)
        substeps_per_control = max(1, int(np.round(dt / dt_sim)))
        
        if not vehicle.joined_platoon:
            for step in range(Np):
                accel_sequence[step] = self.target_speed - vehicle.state.vx
                state_sequence[step, :] = [vehicle.state.x, vehicle.state.vx]
            return accel_sequence, state_sequence
        
        for step in range(Np):
            # Get human's intended acceleration
            accel_human = human_driver_copy.update(dt, platoon_vehicles=[])
            human_driver_copy.vehicle.a_desired = accel_human
            # Run multiple simulation substeps
            for substep in range(substeps_per_control):
                vehicle_copy.update_dynamics(dt_sim)
            
            # vehicle_copy.update_dynamics(dt)
            
            # Save state after control period
            accel_sequence[step] = accel_human
            state_sequence[step, :] = [vehicle_copy.state.x, vehicle_copy.state.vx]
        
        return accel_sequence, state_sequence 
    
    def get_human_acceleration_and_state_prediction(self, dt: float, Np: int, vehicle: Vehicle) -> np.ndarray:
        """Get final predicted acceleration."""
        return self.get_acceleration_and_state_sequence(dt=dt, Np=Np, vehicle=vehicle)[-1] if vehicle.joined_platoon else None, None


__all__ = ['HumanDriver']

if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    print("🚗 Testing Human Driver Model - Step-by-Step Prediction Accuracy")
    
    dt = 0.1
    simulation_time = 5.0  # 5 seconds simulation
    
    # Create vehicle and driver
    vehicle = Vehicle()
    driver = HumanDriver(vehicle)
    driver.set_motion_model(use_kinematic=False)
    vehicle.joined_platoon = True
    
    print("Step-by-Step Prediction vs Reality:")
    print("Time\t| Speed\t\t| Current Accel\t| Predicted Next\t| Actual Next\t| Error\t\t| Error %")
    print("-" * 100)
    
    prediction_errors = []
    prev_predicted_next = None
    
    for i, t in enumerate(np.arange(0, simulation_time, dt)):
        # Get current acceleration
        current_accel = driver.update(dt, platoon_vehicles=[])
        current_speed = vehicle.state.vx
        
        # Make prediction for NEXT step (1-step ahead)
        # Create a copy of current state for prediction
        vehicle_copy = copy.deepcopy(vehicle)
        driver_copy = copy.deepcopy(driver)
        driver_copy.vehicle = vehicle_copy
        
        # Predict what will happen in next step
        predicted_accel, predicted_state = driver_copy.get_acceleration_and_state_sequence(dt=dt, Np=10, vehicle=vehicle_copy)
        predicted_next_accel, state_next = predicted_accel[1], predicted_state[1]  # Get the next step prediction
        # Update the real vehicle
        vehicle.update_dynamics(dt)
        
        # Compare with previous prediction if available
        if prev_predicted_next is not None:
            error = current_accel - prev_predicted_next
            error_percent = (abs(error) / abs(prev_predicted_next) * 100) if abs(prev_predicted_next) > 1e-6 else 0
            prediction_errors.append(abs(error))
            
            print(f"{t:.1f}s\t| {current_speed:.2f} m/s\t| {current_accel:.3f} m/s²\t| {prev_predicted_next:.3f} m/s²\t\t| {current_accel:.3f} m/s²\t| {error:.6f}\t| {error_percent:.2f}%")
        else:
            print(f"{t:.1f}s\t| {current_speed:.2f} m/s\t| {current_accel:.3f} m/s²\t| --\t\t\t| {current_accel:.3f} m/s²\t| --\t\t| --")
        
        # Store prediction for next iteration
        prev_predicted_next = predicted_next_accel
    
    # Final statistics
    if prediction_errors:
        print(f"\n📊 Prediction Accuracy Statistics:")
        print(f"Mean Absolute Error: {np.mean(prediction_errors):.6f} m/s²")
        print(f"Max Error: {np.max(prediction_errors):.6f} m/s²")
        print(f"Min Error: {np.min(prediction_errors):.6f} m/s²")
        print(f"Standard Deviation: {np.std(prediction_errors):.6f} m/s²")
        print(f"RMS Error: {np.sqrt(np.mean(np.array(prediction_errors)**2)):.6f} m/s²")
    
    # Additional test: Test prediction consistency at different speeds
    print(f"\n🎯 Testing prediction accuracy at different speeds:")
    print("Speed\t\t| Current Accel\t| Predicted Accel\t| Error\t\t| Error %")
    print("-" * 75)
    
    test_speeds = [5, 10, 15, 20, 25]  # Different speeds to test
    
    for target_speed in test_speeds:
        # Create new vehicle and run to target speed
        test_vehicle = Vehicle()
        test_driver = HumanDriver(test_vehicle, target_speed=target_speed)
        test_driver.set_motion_model(use_kinematic=True)
        test_vehicle.joined_platoon = True
        
        # Run until close to target speed
        while test_vehicle.state.vx < target_speed * 0.8:
            test_driver.update(dt, platoon_vehicles=[])
            test_vehicle.update_dynamics(dt)
        
        # Test prediction accuracy at this speed
        current_speed = test_vehicle.state.vx
        current_accel = test_driver.update(dt, platoon_vehicles=[])
        
        # Make prediction
        predicted_accel, predicted_state = test_driver.get_acceleration_and_state_sequence(dt=dt, Np=10, vehicle=test_vehicle)
        predicted_accel = predicted_accel[1]  # Immediate next acceleration prediction
        predicted_state = predicted_state[1]  # Immediate next state prediction
        # Get actual next acceleration
        test_vehicle.update_dynamics(dt)
        actual_next_accel = test_driver.update(dt, platoon_vehicles=[])
        
        error = predicted_accel - actual_next_accel
        error_percent = (abs(error) / abs(actual_next_accel) * 100) if abs(actual_next_accel) > 1e-6 else 0

        print(f"{current_speed:.2f} m/s\t| {current_accel:.3f} m/s²\t| {predicted_accel:.3f} m/s²\t\t| {error:.6f}\t| {error_percent:.2f}%")
    
    # Test: Prediction accuracy over different time horizons
    print(f"\n🔮 Testing prediction accuracy over different horizons:")
    print("Horizon\t| Speed\t\t| Predicted\t| Actual\t| Error\t\t| Error %")
    print("-" * 75)
    
    base_vehicle = Vehicle()
    base_driver = HumanDriver(base_vehicle)
    base_driver.set_motion_model(use_kinematic=True)
    base_vehicle.joined_platoon = True
    
    # Run to steady state
    for t in np.arange(0, 2, dt):
        base_driver.update(dt, platoon_vehicles=[])
        base_vehicle.update_dynamics(dt)
    
    for horizon in [1, 2, 3, 5, 10]:
        # Create copy for prediction
        vehicle_pred = copy.deepcopy(base_vehicle)
        driver_pred = copy.deepcopy(base_driver)
        driver_pred.vehicle = vehicle_pred
        
        # Make prediction
        predictions = driver_pred.get_acceleration_and_state_sequence(dt=dt, Np=horizon, vehicle=vehicle_pred)
        predicted_accel = predictions[0][-1]
        predicted_state = predictions[1][-1]

        # Get actual result
        vehicle_actual = copy.deepcopy(base_vehicle)
        driver_actual = copy.deepcopy(base_driver)
        driver_actual.vehicle = vehicle_actual
        
        for step in range(horizon-1):
            driver_actual.update(dt, platoon_vehicles=[])
            vehicle_actual.update_dynamics(dt)
        
        actual_accel = driver_actual.update(dt, platoon_vehicles=[])
        
        error = predicted_accel - actual_accel
        error_percent = (abs(error) / abs(actual_accel) * 100) if abs(actual_accel) > 1e-6 else 0
        
        print(f"{horizon} step\t| {base_vehicle.state.vx:.2f} m/s\t| {predicted_accel:.3f} m/s²\t| {actual_accel:.3f} m/s²\t| {error:.6f}\t| {error_percent:.2f}%")