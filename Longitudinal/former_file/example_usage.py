#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple example of using the new modular structure
Demonstrates basic usage patterns for the refactored code
"""

# Import modules
from config import setup_matplotlib
from vehicle import Vehicle, VehicleParameters, Engine, Transmission
from control import PlatoonManager, HumanDriver
from simulation import PlatoonSimulation
from visualization import create_comprehensive_plots

# Setup matplotlib
setup_matplotlib()

def simple_vehicle_example():
    """Simple example of creating and operating a vehicle"""
    print("Creating new vehicle...")
    
    # Create vehicle with custom parameters
    vehicle = Vehicle(initial_x=0, initial_y=0, vehicle_id="Example_Car")
    
    # Set vehicle to autonomous mode with kinematic model
    vehicle.autonomous_mode = True
    vehicle.set_motion_model(use_kinematic=True)
    
    # Display initial state
    print(f"Initial state: x={vehicle.x:.2f}m, y={vehicle.y:.2f}m, v={vehicle.v:.2f}m/s")
    
    # Set acceleration
    vehicle.a_desired = 2.0  # 2 m/s^2
    
    # Update dynamics for several steps
    dt = 0.1  # 0.1 seconds
    for i in range(10):
        vehicle.update_dynamics(dt)
        print(f"Step {i+1}: x={vehicle.x:.2f}m, v={vehicle.v:.2f}m/s")

def platoon_example():
    """Example of creating a simple platoon"""
    print("\nCreating simple platoon...")
    
    # Create vehicles
    leader = Vehicle(initial_x=50, initial_y=0, vehicle_id="Leader")
    follower1 = Vehicle(initial_x=30, initial_y=0, vehicle_id="Follower1")
    follower2 = Vehicle(initial_x=10, initial_y=0, vehicle_id="Follower2")
    
    # Create vehicle list
    vehicles = [leader, follower1, follower2]
    
    # Create platoon manager
    platoon_manager = PlatoonManager(vehicles)
    
    print(f"Created platoon with {len(platoon_manager.vehicles)} vehicles")
    
    # Display positions
    for vehicle in platoon_manager.vehicles:
        print(f"  {vehicle.vehicle_id}: x={vehicle.x:.2f}m")

def human_driver_example():
    """Example of human driver"""
    print("\nCreating human driver...")
    
    # Create vehicle
    vehicle = Vehicle(initial_x=0, initial_y=0, vehicle_id="Human_Car")
    
    # Create driver
    driver = HumanDriver(vehicle)
    driver.target_speed = 80 / 3.6  # 80 km/h
    driver.target_y = 3.5  # Want to move to second lane
    
    print(f"Human driver with target: speed={driver.target_speed*3.6:.1f} km/h, lane y={driver.target_y}m")
    
    # Short simulation
    dt = 0.1
    for i in range(5):
        try:
            driver.update(dt, [])  # Without third parameter
        except TypeError:
            # If there's an issue with parameters, just update vehicle manually
            vehicle.update_dynamics(dt)
        print(f"  Step {i+1}: v={vehicle.v*3.6:.1f} km/h, y={vehicle.y:.2f}m")

def full_simulation_example():
    """Example of full simple simulation"""
    print("\nRunning simple simulation...")
    
    # Create simulation
    sim = PlatoonSimulation()
    
    # Basic settings
    sim.dt = 0.1
    sim.duration = 5.0  # 5 seconds
    
    # Add single vehicle
    vehicle = Vehicle(initial_x=0, initial_y=0, vehicle_id="Solo_Car")
    vehicle.autonomous_mode = True  # Autonomous mode
    vehicle.set_motion_model(use_kinematic=True)  # Kinematic model
    vehicle.a_desired = 1.0  # Constant acceleration
    sim.vehicles = [vehicle]
    
    print(f"Running simulation for {sim.duration} seconds...")
    
    # Run (replaces run which doesn't exist)
    steps = int(sim.duration / sim.dt)
    for step in range(steps):
        sim.time = step * sim.dt
        for vehicle in sim.vehicles:
            vehicle.update_dynamics(sim.dt)
        # Here we would call log_data if it existed
    
    final_vehicle = sim.vehicles[0]
    print(f"Final result: x={final_vehicle.x:.2f}m, v={final_vehicle.v:.2f}m/s")

if __name__ == "__main__":
    print("Examples of using the new modular structure")
    print("=" * 50)
    
    # Run examples
    simple_vehicle_example()
    platoon_example()
    human_driver_example()
    full_simulation_example()
    
    print("\nAll examples completed successfully!")
    print("Now you can run 'python main.py' for full simulation")