"""
Simulation module containing the main PlatoonSimulation class.
Manages the overall simulation state, data logging, and scenario execution.
"""

import numpy as np
import time
import sys
import os
from typing import List, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SIMULATION_DT, DEFAULT_SIMULATION_TIME
from vehicle.vehicle import Vehicle
from control.platoon_control import PlatoonManager
from control.human_driver import HumanDriver


class PlatoonSimulation:
    """Main simulation class"""

    def __init__(self,Np=None,initial_x_platoon=None,initial_velocity_platoon=None, num_cars_platoon=4, target_speed_platoon=120.0 / 3.6, initial_x_human=0.0, initial_velocity_human=0.0, use_state_space: bool = True, driver_type='normal'):
        # Simulation parameters - match platoon_control.py exactly
        self.dt = SIMULATION_DT  # 0.02s timestep like platoon_control.py
        self.time = 0.0
        self.start_time = time.time()
        self.running = True
        if Np is None:
            self.T_sim = DEFAULT_SIMULATION_TIME  # simulation time [s]
        else:
            self.T_sim = Np*self.dt
        self.num_cars_platoon = num_cars_platoon # Set default number of cars for initialization
        # num_cars = 4  # like platoon_control.py
        if initial_x_platoon is None:
            self.initial_x_platoon = [-i * 15 for i in range(self.num_cars_platoon)]  # initial gap between platoon vehicles [m]
        else:
            self.initial_x_platoon = initial_x_platoon
        if initial_velocity_platoon is None:
            self.initial_velocity_platoon = [0 for _ in range(self.num_cars_platoon)]  # initial velocity for all vehicles [m/s]
        else:
            self.initial_velocity_platoon = initial_velocity_platoon

        self.is_prediction_mode = False  # Placeholder for prediction mode
        
        # Human vehicle initial conditions
        self.driver_type = driver_type
        self.initial_x_human = initial_x_human
        self.initial_velocity_human = initial_velocity_human
        self.use_state_space = use_state_space  # Whether to use state-space model for platoon vehicles
        # Data logging
        self.time_history = []
        self.position_history = []
        self.velocity_history = []
        self.gap_history = []
        self.desired_gap_history = []
        self.acceleration_history = []
        
        # Human vehicle distance tracking (for animation)
        self.human_distance_history = {}
        self.merge_start_time = None
        
        # Track vehicle indices for consistent plotting
        self.vehicle_indices = {}
        self.target_speed_platoon = target_speed_platoon
        
        # Create vehicles
        self.setup_vehicles()

    def setup_vehicles(self):
        """Initialize vehicles and managers"""
        # Platoon vehicles - exactly like platoon_control.py
        self.platoon_vehicles = []
        for i in range(self.num_cars_platoon):
            vehicle = Vehicle(initial_x=self.initial_x_platoon[i], initial_y=0.0, vehicle_id=f"Platoon_{i+1}", initial_velocity=self.initial_velocity_platoon[i])
            vehicle.autonomous_mode = True
            vehicle.use_state_space_model = self.use_state_space  # Use state-space bicycle model if specified
            vehicle.use_kinematic_model = not self.use_state_space  # Platoon vehicles always use kinematic model
            vehicle.use_kinematic = not self.use_state_space
            vehicle.target_velocity = self.target_speed_platoon  # Set target velocity for platoon vehicles
            self.platoon_vehicles.append(vehicle)
        
        # Human vehicle configuration - SET THIS TO CHOOSE MOTION MODEL
        human_use_kinematic = not self.use_state_space  # Set to True for kinematic model, False for complex dynamics
        human_use_state_space = self.use_state_space  # Set to True for state-space bicycle model

        # Human vehicle (left lane)
        self.human_vehicle = Vehicle(initial_x=self.initial_x_human, initial_y=-2, vehicle_id="Human", initial_velocity=self.initial_velocity_human)
        
        # Create human driver and set motion model
        self.human_driver = HumanDriver(self.human_vehicle, driver_type=self.driver_type)
        self.human_driver.set_motion_model( human_use_kinematic, human_use_state_space)
        
        # Managers
        self.platoon_manager = PlatoonManager(self.platoon_vehicles,human_use_state_space)
        
        # Print configuration
        if human_use_state_space:
            motion_type = "state-space bicycle model"
        else:
            motion_type = "kinematic" if human_use_kinematic else "complex dynamics"
        print(f"Human vehicle using: {motion_type} model")
        
        
        # Initialize vehicle tracking for consistent plotting
        self.all_vehicles = self.platoon_vehicles + [self.human_vehicle]
        for i, vehicle in enumerate(self.all_vehicles):
            self.vehicle_indices[vehicle.vehicle_id] = i
        
    def update(self):
        """Single simulation step"""
        # Update platoon using exact platoon_control.py algorithms
        self.platoon_manager.update(self.dt, self.is_prediction_mode)
        
        
        # Update human vehicle dynamics (only if not in platoon)
        if not self.human_vehicle.joined_platoon:
            # Update human driver
            human_acc = self.human_driver.update(self.dt, self.platoon_vehicles)
            self.human_vehicle.a_desired = human_acc
            self.human_vehicle.update_dynamics(self.dt)
        else:
            # If joined platoon, match platoon kinematic update
            human_acc = self.human_vehicle.a_desired
        # Log data
        self.log_data()
        
        # Update time
        self.time += self.dt
        return human_acc if self.human_vehicle.joined_platoon else None
    
    def log_data(self):
        """Log simulation data with consistent vehicle ordering"""
        self.time_history.append(self.time)
        
        # Create ordered vehicle list for consistent plotting
        all_current_vehicles = self.platoon_manager.vehicles.copy()
        if not self.human_vehicle.joined_platoon:
            all_current_vehicles.append(self.human_vehicle)
        
        positions = []
        velocities = []
        accelerations = []
        
        # Log data in original order for consistency
        for vehicle in self.all_vehicles:
            if vehicle in all_current_vehicles:
                positions.append([vehicle.state.x, vehicle.state.y])
                velocities.append(vehicle.state.vx * 3.6)  # Convert to km/h
                accelerations.append(vehicle.state.ax)
            else:
                # Keep placeholder for removed vehicles to maintain indices
                positions.append([float('nan'), float('nan')])
                velocities.append(float('nan'))
                accelerations.append(float('nan'))
            
        self.position_history.append(positions)
        self.velocity_history.append(velocities)
        self.acceleration_history.append(accelerations)
        
        # Track human vehicle distance to closest platoon vehicle (only during merging attempt)
        if not hasattr(self, 'human_closest_distance_history'):
            self.human_closest_distance_history = []
        
        # Only track distance when the human is attempting to merge
        if self.human_driver.merging:
            # Mark the start of merging attempt if not already marked
            if self.merge_start_time is None:
                self.merge_start_time = self.time
                
            # Calculate distance from human vehicle to closest platoon vehicle
            human_pos = self.human_vehicle.state.x
            min_distance = float('inf')
            closest_vehicle_id = None
            
            for platoon_vehicle in self.platoon_manager.vehicles:
                distance = abs(human_pos - platoon_vehicle.state.x)
                if distance < min_distance:
                    min_distance = distance
                    closest_vehicle_id = platoon_vehicle.vehicle_id
            
            # Store the minimum distance for animation plotting
            if min_distance < float('inf'):
                self.human_closest_distance_history.append(min_distance)
                # Also store in the dictionary format that the animation expects
                self.human_distance_history[self.time] = min_distance
            else:
                self.human_closest_distance_history.append(None)
        else:
            # Before merging attempt - don't track distance
            self.human_closest_distance_history.append(None)
        
        # Store platoon gaps from PlatoonManager (including human when part of platoon)
        current_gaps = []
        current_desired_gaps = []
        
        if len(self.platoon_manager.vehicles) > 1:
            # Get all platoon vehicles (including human if joined) sorted by position
            all_platoon_vehicles = sorted(self.platoon_manager.vehicles, key=lambda v: v.state.x, reverse=True)
            
            # Calculate gaps between consecutive vehicles in the platoon
            for i in range(len(all_platoon_vehicles) - 1):
                leader = all_platoon_vehicles[i]
                follower = all_platoon_vehicles[i + 1]
                actual_gap = leader.state.x - follower.state.x
                h = 1.5  # Desired time headway
                desired_gap = leader.L + h * follower.state.vx  # Default desired gap

                current_gaps.append(actual_gap)
                current_desired_gaps.append(desired_gap)
        
        # Store the gaps if we have any
        if current_gaps:
            self.gap_history.append(current_gaps)
            self.desired_gap_history.append(current_desired_gaps)
        else:
            # If no gaps (single vehicle or empty platoon), append empty lists to maintain consistency
            self.gap_history.append([])
            self.desired_gap_history.append([])
        
    def get_vehicle_positions(self) -> List[Tuple[float, float]]:
        """Get current vehicle positions for visualization"""
        return [(v.state.x, v.state.y) for v in self.all_vehicles]
    
    def get_vehicle_velocities(self) -> List[float]:
        """Get current vehicle velocities in km/h"""
        return [v.state.vx * 3.6 for v in self.all_vehicles]
    
    def print_status(self):
        """Print current simulation status"""
        print(f"\n⏱️Time: {self.time:.1f}s")
        for vehicle in self.all_vehicles:
            if vehicle.use_state_space_model:
                model_type = "state-space"
            elif vehicle.use_kinematic_model:
                model_type = "kinematic"
            else:
                model_type = "complex"
            
            # Determine more accurate status for human vehicle
            if vehicle.vehicle_id == "Human":
                if vehicle.joined_platoon:
                    if self.human_driver.merging and self.human_driver.lane_change_progress < 1.0:
                        status = "joining platoon"
                    else:
                        status = "platoon member"
                elif self.human_driver.merging:
                    status = "joining platoon"
                else:
                    status = "independent"
            else:
               # Platoon vehicles are always part of the platoon
                status = "platoon"
                
            print(f"{vehicle.vehicle_id} ({model_type}, {status}): "
                  f"Pos=({vehicle.state.x:.1f}, {vehicle.state.y:.1f}), "
                  f"Speed={vehicle.state.vx*3.6:.1f} km/h",f"acceleration={vehicle.state.ax:.2f} m/s^2")

def run_simulation():
    """Run the basic simulation"""
    print("Creating simulation...")
    sim = PlatoonSimulation()
    print("Simulation created successfully")
    
    # Run simulation for fixed duration - matching platoon_control.py
    max_time = sim.T_sim  # 180 seconds
    T = np.arange(0, max_time, sim.dt)
    max_iterations = len(T)
    print_interval = 20.0  # Print every 20 seconds
    next_print_time = 0.0
    iteration = 0
    human_joined = False
    
    print(f"Starting simulation for {max_time} seconds, {max_iterations} iterations")
    print(f"dt = {sim.dt}, target velocity = {sim.platoon_manager.target_velocity * 3.6:.1f} km/h")
    
    try:
        for t in T:
            if iteration % 1000 == 0:  # Print every 1000 iterations
                progress = (iteration / max_iterations) * 100
                print(f"Progress: {progress:.1f}% (t={sim.time:.1f}s)")
                
            human_acc=sim.update()
            iteration += 1
            
            # Print status periodically
            if sim.time >= next_print_time:
                sim.print_status()
                next_print_time += print_interval
                if human_joined:
                    print(f"human acceleration at {sim.time:.1f}s: {human_acc}")
                
            # Trigger lane change for human driver
            if sim.time >= 20.0 and not human_joined and not sim.human_driver.merging:
                sim.human_driver.merging = True
                human_joined = True
                sim.human_vehicle.joined_platoon = True  # Update status immediately
                print(f"🚗 Human vehicle starting to join platoon at t={sim.time:.1f}s")
                print(f"human vehicle position before lane change: x={sim.human_vehicle.state.x:.1f}")
                print(f"platoon vehicles positions before lane change: {[f'x={v.state.x:.1f}' for v in sim.platoon_vehicles]}")
                sim.platoon_manager.add_vehicle(sim.human_vehicle)
                sim.human_vehicle.target_velocity=sim.platoon_manager.target_velocity
                print(f"Human driver starting lane change at t={sim.time:.1f}s")
            
                
    except Exception as e:
        print(f"Error during simulation at iteration {iteration}: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nSimulation completed after {sim.time:.1f} seconds, {iteration} iterations")
    return sim


__all__ = ['PlatoonSimulation', 'run_simulation']

if __name__ == "__main__":
    sim = run_simulation()
    # Optionally, save or plot results here
    # For example, print final positions
    print("\nFinal vehicle positions:")
    for vehicle in sim.all_vehicles:
        print(f"{vehicle.vehicle_id}: x={vehicle.state.x:.1f} m, y={vehicle.state.y:.1f} m, speed={vehicle.state.vx*3.6:.1f} km/h")