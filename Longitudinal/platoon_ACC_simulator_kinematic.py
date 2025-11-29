import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Configure matplotlib for stable display with fallback
import matplotlib
try:
    # Force interactive mode - don't fall back to headless unless absolutely necessary
    matplotlib.use('Qt5Agg', force=True)
    import matplotlib.pyplot as plt
    plt.ion()  # Turn on interactive mode
    print("✅ Using Qt5Agg backend (interactive mode enabled)")
    HEADLESS_MODE = False
except ImportError as e:
    try:
        # Try TkAgg if Qt5Agg not available
        matplotlib.use('TkAgg', force=True)
        import matplotlib.pyplot as plt
        plt.ion()  # Turn on interactive mode
        print("✅ Using TkAgg backend (interactive mode enabled)")
        HEADLESS_MODE = False
    except ImportError:
        # Only use Agg as last resort
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
        print("⚠️ Using Agg backend (headless mode - plots will be saved only)")
        HEADLESS_MODE = True

# Create results directory
RESULTS_DIR = "platoon_sim_kinematic_results"
import os
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
    print(f"📁 Created results directory: {RESULTS_DIR}")
else:
    print(f"📁 Using existing results directory: {RESULTS_DIR}")

# Enable interactive mode for better display
matplotlib.pyplot.ion()

# Disable interactive mode to prevent threading issues
plt.ioff()

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import time
from scipy.integrate import odeint


@dataclass
class VehicleParameters:
    """Audi TT 2.0 TFSI parameters from Unity code"""
    # Vehicle dimensions
    mass: float = 1305.0  # kg
    length: float = 4.177  # m
    width: float = 1.832  # m
    height: float = 1.353  # m
    wheelbase: float = 2.505  # m
    track_width: float = 1.572  # m
    
    # Center of gravity distances
    lf: float = 1.2525  # distance from CG to front axle
    lr: float = 1.2525  # distance from CG to rear axle
    
    # Aerodynamics
    drag_coefficient: float = 0.30
    frontal_area: float = 2.09  # m^2
    
    # Tire parameters
    wheel_radius: float = 0.3175  # m (225/50 R17)
    tire_friction_coeff: float = 0.8
    
    # Cornering stiffness
    Caf: float = 15000.0  # Front tire cornering stiffness N/rad
    Car: float = 18000.0  # Rear tire cornering stiffness N/rad
    
    # Moment of inertia
    Iz: float = 2500.0  # kg⋅m^2
    
    # Performance
    max_velocity: float = 249.0 / 3.6  # m/s (249 km/h)
    max_acceleration: float = 2.5  # m/s^2
    
    # Steering
    max_steering_angle: float = np.radians(30.0)  # rad
    steering_ratio: float = 14.6
    
    # Constants
    gravity: float = 9.81

class Engine:
    """Simplified engine model based on Audi TT 2.0 TFSI"""
    def __init__(self):
        # Engine characteristics
        self.max_torque = 370.0  # Nm
        self.max_power_kw = 169.0  # kW
        self.idle_rpm = 800.0
        self.redline_rpm = 6700.0
        
        # Current state
        self.rpm = self.idle_rpm
        self.state = 2  # 0=off, 1=starting, 2=running
        
    def get_torque(self, rpm: float) -> float:
        """Calculate engine torque at given RPM"""
        if rpm < self.idle_rpm or rpm > self.redline_rpm:
            return 0.0
            
        # Simplified torque curve
        if rpm <= 1200:
            return np.interp(rpm, [self.idle_rpm, 1200], [180, 240])
        elif rpm <= 1600:
            return np.interp(rpm, [1200, 1600], [240, self.max_torque])
        elif rpm <= 4300:
            return self.max_torque
        elif rpm <= 6200:
            # Power limited region
            angular_velocity = (2 * np.pi * rpm) / 60
            return (self.max_power_kw * 1000) / angular_velocity
        else:
            # Power drops after peak
            power_mult = np.interp(rpm, [6200, self.redline_rpm], [1.0, 0.7])
            adjusted_power = self.max_power_kw * power_mult
            angular_velocity = (2 * np.pi * rpm) / 60
            return (adjusted_power * 1000) / angular_velocity
    
    def update_rpm(self, target_rpm: float, dt: float):
        """Update engine RPM with realistic dynamics"""
        rpm_rate = 2500.0 if target_rpm > self.rpm else 2000.0
        self.rpm = np.clip(
            self.rpm + np.sign(target_rpm - self.rpm) * rpm_rate * dt,
            self.idle_rpm, self.redline_rpm
        )

class Transmission:
    """Automatic transmission model"""
    def __init__(self):
        # Gear ratios from Unity code
        self.gear_ratios = [3.769, 2.087, 1.481, 1.152, 1.167, 0.970]
        self.final_drive_ratios = [3.238, 3.238, 3.238, 3.238, 2.615, 2.615]
        
        # Shift points
        self.upshift_rpm = [2800, 3200, 3600, 4000, 4400]
        self.downshift_rpm = [1400, 1600, 1800, 2000, 2200]
        
        self.current_gear = 0  # 0-based index
        self.is_shifting = False
        self.shift_timer = 0.0
        
    def get_total_ratio(self) -> float:
        """Get current total gear reduction"""
        return self.gear_ratios[self.current_gear] * self.final_drive_ratios[self.current_gear]
    
    def update(self, engine_rpm: float, vehicle_speed: float, dt: float):
        """Update transmission state and gear selection"""
        if self.is_shifting:
            self.shift_timer += dt
            if self.shift_timer >= 0.2:  # 200ms shift time
                self.is_shifting = False
                self.shift_timer = 0.0
            return
        
        # Check for upshift
        if (self.current_gear < len(self.gear_ratios) - 1 and 
            engine_rpm > self.upshift_rpm[self.current_gear]):
            self.current_gear += 1
            self.is_shifting = True
            
        # Check for downshift  
        elif (self.current_gear > 0 and 
              engine_rpm < self.downshift_rpm[self.current_gear - 1]):
            self.current_gear -= 1
            self.is_shifting = True

class VehicleState:
    """Vehicle state vector"""
    def __init__(self):
        # Position and orientation
        self.x = 0.0      # longitudinal position (m)
        self.y = 0.0      # lateral position (m)
        self.psi = 0.0    # yaw angle (rad)
        
        # Velocities
        self.vx = 0.0     # longitudinal velocity (m/s)
        self.vy = 0.0     # lateral velocity (m/s)
        self.psi_dot = 0.0  # yaw rate (rad/s)
        
        # Accelerations
        self.ax = 0.0     # longitudinal acceleration (m/s^2)
        self.ay = 0.0     # lateral acceleration (m/s^2)

class Vehicle:
    """Main vehicle class implementing bicycle model dynamics"""
    
    def __init__(self, initial_x: float = 0.0, initial_y: float = 0.0, 
                 initial_heading: float = 0.0, vehicle_id: str = "Vehicle",
                 initial_velocity: float = 0.0):
        self.params = VehicleParameters()
        self.engine = Engine()
        self.transmission = Transmission()
        self.state = VehicleState()
        self.vehicle_id = vehicle_id
        
        # Initialize position
        self.state.x = initial_x
        self.state.y = initial_y
        self.state.psi = initial_heading
        self.state.vx = initial_velocity
        
        # Control inputs
        self.direct_force = 0.0      # Direct longitudinal force [N]
        self.steering_input = 0.0    # -1 to 1
        
        # For compatibility with human driver
        self.throttle_input = 0.0    # 0 to 1
        self.brake_input = 0.0       # 0 to 1
        
        # Mode controls
        self.autonomous_mode = False
        self.use_kinematic_model = False  # False = complex dynamics, True = kinematic model
        self.target_velocity = 0.0
        self.target_acceleration = 0.0
        
        # Platoon compatibility variables
        self.L = 1.2 + 1.8  # wheelbase for platoon compatibility (l_f + l_r from platoon_control.py)
        self.v = initial_velocity  # velocity shorthand for platoon compatibility
        self.a = 0.0  # acceleration shorthand for platoon compatibility
        
        # Track if this vehicle joined the platoon
        self.joined_platoon = False
        self.joined_time=None
        
    def set_motion_model(self, use_kinematic: bool):
        """Set which motion model to use"""
        self.use_kinematic_model = use_kinematic
        print(f"{self.vehicle_id}: Motion model set to {'kinematic' if use_kinematic else 'complex dynamics'}")
        
    def calculate_forces(self) -> Tuple[float, float]:
        """Calculate longitudinal and lateral forces"""
        if self.autonomous_mode:
            # Use direct force control for autonomous vehicles
            air_density = 1.225  # kg/m³
            aero_drag = 0.5 * air_density * self.params.drag_coefficient * \
                       self.params.frontal_area * self.state.vx * abs(self.state.vx)
            
            rolling_resistance = 0.012 * self.params.mass * self.params.gravity if abs(self.state.vx) > 0.01 else 0.0
            total_force = self.direct_force - aero_drag - rolling_resistance
            
            return total_force, 0.0
        else:
            # Original complex calculation for human-driven vehicles
            # Engine force
            if self.throttle_input > 0.001:
                engine_torque = self.engine.get_torque(self.engine.rpm)
                total_ratio = self.transmission.get_total_ratio()
                wheel_torque = engine_torque * total_ratio
                engine_force = wheel_torque / self.params.wheel_radius
                engine_force *= self.throttle_input
            else:
                engine_force = 0.0
                
            # Brake force
            if self.brake_input > 0.001:
                max_brake_force = self.params.mass * self.params.gravity * self.params.tire_friction_coeff
                brake_force = max_brake_force * self.brake_input
            else:
                brake_force = 0.0
                
            # Aerodynamic drag
            air_density = 1.225  # kg/m³
            aero_drag = 0.5 * air_density * self.params.drag_coefficient * \
                       self.params.frontal_area * self.state.vx * abs(self.state.vx)
            
            # Rolling resistance
            if abs(self.state.vx) > 0.01:
                rolling_resistance = 0.012 * self.params.mass * self.params.gravity
            else:
                rolling_resistance = 0.0
                
            # Total longitudinal force
            Fx_total = engine_force - brake_force - aero_drag - rolling_resistance
            
            return Fx_total, 0.0
    
    def state_equation_platoon(self, delta_f, delta_r, V, dt):
        """Update vehicle state using platoon_control.py kinematic model"""
        # Use platoon_control.py parameters
        l_f = 1.2  # from platoon_control.py
        l_r = 1.8  # from platoon_control.py
        L = l_f + l_r
        
        # kinematic model - exactly like platoon_control.py
        beta = np.arctan((l_f * np.tan(delta_r) + l_r * np.tan(delta_f)) / L)
        x_dot = V * np.cos(self.state.psi + beta)  # [m/s]
        y_dot = V * np.sin(self.state.psi + beta)  # [m/s]
        psi_dot = (np.tan(delta_f) - np.tan(delta_r)) * np.cos(beta) * V / L  # [rad/s]

        # update state variables
        self.state.x += x_dot * dt  # [m]
        self.state.y += y_dot * dt  # [m]
        self.state.psi += psi_dot * dt  # [rad]

        self.a = (V - self.v) / dt
        self.v = V
        self.state.vx = V
        self.state.ax = self.a
    
    def update_dynamics(self, dt: float):
        """Update vehicle dynamics - choose between complex or kinematic model"""
        if self.use_kinematic_model:
            self.update_dynamics_kinematic(dt)
        else:
            self.update_dynamics_complex(dt)
    
    def update_dynamics_complex(self, dt: float):
        """Update vehicle dynamics using complex model - original implementation"""
        # Calculate forces
        Fx, _ = self.calculate_forces()
        
        # Steering angle
        steering_angle = self.steering_input * self.params.max_steering_angle
        
        # Longitudinal dynamics
        self.state.ax = Fx / self.params.mass
        self.state.vx += self.state.ax * dt
        self.state.vx = np.clip(self.state.vx, 0, self.params.max_velocity)
        
        # Update platoon compatibility variables
        self.v = self.state.vx
        self.a = self.state.ax
        
        # Bicycle model for lateral dynamics (simplified)
        if abs(self.state.vx) > 0.1:  # Avoid division by zero
            # Simple bicycle model
            beta = np.arctan(self.params.lr * np.tan(steering_angle) / self.params.wheelbase)
            self.state.psi_dot = (self.state.vx * np.cos(beta) * np.tan(steering_angle)) / self.params.wheelbase
            
            # Update lateral velocity (simplified)
            self.state.vy = self.state.vx * np.sin(beta)
        else:
            self.state.psi_dot = 0.0
            self.state.vy = 0.0
            
        # Update position and orientation
        self.state.psi += self.state.psi_dot * dt
        self.state.x += self.state.vx * np.cos(self.state.psi) * dt - self.state.vy * np.sin(self.state.psi) * dt
        self.state.y += self.state.vx * np.sin(self.state.psi) * dt + self.state.vy * np.cos(self.state.psi) * dt
        
        # Update engine and transmission
        if abs(self.state.vx) > 0.1:
            wheel_rpm = (self.state.vx * 60) / (2 * np.pi * self.params.wheel_radius)
            target_engine_rpm = wheel_rpm * self.transmission.get_total_ratio()
            target_engine_rpm = max(target_engine_rpm, self.engine.idle_rpm)
            
            if self.throttle_input > 0.001:
                target_engine_rpm += self.throttle_input * 1200  # Additional RPM from throttle
                
            self.engine.update_rpm(target_engine_rpm, dt)
        else:
            self.engine.update_rpm(self.engine.idle_rpm, dt)
            
        self.transmission.update(self.engine.rpm, self.state.vx, dt)
    
    def update_dynamics_kinematic(self, dt: float):
        """Update vehicle dynamics using kinematic model"""
        # Calculate target acceleration from forces (for human driver inputs)
        if not self.autonomous_mode:
            Fx, _ = self.calculate_forces()
            target_acceleration = Fx / self.params.mass
        else:
            target_acceleration = self.target_acceleration
        
        # Calculate new velocity
        new_velocity = self.v + target_acceleration * dt
        new_velocity = np.clip(new_velocity, 0, 250)  # velocity limits
        
        # Update state using kinematic model (with steering)
        delta_f = self.steering_input * self.params.max_steering_angle
        delta_r = 0  # rear wheel steering (zero for normal cars)
        
        self.state_equation_platoon(delta_f, delta_r, new_velocity, dt)
    
    def set_manual_inputs(self, throttle: float, brake: float, steering: float):
        """Set manual inputs for non-autonomous vehicles (human driver)"""
        if not self.autonomous_mode:
            self.throttle_input = throttle
            self.brake_input = brake
            self.steering_input = steering
            
            # Convert throttle/brake to direct force for non-autonomous vehicles
            if throttle > 0.001:
                max_drive_force = self.params.mass * self.params.max_acceleration
                self.direct_force = throttle * max_drive_force
            elif brake > 0.001:
                max_brake_force = self.params.mass * self.params.gravity * self.params.tire_friction_coeff
                self.direct_force = -brake * max_brake_force
            else:
                self.direct_force = 0.0
                
        self.steering_input = np.clip(steering, -1.0, 1.0)
    
    def set_direct_force(self, force: float):
        """Set direct force for autonomous vehicles"""
        self.direct_force = force
    
    def set_autonomous_target(self, target_velocity: float, target_acceleration: float):
        """Set autonomous control targets"""
        self.target_velocity = target_velocity
        self.target_acceleration = target_acceleration
        
    def update_autonomous_control(self, dt: float):
        """Update autonomous control inputs"""
        if not self.autonomous_mode:
            return
        
        target_force = self.target_acceleration * self.params.mass
        self.direct_force = target_force

# פונקציות הפלטון - זהות לחלוטין ל-platoon_control.py
def rajamani(Car_1, Car_2):
    """Rajamani controller - identical to platoon_control.py"""
    h = 1.5   # [s] desired time headway
    tau = 0.1 # [s] time lag

    k1, k5 = -0.12, 0.1 # k1 < -tau/h, k5 > 0
    k2, k3, k4 = -k1-h*k1*k5, 1/h-k1*k5, k5/h

    e = Car_2.state.x - Car_1.state.x + Car_1.L  # [m] actual gap
    e_dot = Car_2.v - Car_1.v         # [m/s] relative velocity

    s_des = Car_1.L + h * Car_2.v # [m] desired gap
    a_des = -k1*Car_1.a - k2*Car_2.a - k3*e_dot - k4*e - k5*Car_2.v

    return a_des, s_des

def free_road_acc(v, t, v_target, a_max):
    """Free road acceleration - identical to platoon_control.py"""
    delta = 4  # exponent

    if (v_target >= v):
        dv_dt = a_max * (1 - (v/v_target)**delta)
    else:
        dv_dt = -a_max * (1 - (v_target/v)**delta)
    return dv_dt

class PlatoonManager:
    """Manages autonomous platoon behavior - using exact platoon_control.py algorithms"""
    
    def __init__(self, vehicles: List[Vehicle]):
        self.vehicles = vehicles
        self.target_velocity = 120.0 / 3.6  # 120 km/h in m/s (matching platoon_control)
        self.max_velocity = 250.0  # m/s (matching platoon_control.py)
        self.max_acceleration = 2.5  # m/s^2 (matching platoon_control.py)
        
        # Data storage like platoon_control.py
        self.actual_gaps_history = [[] for _ in range(len(vehicles)-1)]
        self.desired_gaps_history = [[] for _ in range(len(vehicles)-1)]

        # Set all vehicles to autonomous mode
        for vehicle in vehicles:
            vehicle.autonomous_mode = True
    
    def update(self, dt: float):
        """Update platoon control - implementing exact algorithm from platoon_control.py"""
        if not self.vehicles:
            return
            
        # Leader vehicle - free road acceleration (exactly like platoon_control.py)
        Car_1 = self.vehicles[0]  # Lead vehicle
        
        # Calculate lead acceleration using odeint like platoon_control.py
        lead_acc = odeint(free_road_acc, Car_1.v, [0, dt], args=(self.target_velocity, self.max_acceleration))[-1][0]
        
        # Update leader using platoon kinematic model
        delta_f = 0  # front wheel steering angle (straight line)
        delta_r = 0  # rear wheel steering angle
        Car_1.state_equation_platoon(delta_f, delta_r, lead_acc, dt)
        
        # Following vehicles - Rajamani controller (exactly like platoon_control.py)
        for car_num in range(1, len(self.vehicles)):
            leader = self.vehicles[car_num-1]
            follower = self.vehicles[car_num]
            
            # Calculate actual gap like platoon_control.py
            actual_gap = leader.state.x - follower.state.x 
            
            # Ensure we have enough history arrays
            while len(self.actual_gaps_history) < len(self.vehicles) - 1:
                self.actual_gaps_history.append([])
            while len(self.desired_gaps_history) < len(self.vehicles) - 1:
                self.desired_gaps_history.append([])
                
            self.actual_gaps_history[car_num-1].append(actual_gap)
            
            # Apply Rajamani control law - exactly like platoon_control.py
            a_des, s_des = rajamani(leader, follower)
            self.desired_gaps_history[car_num-1].append(s_des)
            
            # Update velocity with acceleration constraint like platoon_control.py
            new_velocity = follower.v + a_des * dt
            new_velocity = np.clip(new_velocity, 0, 250)  # velocity limits
            
            # Update vehicle state using platoon kinematic model
            follower.state_equation_platoon(delta_f, delta_r, new_velocity, dt)
    
    def add_vehicle(self, vehicle: Vehicle):
        """Add a vehicle to the platoon"""
        vehicle.autonomous_mode = True
        vehicle.joined_platoon = True
        vehicle.joined_time = time.time()
        new_index = self.get_new_vehicle_index(vehicle)
        self.vehicles.insert(new_index, vehicle)
        
        # Fix history arrays
        if new_index == 0:
            # Vehicle joined at the front - add new gap at beginning
            self.actual_gaps_history.insert(0, [])
            self.desired_gaps_history.insert(0, [])
        elif new_index == len(self.vehicles) - 1:
            # Vehicle joined at the end - add new gap at end  
            self.actual_gaps_history.append([])
            self.desired_gaps_history.append([])
        else:
            # Vehicle joined in middle - add new gap at the position
            self.actual_gaps_history.insert(new_index, [])
            self.desired_gaps_history.insert(new_index, [])

    def get_new_vehicle_index(self, vehicle: Vehicle) -> int:
        """Get position for a new vehicle joining the platoon"""
        if not self.vehicles:
            print("No vehicles in platoon, adding at index 0")
            return 0
        
        # Find the right position based on x coordinate
        for i in range(len(self.vehicles)):
            if vehicle.state.x > self.vehicles[i].state.x:
                print(f"New vehicle joining platoon at index {i}")
                return i
        
        # If we get here, vehicle goes at the end
        print(f"New vehicle joining platoon at the end (index {len(self.vehicles)})")
        return len(self.vehicles)

class HumanDriver:
    """Human driver model with motion model selection"""
    
    def __init__(self, vehicle: Vehicle):
        self.vehicle = vehicle
        # תיקון: הגברת מהירות המטרה ל-120 קמ"ש
        self.target_speed = 120.0 / 3.6  # 120 km/h במקום 20 km/h
        self.max_acceleration = 2.0  # m/s^2
        self.max_velocity = 250.0  # m/s
        self.lane_change_progress = 0.0
        self.merging = False
        print(f"Human driver target speed set to: {self.target_speed * 3.6:.1f} km/h")
        
    def set_motion_model(self, use_kinematic: bool):
        """Set motion model for the human vehicle"""
        self.vehicle.set_motion_model(use_kinematic)
        
    def update(self, dt: float, platoon_vehicles: List[Vehicle]):
        """Update human driver inputs"""
        # Improved speed control with smoother transitions
        speed_error = self.target_speed - self.vehicle.state.vx
        speed_error_rel = speed_error / self.target_speed
        # # Smooth PID-like control for acceleration
        # # Proportional gain for smooth response
        # kp = 0.5  # Reduced from aggressive values
        
        # Calculate desired acceleration based on speed error
        # desired_accel = kp * speed_error
        
        # Limit acceleration to realistic values
        max_accel = 2.0  # m/s^2
        max_decel = -3.0  # m/s^2
        delta = 2  # exponent for smooth approach to target speed
        if (self.target_speed  >= self.vehicle.state.vx):
            desired_accel = max_accel * (1 - (self.vehicle.state.vx/self.target_speed)**delta)
        else:
            desired_accel = max_decel * (1 - (self.target_speed/self.vehicle.state.vx)**delta)
        
        # if speed_error_rel > 0:
        #     desired_accel = min(speed_error_rel * max_accel, max_accel)
        # else:
        #     desired_accel = max(speed_error_rel * abs(max_decel), max_decel)
        # desired_accel = np.clip(desired_accel, max_decel, max_accel)
        
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

class PlatoonSimulation:
    """Main simulation class"""
    
    def __init__(self):
        # Simulation parameters - match platoon_control.py exactly
        self.dt = 0.02  # 0.02s timestep like platoon_control.py
        self.time = 0.0
        self.start_time = time.time()
        self.running = True
        self.T_sim = 60 * 3  # simulation time [s] 
        
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
        
        # Create vehicles
        self.setup_vehicles()
        
    def setup_vehicles(self):
        """Initialize vehicles and managers"""
        # Platoon vehicles - exactly like platoon_control.py
        num_cars = 4  # like platoon_control.py
        initial_velocity = 0  # Start from zero like in platoon_control.py
        
        self.platoon_vehicles = []
        for i in range(num_cars):
            vehicle = Vehicle(initial_x=-i * 15, initial_y=0.0, vehicle_id=f"Platoon_{i+1}", initial_velocity=initial_velocity)
            vehicle.autonomous_mode = True
            vehicle.use_kinematic_model = True  # Platoon vehicles always use kinematic model
            self.platoon_vehicles.append(vehicle)
        
        # Human vehicle configuration - SET THIS TO CHOOSE MOTION MODEL
        human_use_kinematic = True  # Set to True for kinematic model, False for complex dynamics
        
        # Human vehicle (left lane)
        self.human_vehicle = Vehicle(initial_x=0.0, initial_y=-2, vehicle_id="Human")
        
        # Create human driver and set motion model
        self.human_driver = HumanDriver(self.human_vehicle)
        self.human_driver.set_motion_model(human_use_kinematic)
        
        # Managers
        self.platoon_manager = PlatoonManager(self.platoon_vehicles)
        
        # Print configuration
        motion_type = "kinematic" if human_use_kinematic else "complex dynamics"
        print(f"Human vehicle using: {motion_type} model")
        
        # Initialize vehicle tracking for consistent plotting
        self.all_vehicles = self.platoon_vehicles + [self.human_vehicle]
        for i, vehicle in enumerate(self.all_vehicles):
            self.vehicle_indices[vehicle.vehicle_id] = i
        
    def update(self):
        """Single simulation step"""
        # Update platoon using exact platoon_control.py algorithms
        self.platoon_manager.update(self.dt)
        
        # Update human driver
        self.human_driver.update(self.dt, self.platoon_vehicles)
        
        # Update human vehicle dynamics (only if not in platoon)
        if not self.human_vehicle.joined_platoon:
            self.human_vehicle.update_dynamics(self.dt)
        
        # Log data
        self.log_data()
        
        # Update time
        self.time += self.dt
        
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
        print(f"\nTime: {self.time:.1f}s")
        for vehicle in self.all_vehicles:
            model_type = "kinematic" if vehicle.use_kinematic_model else "complex"
            
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
                  f"Speed={vehicle.state.vx*3.6:.1f} km/h")

def create_platoon_animation(simulation, title: str = "Platoon Simulation Animation"):
    """Create animated visualization of platoon simulation - with improved stability"""
    try:
        # Import here to ensure proper backend is set
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        
        # Check if we're in headless mode
        global HEADLESS_MODE
        
        if HEADLESS_MODE:
            print("🎬 Creating animation in headless mode (will save as GIF file)")
        else:
            print("🎬 Creating interactive animation")
        
        # Force garbage collection before creating animation
        import gc
        gc.collect()
        
        # Create figure with vertical layout - all graphs stacked vertically
        plt.ioff()  # Turn off interactive mode temporarily
        
        # Create a separate figure for animation to avoid conflicts with static plots
        animation_fig_num = plt.gcf().number + 100  # Use a high figure number to avoid conflicts
        fig = plt.figure(num=animation_fig_num, figsize=(14, 12))  # Specific figure number for animation
        
        # Create vertical layout: 4 graphs stacked vertically
        gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 0.25], hspace=0.3)
        
        ax1 = fig.add_subplot(gs[0])  # Top: Spatial view
        ax2 = fig.add_subplot(gs[1])  # Second: Velocities  
        ax3 = fig.add_subplot(gs[2])  # Third: Inter-vehicle gaps
        ax4 = fig.add_subplot(gs[3])  # Bottom: Status bar (narrow)
        
        time_history = simulation.time_history
        position_history = simulation.position_history
        velocity_history = simulation.velocity_history
        gap_history = simulation.gap_history
        desired_gap_history = simulation.desired_gap_history
        
        # Vehicle colors - different color for each vehicle
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        def animate(frame):
            # Clear axes safely with better error handling
            try:
                for ax in [ax1, ax2, ax3, ax4]:
                    ax.clear()
                    ax.set_rasterized(True)  # Improve performance
            except Exception as e:
                print(f"Warning: Could not clear axes: {e}")
                return []
            
            try:
                current_time = time_history[frame]
            except IndexError:
                return []
            
            # Top subplot: Spatial representation of vehicles
            ax1.set_title(f'{title} | Time: {current_time:.1f}s')
            ax1.set_xlabel('Longitudinal Position [m]')
            ax1.set_ylabel('Lane')
            ax1.grid(True, alpha=0.3)
            
            # Get current positions and find plot range
            current_positions = position_history[frame]
            all_x_positions = [pos[0] for pos in current_positions]
            min_x, max_x = min(all_x_positions), max(all_x_positions)
            ax1.set_xlim(min_x - 20, max_x + 50)
            ax1.set_ylim(-3, 3)
            
            # Draw road lanes
            ax1.axhline(y=0, color='yellow', linestyle='-', linewidth=3, alpha=0.8, label='Lane Center')
            ax1.axhline(y=1.5, color='black', linestyle='-', linewidth=2, alpha=0.5)
            ax1.axhline(y=-1.5, color='black', linestyle='-', linewidth=2, alpha=0.5)
            ax1.axhline(y=-2, color='yellow', linestyle='-', linewidth=3, alpha=0.8, label='Left Lane')
            
            # Draw vehicles
            for i, (vehicle, pos) in enumerate(zip(simulation.all_vehicles, current_positions)):
                x_pos, y_pos = pos
                color = colors[i % len(colors)]
                
                # Vehicle rectangle
                vehicle_patch = plt.Rectangle((x_pos - 2, y_pos - 0.4), 4, 0.8, 
                                            color=color, alpha=0.8, 
                                            label=vehicle.vehicle_id if frame == 0 else "")
                ax1.add_patch(vehicle_patch)
                
                # Vehicle label inside the rectangle
                ax1.text(x_pos, y_pos, vehicle.vehicle_id, 
                        ha='center', va='center', fontsize=8, fontweight='bold', color='white')
                
                # Show vehicle speed
                speed_kmh = velocity_history[frame][i]
                ax1.text(x_pos, y_pos + 0.8, f'{speed_kmh:.0f} km/h', 
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            # Show inter-vehicle gaps based on gap_history (proper platoon gaps)
            gap_displayed = False
            
            # First, show platoon inter-vehicle gaps from gap_history
            if gap_history and len(gap_history) > frame and len(gap_history[frame]) > 0:
                current_gaps = gap_history[frame]
                
                # Find ALL vehicles that are part of the platoon (including human after joining)
                platoon_positions = []
                
                for i, vehicle in enumerate(simulation.all_vehicles):
                    if i < len(current_positions):
                        # Include vehicle if it's a platoon vehicle OR if it's human and joined platoon
                        if (vehicle.vehicle_id.startswith("Platoon") or 
                            (vehicle.vehicle_id == "Human" and 
                             hasattr(vehicle, 'joined_platoon') and vehicle.joined_platoon)):
                            platoon_positions.append((current_positions[i], i, vehicle))
                
                # Sort ALL platoon vehicles (including human) by x-position (front to back)
                platoon_positions.sort(key=lambda x: x[0][0], reverse=True)
                
                # Display gaps between consecutive vehicles in the platoon
                for gap_idx, gap_value in enumerate(current_gaps):
                    if gap_idx < len(platoon_positions) - 1:
                        leader_pos, leader_idx, leader_vehicle = platoon_positions[gap_idx]
                        follower_pos, follower_idx, follower_vehicle = platoon_positions[gap_idx + 1]
                        
                        # Position text at midpoint between vehicles
                        mid_x = (leader_pos[0] + follower_pos[0]) / 2
                        mid_y = min(leader_pos[1], follower_pos[1]) - 1.5
                        
                        # Use different colors based on vehicle types
                        if leader_vehicle.vehicle_id == "Human" or follower_vehicle.vehicle_id == "Human":
                            # Gap involving human vehicle - use orange color
                            color = 'darkorange'
                            bgcolor = 'lightyellow'
                        else:
                            # Pure platoon gap - use blue color
                            color = 'darkblue'
                            bgcolor = 'lightcyan'
                        
                        # Display the gap value from gap_history
                        ax1.text(mid_x, mid_y, f'{gap_value:.1f}m', 
                                ha='center', va='top', fontsize=8, color=color, fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor=bgcolor, alpha=0.9))
                
                gap_displayed = True
            
            # Show human vehicle distance to closest platoon vehicle (during merging only)
            human_idx = None
            for i, vehicle in enumerate(simulation.all_vehicles):
                if vehicle.vehicle_id == "Human":
                    human_idx = i
                    break
            
            if (human_idx is not None and human_idx < len(current_positions) and
                hasattr(simulation, 'merge_start_time') and simulation.merge_start_time is not None and
                current_time >= simulation.merge_start_time):
                
                human_pos = current_positions[human_idx]
                min_distance = float('inf')
                closest_platoon_pos = None
                closest_platoon_idx = None
                
                # Find closest platoon vehicle
                for i, vehicle in enumerate(simulation.all_vehicles):
                    if (vehicle.vehicle_id.startswith("Platoon") and i < len(current_positions)):
                        platoon_pos = current_positions[i]
                        distance = abs(human_pos[0] - platoon_pos[0])
                        if distance < min_distance:
                            min_distance = distance
                            closest_platoon_pos = platoon_pos
                            closest_platoon_idx = i
                
                # Display human-to-platoon distance
                if closest_platoon_pos is not None and min_distance < float('inf'):
                    mid_x = (human_pos[0] + closest_platoon_pos[0]) / 2
                    mid_y = min(human_pos[1], closest_platoon_pos[1]) - 1.5
                    
                    # Use red color for human-platoon distance
                    ax1.text(mid_x, mid_y, f'{min_distance:.1f}m', 
                            ha='center', va='top', fontsize=8, color='red', fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.9))
            
            # If no gap_history data, fall back to simple distance calculation
            if not gap_displayed:
                for i in range(len(current_positions) - 1):
                    pos1 = current_positions[i]
                    pos2 = current_positions[i + 1]
                    
                    # Calculate distance
                    distance = abs(pos1[0] - pos2[0])
                    
                    # Position text at midpoint between vehicles
                    mid_x = (pos1[0] + pos2[0]) / 2
                    mid_y = min(pos1[1], pos2[1]) - 1.5  # Below the lower vehicle
                    
                    # Choose color based on whether vehicles are in the same lane
                    if abs(pos1[1] - pos2[1]) < 0.5:  # Same lane
                        color = 'darkblue'
                        bgcolor = 'lightcyan'
                    else:  # Different lanes
                        color = 'red'
                        bgcolor = 'yellow'
                    
                    # Display distance
                    ax1.text(mid_x, mid_y, f'{distance:.1f}m', 
                            ha='center', va='top', fontsize=8, color=color, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor=bgcolor, alpha=0.9))
            
            ax1.legend(loc='upper right')
            
            # Second subplot: Vehicle velocities over time
            ax2.set_xlim(0, time_history[-1])
            ax2.set_ylim(0, max([max(v) for v in velocity_history]) * 1.1)
            ax2.set_xlabel('Time [s]')
            ax2.set_ylabel('Velocity [km/h]')
            ax2.set_title('Vehicle Velocities')
            ax2.grid(True, alpha=0.3)
            
            # Plot velocity histories up to current frame
            current_range = slice(0, frame + 1)
            time_current = time_history[current_range]
            
            for i, vehicle in enumerate(simulation.all_vehicles):
                color = colors[i % len(colors)]
                velocities = [vel[i] for vel in velocity_history[current_range]]
                ax2.plot(time_current, velocities, color=color, linewidth=2, 
                        label=vehicle.vehicle_id)
                
                # Mark current point
                if len(velocities) > 0:
                    ax2.plot(current_time, velocities[-1], 'o', color=color, markersize=6)
            
            # Add target velocity line for platoon
            if hasattr(simulation.platoon_manager, 'target_velocity'):
                target_kmh = simulation.platoon_manager.target_velocity * 3.6
                ax2.axhline(y=target_kmh, color='gray', linestyle='--', alpha=0.7, 
                           label=f'Target: {target_kmh:.0f} km/h')
            
            ax2.legend(loc='upper right')
            
            # Third subplot: Inter-vehicle gaps (ENHANCED like static plot)
            ax3.clear()
            ax3.set_title('Inter-vehicle Gaps')
            ax3.set_xlabel('Time [s]')
            ax3.set_ylabel('Gap [m]')
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(0, time_history[-1])
            
            # Plot gaps between platoon vehicles (matching static analysis exactly)
            if gap_history and len(gap_history) > frame:
                # Get the maximum number of gaps across all frames
                max_gaps = max(len(gap_set) for gap_set in gap_history) if gap_history else 0
                
                # Use same colors as static plot for consistency
                gap_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
                
                for i in range(max_gaps):
                    # Collect data for this gap across time (up to current frame)
                    gaps = []
                    desired_gaps = []
                    valid_times = []
                    
                    for j in range(frame + 1):
                        if (j < len(gap_history) and j < len(desired_gap_history) and
                            i < len(gap_history[j]) and i < len(desired_gap_history[j])):
                            gaps.append(gap_history[j][i])
                            desired_gaps.append(desired_gap_history[j][i])
                            valid_times.append(time_history[j])
                    
                    if gaps and valid_times:
                        color = gap_colors[i % len(gap_colors)]
                        
                        # Plot actual gaps (solid line) - exactly like static plot
                        ax3.plot(valid_times, gaps, color=color, linewidth=2, 
                               label=f'Gap {i+1}', alpha=0.8)
                        
                        # Plot desired gaps (dashed line) - exactly like static plot  
                        ax3.plot(valid_times, desired_gaps, color=color, linestyle='--', 
                               linewidth=2, alpha=0.6, label=f'Desired {i+1}')
                        
                        # Mark current point with a dot
                        if gaps:
                            ax3.plot(current_time, gaps[-1], 'o', color=color, markersize=6)

            
            # Set appropriate y-limits for better visualization
            if gap_history and len(gap_history) > 0:
                all_gaps = [gap for gap_set in gap_history[:frame+1] for gap in gap_set if isinstance(gap, (int, float))]
                if all_gaps:
                    y_min = max(0, min(all_gaps) - 5)  # Don't go below 0
                    y_max = max(all_gaps) + 10
                    ax3.set_ylim(y_min, y_max)
                else:
                    ax3.set_ylim(0, 60)  # Default range
            else:
                ax3.set_ylim(0, 60)  # Default range
            
            ax3.legend(loc='upper right', ncol=2)  # Two columns for better space usage
            
            # Fourth subplot: Status bar (horizontal, narrow)
            ax4.clear()
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')  # Hide axes for clean text display
            
            # Determine human vehicle status based on current time
            human_status = "Independent"
            if current_time >= 20.0:
                if current_time < 22.0:  # During joining process
                    human_status = "Joining Platoon"
                else:  # After joining
                    human_status = "Platoon Member"
            
            # Create horizontal status text layout
            status_text = f"Time: {current_time:.1f}s  |  Platoon Vehicles: {len(simulation.platoon_vehicles)}  |  Human: {human_status}  |  Target Speed: {simulation.platoon_manager.target_velocity * 3.6:.0f} km/h"
            
            # Find the human vehicle index to get correct speed and position
            human_idx = None
            for i, vehicle in enumerate(simulation.all_vehicles):
                if vehicle.vehicle_id == "Human":
                    human_idx = i
                    break
            
            if human_idx is not None and len(velocity_history) > frame and len(velocity_history[frame]) > human_idx:
                human_speed = velocity_history[frame][human_idx]  # Already in km/h, no need to convert
                status_text += f"  |  Human Speed: {human_speed:.0f} km/h"
            
            if human_idx is not None and len(position_history) > frame and len(position_history[frame]) > human_idx:
                human_pos = position_history[frame][human_idx][0]  # Get x position
                status_text += f"  |  Human Position: {human_pos:.0f}m"
            
            # Display status text horizontally across the bar
            ax4.text(0.5, 0.5, status_text, transform=ax4.transAxes, 
                    ha='center', va='center', fontfamily='monospace', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

        # Create animation with robust error handling
        try:
            # Reduce frame count for better performance and slower animation
            frame_step = max(1, len(time_history) // 150)  # Reduced to ~150 frames (was 200)
            frames_to_use = range(0, len(time_history), frame_step)
            
            print(f"🎬 Creating animation with {len(frames_to_use)} frames...")
            
            # Create animation with slower settings
            anim = FuncAnimation(fig, animate, frames=frames_to_use, 
                               interval=200, repeat=False, blit=False)  # Increased interval from 100ms to 200ms
            
            # Use subplots_adjust instead of tight_layout for gridspec compatibility
            plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08)
            
            # Always try to show animation first (user requested always display)
            try:
                if HEADLESS_MODE:
                    print("🎬 Creating animation in headless mode (will save as GIF file)")
                else:
                    print("🎬 Displaying interactive animation...")
                    
                    # Make sure we're in interactive mode
                    plt.ion()
                    
                    # Show the animation with proper interactive display
                    plt.show(block=False)
                    
                    # Force the animation window to appear
                    plt.draw()
                    fig.canvas.flush_events()
                    
                    print("✅ Interactive animation displayed successfully!")
                    print("📋 The animation is now playing in a separate window")
                    print("📋 Close the animation window manually when you're done viewing")
                    print("⚠️  IMPORTANT: Please wait for the animation to finish before closing!")
                    print("📌 Tip: Closing early may cause harmless TkAgg cleanup errors")
                    
                    # Give time for animation to start
                    import time as time_module
                    time_module.sleep(2)
                
            except Exception as show_error:
                print(f"⚠️ Could not display animation: {show_error}")
                print("💡 Animation will still be available for saving")
            
            # Ask user if they want to save the animation
            try:
                # Check if running in automated mode (piped input)
                import sys
                if not sys.stdin.isatty():
                    # Running with piped input (like echo "2" | python script.py)
                    print("💾 Automated mode detected - skipping animation save")
                    save_choice = 'n'
                else:
                    # Interactive mode - ask user
                    save_choice = input("\n💾 Do you want to save the animation as GIF? (y/n): ").strip().lower()
                
                if save_choice in ['y', 'yes', '1']:
                    anim_filename = os.path.join(RESULTS_DIR, f"{title.replace(' ', '_')}_animation.gif")
                    print(f"💾 Saving animation as {anim_filename}...")
                    
                    # Save with better settings for file size
                    anim.save(anim_filename, writer='pillow', fps=8, dpi=80)
                    
                    # Get absolute path for user
                    abs_path = os.path.abspath(anim_filename)
                    print(f"✅ Animation saved successfully!")
                    print(f"📁 Full path: {abs_path}")
                else:
                    print("ℹ️ Animation not saved (user choice)")
                    
            except KeyboardInterrupt:
                print("\n⚠️ Animation save cancelled by user")
            except Exception as save_error:
                print(f"⚠️ Could not save animation: {save_error}")
                
            print("✅ Animation created successfully")
            return anim
                
        except Exception as anim_error:
            print(f"❌ Animation creation failed: {anim_error}")
            return None
        finally:
            # Safe cleanup - only close the animation figure, not all matplotlib windows
            try:
                # Only close this specific animation figure
                if 'fig' in locals():
                    plt.figure(fig.number)  # Select the animation figure
                    plt.close(fig)  # Close only the animation figure
                # Don't call plt.close('all') to preserve other plots
            except Exception as cleanup_error:
                print(f"⚠️ Animation cleanup warning: {cleanup_error}")
                pass

    except ImportError:
        print("⚠️ Animation requires matplotlib.animation to be installed.")
        return None
    except Exception as e:
        print(f"❌ Animation setup failed: {e}")
        return None

def run_simulation():
    """Run the simulation"""
    print("Creating simulation...")
    sim = PlatoonSimulation()
    print("Simulation created successfully")
    
    # Run simulation for fixed duration - matching platoon_control.py
    max_time = sim.T_sim  # 600 seconds
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
                
            sim.update()
            iteration += 1
            
            # Print status periodically
            if sim.time >= next_print_time:
                sim.print_status()
                next_print_time += print_interval
                
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
                sim.human_vehicle.target_acceleration=sim.platoon_manager.max_acceleration
                print(f"Human driver starting lane change at t={sim.time:.1f}s")
                
    except Exception as e:
        print(f"Error during simulation at iteration {iteration}: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nSimulation completed after {sim.time:.1f} seconds, {iteration} iterations")
    return sim

# if __name__ == "__main__":
#     import os
#     os.system('cls' if os.name == 'nt' else 'clear')
    
#     try:
#         print("🚗 Starting Vehicle Platoon Merging Simulation")
#         print("=" * 50)
        
#         # Run the simulation
#         simulation = run_simulation()
        
#         print("\n📊 Simulation completed!")
#         print(f"Total simulation time: {simulation.time:.1f} seconds")
        
#         # Print final statistics like platoon_control.py
#         print("=== Platoon Control Simulation Results ===")
#         print(f"Simulation time: {simulation.T_sim} seconds")
#         print(f"Number of platoon vehicles: {len(simulation.platoon_manager.vehicles)}")
#         print(f"Final platoon positions: {[f'{v.state.x:.1f}m' for v in simulation.platoon_manager.vehicles]}")
#         if simulation.gap_history:
#             print(f"Final gaps: {[f'{simulation.gap_history[-1][i]:.1f}m' for i in range(len(simulation.gap_history[-1]))]}")
#         print(f"Final platoon velocities: {[f'{v.v:.1f}m/s ({v.v*3.6:.1f}km/h)' for v in simulation.platoon_manager.vehicles]}")
        
#         # Calculate average gap errors like platoon_control.py
#         if simulation.gap_history:
#             avg_gap_errors = []
#             num_gaps = len(simulation.gap_history[0])
#             for i in range(num_gaps):
#                 gaps = [gap_set[i] for gap_set in simulation.gap_history if i < len(gap_set)]
#                 desired_gaps = [gap_set[i] for gap_set in simulation.desired_gap_history if i < len(gap_set)]
#                 if gaps and desired_gaps:
#                     gap_errors = [abs(g - d) for g, d in zip(gaps, desired_gaps)]
#                     avg_gap_errors.append(sum(gap_errors) / len(gap_errors))
#             if avg_gap_errors:
#                 print(f"Average gap errors: {[f'{err:.2f}m' for err in avg_gap_errors]}")
        
#         print(f"Target velocity: {simulation.platoon_manager.target_velocity*3.6:.1f} km/h")
        
#         # Create animation first
#         print("\n🎬 Creating animation...")
#         try:
#             animation = create_platoon_animation(simulation, "Vehicle Platoon Simulation")
#             if animation:
#                 print("✅ Animation created successfully!")
#         except Exception as anim_error:
#             print(f"❌ Animation creation failed: {anim_error}")
        
#         # Plot final results with improved error handling
#         print("Plotting results...")
#         try:
#             # Set matplotlib backend explicitly
#             import matplotlib
#             matplotlib.use('TkAgg')  # or 'Qt5Agg' if TkAgg doesn't work
            
#             plt.figure(figsize=(16, 12))
            
#             # Plot 1: Vehicle positions
#             plt.subplot(3, 3, 1)
#             for i, vehicle in enumerate(simulation.all_vehicles):
#                 x_positions = []
#                 valid_times = []
#                 for j, pos in enumerate(simulation.position_history):
#                     if i < len(pos) and not np.isnan(pos[i][0]):
#                         x_positions.append(pos[i][0])
#                         valid_times.append(simulation.time_history[j])
#                 if x_positions:
#                     plt.plot(valid_times, x_positions, label=vehicle.vehicle_id, linewidth=2)
#             plt.xlabel('Time [s]')
#             plt.ylabel('Position [m]')
#             plt.title('Vehicle Positions')
#             plt.legend()
#             plt.grid(True)
            
#             # Plot 2: Vehicle velocities
#             plt.subplot(3, 3, 2)
#             for i, vehicle in enumerate(simulation.all_vehicles):
#                 velocities = []
#                 valid_times = []
#                 for j, vel in enumerate(simulation.velocity_history):
#                     if i < len(vel) and not np.isnan(vel[i]):
#                         velocities.append(vel[i] / 3.6)  # Convert back to m/s for plotting
#                         valid_times.append(simulation.time_history[j])
#                 if velocities:
#                     plt.plot(valid_times, velocities, label=vehicle.vehicle_id, linewidth=2)
#             # Add target velocity line
#             target_line = [simulation.platoon_manager.target_velocity] * len(simulation.time_history)
#             plt.plot(simulation.time_history, target_line, 'k--', label='Target (Platoon)', linewidth=2)
#             plt.xlabel('Time [s]')
#             plt.ylabel('Velocity [m/s]')
#             plt.title('Vehicle Velocities')
#             plt.legend()
#             plt.grid(True)
            
#             # Plot 3: Inter-vehicle gaps (platoon only)
#             plt.subplot(3, 3, 3)
#             if simulation.gap_history:
#                 max_gaps = max(len(gap_set) for gap_set in simulation.gap_history) if simulation.gap_history else 0
#                 for i in range(max_gaps):
#                     gaps = []
#                     desired_gaps = []
#                     valid_times = []
#                     for j, (gap_set, desired_set) in enumerate(zip(simulation.gap_history, simulation.desired_gap_history)):
#                         if i < len(gap_set) and i < len(desired_set):
#                             gaps.append(gap_set[i])
#                             desired_gaps.append(desired_set[i])
#                             valid_times.append(simulation.time_history[j])
#                     if gaps:
#                         plt.plot(valid_times, gaps, label=f'Gap {i+1}', linewidth=2)
#                         plt.plot(valid_times, desired_gaps, '--', label=f'Desired {i+1}', linewidth=2)
#             plt.xlabel('Time [s]')
#             plt.ylabel('Gap [m]')
#             plt.title('Inter-vehicle Gaps')
#             plt.legend()
#             plt.grid(True)
            
#             # Plot 4: Gap errors
#             plt.subplot(3, 3, 4)
#             if simulation.gap_history:
#                 max_gaps = max(len(gap_set) for gap_set in simulation.gap_history) if simulation.gap_history else 0
#                 for i in range(max_gaps):
#                     gap_errors = []
#                     valid_times = []
#                     for j, (gap_set, desired_set) in enumerate(zip(simulation.gap_history, simulation.desired_gap_history)):
#                         if i < len(gap_set) and i < len(desired_set):
#                             gap_errors.append(gap_set[i] - desired_set[i])
#                             valid_times.append(simulation.time_history[j])
#                     if gap_errors:
#                         plt.plot(valid_times, gap_errors, label=f'Error {i+1}', linewidth=2)
#             plt.xlabel('Time [s]')
#             plt.ylabel('Gap Error [m]')
#             plt.title('Gap Tracking Errors')
#             plt.legend()
#             plt.grid(True)
            
#             # Plot 5: Vehicle trajectories (top view)
#             plt.subplot(3, 3, 5)
#             for i, vehicle in enumerate(simulation.all_vehicles):
#                 x_positions = []
#                 y_positions = []
#                 for pos in simulation.position_history:
#                     if i < len(pos) and not np.isnan(pos[i][0]) and not np.isnan(pos[i][1]):
#                         x_positions.append(pos[i][0])
#                         y_positions.append(pos[i][1])
#                 if x_positions:
#                     plt.plot(x_positions, y_positions, 'o-', label=vehicle.vehicle_id, markersize=1, linewidth=2)
            
#             # Draw road lanes
#             if simulation.position_history:
#                 all_x = [pos[i][0] for pos in simulation.position_history for i in range(len(pos)) if not np.isnan(pos[i][0])]
#                 if all_x:
#                     x_range = [min(all_x), max(all_x)]
#                     plt.plot(x_range, [0, 0], 'k-', alpha=0.3, linewidth=3, label='Right Lane')
#                     plt.plot(x_range, [-2, -2], 'k-', alpha=0.3, linewidth=3, label='Left Lane')
#                     plt.plot(x_range, [-1, -1], 'y--', alpha=0.5, linewidth=2, label='Lane Divider')
            
#             plt.xlabel('X Position [m]')
#             plt.ylabel('Y Position [m]')
#             plt.title('Vehicle Trajectories (Top View)')
#             plt.legend()
#             plt.grid(True)
#             plt.axis('equal')
            
#             # Plot 6: String stability analysis - fixed for dynamic platoon
#             plt.subplot(3, 3, 6)
#             # Calculate velocity differences between consecutive platoon vehicles
#             if len(simulation.velocity_history) > 0:
#                 # We'll calculate based on the original 4 platoon vehicles for consistency
#                 max_platoon_vehicles = 4  # Original platoon size
#                 for i in range(max_platoon_vehicles - 1):
#                     vel_leader = []
#                     vel_follower = []
#                     valid_times = []
                    
#                     for j, vel_data in enumerate(simulation.velocity_history):
#                         # Check if we have data for both leader and follower
#                         if (i < len(vel_data) and (i+1) < len(vel_data) and 
#                             not np.isnan(vel_data[i]) and not np.isnan(vel_data[i+1])):
#                             vel_leader.append(vel_data[i] / 3.6)  # Convert to m/s
#                             vel_follower.append(vel_data[i+1] / 3.6)  # Convert to m/s
#                             valid_times.append(simulation.time_history[j])
                    
#                     if vel_leader and vel_follower:
#                         vel_diff = [vl - vf for vl, vf in zip(vel_leader, vel_follower)]
#                         plt.plot(valid_times, vel_diff, label=f'ΔV {i+1}-{i+2}', linewidth=2)
                
#                 # After human joins (t>=20), add human-related velocity differences
#                 if simulation.human_vehicle.joined_platoon:
#                     human_idx = simulation.vehicle_indices.get("Human", -1)
#                     if human_idx >= 0:
#                         # Find human's position in platoon and calculate relevant differences
#                         for i in range(max_platoon_vehicles):
#                             if i != human_idx:  # Skip self-comparison
#                                 vel_vehicle = []
#                                 vel_human = []
#                                 valid_times = []
                                
#                                 for j, vel_data in enumerate(simulation.velocity_history):
#                                     if (simulation.time_history[j] >= 20.0 and  # Only after joining
#                                         i < len(vel_data) and human_idx < len(vel_data) and
#                                         not np.isnan(vel_data[i]) and not np.isnan(vel_data[human_idx])):
#                                         vel_vehicle.append(vel_data[i] / 3.6)
#                                         vel_human.append(vel_data[human_idx] / 3.6)
#                                         valid_times.append(simulation.time_history[j])
                                
#                                 if vel_vehicle and vel_human and len(valid_times) > 10:  # Only if enough data
#                                     vel_diff = [vv - vh for vv, vh in zip(vel_vehicle, vel_human)]
#                                     plt.plot(valid_times, vel_diff, '--', 
#                                            label=f'ΔV Platoon_{i+1}-Human', linewidth=1.5, alpha=0.7)
#                                     break  # Only show one human comparison to avoid clutter
            
#             plt.xlabel('Time [s]')
#             plt.ylabel('Velocity Difference [m/s]')
#             plt.title('String Stability Analysis')
#             plt.legend()
#             plt.grid(True)
            
#             # Plot 7: Accelerations
#             plt.subplot(3, 3, 7)
#             for i, vehicle in enumerate(simulation.all_vehicles):
#                 accelerations = []
#                 valid_times = []
#                 for j, acc in enumerate(simulation.acceleration_history):
#                     if i < len(acc) and not np.isnan(acc[i]):
#                         accelerations.append(acc[i])
#                         valid_times.append(simulation.time_history[j])
#                 if accelerations:
#                     plt.plot(valid_times, accelerations, label=vehicle.vehicle_id, linewidth=2)
#             plt.xlabel('Time [s]')
#             plt.ylabel('Acceleration [m/s^2]')
#             plt.title('Vehicle Accelerations')
#             plt.legend()
#             plt.grid(True)
            
#             # Plot 8: Human vehicle lane change detail
#             plt.subplot(3, 3, 8)
#             human_idx = simulation.vehicle_indices.get("Human", -1)
#             if human_idx >= 0:
#                 y_positions = []
#                 valid_times = []
#                 for j, pos in enumerate(simulation.position_history):
#                     if human_idx < len(pos) and not np.isnan(pos[human_idx][1]):
#                         y_positions.append(pos[human_idx][1])
#                         valid_times.append(simulation.time_history[j])
#                 if y_positions:
#                     plt.plot(valid_times, y_positions, 'r-', linewidth=3, label='Human Y-position')
#             plt.axhline(y=0, color='k', linestyle='-', alpha=0.3, label='Right Lane Center')
#             plt.axhline(y=-2, color='k', linestyle='-', alpha=0.3, label='Left Lane Center')
#             plt.axvline(x=20.0, color='g', linestyle='--', alpha=0.7, label='Merge Start')
#             plt.xlabel('Time [s]')
#             plt.ylabel('Lateral Position [m]')
#             plt.title('Human Vehicle Lane Change')
#             plt.legend()
#             plt.grid(True)
            
#             # Plot 9: Vehicle count over time
#             plt.subplot(3, 3, 9)
#             platoon_count = []
#             for t in simulation.time_history:
#                 if t < 20.0:
#                     platoon_count.append(4)  # Original platoon size
#                 else:
#                     platoon_count.append(5)  # After human joins
#             plt.plot(simulation.time_history, platoon_count, 'b-', linewidth=3, label='Platoon Size')
#             plt.axvline(x=20.0, color='g', linestyle='--', alpha=0.7, label='Human Joins')
#             plt.xlabel('Time [s]')
#             plt.ylabel('Number of Vehicles')
#             plt.title('Platoon Size Over Time')
#             plt.legend()
#             plt.grid(True)
            
#             plt.tight_layout()
#             plt.show()
#             plt.savefig('platoon_results.png', dpi=150, bbox_inches='tight')
#             print("Results saved to platoon_results.png")
            
#         except Exception as plot_error:
#             print(f"Could not create plots: {plot_error}")
#             print("Trying alternative plotting...")
#             try:
#                 # Simple fallback plot
#                 plt.figure(figsize=(10, 6))
#                 plt.subplot(1, 2, 1)
#                 for i, vehicle in enumerate(simulation.all_vehicles):
#                     x_positions = [pos[i][0] for j, pos in enumerate(simulation.position_history) 
#                                  if i < len(pos) and not np.isnan(pos[i][0])]
#                     times = simulation.time_history[:len(x_positions)]
#                     if x_positions:
#                         plt.plot(times, x_positions, label=vehicle.vehicle_id)
#                 plt.xlabel('Time [s]')
#                 plt.ylabel('Position [m]')
#                 plt.title('Vehicle Positions')
#                 plt.legend()
#                 plt.grid(True)
                
#                 plt.subplot(1, 2, 2)
#                 for i, vehicle in enumerate(simulation.all_vehicles):
#                     velocities = [vel[i] for j, vel in enumerate(simulation.velocity_history)
#                                 if i < len(vel) and not np.isnan(vel[i])]
#                     times = simulation.time_history[:len(velocities)]
#                     if velocities:
#                         plt.plot(times, velocities, label=vehicle.vehicle_id)
#                 plt.xlabel('Time [s]')
#                 plt.ylabel('Velocity [km/h]')
#                 plt.title('Vehicle Velocities')
#                 plt.legend()
#                 plt.grid(True)
                
#                 plt.tight_layout()
#                 plt.show()
#                 print("Fallback plots created successfully")
#             except Exception as fallback_error:
#                 print(f"Fallback plotting also failed: {fallback_error}")
#                 print("Simulation data is still available for manual analysis")
            
#     except Exception as e:
#         print(f"Error occurred: {e}")
#         import traceback
#         traceback.print_exc()

# platoon Joining Scenarios - Three Separate Scenarios
def create_detailed_scenario_summary(simulation, scenario_name, execution_time):
    """Create comprehensive unified summary for each scenario"""
    print(f"\n{'=' * 70}")
    print(f"✅ Simulation {scenario_name} completed!")
    print(f"⏱️ Execution time: {execution_time:.2f} seconds")
    print(f"[ZAP] Speed: {simulation.time_history[-1]/execution_time:.1f}x real time")
    print(f"{'=' * 70}")
    
    print(f"\n📊 === {scenario_name} Platoon Control Simulation Results ===")
    
    # Simulation performance summary  
    print(f"\n🎯 Simulation Performance:")
    print(f"   ⏱️ Total simulation time: {simulation.time_history[-1]:.1f} seconds")
    print(f"   [ZAP] Execution time: {execution_time:.2f} seconds")
    print(f"   🚀 Speed factor: {simulation.time_history[-1]/execution_time:.1f}x real time")
    
    # platoon composition
    print(f"\n[TRUCK] platoon Composition:")
    print(f"   📊 Total platoon vehicles: {len(simulation.platoon_manager.vehicles)}")
    print(f"   🎯 Target velocity: {simulation.platoon_manager.target_velocity*3.6:.1f} km/h")
    
    # Human vehicle status
    print(f"\n🏎️ Human Vehicle Status:")
    print(f"   📍 Final position: ({simulation.human_vehicle.state.x:.1f}, {simulation.human_vehicle.state.y:.1f})")
    print(f"   🏃 Final speed: {simulation.human_vehicle.state.vx*3.6:.1f} km/h")
    print(f"   ✅ Successfully joined platoon: {'Yes' if simulation.human_vehicle.joined_platoon else 'No'}")
    
    # Final positions
    final_positions = []
    for vehicle in simulation.platoon_manager.vehicles:
        final_positions.append(f"{vehicle.state.x:.1f}m")
    print(f"\n📍 Final platoon Positions: {final_positions}")
    
    # Final gaps between vehicles
    final_gaps = []
    if len(simulation.platoon_manager.vehicles) > 1:
        for i in range(len(simulation.platoon_manager.vehicles) - 1):
            leader = simulation.platoon_manager.vehicles[i]
            follower = simulation.platoon_manager.vehicles[i + 1]
            gap = leader.state.x - follower.state.x
            final_gaps.append(f"{gap:.1f}m")
    print(f"📏 Final inter-vehicle gaps: {final_gaps}")
    
    # Calculate and display average gap
    if final_gaps:
        avg_gap = np.mean([float(gap.replace('m', '')) for gap in final_gaps])
        print(f"📊 Average platoon spacing: {avg_gap:.1f} meters")
    
    # Final velocities
    final_velocities = []
    for vehicle in simulation.platoon_manager.vehicles:
        vel_ms = vehicle.state.vx
        vel_kmh = vel_ms * 3.6
        final_velocities.append(f"{vel_ms:.1f}m/s ({vel_kmh:.1f}km/h)")
    print(f"🏃 Final platoon velocities: {final_velocities}")
    
    # Gap errors analysis
    if hasattr(simulation.platoon_manager, 'actual_gaps_history') and simulation.platoon_manager.actual_gaps_history:
        gap_errors = []
        for i in range(len(simulation.platoon_manager.actual_gaps_history)):
            if (simulation.platoon_manager.actual_gaps_history[i] and 
                simulation.platoon_manager.desired_gaps_history[i]):
                actual_gaps = simulation.platoon_manager.actual_gaps_history[i]
                desired_gaps = simulation.platoon_manager.desired_gaps_history[i]
                
                # Calculate average error for this gap
                errors = [abs(actual - desired) for actual, desired in zip(actual_gaps, desired_gaps)]
                avg_error = sum(errors) / len(errors) if errors else 0
                gap_errors.append(f"{avg_error:.2f}m")
        
        if gap_errors:
            print(f"📊 Average gap tracking errors: {gap_errors}")
    
    print(f"\n{'=' * 70}")
    print(f"🎯 Scenario {scenario_name} Summary Complete")
    print(f"{'=' * 70}")
    
    # Joining status
    joining_status = "Yes" if simulation.human_vehicle.joined_platoon else "No"
    print(f"Human vehicle joined platoon: {joining_status}")
    
    # Performance metrics
    print(f"Execution time: {execution_time:.2f} seconds")
    speed_factor = simulation.time_history[-1] / execution_time if execution_time > 0 else 0
    print(f"Speed factor: {speed_factor:.1f}x real time")
    
    print(f"{'=' * 60}")


def run_scenario_join_before():
    """Scenario 1: Join Before platoon"""
    print("\n" + "="*60)
    print("🎯 Scenario 1: Join Before platoon")
    print("Vehicle starts ahead, slows down and joins at the front of platoon")
    print("="*60)
    
    # Create adapted simulation
    sim = PlatoonSimulation()
    
    # Scenario adaptations
    sim.human_vehicle.state.x = 100.0  # Start ahead
    sim.human_vehicle.state.y = -2.0   # In left lane
    sim.human_driver.target_speed = 100.0 / 3.6  # Lower speed (100 km/h)

    print(f"🚗 Scenario settings:")
    print(f"   📍 Initial human vehicle position: ({sim.human_vehicle.state.x:.1f}, {sim.human_vehicle.state.y:.1f})")
    print(f"   🎯 Human vehicle target speed: {sim.human_driver.target_speed*3.6:.0f} km/h")
    print(f"   [TRUCK] platoon target speed: {sim.platoon_manager.target_velocity*3.6:.0f} km/h")
    
    # Run simulation
    run_single_simulation(sim, "JOIN_BEFORE", join_trigger_time=25.0)
    return sim

def run_scenario_join_middle():
    """Scenario 2: Join Middle of platoon"""
    print("\n" + "="*60)
    print("🎯 Scenario 2: Join Middle of platoon")
    print("Vehicle penetrates into gap in the middle of platoon")
    print("="*60)
    
    # Create adapted simulation
    sim = PlatoonSimulation()
    
    # Scenario adaptations
    sim.human_vehicle.state.x = -10.0  # In middle position
    sim.human_vehicle.state.y = -2.0   # In left lane
    sim.human_driver.target_speed = 70.0 / 3.6  # High speed for penetration (70 km/h)

    print(f"🚗 Scenario settings:")
    print(f"   📍 Initial human vehicle position: ({sim.human_vehicle.state.x:.1f}, {sim.human_vehicle.state.y:.1f})")
    print(f"   🎯 Human vehicle target speed: {sim.human_driver.target_speed*3.6:.0f} km/h")
    print(f"   [TRUCK] platoon target speed: {sim.platoon_manager.target_velocity*3.6:.0f} km/h")
    
    # Run simulation
    run_single_simulation(sim, "JOIN_MIDDLE", join_trigger_time=20.0)
    return sim

def run_scenario_join_after():
    """Scenario 3: Join After platoon"""
    print("\n" + "="*60)
    print("🎯 Scenario 3: Join After platoon")
    print("Vehicle starts from behind, accelerates and joins at the back of platoon")
    print("="*60)
    
    # Create adapted simulation
    sim = PlatoonSimulation()
    
    # Scenario adaptations
    sim.human_vehicle.state.x = -100.0  # Start from behind
    sim.human_vehicle.state.y = -2.0    # In left lane
    sim.human_driver.target_speed = 50.0 / 3.6  # High speed to catch up (50 km/h)

    print(f"🚗 Scenario settings:")
    print(f"   📍 Initial human vehicle position: ({sim.human_vehicle.state.x:.1f}, {sim.human_vehicle.state.y:.1f})")
    print(f"   🎯 Human vehicle target speed: {sim.human_driver.target_speed*3.6:.0f} km/h")
    print(f"   [TRUCK] platoon target speed: {sim.platoon_manager.target_velocity*3.6:.0f} km/h")
    
    # Run simulation
    run_single_simulation(sim, "JOIN_AFTER", join_trigger_time=15.0)
    return sim

def run_single_simulation(sim, scenario_name, join_trigger_time=20.0):
    """Run single simulation"""
    print(f"\n🚀 Starting simulation {scenario_name}")
    
    # Simulation parameters
    max_time = sim.T_sim  # 180 seconds
    T = np.arange(0, max_time, sim.dt)
    max_iterations = len(T)
    human_joined = False
    
    print(f"⏱️ Simulation time: {max_time:.0f} seconds")
    print(f"🔢 Number of iterations: {max_iterations:,}")
    
    start_time = time.time()
    
    try:
        for iteration, t in enumerate(T):
            # Report progress every 2000 iterations
            if iteration % 2000 == 0:
                progress = (iteration / max_iterations) * 100
                print(f"📈 Progress: {progress:5.1f}% (t={sim.time:6.1f}s)")
            
            # Update simulation
            sim.update()
            
            # Trigger joining at the right time
            if (sim.time >= join_trigger_time and 
                not human_joined and not sim.human_driver.merging):
                
                print(f"\n🚨 Activating joining at t={sim.time:.1f}s")
                print(f"📍 Human vehicle position: x={sim.human_vehicle.state.x:.1f}m")
                print(f"📍 platoon positions: {[f'{v.state.x:.1f}m' for v in sim.platoon_vehicles]}")
                
                # Activate joining
                sim.human_driver.merging = True
                human_joined = True
                sim.human_vehicle.joined_platoon = True
                sim.platoon_manager.add_vehicle(sim.human_vehicle)
                sim.human_vehicle.target_velocity = sim.platoon_manager.target_velocity
                sim.human_vehicle.target_acceleration = sim.platoon_manager.max_acceleration
                
                print(f"✅ Human vehicle started joining the platoon")

            # Status report every 20 seconds
            if sim.time % 20.0 < sim.dt:
                print(f"\n⏱️ Time: {sim.time:.1f}s")
                for vehicle in sim.all_vehicles:
                    # Determine correct status based on actual vehicle state
                    if vehicle.vehicle_id.startswith("Platoon"):
                        status = "platoon"
                    elif vehicle.vehicle_id == "Human":
                        if vehicle.joined_platoon:
                            if sim.human_driver.merging and sim.human_driver.lane_change_progress < 1.0:
                                status = "Joining platoon"
                            else:
                                status = "platoon Member"
                        else:
                            status = "Independent"
                    else:
                        status = "Unknown"
                    
                    print(f"   {vehicle.vehicle_id}: position=({vehicle.state.x:.1f}, {vehicle.state.y:.1f}), "
                          f"speed={vehicle.state.vx*3.6:.0f}km/h ({status})")
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Display graphs and animation after scenario completion
        print(f"\n📊 Displaying graphs and animation for {scenario_name}...")
        
        # Import matplotlib to avoid scope issues
        import matplotlib.pyplot as plt
        
        # Create comprehensive 9-plot analysis
        static_plots_fig = create_comprehensive_plots(sim, scenario_name)
        
        # Keep the static plots active but temporary disable interaction during animation
        import time as time_module
        print("⏳ Waiting for static plots to stabilize...")
        time_module.sleep(3)  # Give static plots time to fully render
        
        # Create animation
        try:
            print(f"🎬 Creating animation for {scenario_name}...")
            # Temporarily turn off interactive mode for stability
            plt.ioff()
            anim = create_platoon_animation(sim, f"{scenario_name} Animation")
            if anim:
                # Keep animation reference to prevent garbage collection warning
                sim._last_animation = anim
                print(f"✅ Animation created successfully")
            else:
                print("⚠️ Animation creation skipped or failed")
                
            # Re-enable interactive mode and refresh static plots
            plt.ion()
            if static_plots_fig:
                plt.figure(static_plots_fig.number)  # Select static plots figure
                plt.draw()
                static_plots_fig.canvas.flush_events()
                print("🔄 Static plots refreshed after animation")
                
        except Exception as anim_error:
            print(f"❌ Animation failed: {anim_error}")
            # Re-enable interactive mode even if animation failed
            plt.ion()
            # Don't re-raise the error, just continue
        
        # Create detailed summary for the scenario
        create_detailed_scenario_summary(sim, scenario_name, execution_time)
        
        # Clean up matplotlib to prevent memory issues and backend conflicts
        try:
            # Give time for any animations to finish
            import time as time_module
            time_module.sleep(1)
            
            # Gentle cleanup that won't cause TkAgg errors
            import matplotlib.pyplot as plt
            # Don't force close all figures - just clear them
            for i in plt.get_fignums():
                try:
                    fig = plt.figure(i)
                    fig.clf()  # Clear figure content instead of closing
                except:
                    pass
                    
            # Force garbage collection
            import gc
            gc.collect()
            
        except Exception as cleanup_error:
            # Only show non-TkAgg related cleanup errors
            if "Tcl command" not in str(cleanup_error) and "TclError" not in str(cleanup_error):
                print(f"⚠️ Cleanup note: {cleanup_error}")
        
    except Exception as e:
        print(f"❌ Error in simulation {scenario_name}: {e}")
        import traceback
        traceback.print_exc()
        # Also clean up on error with safe cleanup
        try:
            import matplotlib.pyplot as plt
            for i in plt.get_fignums():
                try:
                    fig = plt.figure(i)
                    fig.clf()
                except:
                    pass
            import gc
            gc.collect()
        except:
            pass
    
    print(f"\n{'='*60}")
    
def create_comprehensive_plots(simulation, scenario_name="Simulation"):
    """Create comprehensive 9-plot analysis for any simulation"""
    try:
        # Import here to ensure proper backend is set
        import matplotlib.pyplot as plt
        
        # Create figure with proper cleanup
        fig = plt.figure(figsize=(16, 12))
        
        # Plot 1: Vehicle positions
        plt.subplot(3, 3, 1)
        for i, vehicle in enumerate(simulation.all_vehicles):
            x_positions = []
            valid_times = []
            for j, pos in enumerate(simulation.position_history):
                if i < len(pos) and not np.isnan(pos[i][0]):
                    x_positions.append(pos[i][0])
                    valid_times.append(simulation.time_history[j])
            if x_positions:
                plt.plot(valid_times, x_positions, label=vehicle.vehicle_id, linewidth=2)
        plt.xlabel('Time [s]')
        plt.ylabel('Position [m]')
        plt.title('Vehicle Positions')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Vehicle velocities
        plt.subplot(3, 3, 2)
        for i, vehicle in enumerate(simulation.all_vehicles):
            velocities = []
            valid_times = []
            for j, vel in enumerate(simulation.velocity_history):
                if i < len(vel) and not np.isnan(vel[i]):
                    velocities.append(vel[i] / 3.6)  # Convert back to m/s for plotting
                    valid_times.append(simulation.time_history[j])
            if velocities:
                plt.plot(valid_times, velocities, label=vehicle.vehicle_id, linewidth=2)
        # Add target velocity line
        target_line = [simulation.platoon_manager.target_velocity] * len(simulation.time_history)
        plt.plot(simulation.time_history, target_line, 'k--', label='Target (Platoon)', linewidth=2)
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.title('Vehicle Velocities')
        plt.legend()
        plt.grid(True)
        
        # Plot 3: Inter-vehicle gaps (platoon only)
        plt.subplot(3, 3, 3)
        if simulation.gap_history:
            max_gaps = max(len(gap_set) for gap_set in simulation.gap_history) if simulation.gap_history else 0
            for i in range(max_gaps):
                gaps = []
                desired_gaps = []
                valid_times = []
                for j, (gap_set, desired_set) in enumerate(zip(simulation.gap_history, simulation.desired_gap_history)):
                    if i < len(gap_set) and i < len(desired_set):
                        gaps.append(gap_set[i])
                        desired_gaps.append(desired_set[i])
                        valid_times.append(simulation.time_history[j])
                if gaps:
                    plt.plot(valid_times, gaps, label=f'Gap {i+1}', linewidth=2)
                    plt.plot(valid_times, desired_gaps, '--', label=f'Desired {i+1}', linewidth=2)
        plt.xlabel('Time [s]')
        plt.ylabel('Gap [m]')
        plt.title('Inter-vehicle Gaps')
        plt.legend()
        plt.grid(True)
        
        # Plot 4: Gap errors
        plt.subplot(3, 3, 4)
        if simulation.gap_history:
            max_gaps = max(len(gap_set) for gap_set in simulation.gap_history) if simulation.gap_history else 0
            for i in range(max_gaps):
                gap_errors = []
                valid_times = []
                for j, (gap_set, desired_set) in enumerate(zip(simulation.gap_history, simulation.desired_gap_history)):
                    if i < len(gap_set) and i < len(desired_set):
                        gap_errors.append(gap_set[i] - desired_set[i])
                        valid_times.append(simulation.time_history[j])
                if gap_errors:
                    plt.plot(valid_times, gap_errors, label=f'Error {i+1}', linewidth=2)
        plt.xlabel('Time [s]')
        plt.ylabel('Gap Error [m]')
        plt.title('Gap Tracking Errors')
        plt.legend()
        plt.grid(True)
        
        # Plot 5: Vehicle trajectories (top view)
        plt.subplot(3, 3, 5)
        for i, vehicle in enumerate(simulation.all_vehicles):
            x_positions = []
            y_positions = []
            for pos in simulation.position_history:
                if i < len(pos) and not np.isnan(pos[i][0]) and not np.isnan(pos[i][1]):
                    x_positions.append(pos[i][0])
                    y_positions.append(pos[i][1])
            if x_positions:
                plt.plot(x_positions, y_positions, 'o-', label=vehicle.vehicle_id, markersize=1, linewidth=2)
        
        # Draw road lanes
        if simulation.position_history:
            all_x = [pos[i][0] for pos in simulation.position_history for i in range(len(pos)) if not np.isnan(pos[i][0])]
            if all_x:
                x_range = [min(all_x), max(all_x)]
                plt.plot(x_range, [0, 0], 'k-', alpha=0.3, linewidth=3, label='Right Lane')
                plt.plot(x_range, [-2, -2], 'k-', alpha=0.3, linewidth=3, label='Left Lane')
                plt.plot(x_range, [-1, -1], 'y--', alpha=0.5, linewidth=2, label='Lane Divider')
        
        plt.xlabel('X Position [m]')
        plt.ylabel('Y Position [m]')
        plt.title('Vehicle Trajectories (Top View)')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        # Plot 6: String stability analysis - fixed for dynamic platoon
        plt.subplot(3, 3, 6)
        # Calculate velocity differences between consecutive platoon vehicles
        if len(simulation.velocity_history) > 0:
            # We'll calculate based on the original 4 platoon vehicles for consistency
            max_platoon_vehicles = 4  # Original platoon size
            for i in range(max_platoon_vehicles - 1):
                vel_leader = []
                vel_follower = []
                valid_times = []
                
                for j, vel_data in enumerate(simulation.velocity_history):
                    # Check if we have data for both leader and follower
                    if (i < len(vel_data) and (i+1) < len(vel_data) and 
                        not np.isnan(vel_data[i]) and not np.isnan(vel_data[i+1])):
                        vel_leader.append(vel_data[i] / 3.6)  # Convert to m/s
                        vel_follower.append(vel_data[i+1] / 3.6)  # Convert to m/s
                        valid_times.append(simulation.time_history[j])
                
                if vel_leader and vel_follower:
                    vel_diff = [vl - vf for vl, vf in zip(vel_leader, vel_follower)]
                    plt.plot(valid_times, vel_diff, label=f'ΔV {i+1}-{i+2}', linewidth=2)
            
            # After human joins, add human-related velocity differences
            if simulation.human_vehicle.joined_platoon:
                human_idx = simulation.vehicle_indices.get("Human", -1)
                if human_idx >= 0:
                    # Find human's position in platoon and calculate relevant differences
                    for i in range(max_platoon_vehicles):
                        if i != human_idx:  # Skip self-comparison
                            vel_vehicle = []
                            vel_human = []
                            valid_times = []
                            
                            for j, vel_data in enumerate(simulation.velocity_history):
                                if (simulation.time_history[j] >= 20.0 and  # Only after joining
                                    i < len(vel_data) and human_idx < len(vel_data) and
                                    not np.isnan(vel_data[i]) and not np.isnan(vel_data[human_idx])):
                                    vel_vehicle.append(vel_data[i] / 3.6)
                                    vel_human.append(vel_data[human_idx] / 3.6)
                                    valid_times.append(simulation.time_history[j])
                            
                            if vel_vehicle and vel_human and len(valid_times) > 10:  # Only if enough data
                                vel_diff = [vv - vh for vv, vh in zip(vel_vehicle, vel_human)]
                                plt.plot(valid_times, vel_diff, '--', 
                                       label=f'ΔV Platoon_{i+1}-Human', linewidth=1.5, alpha=0.7)
                                break  # Only show one human comparison to avoid clutter
        
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity Difference [m/s]')
        plt.title('String Stability Analysis')
        plt.legend()
        plt.grid(True)
        
        # Plot 7: Accelerations
        plt.subplot(3, 3, 7)
        for i, vehicle in enumerate(simulation.all_vehicles):
            accelerations = []
            valid_times = []
            for j, acc in enumerate(simulation.acceleration_history):
                if i < len(acc) and not np.isnan(acc[i]):
                    accelerations.append(acc[i])
                    valid_times.append(simulation.time_history[j])
            if accelerations:
                plt.plot(valid_times, accelerations, label=vehicle.vehicle_id, linewidth=2)
        plt.xlabel('Time [s]')
        plt.ylabel('Acceleration [m/s^2]')
        plt.title('Vehicle Accelerations')
        plt.legend()
        plt.grid(True)
        
        # Plot 8: Human vehicle lane change detail
        plt.subplot(3, 3, 8)
        human_idx = simulation.vehicle_indices.get("Human", -1)
        if human_idx >= 0:
            y_positions = []
            valid_times = []
            for j, pos in enumerate(simulation.position_history):
                if human_idx < len(pos) and not np.isnan(pos[human_idx][1]):
                    y_positions.append(pos[human_idx][1])
                    valid_times.append(simulation.time_history[j])
            if y_positions:
                plt.plot(valid_times, y_positions, 'r-', linewidth=3, label='Human Y-position')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3, label='Right Lane Center')
        plt.axhline(y=-2, color='k', linestyle='-', alpha=0.3, label='Left Lane Center')
        plt.axvline(x=20.0, color='g', linestyle='--', alpha=0.7, label='Merge Start')
        plt.xlabel('Time [s]')
        plt.ylabel('Lateral Position [m]')
        plt.title('Human Vehicle Lane Change')
        plt.legend()
        plt.grid(True)
        
        # Plot 9: Vehicle count over time
        plt.subplot(3, 3, 9)
        platoon_count = []
        for t in simulation.time_history:
            if t < 20.0:
                platoon_count.append(4)  # Original platoon size
            else:
                platoon_count.append(5)  # After human joins
        plt.plot(simulation.time_history, platoon_count, 'b-', linewidth=3, label='Platoon Size')
        plt.axvline(x=20.0, color='g', linestyle='--', alpha=0.7, label='Human Joins')
        plt.xlabel('Time [s]')
        plt.ylabel('Number of Vehicles')
        plt.title('Platoon Size Over Time')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.suptitle(f'{scenario_name} - Comprehensive Analysis', fontsize=16, y=0.98)
        
        # Always save plot to results directory
        filename = os.path.join(RESULTS_DIR, f'{scenario_name.replace(" ", "_").replace(":", "")}_results.png')
        try:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            abs_path = os.path.abspath(filename)
            print(f"📁 Results saved to {filename}")
            print(f"📁 Full path: {abs_path}")
            
            # Keep a global reference to prevent garbage collection
            global _static_plots_figure
            _static_plots_figure = fig  # Store reference to prevent cleanup
        except Exception as save_error:
            print(f"⚠️ Could not save plot: {save_error}")
        
        # Always try to display plots with better error handling
        try:
            if HEADLESS_MODE:
                # In headless mode, try to open the saved image with system viewer
                print("📊 Plots saved in headless mode")
                try:
                    import subprocess
                    import platform
                    if platform.system() == 'Windows':
                        os.startfile(filename)
                        print("📊 Opening plots with system image viewer")
                    else:
                        subprocess.run(['xdg-open', filename], check=False)
                        print("📊 Opening plots with system image viewer")
                except:
                    print("📊 Plots saved - you can view them manually")
            else:
                # Interactive mode - show the plots properly
                print("📊 Displaying interactive plots...")
                
                # Make sure we're in interactive mode
                plt.ion()
                
                # Show the plot with blocking=False for interactive display
                plt.show(block=False)
                
                # Force the window to appear and draw
                plt.draw()
                fig.canvas.flush_events()
                
                # Try to keep the figure active by preventing early cleanup
                try:
                    # Windows specific - try to bring window to front
                    if hasattr(fig.canvas.manager.window, 'wm_attributes'):
                        fig.canvas.manager.window.wm_attributes("-topmost", 1)
                        fig.canvas.manager.window.wm_attributes("-topmost", 0)
                except:
                    # Ignore errors on non-Windows or different backends
                    pass
                
                print("✅ Interactive plots displayed successfully!")
                print("📋 Close the plot window manually when you're done viewing")
                print("⚠️  Note: Please wait for the animation to finish before closing windows")
                print("📌 Tip: Let the animation complete its cycle to avoid cleanup errors")
                
                # Give enough time for the plot to render and stay active
                import time as time_module
                time_module.sleep(2)  # Increased from 1 to 2 seconds
                
        except Exception as show_error:
            print(f"⚠️ Could not display plots interactively: {show_error}")
            print("� Plots were saved to file - trying to open with system viewer...")
            try:
                import subprocess
                import platform
                if platform.system() == 'Windows':
                    os.startfile(filename)
                    print("📊 Opened with system image viewer")
            except:
                print("📊 Please open the saved plot file manually")
        
        # Safer cleanup - avoid TkAgg destruction errors
        try:
            # Give the animation time to finish properly before any cleanup
            import time as time_module
            time_module.sleep(0.5)  # Short delay to let animation settle
            
            # Try gentle cleanup without forcing destruction
            if hasattr(fig, 'canvas'):
                try:
                    # Disconnect any event handlers first
                    fig.canvas.mpl_disconnect_all = lambda: None
                    # Clear the figure gently
                    plt.clf()
                except:
                    pass  # Ignore cleanup errors
                    
        except Exception as cleanup_error:
            # Don't print TkAgg cleanup errors as they're cosmetic
            if "Tcl command" not in str(cleanup_error):
                print(f"⚠️ Minor cleanup warning: {cleanup_error}")
        
        # Safer cleanup - avoid TkAgg destruction errors
        try:
            # Don't close immediately - let it stay open
            pass
        except:
            pass  # Ignore cleanup errors
        
    except Exception as plot_error:
        print(f"Could not create comprehensive plots: {plot_error}")
        print("Trying simple fallback plot...")
        try:
            # Simple fallback plot
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            for i, vehicle in enumerate(simulation.all_vehicles):
                x_positions = [pos[i][0] for j, pos in enumerate(simulation.position_history) 
                             if i < len(pos) and not np.isnan(pos[i][0])]
                times = simulation.time_history[:len(x_positions)]
                if x_positions:
                    plt.plot(times, x_positions, label=vehicle.vehicle_id)
            plt.xlabel('Time [s]')
            plt.ylabel('Position [m]')
            plt.title('Vehicle Positions')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            for i, vehicle in enumerate(simulation.all_vehicles):
                velocities = [vel[i] for j, vel in enumerate(simulation.velocity_history)
                            if i < len(vel) and not np.isnan(vel[i])]
                times = simulation.time_history[:len(velocities)]
                if velocities:
                    plt.plot(times, velocities, label=vehicle.vehicle_id)
            plt.xlabel('Time [s]')
            plt.ylabel('Velocity [km/h]')
            plt.title('Vehicle Velocities')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            
            # Save fallback plot
            fallback_filename = os.path.join(RESULTS_DIR, f'{scenario_name.replace(" ", "_")}_fallback_results.png')
            try:
                plt.savefig(fallback_filename, dpi=150, bbox_inches='tight')
                print(f"📁 Fallback results saved to {fallback_filename}")
            except:
                pass
            
            # Show fallback plot
            plt.show(block=False)
            print("📊 Fallback plots displayed successfully")
            
            # Give time for the plot to render
            import time
            time.sleep(1)
            
            # Return the current figure for reference
            return plt.gcf()
            
        except Exception as fallback_error:
            print(f"Even fallback plotting failed: {fallback_error}")
            return None
    
    # Return the figure if successful
    return fig if 'fig' in locals() else None

def run_all_scenarios_separately():
    """Run all scenarios separately"""
    print("[TRUCK] platoon Joining Scenarios System")
    print("=" * 60)
    
    scenarios = [
        ("Scenario 1: Join Before platoon", run_scenario_join_before),
        ("Scenario 2: Join Middle of platoon", run_scenario_join_middle),
        ("Scenario 3: Join After platoon", run_scenario_join_after)
    ]
    
    results = []
    
    for i, (name, scenario_func) in enumerate(scenarios, 1):
        print(f"\n{'🎬' * 20}")
        print(f"Starting {name} [{i}/3]")
        print(f"{'🎬' * 20}")
        
        try:
            result = scenario_func()
            results.append((name, result))
            print(f"✅ {name} completed successfully!")
            
            # Pause between scenarios
            if i < len(scenarios):
                print("\n⏸️ 3-second pause before next scenario...")
                time.sleep(3)
                
        except Exception as e:
            print(f"❌ Error in {name}: {e}")
            results.append((name, None))
    
    # Overall summary
    print(f"\n{'🏁' * 25}")
    print("All Scenarios Summary")
    print(f"{'🏁' * 25}")
    
    successful_scenarios = sum(1 for _, result in results if result is not None)
    
    print(f"📊 Overall statistics:")
    print(f"   ✅ Completed scenarios: {successful_scenarios}/3")
    print(f"   📈 Success rate: {successful_scenarios/3*100:.0f}%")
    
    for name, result in results:
        if result:
            human_speed = result.human_vehicle.state.vx * 3.6
            target_speed = result.platoon_manager.target_velocity * 3.6
            joined = "Yes" if result.human_vehicle.joined_platoon else "No"
            print(f"\n🎯 {name}:")
            print(f"   🏎️ Final speed: {human_speed:.1f} km/h (target: {target_speed:.1f})")
            print(f"   📍 Final position: ({result.human_vehicle.state.x:.1f}, {result.human_vehicle.state.y:.1f})")
            print(f"   ✅ Joined platoon: {joined}")
        else:
            print(f"\n❌ {name}: Failed")
    
    return results

# Addition to main function
if __name__ == "__main__":
    import os
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("🚗 Choose what to run:")
    print("1. Single scenario - Join before platoon")
    print("2. Single scenario - Join middle of platoon") 
    print("3. Single scenario - Join after platoon")
    print("4. All three scenarios one after another")
    print("5. Original simulation")
    print("6. Switch to interactive plotting mode (may cause TkAgg errors)")
    
    try:
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == "6":
            # Switch to interactive mode
            try:
                import matplotlib
                matplotlib.use('Qt5Agg')
                HEADLESS_MODE = False
                print("✅ Switched to interactive Qt5Agg backend")
                print("⚠️ Note: This may cause destruction errors on exit")
                print("\nNow choose an option to run:")
                print("1-5. Run scenarios with interactive plots")
                choice = input("Enter choice (1-5): ").strip()
            except Exception as backend_error:
                print(f"❌ Could not switch backend: {backend_error}")
                exit(1)
        
        if choice == "1":
            run_scenario_join_before()
        elif choice == "2":
            run_scenario_join_middle()
        elif choice == "3":
            run_scenario_join_after()
        elif choice == "4":
            run_all_scenarios_separately()
        elif choice == "5":
            # Original simulation
            print("🚗 Starting Original Vehicle Platoon Merging Simulation")
            print("=" * 50)
            simulation = run_simulation()
            print("\n📊 Original simulation completed!")
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n\n⚠️ User interrupted the program")
    except Exception as e:
        print(f"\n❌ Error: {e}")