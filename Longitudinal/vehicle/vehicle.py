"""
Main Vehicle class implementing bicycle model dynamics.
Contains the vehicle physics, control inputs, and state management.
"""

import numpy as np
import time
from typing import Tuple
from .components import VehicleParameters, Engine, Transmission, VehicleState
from config import (SIMULATION_DT)

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
        if vehicle_id == "Human":
            self.human_vehicle = True
        else:
            self.human_vehicle = False
        self.max_velocity = self.params.max_velocity  # m/s
        # Initialize position
        self.state.x = initial_x
        self.state.y = initial_y
        self.state.psi = initial_heading
        self.state.vx = initial_velocity
        
        # Control inputs
        self.direct_force = 0.0      # Direct longitudinal force [N]
        self.steering_input = 0.0    # -1 to 1
        self.a_desired = 0.0         # Desired acceleration [m/s^2]
        
        # For compatibility with human driver
        self.throttle_input = 0.0    # 0 to 1
        self.brake_input = 0.0       # 0 to 1
        
        # Mode controls
        self.autonomous_mode = False
        self.use_kinematic_model = False  # False = complex dynamics, True = kinematic model
        self.use_state_space_model = False  # New state-space bicycle model
        self.use_hierarchical_model = False  # Hierarchical: upper (double integrator) + lower (complex dynamics)
        self.lower_level_controller = None  # Assigned when hierarchical mode is activated
        self.target_velocity = 0.0
        self.target_acceleration = 0.0
        
        # Platoon compatibility variables
        self.L = self.params.length # vehicle length
        self.v = initial_velocity  # velocity shorthand for platoon compatibility
        self.a = 0.0  # acceleration shorthand for platoon compatibility
        
        # Track if this vehicle joined the platoon
        self.joined_platoon = False
        self.joined_time = None
    
        self._debug_counter = 0  # For debug printing frequency control
        
        self.nash_acceleration = None  # For Nash equilibrium control input
        
        # Acceleration output filter for hierarchical mode
        # Smooths noisy engine-level acceleration before Rajamani/Nash feedback
        self._a_filtered = 0.0
        self._a_filter_initialized = False
        
        # State-space matrices cache
        self.A=None
        self.B1=None
        self.B2=None
        self.C=None
        
    # Properties for easy access to state variables
    @property
    def x(self) -> float:
        """Current x position (m)"""
        return self.state.x
    
    @property
    def y(self) -> float:
        """Current y position (m)"""
        return self.state.y
    
    @property
    def psi(self) -> float:
        """Current yaw angle (rad)"""
        return self.state.psi
    
    @property
    def vx(self) -> float:
        """Current longitudinal velocity (m/s)"""
        return self.state.vx
    
    @property
    def vy(self) -> float:
        """Current lateral velocity (m/s)"""
        return self.state.vy
        
    def set_motion_model(self, use_kinematic: bool, use_state_space: bool = False,
                         use_hierarchical: bool = False):
        """Set which motion model to use.
        
        Parameters
        ----------
        use_kinematic : bool
            Use simple kinematic model.
        use_state_space : bool
            Use double-integrator state-space model.
        use_hierarchical : bool
            Use hierarchical model: upper controller plans with double integrator,
            lower controller converts a_desired to throttle/brake,
            vehicle propagates with full longitudinal dynamics (engine, drag, etc.).
        """
        # Reset all flags first
        self.use_kinematic_model = False
        self.use_state_space_model = False
        self.use_hierarchical_model = False
        
        if use_hierarchical:
            self.use_hierarchical_model = True
            # Also keep state-space flag for get_state_space_matrices() —
            # Nash solver still uses the double integrator for planning.
            self.use_state_space_model = True
            model_name = "hierarchical (upper: double integrator, lower: complex dynamics)"
            # Initialize lower-level controller
            from control.lower_level_controller import LowerLevelController
            self.lower_level_controller = LowerLevelController(self)
            # Ensure engine/transmission are in a reasonable state
            self._sync_engine_to_velocity()
        elif use_state_space:
            self.use_state_space_model = True
            model_name = "state-space bicycle model"
        elif use_kinematic:
            self.use_kinematic_model = True
            model_name = "kinematic"
        else:
            model_name = "complex dynamics"
        
        print(f"{self.vehicle_id}: Motion model set to {model_name}")
    
    def _sync_engine_to_velocity(self):
        """Synchronize engine RPM and gear to current velocity.
        
        Simulates sequential upshifting (same thresholds as Transmission.update)
        so the initial gear is consistent with what the transmission would
        naturally reach through gear progression at the current speed.
        """
        if abs(self.state.vx) > 0.1:
            wheel_rpm = (self.state.vx * 60) / (2 * np.pi * self.params.wheel_radius)
            # Walk gears from lowest, shifting up when RPM exceeds upshift threshold
            # (mirrors Transmission.update sequential shift logic)
            selected_gear = 0
            for gear_idx in range(len(self.transmission.gear_ratios)):
                ratio = (self.transmission.gear_ratios[gear_idx]
                         * self.transmission.final_drive_ratios[gear_idx])
                engine_rpm = wheel_rpm * ratio
                # Skip gears where RPM exceeds redline
                if engine_rpm > self.engine.redline_rpm:
                    continue
                # RPM below idle — gear too high for this speed
                if engine_rpm < self.engine.idle_rpm:
                    break
                selected_gear = gear_idx
                # Would the transmission upshift from this gear?
                if (gear_idx < len(self.transmission.upshift_rpm) and
                        engine_rpm > self.transmission.upshift_rpm[gear_idx]):
                    continue  # yes — check next gear
                else:
                    break     # no — stay in this gear
            self.transmission.current_gear = selected_gear
            target_rpm = wheel_rpm * self.transmission.get_total_ratio()
            target_rpm = np.clip(target_rpm, self.engine.idle_rpm, self.engine.redline_rpm)
            self.engine.rpm = target_rpm
        else:
            self.engine.rpm = self.engine.idle_rpm
            self.transmission.current_gear = 0
        
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
                # Engine braking: pumping losses + internal friction when
                # drivetrain is connected but throttle is released.
                # Real engines produce ~50-80 Nm of retarding torque at
                # typical highway RPMs due to air compression in cylinders,
                # bearing friction and auxiliary loads.
                from config import ENGINE_BRAKING_BASE_TORQUE, ENGINE_BRAKING_MAX_TORQUE
                if abs(self.state.vx) > 1.0 and self.brake_input < 0.001:
                    braking_torque = ENGINE_BRAKING_BASE_TORQUE * (self.engine.rpm / 2000.0)
                    braking_torque = min(braking_torque, ENGINE_BRAKING_MAX_TORQUE)
                    total_ratio = self.transmission.get_total_ratio()
                    engine_force = -(braking_torque * total_ratio / self.params.wheel_radius)
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
        l_f = self.params.lf  # from platoon_control.py
        l_r = self.params.lr  # from platoon_control.py
        L = self.params.wheelbase  # vehicle wheelbase
        
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
        """Update vehicle dynamics - choose between complex, kinematic, state-space,
        or hierarchical model.
        
        In hierarchical mode the upper controller (Nash/Rajamani) has already
        written a_desired using the double-integrator planning model.  Here the
        lower-level controller converts that into throttle/brake, and the complex
        dynamics model propagates the actual vehicle state.
        """
        if self.use_hierarchical_model:
            self.update_dynamics_hierarchical(dt)
        elif self.use_state_space_model:
            self.update_dynamics_state_space(dt)
        elif self.use_kinematic_model:
            self.update_dynamics_kinematic(dt)
        else:
            self.update_dynamics_complex(dt)
    
    def update_dynamics_complex(self, dt: float):
        """Update vehicle dynamics using complex model with ZOH-consistent integration.
        
        Integration order must match the ZOH double integrator used by the
        upper controller for planning:
            x[k+1] = x[k] + v[k]*dt + 0.5*a*dt²   (position first, OLD velocity)
            v[k+1] = v[k] + a*dt
        
        The previous semi-implicit Euler (v first, then x using NEW v) introduced
        a systematic O(a*dt²) position bias per step relative to ZOH.
        """
        # Calculate forces
        Fx, _ = self.calculate_forces()
        
        # Steering angle
        steering_angle = self.steering_input * self.params.max_steering_angle
        
        # Longitudinal acceleration from Newton's 2nd law
        self.state.ax = Fx / self.params.mass
        
        # Bicycle model for lateral dynamics (simplified)
        if abs(self.state.vx) > 0.1:  # Avoid division by zero
            beta = np.arctan(self.params.lr * np.tan(steering_angle) / self.params.wheelbase)
            self.state.psi_dot = (self.state.vx * np.cos(beta) * np.tan(steering_angle)) / self.params.wheelbase
            self.state.vy = self.state.vx * np.sin(beta)
        else:
            self.state.psi_dot = 0.0
            self.state.vy = 0.0
        
        # === ZOH-consistent integration: POSITION first (using OLD velocity) ===
        cos_psi = np.cos(self.state.psi)
        sin_psi = np.sin(self.state.psi)
        
        self.state.psi += self.state.psi_dot * dt
        self.state.x += (self.state.vx * cos_psi - self.state.vy * sin_psi) * dt \
                       + 0.5 * self.state.ax * cos_psi * dt * dt
        self.state.y += (self.state.vx * sin_psi + self.state.vy * cos_psi) * dt \
                       + 0.5 * self.state.ax * sin_psi * dt * dt
        
        # === VELOCITY update (after position) ===
        self.state.vx += self.state.ax * dt
        self.state.vx = np.clip(self.state.vx, 0, self.params.max_velocity)
        
        # Update platoon compatibility variables
        self.v = self.state.vx
        self.a = self.state.ax
        
        # Update engine and transmission
        if abs(self.state.vx) > 0.1:
            wheel_rpm = (self.state.vx * 60) / (2 * np.pi * self.params.wheel_radius)
            target_engine_rpm = wheel_rpm * self.transmission.get_total_ratio()
            target_engine_rpm = max(target_engine_rpm, self.engine.idle_rpm)
            
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
            # Use a_desired for autonomous mode
            target_acceleration = self.a_desired if hasattr(self, 'a_desired') else self.target_acceleration
        
        # Calculate new velocity
        new_velocity = self.v + target_acceleration * dt
        new_velocity = np.clip(new_velocity, 0, self.max_velocity)  # velocity limits
        
        # Update state using kinematic model (with steering)
        delta_f = self.steering_input * self.params.max_steering_angle
        delta_r = 0  # rear wheel steering (zero for normal cars)
        
        self.state_equation_platoon(delta_f, delta_r, new_velocity, dt)
    
    def update_dynamics_hierarchical(self, dt: float):
        """Update vehicle dynamics using hierarchical control architecture.
        
        Architecture (Rajamani Ch.6 / Belousov et al.):
        ================================================
        Upper Controller (already executed before this call):
            - Nash solver / Rajamani / IDM used double-integrator model
            - Result stored in self.a_desired
        
        Lower Controller (executed here):
            1. Feedforward: F_ff = m*a_desired + F_drag + F_roll
            2. PI feedback:  F_fb = Kp*m*(a_des - a_actual) + Ki*m*∫error
            3. Actuator mapping: F_total → (throttle, brake)
            
        Complex Dynamics (executed here via update_dynamics_complex):
            - Engine torque → wheel force
            - Brake force
            - Aerodynamic drag, rolling resistance
            - Newton's 2nd law → actual acceleration
        
        The get_state_space_matrices() method still returns the double-integrator
        matrices so the Nash solver continues to plan with the simplified model.
        """
        if self.lower_level_controller is None:
            # Fallback: use state-space if controller not initialized
            self.update_dynamics_state_space(dt)
            return
        
        # --- Lower-level controller: a_desired → (throttle, brake) ---
        a_desired = self.a_desired if hasattr(self, 'a_desired') else self.target_acceleration
        throttle, brake = self.lower_level_controller.compute_control(a_desired, dt)
        
        # Set throttle/brake on the vehicle (used by calculate_forces / update_dynamics_complex)
        self.throttle_input = throttle
        self.brake_input = brake
        self.steering_input = 0.0  # Longitudinal only
        
        # Choose force computation path:
        # During gear shifts the blending gear ratio causes large force
        # transients even with frozen throttle (e.g. gear 1→2: ratio drops
        # ~45%, halving engine force).  Bypass the engine path entirely and
        # use direct force control — mirroring Unity AutoMode where
        # Fx = m·a_desired, giving perfectly smooth acceleration during
        # shifts.  Outside shifts, the normal engine/transmission path runs.
        was_autonomous = self.autonomous_mode
        if hasattr(self, 'transmission') and self.transmission.is_shifting:
            # Direct force: set F so that calculate_forces returns m·a_desired
            # (calculate_forces subtracts aero+rolling, so add them here)
            air_density = 1.225
            F_aero = (0.5 * air_density * self.params.drag_coefficient
                      * self.params.frontal_area * self.state.vx * abs(self.state.vx))
            F_roll = (0.012 * self.params.mass * self.params.gravity
                      if abs(self.state.vx) > 0.01 else 0.0)
            self.direct_force = self.params.mass * a_desired + F_aero + F_roll
            self.autonomous_mode = True
        else:
            self.autonomous_mode = False
        
        # --- Propagate with complex dynamics (engine, aero, rolling resistance) ---
        self.update_dynamics_complex(dt)
        
        # Restore mode
        self.autonomous_mode = was_autonomous
        
        # ---------------------------------------------------------------
        # Acceleration output filter (EMA)
        # Smooths the noisy engine-level acceleration before it feeds back
        # to the upper controller (Rajamani uses vehicle.a directly).
        # Without this filter, gear-shift force discontinuities and RPM
        # slew-rate limiting inject high-frequency noise into Rajamani's
        # a_des = -k1*Car_1.a - k2*Car_2.a - ..., which then cascades
        # back through the PI loop creating sustained oscillations.
        # ---------------------------------------------------------------
        from config import LOWER_CTRL_ACCEL_FILTER_ALPHA
        a_raw = self.a  # set by update_dynamics_complex()
        
        if not self._a_filter_initialized:
            self._a_filtered = a_raw
            self._a_filter_initialized = True
        else:
            alpha = LOWER_CTRL_ACCEL_FILTER_ALPHA
            self._a_filtered = alpha * a_raw + (1.0 - alpha) * self._a_filtered
        
        # Use filtered acceleration for all feedback paths
        self.a = self._a_filtered
        self.state.ax = self._a_filtered
    
    def update_dynamics_state_space(self, dt: float):
        """Update vehicle dynamics using state-space bicycle model with ZOH discretization
        
        State-space representation (discrete-time with ZOH):
        x[k+1] = A_d*x[k] + B_d*u[k]
        y[k] = C*x[k]
        
        State vector x = [position, velocity]^T
        Input u = acceleration
        """
        # Get ZOH discrete-time matrices
        A_d, B_d, C = self.get_state_space_matrices(dt=dt)
        
        # Current state vector [x, vx]
        x_current = np.array([[self.state.x],
                             [self.state.vx]])
        
        # Calculate control input (acceleration)
        if not self.autonomous_mode:
            # For manual control, use forces from throttle/brake
            Fx, _ = self.calculate_forces()
            # u_acceleration = Fx / self.params.mass
            u_acceleration = self.a_desired
        else:
            # For autonomous mode, use desired acceleration
            u_acceleration = self.a_desired if hasattr(self, 'a_desired') else self.target_acceleration
        
        # Apply acceleration limits
        max_accel = 2.5  # m/s^2
        u_acceleration = np.clip(u_acceleration, -max_accel, max_accel)
        self.a = u_acceleration  # Update acceleration for logging
        self.state.ax = self.a  # Store desired acceleration
        
        # Control input vector
        u1 = np.array([[u_acceleration]]) # Control input for acceleration
        u2 = np.zeros(u1.shape) if not self.autonomous_mode and self.vehicle_id=="Human" else None # Control input for steering (zero for no steering)

        # **ZOH DISCRETE-TIME UPDATE**
        # x[k+1] = A_d*x[k] + B_d*u[k]
        x_next = A_d @ x_current + B_d @ u1 + (B_d @ u2 if u2 is not None else 0)
        
        # Extract new state values
        self.state.x = float(x_next[0, 0])    # new position
        self.state.vx = float(x_next[1, 0])   # new velocity
        
        # Apply velocity limits
        self.state.vx = np.clip(self.state.vx, 0, self.params.max_velocity)
        
        # Update acceleration for logging and platoon compatibility
        self.state.ax = u_acceleration
        self.a = u_acceleration
        self.v = self.state.vx
        
        # For lateral motion - simplified (straight line motion)
        self.state.vy = 0.0
        self.state.psi_dot = 0.0
        
        # # Debug output for human vehicle
        # if self.vehicle_id == "Human" and hasattr(self, '_debug_counter'):
        #     if self._debug_counter % 100 == 0:  # Print every 100 iterations
        #         print(f"ZOH Debug {self.vehicle_id}: x={self.state.x:.2f}, vx={self.state.vx:.2f}, ax={u_acceleration:.2f}")
        #     self._debug_counter += 1
        # elif self.vehicle_id == "Human":
        #     self._debug_counter = 0
        
        # if hasattr(self, '_debug_counter'):
        #     if self._debug_counter % 100 == 0:  # Print every 100 iterations
        #         print(f"ZOH Debug {self.vehicle_id}: x={self.state.x:.2f}, vx={self.state.vx:.2f}, ax={u_acceleration:.2f}")
        #     self._debug_counter += 1
    
    def get_state_space_matrices(self, dt: float = SIMULATION_DT):
        """Get state-space matrices A, B, C for the longitudinal bicycle model
        
        Args:
            dt: Time step for discrete-time matrices (if None, returns continuous-time)
            
        Returns:
            tuple: (A, B, C) matrices
        """
        # Continuous-time state-space matrices
        A_continuous = np.array([[0.0, 1.0],   # dx/dt = vx
                                [0.0, 0.0]])   # dvx/dt = ax (from input)
        
        B_continuous = np.array([[0.0],        # position not directly affected by acceleration
                                [1.0]])        # velocity affected by acceleration
        
        C = np.array([[1.0, 0.0],   # output position = x
                      [0.0, 1.0]])  # output velocity = vx
        
        if dt is None:
            # Return continuous-time matrices
            return A_continuous, B_continuous, C
        else:
            # Convert to discrete-time using ZOH (Zero-Order Hold)
            # For the double integrator system [position, velocity]:
            # 
            # A_continuous = [0  1]    B_continuous = [0]
            #                [0  0]                   [1]
            #
            # The ZOH discretization gives:
            # A_d = exp(A*dt) = [1  dt]
            #                   [0   1]
            #
            # B_d = ∫[0 to dt] exp(A*τ) dτ * B = [dt^2/2]
            #                                     [dt  ]
        
            from scipy.linalg import expm
        
            # Method 1: Using matrix exponential (exact ZOH)
            try:
                # Create the augmented matrix for ZOH conversion
                # [A  B] 
                # [0  0]
                n = A_continuous.shape[0]  # 2
                m = B_continuous.shape[1]  # 1
            
                augmented = np.zeros((n + m, n + m))
                augmented[0:n, 0:n] = A_continuous * dt
                augmented[0:n, n:n+m] = B_continuous * dt
            
                # Matrix exponential of augmented matrix
                exp_augmented = expm(augmented)
            
                # Extract discrete matrices
                A_discrete = exp_augmented[0:n, 0:n]
                B_discrete = exp_augmented[0:n, n:n+m]
            
            except ImportError:
                # Fallback: Analytical solution for double integrator
                # This is the exact ZOH solution for our specific A and B matrices
                A_discrete = np.array([[1.0, dt],      # x[k+1] = x[k] + vx[k]*dt
                                      [0.0, 1.0]])     # vx[k+1] = vx[k] + ax[k]*dt
            
                B_discrete = np.array([[0.5*dt*dt],    # x[k+1] += 0.5*ax*dt^2 (ZOH effect)
                                      [dt]])           # vx[k+1] += ax*dt
            self.A=A_discrete
            self.B1=B_discrete
            self.B2=B_discrete if self.autonomous_mode and self.vehicle_id=="Human" else None
            self.C=C
            return A_discrete, B_discrete, C
    
    def get_state_vector(self):
        """Get current state vector [x, vx]^T"""
        return np.array([[self.state.x],
                        [self.state.vx]])
    
    def set_state_vector(self, x_state):
        """Set state vector [x, vx]^T"""
        self.state.x = float(x_state[0, 0])
        self.state.vx = float(x_state[1, 0])
        self.v = self.state.vx  # Update platoon compatibility variable
    
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


__all__ = ['Vehicle']