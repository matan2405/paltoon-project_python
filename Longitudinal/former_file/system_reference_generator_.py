#!/usr/bin/env python3
"""
File: system_reference_generator.py  
Description: Generates target trajectories for the System player (R1).

VERSION 2.0 - WITH TRANSITIONAL CONTROLLER
============================================
Based on Rajamani Chapter 6.7.1: "The need for a transitional controller"

Key Insight: The regular CTG (Constant Time-Gap) control law cannot directly 
be used to follow a newly encountered vehicle. A transitional trajectory 
needs to be designed before the CTG control law can be used.

This version adds THREE operating modes:
1. CRUISE MODE: No leader or leader very far - accelerate to target speed
2. GAP CLOSING MODE: Leader exists but gap is larger than desired - 
   use transitional controller to close the gap safely
3. FOLLOWING MODE: Gap is close to desired - use Rajamani CTG controller

The Gap Closing Mode is the key addition that solves the "Join After Platoon"
persistent gap error problem.
"""
import numpy as np
import copy
import sys
import os

# Add parent directory to path to allow importing from control
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control.platoon_control import free_road_acc, rajamani


class SystemReferenceGenerator:
    """
    Generates target trajectories for the System player (R1) in Nash equilibrium.
    
    The generator uses three modes based on the situation:
    - Cruise: Accelerate to platoon target speed
    - Gap Closing: Transitional controller to close large gaps safely
    - Following: Rajamani CTG controller for steady-state following
    """
    
    def __init__(self, Np: int = 20, dt: float = 0.1):
        """
        Initialize the System Reference Generator.
        
        Args:
            Np: Prediction horizon (number of steps)
            dt: Time step for prediction [seconds]
        """
        self.Np = Np
        self.dt = dt
        
        # === MODE THRESHOLDS ===
        # Distance beyond which we consider ourselves "alone" (no leader influence)
        self.DETECTION_RANGE = 150.0  # [m]
        
        # Gap error threshold for switching between Gap Closing and Following modes
        # If gap_error > GAP_CLOSING_THRESHOLD, use transitional controller
        self.GAP_CLOSING_THRESHOLD = 3.0  # [m] - increased from 5.0 for smoother transition
        
        # Minimum gap error below which we use pure Rajamani
        self.STABLE_GAP_THRESHOLD = 1.0  # [m]
        
        # === GAP CLOSING PARAMETERS ===
        # Maximum overspeed factor relative to leader when closing gap
        self.MAX_OVERSPEED_FACTOR = 1.20  # Can go 20% faster than leader
        
        # Desired closing rate [m/s] - how fast we want to close the gap
        self.DESIRED_CLOSING_RATE = 3.0  # [m/s]
        
        # Gain for proportional gap closing controller
        self.GAP_CLOSING_GAIN = 0.25  # [1/s] - acceleration per meter of gap error
        
        # === CATCH-UP PARAMETERS (for cruise mode when leader exists but far) ===
        self.CATCHUP_FACTOR = 1.15  # 10% overspeed when catching up
        
        # === COMFORT LIMITS ===
        self.MAX_ACCEL = 2.0   # [m/s²]
        self.MAX_DECEL = -3.5  # [m/s²]
        self.MAX_JERK = 1.5    # [m/s³] - limits rate of change of acceleration
        
        # === RATE LIMITING STATE ===
        self.prev_a_ref = 0.0  # Previous acceleration reference for jerk limiting
        
        print(f"🚀 System Reference Generator V2.1 (Smoothstep + Jerk Limiting)")
        print(f"   📊 Prediction: dt={self.dt}s, Np={self.Np}")
        print(f"   🎯 Gap Closing Threshold: {self.GAP_CLOSING_THRESHOLD}m")
        print(f"   ⚡ Max Overspeed Factor: {self.MAX_OVERSPEED_FACTOR}")
        print(f"   🎢 Max Jerk: {self.MAX_JERK} m/s³")

    def get_system_acceleration_and_state_sequence(self, simulation):
        """
        Generates the target trajectory using appropriate controller based on situation.
        
        Three modes:
        1. CRUISE: No leader or leader > DETECTION_RANGE → free_road_acc to target speed
        2. GAP_CLOSING: gap_error > GAP_CLOSING_THRESHOLD → transitional controller
        3. FOLLOWING: gap_error < GAP_CLOSING_THRESHOLD → Rajamani CTG
        
        Args:
            simulation: The simulation object containing vehicle states
            
        Returns:
            tuple: (accel_sequence, state_sequence) - predicted trajectory
        """
        accel_sequence = np.zeros(self.Np)
        state_sequence = np.zeros((self.Np, 2))  # [position, velocity]
        
        # Clone current state for forward simulation
        sim_vehicle = copy.deepcopy(simulation.human_vehicle)
        
        # ========================================================================
        # Identify Leader - USE LOCKED if available
        # ========================================================================
        leader = None
        
        # Check if we have locked targets (after t_merge)
        if hasattr(simulation.human_vehicle, 'target_leader_id'):
            # === POST-MERGE: Use LOCKED leader ===
            target_id = simulation.human_vehicle.target_leader_id
            
            if target_id is not None:
                # Find the locked leader by ID
                for v in simulation.platoon_manager.vehicles:
                    if v.vehicle_id == target_id:
                        leader = v
                        break
                
                if leader is None:
                    # Locked leader not found - this shouldn't happen!
                    print(f"⚠️  WARNING (RefGen): Locked leader '{target_id}' not found!")
                    
                    # Fallback to dynamic detection
                    if hasattr(simulation, 'safety_field'):
                        leader = simulation.safety_field._get_leader(
                            simulation.human_vehicle, 
                            simulation.platoon_manager
                        )
            else:
                # Locked target is None - merging at front of platoon
                leader = None
                
        else:
            # === PRE-MERGE: Use DYNAMIC closest leader ===
            if hasattr(simulation, 'safety_field'):
                leader = simulation.safety_field._get_leader(
                    simulation.human_vehicle, 
                    simulation.platoon_manager
                )
        
        # Debug logging (only once)
        if not hasattr(self, '_leader_mode_logged'):
            self._leader_mode_logged = True
            
            if hasattr(simulation.human_vehicle, 'target_leader_id'):
                target_id = simulation.human_vehicle.target_leader_id
                print(f"🎯 System Reference Generator: LOCKED mode")
                print(f"   Target: {target_id if target_id else 'None (front)'}")
                print(f"   Found:  {leader.vehicle_id if leader else 'None'}")
            else:
                print(f"🎯 System Reference Generator: DYNAMIC mode (pre-merge)")
                print(f"   Closest: {leader.vehicle_id if leader else 'None'}")
        
        # Clone leader for prediction (constant velocity assumption)
        sim_leader = copy.deepcopy(leader) if leader else None
        
        # Get target platoon speed
        target_platoon_speed = simulation.platoon_manager.target_velocity
        
        # Time headway for desired gap calculation
        h = 1.5  # [s] - constant time gap
        
        for i in range(self.Np):
            # === Step 1: Update Leader Prediction ===
            if sim_leader:
                sim_leader.state.x += sim_leader.state.vx * self.dt
            
            # === Step 2: Calculate Distance and Gap Error ===
            dist_to_leader = float('inf')
            gap_error = 0.0
            desired_gap = 50.0  # Default
            
            if sim_leader:
                dist_to_leader = sim_leader.state.x - sim_vehicle.state.x
                
                # Calculate desired gap (Rajamani formula)
                L = sim_leader.L if hasattr(sim_leader, 'L') else 5.0
                desired_gap = L + h * sim_vehicle.state.vx
                
                # Gap error: positive means we are too far behind
                gap_error = dist_to_leader - desired_gap
            
            # === Step 3: Select Operating Mode and Calculate Acceleration ===
            a_ref = 0.0
            
            if sim_leader is None or dist_to_leader > self.DETECTION_RANGE:
                # =============================================
                # MODE 1: CRUISE (No leader or leader very far)
                # =============================================
                target_v = target_platoon_speed
                
                # If we know there's a leader ahead (just far), allow overspeed
                if sim_leader:
                    target_v *= self.CATCHUP_FACTOR
                
                a_ref = free_road_acc(
                    v=sim_vehicle.state.vx,
                    t=0,
                    v_target=target_v,
                    a_max=sim_vehicle.params.max_acceleration if hasattr(sim_vehicle, 'params') else self.MAX_ACCEL
                )
                
            elif gap_error > self.GAP_CLOSING_THRESHOLD:
                # =============================================
                # MODE 2: GAP CLOSING (Transitional Controller)
                # =============================================
                # This is the KEY addition for solving "Join After Platoon"
                
                a_ref = self._compute_gap_closing_acceleration(
                    sim_vehicle=sim_vehicle,
                    sim_leader=sim_leader,
                    gap_error=gap_error,
                    desired_gap=desired_gap,
                    target_platoon_speed=target_platoon_speed
                )
                
            elif gap_error > self.STABLE_GAP_THRESHOLD:
                # =============================================
                # MODE 2.5: TRANSITION ZONE (Blend between modes)
                # =============================================
                # Smooth transition between gap closing and following
                
                # Calculate both accelerations
                a_closing = self._compute_gap_closing_acceleration(
                    sim_vehicle=sim_vehicle,
                    sim_leader=sim_leader,
                    gap_error=gap_error,
                    desired_gap=desired_gap,
                    target_platoon_speed=target_platoon_speed
                )
                
                a_following, _ = rajamani(sim_leader, sim_vehicle)
                
                # Blend factor: 1.0 at GAP_CLOSING_THRESHOLD, 0.0 at STABLE_GAP_THRESHOLD
                # Using SMOOTHSTEP instead of linear for smoother transition (zero derivative at endpoints)
                t = (gap_error - self.STABLE_GAP_THRESHOLD) / \
                        (self.GAP_CLOSING_THRESHOLD - self.STABLE_GAP_THRESHOLD)
                t = np.clip(t, 0.0, 1.0)
                
                # Smoothstep function: 3t² - 2t³
                blend = t * t * (3.0 - 2.0 * t)
                
                a_ref = blend * a_closing + (1 - blend) * a_following
                
            else:
                # =============================================
                # MODE 3: FOLLOWING (Rajamani CTG Controller)
                # =============================================
                # Steady-state car following
                
                a_ref, _ = rajamani(sim_leader, sim_vehicle)
            
            # === Step 4: Apply Comfort Constraints ===
            a_ref = np.clip(a_ref, self.MAX_DECEL, self.MAX_ACCEL)
            
            # === Step 4.5: Apply Jerk Limiting ===
            # Limit rate of change of acceleration for smoother transitions
            if i == 0:
                # First step uses actual previous acceleration
                max_delta_a = self.MAX_JERK * self.dt
                a_ref = np.clip(a_ref, 
                               self.prev_a_ref - max_delta_a,
                               self.prev_a_ref + max_delta_a)
            
            # === Step 5: Update Simulated Vehicle State ===
            sim_vehicle.state.ax = a_ref
            if hasattr(sim_vehicle, 'a'):
                sim_vehicle.a = a_ref
            
            # Euler integration
            sim_vehicle.state.x += sim_vehicle.state.vx * self.dt + 0.5 * a_ref * self.dt**2
            sim_vehicle.state.vx += a_ref * self.dt
            sim_vehicle.state.vx = max(0.0, sim_vehicle.state.vx)  # No reverse
            
            if hasattr(sim_vehicle, 'v'):
                sim_vehicle.v = sim_vehicle.state.vx
            
            # Store prediction
            accel_sequence[i] = a_ref
            state_sequence[i, 0] = sim_vehicle.state.x
            state_sequence[i, 1] = sim_vehicle.state.vx
        
        # Update previous acceleration for next call (jerk limiting)
        if len(accel_sequence) > 0:
            self.prev_a_ref = accel_sequence[0]
        
        return accel_sequence, state_sequence

    def _compute_gap_closing_acceleration(self, sim_vehicle, sim_leader, 
                                          gap_error, desired_gap, target_platoon_speed):
        """
        Transitional Controller for closing large gaps.
        
        Based on Rajamani Chapter 6.7.1, this controller:
        1. Computes a safe approach speed based on distance
        2. Uses proportional control to achieve desired closing rate
        3. Limits overspeed for safety and comfort
        
        The key insight is that we want to approach the leader at a controlled
        relative velocity, not just match speeds.
        
        Args:
            sim_vehicle: Simulated ego vehicle
            sim_leader: Simulated leader vehicle
            gap_error: Current gap error (positive = too far)
            desired_gap: Target gap distance
            target_platoon_speed: Platoon target speed
            
        Returns:
            float: Desired acceleration for gap closing
        """
        ego_v = sim_vehicle.state.vx
        leader_v = sim_leader.state.vx
        
        # === Method 1: Safe Approach Speed ===
        # Maximum safe speed based on being able to stop before collision
        # v_safe = v_leader + sqrt(2 * a_comfortable * gap_error)
        # This ensures we can always decelerate to leader's speed
        comfortable_decel = 1.5  # [m/s²] - comfortable deceleration
        v_safe = leader_v + np.sqrt(2 * comfortable_decel * max(gap_error, 0.1))
        
        # === Method 2: Desired Closing Rate ===
        # We want to approach at a steady rate, faster when gap is larger
        # Proportional controller: closing_rate = K * gap_error
        desired_closing_rate = self.GAP_CLOSING_GAIN * gap_error
        desired_closing_rate = np.clip(desired_closing_rate, 0.5, self.DESIRED_CLOSING_RATE * 2)
        
        # Target velocity to achieve this closing rate
        v_closing = leader_v + desired_closing_rate
        
        # === Combine Methods ===
        # Take minimum of safe speed and closing speed
        v_target = min(v_safe, v_closing)
        
        # Also limit by overspeed factor relative to platoon speed
        v_max_overspeed = target_platoon_speed * self.MAX_OVERSPEED_FACTOR
        v_target = min(v_target, v_max_overspeed)
        
        # === Calculate Required Acceleration ===
        # Simple proportional control to reach target velocity
        velocity_error = v_target - ego_v
        
        # Gain for velocity tracking (higher = faster response)
        K_v = 0.8  # [1/s]
        
        a_ref = K_v * velocity_error
        
        # Add small feedforward based on leader acceleration if available
        if hasattr(sim_leader.state, 'ax') and sim_leader.state.ax is not None:
            a_ref += 0.3 * sim_leader.state.ax
        
        return a_ref


# Export
__all__ = ['SystemReferenceGenerator']


# === UNIT TEST ===
if __name__ == "__main__":
    print("\n" + "="*60)
    print("System Reference Generator V2.0 - Unit Test")
    print("="*60)
    
    # Create mock objects for testing
    class MockState:
        def __init__(self, x, vx, ax=0):
            self.x = x
            self.vx = vx
            self.ax = ax
    
    class MockParams:
        def __init__(self):
            self.max_acceleration = 2.5
            self.length = 5.0
    
    class MockVehicle:
        def __init__(self, x, v):
            self.state = MockState(x, v)
            self.params = MockParams()
            self.L = 5.0
            self.v = v
            self.a = 0
    
    class MockPlatoonManager:
        def __init__(self):
            self.target_velocity = 33.33  # 120 km/h
    
    class MockSafetyField:
        def __init__(self, leader):
            self.leader = leader
        def _get_leader(self, ego, pm):
            return self.leader
    
    class MockSimulation:
        def __init__(self, ego_x, ego_v, leader_x, leader_v):
            self.human_vehicle = MockVehicle(ego_x, ego_v)
            self.platoon_manager = MockPlatoonManager()
            if leader_x is not None:
                leader = MockVehicle(leader_x, leader_v)
                self.safety_field = MockSafetyField(leader)
            else:
                self.safety_field = MockSafetyField(None)
    
    # Initialize generator
    gen = SystemReferenceGenerator(Np=20, dt=0.1)
    
    print("\n--- Test 1: Cruise Mode (No Leader) ---")
    sim1 = MockSimulation(ego_x=0, ego_v=25, leader_x=None, leader_v=None)
    accel1, states1 = gen.get_system_acceleration_and_state_sequence(sim1)
    print(f"Initial velocity: 25 m/s, Target: 33.33 m/s")
    print(f"Accelerations: {accel1[:5].round(2)}")
    print(f"Final velocity: {states1[-1, 1]:.2f} m/s")
    
    print("\n--- Test 2: Gap Closing Mode (Large Gap) ---")
    # Ego at x=0, v=30. Leader at x=80, v=30. Desired gap ~50m. Gap error = 30m
    sim2 = MockSimulation(ego_x=0, ego_v=30, leader_x=80, leader_v=30)
    accel2, states2 = gen.get_system_acceleration_and_state_sequence(sim2)
    print(f"Initial gap: 80m, Desired: ~50m, Gap error: ~30m")
    print(f"Accelerations: {accel2[:5].round(2)}")
    print(f"Should be POSITIVE to close gap")
    
    print("\n--- Test 3: Following Mode (Small Gap) ---")
    # Ego at x=0, v=30. Leader at x=52, v=30. Desired gap ~50m. Gap error = 2m
    sim3 = MockSimulation(ego_x=0, ego_v=30, leader_x=52, leader_v=30)
    accel3, states3 = gen.get_system_acceleration_and_state_sequence(sim3)
    print(f"Initial gap: 52m, Desired: ~50m, Gap error: ~2m")
    print(f"Accelerations: {accel3[:5].round(2)}")
    print(f"Should be near ZERO (steady state)")
    
    print("\n--- Test 4: Join After Platoon Scenario ---")
    # Simulating the problematic scenario
    # Ego behind platoon, just joined, gap error = +20m
    sim4 = MockSimulation(ego_x=0, ego_v=33, leader_x=70, leader_v=33)
    accel4, states4 = gen.get_system_acceleration_and_state_sequence(sim4)
    h = 1.5
    desired = 5 + h * 33  # ~55m
    actual = 70
    gap_err = actual - desired  # ~15m
    print(f"Initial gap: 70m, Desired: {desired:.1f}m, Gap error: {gap_err:.1f}m")
    print(f"Accelerations: {accel4[:5].round(2)}")
    print(f"Should show POSITIVE acceleration to close gap!")
    
    print("\n" + "="*60)
    print("✅ Unit tests complete")
    print("="*60)