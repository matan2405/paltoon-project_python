#!/usr/bin/env python3
"""
File: system_reference_generator.py  
Description: Generates target trajectories for the System player (R1).

VERSION 5.0 - RAJAMANI CHAPTER 6.7 TRANSITIONAL CONTROLLER
===========================================================
Exact implementation of Rajamani "Vehicle Dynamics and Control" Chapter 6.7.

The Transitional Controller operates in the R-Ṙ (Range vs Range-Rate) phase plane:
    R = actual gap to preceding vehicle [m]
    Ṙ = range-rate = v_leader - v_ego [m/s] (positive = gap increasing)

Key Equations from Rajamani Section 6.7.2:

1. Parabolic Switching Line:
   R_switch = R_desired + Ṙ² / (2 * a_comfort)
   
2. Gap-Closing Controller (above parabola):
   a_des = -a_comfort  (constant comfortable deceleration)
   
3. Parabola-Following Controller (on/near parabola):
   v_target = v_leader - sqrt(2 * a_comfort * (R - R_desired))
   a_des = -K_v * (v_ego - v_target)
   
4. CTG Following Controller (at equilibrium):
   a_des = K1 * (R - R_desired) + K2 * Ṙ

The controller uses CONTINUOUS parabola-based velocity tracking
without discrete mode switching to eliminate oscillations.
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
    
    VERSION 5.0: Exact Rajamani Chapter 6.7 Transitional Controller.
    
    Uses CONTINUOUS parabola-based velocity control:
    - Compute target velocity from parabola: v_target = v_leader - sqrt(2*a_comfort*gap_error)
    - Track this velocity with proportional controller
    - No discrete mode switching = no oscillations
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
        
        # === DETECTION ===
        self.DETECTION_RANGE = 150.0  # [m]
        
        # === RAJAMANI CHAPTER 6.7 PARAMETERS ===
        # Time headway for CTG policy (Section 6.3)
        self.h = 1.5  # [s]
        
        # Standstill distance
        self.d0 = 5.0  # [m]
        
        # Comfortable deceleration (Section 6.7.2)
        # Typical: 0.1g - 0.2g = 1.0 - 2.0 m/s²
        self.a_comfort = 1.5  # [m/s²]
        
        # Velocity tracking gain (Section 6.7.2)
        # Typical: 0.3 - 0.5 s⁻¹
        self.K_v = 0.4  # [1/s]
        
        # CTG Controller gains (Section 6.4, Eq. 6.15-6.16)
        # K1: position gain, K2: velocity gain
        self.K1 = 0.23  # [1/s²] - typical 0.2-0.4
        self.K2 = 0.6  # [1/s] - typical 0.6-1.0 (reduced for smoother response)
        
        # === COMFORT LIMITS ===
        self.MAX_ACCEL = 2.5   # [m/s²]
        self.MAX_DECEL = -3.5  # [m/s²]
        
        # === CRUISE PARAMETERS ===
        self.CATCHUP_FACTOR = 1.15
        
        # === DEBUG ===
        self._debug_counter = 0
        
        print(f"🚀 System Reference Generator V5.0 (RAJAMANI Ch.6.7)")
        print(f"   📊 Prediction: dt={self.dt}s, Np={self.Np}")
        print(f"   🎯 CTG: h={self.h}s, d0={self.d0}m")
        print(f"   📐 Comfort decel: a_comfort={self.a_comfort} m/s²")
        print(f"   📈 Gains: K_v={self.K_v}, K1={self.K1}, K2={self.K2}")

    def get_system_acceleration_and_state_sequence(self, simulation):
        """
        Generates the target trajectory using Rajamani Chapter 6.7 Transitional Controller.
        
        This is a CONTINUOUS controller based on parabola velocity tracking:
        
        1. CRUISE MODE: No leader detected
           - Accelerate to target platoon speed
           
        2. TRANSITIONAL MODE (Rajamani Section 6.7.2):
           - Compute target velocity from parabola equation:
             v_target = v_leader + sign(gap_error) * sqrt(2 * a_comfort * |gap_error|)
           - Track this velocity with proportional controller:
             a_des = K_v * (v_target - v_ego)
           
        The parabola equation ensures that if we track v_target, we will
        decelerate at exactly a_comfort and arrive at R_desired with v_rel = 0.
        
        This eliminates discrete mode switching and the oscillations it causes.
        """
        accel_sequence = np.zeros(self.Np)
        state_sequence = np.zeros((self.Np, 2))  # [position, velocity]
        
        # Clone current state for forward simulation
        sim_vehicle = copy.deepcopy(simulation.human_vehicle)
        
        # ========================================================================
        # Identify Leader - USE LOCKED if available
        # ========================================================================
        leader = None
        
        if hasattr(simulation.human_vehicle, 'target_leader_id'):
            target_id = simulation.human_vehicle.target_leader_id
            
            if target_id is not None:
                for v in simulation.platoon_manager.vehicles:
                    if v.vehicle_id == target_id:
                        leader = v
                        break
                
                if leader is None:
                    print(f"⚠️  WARNING (RefGen): Locked leader '{target_id}' not found!")
                    if hasattr(simulation, 'safety_field'):
                        leader = simulation.safety_field._get_leader(
                            simulation.human_vehicle, 
                            simulation.platoon_manager
                        )
            else:
                leader = None
                
        else:
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
        
        # Track for debug output
        initial_mode = "UNKNOWN"
        initial_gap_error = 0.0
        initial_R_dot = 0.0
        
        for i in range(self.Np):
            # === Step 1: Update Leader Prediction ===
            if sim_leader:
                sim_leader.state.x += sim_leader.state.vx * self.dt
            
            # === Step 2: Calculate Distance and Gap Error ===
            dist_to_leader = float('inf')
            gap_error = 0.0
            desired_gap = 50.0  # Default
            
            if sim_leader:
                # R = Range (actual gap, bumper to bumper)
                L = sim_leader.L if hasattr(sim_leader, 'L') else 5.0
                R = sim_leader.state.x - sim_vehicle.state.x - L
                
                # R_dot = Range rate (Rajamani convention: positive = gap increasing)
                R_dot = sim_leader.state.vx - sim_vehicle.state.vx
                
                # R_desired = Desired gap (CTG policy, Rajamani Eq. 6.5)
                R_desired = self.d0 + self.h * sim_vehicle.state.vx
                
                # Gap error: positive = too far behind, negative = too close
                gap_error = R - R_desired
                
                # Store for distance calculations
                dist_to_leader = sim_leader.state.x - sim_vehicle.state.x
                desired_gap = R_desired + L
                
                if i == 0:
                    initial_gap_error = gap_error
                    initial_R_dot = R_dot
            
            # === Step 3: Rajamani Section 6.7.2 Transitional Controller ===
            a_ref = 0.0
            mode = "UNKNOWN"
            
            if sim_leader is None or dist_to_leader > self.DETECTION_RANGE:
                # =============================================
                # CRUISE MODE: No leader or leader very far
                # =============================================
                mode = "CRUISE"
                target_v = target_platoon_speed
                
                if sim_leader:
                    target_v *= self.CATCHUP_FACTOR
                
                a_ref = free_road_acc(
                    v=sim_vehicle.state.vx,
                    t=0,
                    v_target=target_v,
                    a_max=self.MAX_ACCEL
                )
                
            else:
                # =============================================
                # TRANSITIONAL CONTROLLER (Rajamani Section 6.7.2)
                #
                # CONTINUOUS parabola-based velocity tracking.
                # No discrete mode switching!
                #
                # The parabola equation defines the target velocity:
                #   v_target = v_leader + sign(gap_error) * sqrt(2 * a_comfort * |gap_error|)
                #
                # If gap_error > 0 (too far): v_target > v_leader (approach)
                # If gap_error < 0 (too close): v_target < v_leader (back off)
                # If gap_error = 0: v_target = v_leader (maintain)
                # =============================================
                mode = "TRANSITIONAL"
                
                # Compute target velocity from parabola (Rajamani Eq. in Section 6.7.2)
                if gap_error >= 0:
                    # Too far behind - need to approach (v_ego > v_leader)
                    v_target = sim_leader.state.vx + np.sqrt(2 * self.a_comfort * gap_error)
                else:
                    # Too close - need to back off (v_ego < v_leader)
                    v_target = sim_leader.state.vx - np.sqrt(2 * self.a_comfort * abs(gap_error))
                
                # Velocity tracking controller (Rajamani Section 6.7.2)
                # a_des = K_v * (v_target - v_ego)
                v_error = v_target - sim_vehicle.state.vx
                a_ref = self.K_v * v_error
                
                # Add CTG feedback for better steady-state tracking (Rajamani Eq. 6.15)
                # This provides damping when near the equilibrium point
                # a_ctg = K1 * gap_error + K2 * R_dot
                a_ctg = self.K1 * gap_error + self.K2 * R_dot
                
                # Blend: use more CTG as we get closer to equilibrium
                # This provides smooth transition without discrete switching
                gap_magnitude = abs(gap_error)
                if gap_magnitude < 5.0:
                    # Near equilibrium - blend in CTG for fine control
                    blend = 1.0 - (gap_magnitude / 5.0)
                    blend = blend * blend  # Quadratic for smoother transition
                    a_ref = (1 - blend) * a_ref + blend * a_ctg
                
                # Feedforward from leader acceleration
                if hasattr(sim_leader.state, 'ax') and sim_leader.state.ax is not None:
                    a_ref += 0.5 * sim_leader.state.ax
            
            if i == 0:
                initial_mode = mode
            
            # === Step 4: Apply Comfort Constraints ===
            a_ref = np.clip(a_ref, self.MAX_DECEL, self.MAX_ACCEL)
            
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
        
        # Debug output (every 100 calls)
        self._debug_counter += 1
        if self._debug_counter % 100 == 1:
            print(f"   📍 RefGen: mode={initial_mode}, gap_err={initial_gap_error:.1f}m, "
                  f"R_dot={initial_R_dot:.2f}m/s, a_ref[0]={accel_sequence[0]:.2f} m/s²")
        
        return accel_sequence, state_sequence


# Export
__all__ = ['SystemReferenceGenerator']


# === UNIT TEST ===
if __name__ == "__main__":
    print("\n" + "="*60)
    print("System Reference Generator V5.0 - Unit Test")
    print("Rajamani Chapter 6.7 Transitional Controller")
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
            self.vehicle_id = "mock"
    
    class MockPlatoonManager:
        def __init__(self):
            self.target_velocity = 33.33  # 120 km/h
            self.vehicles = []
    
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
    print(f"Expected: Positive acceleration toward 33.33 m/s")
    
    print("\n--- Test 2: Large Gap (gap_error = 13.4m) ---")
    # Ego at x=0, v=28.6 (103 km/h). Leader at x=66.6, v=30.3 (109 km/h)
    # R = 66.6 - 5 = 61.6m (actual gap)
    # R_desired = 5 + 1.5*28.6 = 47.9m
    # gap_error = 61.6 - 47.9 = 13.7m
    # v_target = 30.3 + sqrt(2*1.5*13.7) = 30.3 + 6.4 = 36.7 m/s
    # v_error = 36.7 - 28.6 = 8.1 m/s
    # a_ref = 0.4 * 8.1 = 3.24 m/s² (capped to 2.5)
    sim2 = MockSimulation(ego_x=0, ego_v=28.6, leader_x=66.6, leader_v=30.3)
    accel2, states2 = gen.get_system_acceleration_and_state_sequence(sim2)
    print(f"Ego: x=0, v=28.6 m/s (103 km/h)")
    print(f"Leader: x=66.6m, v=30.3 m/s (109 km/h)")
    print(f"Gap error: ~14m")
    print(f"Accelerations: {accel2[:5].round(2)}")
    print(f"Expected: Positive acceleration ~2.5 m/s² (saturated)")
    
    if accel2[0] > 1.5:
        print("✅ PASS: Acceleration is sufficient to close gap")
    else:
        print("❌ FAIL: Acceleration too low!")
    
    print("\n--- Test 3: Approaching Too Fast ---")
    # Ego at x=0, v=35. Leader at x=60, v=30.
    # R = 60 - 5 = 55m
    # R_desired = 5 + 1.5*35 = 57.5m
    # gap_error = 55 - 57.5 = -2.5m (too close!)
    # v_target = 30 - sqrt(2*1.5*2.5) = 30 - 2.74 = 27.26 m/s
    # v_error = 27.26 - 35 = -7.74 m/s
    # a_ref = 0.4 * (-7.74) = -3.1 m/s² (decelerate)
    sim3 = MockSimulation(ego_x=0, ego_v=35, leader_x=60, leader_v=30)
    accel3, states3 = gen.get_system_acceleration_and_state_sequence(sim3)
    print(f"Ego: x=0, v=35 m/s")
    print(f"Leader: x=60m, v=30 m/s")
    print(f"Gap error: -2.5m (too close)")
    print(f"Accelerations: {accel3[:5].round(2)}")
    print(f"Expected: NEGATIVE acceleration to back off")
    
    if accel3[0] < -1.0:
        print("✅ PASS: Decelerating to back off")
    else:
        print("❌ FAIL: Should be decelerating!")
    
    print("\n--- Test 4: Steady State Following ---")
    # Ego at x=0, v=30. Leader at x=55, v=30.
    # R = 55 - 5 = 50m
    # R_desired = 5 + 1.5*30 = 50m
    # gap_error = 0
    # v_target = v_leader = 30 m/s
    # a_ref ≈ 0
    sim4 = MockSimulation(ego_x=0, ego_v=30, leader_x=55, leader_v=30)
    accel4, states4 = gen.get_system_acceleration_and_state_sequence(sim4)
    print(f"Ego: x=0, v=30 m/s")
    print(f"Leader: x=55m, v=30 m/s")
    print(f"Gap error: 0m (perfect)")
    print(f"Accelerations: {accel4[:5].round(2)}")
    print(f"Expected: Near ZERO (steady state)")
    
    if abs(accel4[0]) < 0.5:
        print("✅ PASS: Near-zero acceleration")
    else:
        print("❌ FAIL: Should be near zero!")
    
    print("\n" + "="*60)
    print("✅ Unit tests complete")
    print("="*60)