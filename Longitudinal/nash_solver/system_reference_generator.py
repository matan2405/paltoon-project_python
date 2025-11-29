# #!/usr/bin/env python3
# """
# File: system_reference_generator.py  
# Description: Generates target trajectories for the System player (R1).
# UPDATED: Robust logic for Join Before/After scenarios.
# """
# import numpy as np
# import copy
# import sys
# import os

# # Add parent directory to path to allow importing from control
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from control.platoon_control import free_road_acc, rajamani

# class SystemReferenceGenerator:
#     def __init__(self, Np: int = 20, dt: float = 0.1):
#         self.Np = Np
#         self.dt = dt
#         print(f"👍 System Reference Generator (Robust): dt={self.dt}s, Np={self.Np}")

#     def get_system_acceleration_and_state_sequence(self, simulation):
#         accel_sequence = np.zeros(self.Np)
#         state_sequence = np.zeros((self.Np, 2))
        
#         human_vehicle = simulation.human_vehicle
#         platoon_manager = simulation.platoon_manager
        
#         sim_vehicle = copy.deepcopy(human_vehicle)
#         sim_x = sim_vehicle.state.x
#         sim_v = sim_vehicle.state.vx
        
#         # Identify Leader
#         leader = None
#         if hasattr(simulation, 'safety_field'):
#             leader = simulation.safety_field._get_leader(human_vehicle, platoon_manager)
        
#         leader_sim = copy.deepcopy(leader) if leader else None
        
#         target_platoon_speed = platoon_manager.target_velocity
        
#         # --- ROBUST MODE SELECTION ---
#         # Mode 1: Cruise (No leader OR Leader very far)
#         # Mode 2: Following (Leader close)
        
#         distance_to_leader = float('inf')
#         if leader_sim:
#             distance_to_leader = leader_sim.state.x - sim_x
        
#         # Threshold for switching to Following mode
#         FOLLOWING_DISTANCE_THRESHOLD = 120.0 
        
#         for i in range(self.Np):
#             # Update leader state
#             if leader_sim:
#                 leader_sim.state.x += leader_sim.state.vx * self.dt
#                 distance_to_leader = leader_sim.state.x - sim_x
            
#             a_des = 0.0
            
#             # --- LOGIC ---
#             if leader_sim and distance_to_leader < FOLLOWING_DISTANCE_THRESHOLD:
#                 # === FOLLOWING MODE (Join Middle / Approaching) ===
#                 h = 1.5
#                 L = leader.params.length if hasattr(leader, 'params') else 5.0
#                 desired_gap = L + h * sim_v
                
#                 gap_error = distance_to_leader - desired_gap
                
#                 if gap_error > 5.0:
#                     # Gap Closing: Smoothly approach
#                     # Max safe speed based on distance
#                     v_safe = leader_sim.state.vx + np.sqrt(2 * 1.0 * gap_error)
                    
#                     # Target is slightly faster than leader, but limited by safety
#                     v_target = min(target_platoon_speed * 1.15, v_safe)
                    
#                     a_des = free_road_acc(sim_v, 0, v_target, 2.5)
#                 else:
#                     # Stable Platoon (Rajamani)
#                     # We need to pass the updated virtual vehicle to rajamani
#                     virtual_follower = copy.deepcopy(simulation.human_vehicle)
#                     virtual_follower.state.x = sim_x
#                     virtual_follower.state.vx = sim_v
#                     virtual_follower.state.ax = accel_sequence[i-1] if i > 0 else 0
#                     a_des, _ = rajamani(leader_sim, virtual_follower)
            
#             else:
#                 # === CRUISE MODE (Join Before / Join After far away) ===
#                 # Just accelerate to target speed. Simple.
                
#                 # If we are in "Join Before" (leader is None), we just want target speed.
#                 # If we are in "Join After" (leader far), we also just want target speed.
                
#                 # Allow slight overspeed to catch up if we know there is a platoon ahead
#                 target_v = target_platoon_speed
#                 if leader_sim: # We know there is a leader, just far away
#                     target_v *= 1.1
                
#                 a_des = free_road_acc(
#                     v=sim_v,
#                     t=0,
#                     v_target=target_v,
#                     a_max=simulation.human_vehicle.params.max_acceleration
#                 )

#             # Constraints
#             a_des = np.clip(a_des, -3.0, 2.5)
            
#             # Integrate
#             sim_vehicle.a_desired = a_des
#             sim_vehicle.update_dynamics(self.dt)
            
#             sim_x = sim_vehicle.state.x
#             sim_v = sim_vehicle.state.vx
            
#             accel_sequence[i] = a_des
#             state_sequence[i, 0] = sim_x
#             state_sequence[i, 1] = sim_v
            
#         return accel_sequence, state_sequence

# __all__ = ['SystemReferenceGenerator']

#!/usr/bin/env python3
"""
File: system_reference_generator.py  
Description: Generates target trajectories for the System player (R1).
UPDATED: Elegant switching between Free Flow and Rajamani Car-Following.
"""
import numpy as np
import copy
import sys
import os

# Add parent directory to path to allow importing from control
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control.platoon_control import free_road_acc, rajamani

class SystemReferenceGenerator:
    def __init__(self, Np: int = 20, dt: float = 0.1):
        self.Np = Np
        self.dt = dt
        print(f"👍 System Reference Generator (Rajamani/FreeFlow): dt={self.dt}s, Np={self.Np}")

    def get_system_acceleration_and_state_sequence(self, simulation):
        """
        Generates the target trajectory using 'free_road_acc' for cruising 
        and 'rajamani' for car-following.
        """
        accel_sequence = np.zeros(self.Np)
        state_sequence = np.zeros((self.Np, 2))
        
        # Clone current state to simulate forward
        sim_vehicle = copy.deepcopy(simulation.human_vehicle)
        
        # Identify Leader
        leader = None
        if hasattr(simulation, 'safety_field'):
            leader = simulation.safety_field._get_leader(simulation.human_vehicle, simulation.platoon_manager)
        
        # Clone leader for prediction (simple constant velocity prediction)
        sim_leader = copy.deepcopy(leader) if leader else None
        
        target_platoon_speed = simulation.platoon_manager.target_velocity
        
        # Logic Constants
        # טווח הזיהוי שבו אנחנו מפסיקים להיות "לבד" ומתחילים להתייחס למוביל
        DETECTION_RANGE = 120.0  
        # פקטור שמאפשר לנסוע טיפה יותר מהר כדי לסגור פערים כשאנחנו במצב שיוט
        CATCHUP_FACTOR = 1.1     
        
        for i in range(self.Np):
            # 1. Update Leader Prediction (Constant Velocity assumption for short horizon)
            if sim_leader:
                sim_leader.state.x += sim_leader.state.vx * self.dt
            
            dist_to_leader = float('inf')
            if sim_leader:
                dist_to_leader = sim_leader.state.x - sim_vehicle.state.x

            # 2. Select Control Law (Elegant Switching)
            a_ref = 0.0
            
            # אם יש מוביל והוא בטווח הראייה שלנו (120 מטר) - תן לרג'אמני לנהוג
            if sim_leader and 0 < dist_to_leader < DETECTION_RANGE:
                # === FOLLOW MODE: Use Rajamani ===
                # Rajamani controller calculates optimal accel to maintain gap 'h'
                # We pass the simulated vehicles to it directly
                a_ref, _ = rajamani(sim_leader, sim_vehicle)
                
            else:
                # === CRUISE/CATCH-UP MODE: Use Free Road Acc ===
                # אם אנחנו רחוקים או אין מוביל, נאיץ למהירות המטרה
                target_v = target_platoon_speed
                # אם יש מוביל אבל הוא רחוק, מותר לנסוע קצת יותר מהר כדי להתקרב
                if sim_leader: 
                    target_v *= CATCHUP_FACTOR
                
                a_ref = free_road_acc(
                    v=sim_vehicle.state.vx,
                    t=0,
                    v_target=target_v,
                    a_max=sim_vehicle.params.max_acceleration
                )

            # 3. Apply Limits (Comfort constraints)
            a_ref = np.clip(a_ref, -3.5, 2.0)
            
            # 4. Update Simulated Ego Vehicle (Kinematics) for the next prediction step
            sim_vehicle.state.ax = a_ref  # Important for Rajamani in next step!
            sim_vehicle.a = a_ref         # Sync for compatibility
            
            # Euler integration
            sim_vehicle.state.x += sim_vehicle.state.vx * self.dt + 0.5 * a_ref * self.dt**2
            sim_vehicle.state.vx += a_ref * self.dt
            sim_vehicle.v = sim_vehicle.state.vx # Sync
            
            # Store prediction
            accel_sequence[i] = a_ref
            state_sequence[i, 0] = sim_vehicle.state.x
            state_sequence[i, 1] = sim_vehicle.state.vx
            
        return accel_sequence, state_sequence