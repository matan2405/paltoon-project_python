"""
MOBIL Lane Change Decision Model
Based on Kesting, Treiber, and Helbing (2007)
"General Lane-Changing Model MOBIL for Car-Following Models"

This module implements the MOBIL (Minimizing Overall Braking Induced by Lane changes)
model for deciding WHEN a lane change is safe and beneficial.

The model uses IDM (Intelligent Driver Model) to compute accelerations and then
applies two criteria:
1. Safety Criterion: ã_n >= -b_safe (new follower won't brake too hard)
2. Incentive Criterion: ã_c - a_c + p*(ã_n - a_n) > Δa_th (lane change is beneficial)

Note: This module only decides IF a lane change should occur.
      The actual lane change execution is handled by Nash + Stanley.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DRIVER_PARAMS,
    MOBIL_IDM_V0, MOBIL_IDM_T, MOBIL_IDM_A_MAX, MOBIL_IDM_B,
    MOBIL_IDM_S0, MOBIL_IDM_DELTA, MOBIL_IDM_L,
    MOBIL_P, MOBIL_B_SAFE, MOBIL_A_TH, MOBIL_A_BIAS, MOBIL_MIN_GAP,
)


@dataclass
class IDMParams:
    """Intelligent Driver Model parameters (from Kesting et al.)"""
    v0: float = MOBIL_IDM_V0       # Desired velocity [m/s] (120 km/h)
    T: float = MOBIL_IDM_T         # Safe time headway [s]
    a: float = MOBIL_IDM_A_MAX     # Maximum acceleration [m/s²]
    b: float = MOBIL_IDM_B         # Comfortable deceleration [m/s²]
    s0: float = MOBIL_IDM_S0       # Minimum spacing [m]
    delta: float = MOBIL_IDM_DELTA  # Acceleration exponent
    L: float = MOBIL_IDM_L         # Vehicle length [m]


@dataclass
class MOBILParams:
    """MOBIL model parameters (from Kesting et al. Table 1)"""
    p: float = MOBIL_P             # Politeness factor [0=egoistic, 1=altruistic]
    b_safe: float = MOBIL_B_SAFE   # Maximum safe deceleration [m/s²]
    a_th: float = MOBIL_A_TH       # Changing threshold [m/s²]
    a_bias: float = MOBIL_A_BIAS   # Bias for right lane (asymmetric rules) [m/s²]

    # Mandatory lane change parameters
    mandatory_mode: bool = True         # If True, skip incentive criterion
    min_gap_front: float = MOBIL_MIN_GAP  # Minimum gap to vehicle in front [m]
    min_gap_rear: float = MOBIL_MIN_GAP   # Minimum gap to vehicle behind [m]


class IDM:
    """
    Intelligent Driver Model for computing accelerations.
    
    The IDM acceleration is:
    a = a_max * [1 - (v/v0)^δ - (s*/s)²]
    
    where s* = s0 + v*T + v*Δv/(2*sqrt(a*b))
    """
    
    def __init__(self, params: IDMParams = None):
        self.params = params or IDMParams()
    
    def compute_desired_gap(self, v: float, delta_v: float) -> float:
        """
        Compute desired minimum gap s*.
        
        s*(v, Δv) = s0 + v*T + v*Δv / (2*sqrt(a*b))
        
        Args:
            v: Own velocity [m/s]
            delta_v: Velocity difference (v_self - v_leader) [m/s]
        
        Returns:
            Desired gap [m]
        """
        p = self.params
        
        # Interaction term (only positive when approaching)
        interaction = (v * delta_v) / (2 * np.sqrt(p.a * p.b))
        interaction = max(0, interaction)  # Only active when approaching
        
        s_star = p.s0 + v * p.T + interaction
        return s_star
    
    def compute_acceleration(self, v: float, s: float, delta_v: float) -> float:
        """
        Compute IDM acceleration.
        
        a = a_max * [1 - (v/v0)^δ - (s*/s)²]
        
        Args:
            v: Own velocity [m/s]
            s: Gap to leader (bumper-to-bumper) [m]
            delta_v: Velocity difference (v_self - v_leader) [m/s]
        
        Returns:
            Acceleration [m/s²]
        """
        p = self.params
        
        # Free road term
        if p.v0 > 0.1:
            free_road = 1 - (v / p.v0) ** p.delta
        else:
            free_road = 0
        
        # Interaction term
        if s > 0.1:
            s_star = self.compute_desired_gap(v, delta_v)
            interaction = (s_star / s) ** 2
        else:
            interaction = 100.0  # Very high braking if too close
        
        # Total acceleration
        a_idm = p.a * (free_road - interaction)
        
        # Clip to physical limits
        return np.clip(a_idm, -p.b * 2, p.a)
    
    def compute_free_road_acceleration(self, v: float) -> float:
        """Compute acceleration on free road (no leader)."""
        p = self.params
        if p.v0 > 0.1:
            return p.a * (1 - (v / p.v0) ** p.delta)
        return 0.0


class MOBILLaneChange:
    """
    MOBIL Lane Change Decision Model.
    
    Decides WHETHER a lane change is safe and beneficial based on:
    1. Safety Criterion: The new follower won't have to brake dangerously
    2. Incentive Criterion: The lane change provides sufficient advantage
    
    For our platoon merging scenario:
    - Vehicle 'c' is the human vehicle wanting to join the platoon
    - Vehicle 'n' is the new follower (platoon vehicle behind merge point)
    - Vehicle 'o' is the old follower (not relevant - human is in different lane)
    """
    
    def __init__(self, idm_params: IDMParams = None, mobil_params: MOBILParams = None):
        self.idm = IDM(idm_params)
        self.params = mobil_params or MOBILParams()
        
        # Store last evaluation for debugging
        self.last_evaluation = {}
        
        print(f"🚗 MOBIL Lane Change Model Initialized")
        print(f"   p={self.params.p}, b_safe={self.params.b_safe} m/s², a_th={self.params.a_th} m/s²")
    
    def set_politeness(self, driver_type: str):
        """Set politeness factor based on driver type (reads from DRIVER_PARAMS)."""
        p = DRIVER_PARAMS.get(driver_type, DRIVER_PARAMS['normal'])
        self.params.p     = p['mobil_p']
        self.params.a_th  = p['mobil_a_th']
        print(f"   MOBIL politeness set for {driver_type}: p={self.params.p}")
    
    def check_lane_change(self, 
                          ego_x: float, ego_v: float,
                          leader_x: Optional[float], leader_v: Optional[float],
                          new_follower_x: Optional[float], new_follower_v: Optional[float],
                          new_leader_x: Optional[float], new_leader_v: Optional[float]) -> Tuple[bool, Dict]:
        """
        Check if lane change is safe and beneficial.
        
        Scenario: Ego vehicle (human) wants to merge into platoon lane.
        
        Args:
            ego_x, ego_v: Ego vehicle position and velocity
            leader_x, leader_v: Current lane leader (if any) - often None for merging
            new_follower_x, new_follower_v: Vehicle that will be behind ego after merge
            new_leader_x, new_leader_v: Vehicle that will be in front of ego after merge
        
        Returns:
            (is_safe_and_beneficial, details_dict)
        """
        L = self.idm.params.L  # Vehicle length
        
        # === 1. Compute current acceleration (a_c) ===
        # In current lane, ego might have no leader or a distant one
        if leader_x is not None and leader_v is not None:
            gap_current = leader_x - ego_x - L
            delta_v_current = ego_v - leader_v
            a_c = self.idm.compute_acceleration(ego_v, gap_current, delta_v_current)
        else:
            # Free road
            a_c = self.idm.compute_free_road_acceleration(ego_v)
        
        # === 2. Compute acceleration after lane change (ã_c) ===
        # After merge, ego follows new_leader
        if new_leader_x is not None and new_leader_v is not None:
            gap_new = new_leader_x - ego_x - L
            delta_v_new = ego_v - new_leader_v
            a_c_tilde = self.idm.compute_acceleration(ego_v, gap_new, delta_v_new)
        else:
            # No new leader - free road in target lane
            a_c_tilde = self.idm.compute_free_road_acceleration(ego_v)
        
        # === 3. Compute new follower's acceleration after merge (ã_n) ===
        # This is the SAFETY CHECK - will new follower brake too hard?
        if new_follower_x is not None and new_follower_v is not None:
            # After merge, new_follower follows ego
            gap_follower_new = ego_x - new_follower_x - L
            delta_v_follower = new_follower_v - ego_v
            a_n_tilde = self.idm.compute_acceleration(new_follower_v, gap_follower_new, delta_v_follower)
            
            # Also compute new follower's current acceleration (before merge)
            if new_leader_x is not None:
                # Before merge, new_follower follows new_leader
                gap_follower_current = new_leader_x - new_follower_x - L
                delta_v_follower_current = new_follower_v - new_leader_v
                a_n = self.idm.compute_acceleration(new_follower_v, gap_follower_current, delta_v_follower_current)
            else:
                a_n = self.idm.compute_free_road_acceleration(new_follower_v)
        else:
            # No new follower - no safety concern from behind
            a_n_tilde = 0.0
            a_n = 0.0
        
        # === 4. Safety Criterion ===
        # ã_n >= -b_safe
        safety_ok = a_n_tilde >= -self.params.b_safe
        
        # === 5. Incentive Criterion (Symmetric rule) ===
        # ã_c - a_c + p*(ã_n - a_n + ã_o - a_o) > Δa_th
        # Since ego is changing from different lane, ã_o - a_o ≈ 0 (no old follower impact)
        
        ego_advantage = a_c_tilde - a_c
        follower_disadvantage = a_n_tilde - a_n
        
        incentive = ego_advantage + self.params.p * follower_disadvantage
        incentive_ok = incentive > self.params.a_th
        
        # === 6. Gap Check (additional safety) ===
        gap_front_ok = True
        gap_rear_ok = True
        gap_front = None
        gap_rear = None
        
        if new_leader_x is not None:
            gap_front = new_leader_x - ego_x - L
            # For 'join_after': if no follower, we're joining as last vehicle
            # In this case, we only need to be behind the leader (not necessarily with 8m gap)
            # The gap will be established during the merge maneuver
            if new_follower_x is None:
                # Joining as LAST vehicle - just need to not overlap
                gap_front_ok = gap_front >= -L  # Allow being close or even slightly ahead
            else:
                # Joining in MIDDLE - need proper gap
                gap_front_ok = gap_front >= self.params.min_gap_front
        
        if new_follower_x is not None:
            gap_rear = ego_x - new_follower_x - L
            # For 'join_before': if no leader, we're joining as first vehicle
            if new_leader_x is None:
                # Joining as FIRST vehicle - just need to not overlap
                gap_rear_ok = gap_rear >= -L
            else:
                # Joining in MIDDLE - need proper gap
                gap_rear_ok = gap_rear >= self.params.min_gap_rear
        
        gaps_ok = gap_front_ok and gap_rear_ok
        
        # === 7. Overall Decision ===
        if self.params.mandatory_mode:
            # For mandatory lane change (platoon merging):
            # Only check safety criterion and gap constraints
            # Skip incentive (driver WANTS to merge regardless of advantage)
            lane_change_approved = safety_ok and gaps_ok
        else:
            # Standard MOBIL: need both safety and incentive
            lane_change_approved = safety_ok and incentive_ok and gaps_ok
        
        # === 8. Store details for debugging ===
        self.last_evaluation = {
            'a_c': a_c,
            'a_c_tilde': a_c_tilde,
            'a_n': a_n,
            'a_n_tilde': a_n_tilde,
            'ego_advantage': ego_advantage,
            'follower_disadvantage': follower_disadvantage,
            'incentive': incentive,
            'safety_ok': safety_ok,
            'incentive_ok': incentive_ok,
            'gaps_ok': gaps_ok,
            'gap_front_ok': gap_front_ok,
            'gap_rear_ok': gap_rear_ok,
            'mandatory_mode': self.params.mandatory_mode,
            'approved': lane_change_approved,
            'gap_to_new_leader': gap_front,
            'gap_from_new_follower': gap_rear
        }
        
        return lane_change_approved, self.last_evaluation
    
    def check_platoon_merge(self,
                            human_x: float, human_v: float,
                            platoon_vehicles: List[Dict],
                            merge_position: str = 'middle') -> Tuple[bool, Dict]:
        """
        High-level function to check if merging into platoon is safe.
        
        Args:
            human_x: Human vehicle x position
            human_v: Human vehicle velocity
            platoon_vehicles: List of dicts with 'x', 'v' for each platoon vehicle
                             Sorted by x position (front to back)
            merge_position: 'before', 'middle', 'after' - intended merge position
        
        Returns:
            (is_safe, details_dict)
        """
        if not platoon_vehicles:
            # No platoon - always safe
            return True, {'reason': 'no_platoon', 'approved': True}
        
        # Sort platoon by x position (front to back)
        sorted_platoon = sorted(platoon_vehicles, key=lambda v: v['x'], reverse=True)
        
        # Find merge gap based on position
        if merge_position == 'before':
            # Merge in front of platoon leader
            new_leader_x = None
            new_leader_v = None
            new_follower_x = sorted_platoon[0]['x']
            new_follower_v = sorted_platoon[0]['v']
            
        elif merge_position == 'after':
            # Merge behind platoon (last vehicle)
            new_leader_x = sorted_platoon[-1]['x']
            new_leader_v = sorted_platoon[-1]['v']
            new_follower_x = None
            new_follower_v = None
            
        else:  # 'middle'
            # Find the appropriate gap
            # Ego is at human_x, find gap where ego would fit
            for i in range(len(sorted_platoon) - 1):
                front_x = sorted_platoon[i]['x']
                back_x = sorted_platoon[i + 1]['x']
                
                # Check if human is roughly aligned with this gap
                if back_x < human_x < front_x:
                    new_leader_x = front_x
                    new_leader_v = sorted_platoon[i]['v']
                    new_follower_x = back_x
                    new_follower_v = sorted_platoon[i + 1]['v']
                    break
            else:
                # Human not aligned with any gap - use closest
                # Default to joining after leader
                if len(sorted_platoon) >= 2:
                    new_leader_x = sorted_platoon[0]['x']
                    new_leader_v = sorted_platoon[0]['v']
                    new_follower_x = sorted_platoon[1]['x']
                    new_follower_v = sorted_platoon[1]['v']
                else:
                    new_leader_x = sorted_platoon[0]['x']
                    new_leader_v = sorted_platoon[0]['v']
                    new_follower_x = None
                    new_follower_v = None
        
        # Current lane has no leader (simplification)
        leader_x = None
        leader_v = None
        
        return self.check_lane_change(
            ego_x=human_x, ego_v=human_v,
            leader_x=leader_x, leader_v=leader_v,
            new_follower_x=new_follower_x, new_follower_v=new_follower_v,
            new_leader_x=new_leader_x, new_leader_v=new_leader_v
        )
    
    def get_status_string(self) -> str:
        """Get human-readable status of last evaluation."""
        e = self.last_evaluation
        if not e:
            return "No evaluation yet"
        
        status = "✅ APPROVED" if e.get('approved') else "❌ NOT APPROVED"
        safety = "✓" if e.get('safety_ok') else "✗"
        gaps = "✓" if e.get('gaps_ok') else "✗"
        mode = "MANDATORY" if e.get('mandatory_mode') else "DISCRETIONARY"
        
        result = (f"MOBIL ({mode}): {status}\n"
                  f"  Safety [{safety}]: ã_n={e.get('a_n_tilde', 0):.2f} >= -{self.params.b_safe}\n"
                  f"  Gaps [{gaps}]: front={e.get('gap_to_new_leader')}, rear={e.get('gap_from_new_follower')}")
        
        if not e.get('mandatory_mode'):
            incentive = "✓" if e.get('incentive_ok') else "✗"
            result += f"\n  Incentive [{incentive}]: {e.get('incentive', 0):.2f} > {self.params.a_th}"
        
        return result


# ============================================================================
# UNIT TEST
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("MOBIL Lane Change Model - Unit Test")
    print("="*60)
    
    # Create MOBIL model
    mobil = MOBILLaneChange()
    
    # Test scenario: Human at x=50, platoon vehicles at x=70, 55, 40
    human_x = 50.0
    human_v = 20.0  # 72 km/h
    
    platoon = [
        {'x': 70.0, 'v': 20.0},  # Leader
        {'x': 55.0, 'v': 20.0},  # Middle
        {'x': 40.0, 'v': 20.0},  # Last
    ]
    
    print(f"\n📍 Test Scenario:")
    print(f"   Human: x={human_x}m, v={human_v}m/s")
    print(f"   Platoon: {platoon}")
    
    # Test different merge positions
    for position in ['before', 'middle', 'after']:
        print(f"\n🎯 Testing merge position: {position}")
        approved, details = mobil.check_platoon_merge(human_x, human_v, platoon, position)
        print(mobil.get_status_string())
        if details.get('gap_to_new_leader'):
            print(f"   Gap to new leader: {details['gap_to_new_leader']:.1f}m")
        if details.get('gap_from_new_follower'):
            print(f"   Gap from new follower: {details['gap_from_new_follower']:.1f}m")
    
    # Test with different driver types
    print(f"\n" + "="*60)
    print("Testing driver personalities")
    print("="*60)
    
    for driver in ['cautious', 'normal', 'aggressive']:
        mobil.set_politeness(driver)
        approved, details = mobil.check_platoon_merge(human_x, human_v, platoon, 'middle')
        status = "✅" if approved else "❌"
        print(f"   {driver}: {status} (incentive={details['incentive']:.3f})")
    
    print("\n" + "="*60)
    print("✅ Unit test complete")
    print("="*60)
