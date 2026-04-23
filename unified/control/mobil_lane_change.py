"""
unified/control/mobil_lane_change.py — MOBIL Lane Change Decision Model.

Copied from lateral/control/mobil_lane_change.py with all imports
redirected to unified/config.py.
No dependency on the split Longitudinal/ or lateral/ module trees.

Based on Kesting, Treiber, and Helbing (2007)
"General Lane-Changing Model MOBIL for Car-Following Models"
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

from unified.config import (
    DRIVER_PARAMS,
    MOBIL_IDM_V0, MOBIL_IDM_T, MOBIL_IDM_A_MAX, MOBIL_IDM_B,
    MOBIL_IDM_S0, MOBIL_IDM_DELTA, MOBIL_IDM_L,
    MOBIL_P, MOBIL_B_SAFE, MOBIL_A_TH, MOBIL_A_BIAS, MOBIL_MIN_GAP,
)


@dataclass
class IDMParams:
    """Intelligent Driver Model parameters (from Kesting et al.)"""
    v0: float = MOBIL_IDM_V0
    T: float = MOBIL_IDM_T
    a: float = MOBIL_IDM_A_MAX
    b: float = MOBIL_IDM_B
    s0: float = MOBIL_IDM_S0
    delta: float = MOBIL_IDM_DELTA
    L: float = MOBIL_IDM_L


@dataclass
class MOBILParams:
    """MOBIL model parameters (from Kesting et al. Table 1)"""
    p: float = MOBIL_P
    b_safe: float = MOBIL_B_SAFE
    a_th: float = MOBIL_A_TH
    a_bias: float = MOBIL_A_BIAS

    mandatory_mode: bool = True
    min_gap_front: float = MOBIL_MIN_GAP
    min_gap_rear: float = MOBIL_MIN_GAP


class IDM:
    """Intelligent Driver Model for computing accelerations."""

    def __init__(self, params: IDMParams = None):
        self.params = params or IDMParams()

    def compute_desired_gap(self, v: float, delta_v: float) -> float:
        p = self.params
        interaction = (v * delta_v) / (2 * np.sqrt(p.a * p.b))
        interaction = max(0, interaction)
        return p.s0 + v * p.T + interaction

    def compute_acceleration(self, v: float, s: float, delta_v: float) -> float:
        p = self.params

        if p.v0 > 0.1:
            free_road = 1 - (v / p.v0) ** p.delta
        else:
            free_road = 0

        if s > 0.1:
            s_star = self.compute_desired_gap(v, delta_v)
            interaction = (s_star / s) ** 2
        else:
            interaction = 100.0

        a_idm = p.a * (free_road - interaction)
        return np.clip(a_idm, -p.b * 2, p.a)

    def compute_free_road_acceleration(self, v: float) -> float:
        p = self.params
        if p.v0 > 0.1:
            return p.a * (1 - (v / p.v0) ** p.delta)
        return 0.0


class MOBILLaneChange:
    """MOBIL Lane Change Decision Model."""

    def __init__(self, idm_params: IDMParams = None, mobil_params: MOBILParams = None):
        self.idm = IDM(idm_params)
        self.params = mobil_params or MOBILParams()
        self.last_evaluation = {}

        print(f"MOBIL Lane Change Model Initialized")
        print(f"   p={self.params.p}, b_safe={self.params.b_safe} m/s², a_th={self.params.a_th} m/s²")

    def set_politeness(self, driver_type: str):
        """Set politeness factor based on driver type."""
        p = DRIVER_PARAMS.get(driver_type, DRIVER_PARAMS['normal'])
        self.params.p = p['mobil_p']
        self.params.a_th = p['mobil_a_th']
        print(f"   MOBIL politeness set for {driver_type}: p={self.params.p}")

    def check_lane_change(self,
                          ego_x: float, ego_v: float,
                          leader_x: Optional[float], leader_v: Optional[float],
                          new_follower_x: Optional[float], new_follower_v: Optional[float],
                          new_leader_x: Optional[float], new_leader_v: Optional[float]) -> Tuple[bool, Dict]:
        """Check if lane change is safe and beneficial."""
        L = self.idm.params.L

        # 1. Current acceleration (a_c)
        if leader_x is not None and leader_v is not None:
            gap_current = leader_x - ego_x - L
            delta_v_current = ego_v - leader_v
            a_c = self.idm.compute_acceleration(ego_v, gap_current, delta_v_current)
        else:
            a_c = self.idm.compute_free_road_acceleration(ego_v)

        # 2. Acceleration after lane change (ã_c)
        if new_leader_x is not None and new_leader_v is not None:
            gap_new = new_leader_x - ego_x - L
            delta_v_new = ego_v - new_leader_v
            a_c_tilde = self.idm.compute_acceleration(ego_v, gap_new, delta_v_new)
        else:
            a_c_tilde = self.idm.compute_free_road_acceleration(ego_v)

        # 3. New follower's acceleration after merge (ã_n) — safety check
        if new_follower_x is not None and new_follower_v is not None:
            gap_follower_new = ego_x - new_follower_x - L
            delta_v_follower = new_follower_v - ego_v
            a_n_tilde = self.idm.compute_acceleration(new_follower_v, gap_follower_new, delta_v_follower)

            if new_leader_x is not None:
                gap_follower_current = new_leader_x - new_follower_x - L
                delta_v_follower_current = new_follower_v - new_leader_v
                a_n = self.idm.compute_acceleration(new_follower_v, gap_follower_current, delta_v_follower_current)
            else:
                a_n = self.idm.compute_free_road_acceleration(new_follower_v)
        else:
            a_n_tilde = 0.0
            a_n = 0.0

        # 4. Safety criterion
        safety_ok = a_n_tilde >= -self.params.b_safe

        # 5. Incentive criterion
        ego_advantage = a_c_tilde - a_c
        follower_disadvantage = a_n_tilde - a_n
        incentive = ego_advantage + self.params.p * follower_disadvantage
        incentive_ok = incentive > self.params.a_th

        # 6. Gap check
        gap_front_ok = True
        gap_rear_ok = True
        gap_front = None
        gap_rear = None

        if new_leader_x is not None:
            gap_front = new_leader_x - ego_x - L
            if new_follower_x is None:
                gap_front_ok = gap_front >= -L
            else:
                gap_front_ok = gap_front >= self.params.min_gap_front

        if new_follower_x is not None:
            gap_rear = ego_x - new_follower_x - L
            if new_leader_x is None:
                gap_rear_ok = gap_rear >= -L
            else:
                gap_rear_ok = gap_rear >= self.params.min_gap_rear

        gaps_ok = gap_front_ok and gap_rear_ok

        # 7. Decision
        if self.params.mandatory_mode:
            lane_change_approved = safety_ok and gaps_ok
        else:
            lane_change_approved = safety_ok and incentive_ok and gaps_ok

        self.last_evaluation = {
            'a_c': a_c, 'a_c_tilde': a_c_tilde,
            'a_n': a_n, 'a_n_tilde': a_n_tilde,
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
            'gap_from_new_follower': gap_rear,
        }

        return lane_change_approved, self.last_evaluation

    def check_platoon_merge(self,
                            human_x: float, human_v: float,
                            platoon_vehicles: List[Dict],
                            merge_position: str = 'middle') -> Tuple[bool, Dict]:
        """High-level check if merging into platoon is safe."""
        if not platoon_vehicles:
            return True, {'reason': 'no_platoon', 'approved': True}

        sorted_platoon = sorted(platoon_vehicles, key=lambda v: v['x'], reverse=True)

        if merge_position == 'before':
            new_leader_x = None
            new_leader_v = None
            new_follower_x = sorted_platoon[0]['x']
            new_follower_v = sorted_platoon[0]['v']

        elif merge_position == 'after':
            new_leader_x = sorted_platoon[-1]['x']
            new_leader_v = sorted_platoon[-1]['v']
            new_follower_x = None
            new_follower_v = None

        else:  # 'middle'
            for i in range(len(sorted_platoon) - 1):
                front_x = sorted_platoon[i]['x']
                back_x = sorted_platoon[i + 1]['x']
                if back_x < human_x < front_x:
                    new_leader_x = front_x
                    new_leader_v = sorted_platoon[i]['v']
                    new_follower_x = back_x
                    new_follower_v = sorted_platoon[i + 1]['v']
                    break
            else:
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

        return self.check_lane_change(
            ego_x=human_x, ego_v=human_v,
            leader_x=None, leader_v=None,
            new_follower_x=new_follower_x, new_follower_v=new_follower_v,
            new_leader_x=new_leader_x, new_leader_v=new_leader_v,
        )

    def get_status_string(self) -> str:
        e = self.last_evaluation
        if not e:
            return "No evaluation yet"

        status = "APPROVED" if e.get('approved') else "NOT APPROVED"
        safety = "ok" if e.get('safety_ok') else "FAIL"
        gaps = "ok" if e.get('gaps_ok') else "FAIL"
        mode = "MANDATORY" if e.get('mandatory_mode') else "DISCRETIONARY"

        result = (f"MOBIL ({mode}): {status}\n"
                  f"  Safety [{safety}]: a_n_tilde={e.get('a_n_tilde', 0):.2f} >= -{self.params.b_safe}\n"
                  f"  Gaps [{gaps}]: front={e.get('gap_to_new_leader')}, rear={e.get('gap_from_new_follower')}")

        if not e.get('mandatory_mode'):
            incentive = "ok" if e.get('incentive_ok') else "FAIL"
            result += f"\n  Incentive [{incentive}]: {e.get('incentive', 0):.2f} > {self.params.a_th}"

        return result


__all__ = ['MOBILLaneChange', 'IDM', 'IDMParams', 'MOBILParams']
