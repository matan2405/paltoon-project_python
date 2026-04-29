"""
unified/simulation/simulator.py — Unified platoon merge simulation.

Orchestrates:
  - N platoon vehicles (Longitudinal Vehicle class, longitudinal dynamics only)
  - 1 ego vehicle (Vehicle6D, full 6D dynamics)
  - UnifiedCoordinator (Nash shared control, phase management, MOBIL)

Loop structure:
  ┌─ for each sim step (100 Hz) ──────────────────────────────────────────┐
  │  1. platoon_manager.update(dt)           — Rajamani + Nash injection   │
  │  2. (u_long, delta) = coordinator.step() — Nash solves at sub-rates    │
  │  3. ego.update_dynamics_6d(u_long, delta, dt)                          │
  │  4. log_data()                                                          │
  └───────────────────────────────────────────────────────────────────────┘
"""

import os
import sys
import numpy as np
import time as time_module
from typing import Dict, List, Optional

# ── paths ────────────────────────────────────────────────────────────────────
_REPO_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_UNIFIED_DIR = os.path.join(_REPO_ROOT, 'unified')

sys.path.insert(0, _REPO_ROOT)
if _UNIFIED_DIR not in sys.path:
    sys.path.insert(1, _UNIFIED_DIR)
from unified.config import (
    SIMULATION_DT, DEFAULT_SIMULATION_TIME, RESULTS_DIR,
    PLATOON_TARGET_VELOCITY, PLATOON_VEHICLE_LENGTH, PLATOON_TIME_GAP,
    PLATOON_STANDSTILL_DISTANCE,
    NOMINAL_VELOCITY, DRIVER_PARAMS,
    PLATOON_LANE_Y, HUMAN_INITIAL_LANE_Y,
    VEHICLE_LENGTH, MAX_ACCELERATION, MAX_DECELERATION,
    MOBIL_IDM_A_MAX, MOBIL_IDM_DELTA,
    LAT_DSF_TS, LAT_DSF_A_MIN,
)
from unified.vehicle.vehicle_6d import Vehicle6D
from unified.control.coordinator import UnifiedCoordinator, MergePhase
from unified.vehicle.platoon_vehicle import Vehicle as LongVehicle
from unified.control.platoon_control import PlatoonManager as LongPlatoonManager


# =============================================================================
# UnifiedSimulation
# =============================================================================

class UnifiedSimulation:
    """
    End-to-end unified longitudinal + lateral platoon merge simulation.

    Parameters
    ----------
    num_platoon_vehicles : int
        Number of pre-existing platoon members.
    driver_type : str
        'cautious' | 'normal' | 'aggressive' — ego driver profile.
    merge_scenario : str
        'back' | 'middle' | 'front' — where ego joins the platoon.
    ego_initial_x : float
        Ego starting x-position override [m].  If None, computed from scenario.
    T_sim : float
        Total simulation time [s].
    merge_trigger_time : float
        Time [s] at which MOBIL checking starts (platoon approaching target speed).
    """

    def __init__(self,
                 num_platoon_vehicles: int = 3,
                 driver_type: str = 'normal',
                 merge_scenario: str = 'back',
                 ego_initial_x: Optional[float] = None,
                 T_sim: float = DEFAULT_SIMULATION_TIME,
                 merge_trigger_time: float = 30.0,
                 noise_mode: str = 'gp',
                 fixed_lambda_long: Optional[float] = None,
                 fixed_lambda_lat: Optional[float] = None,
                 weight_overrides: Optional[Dict] = None):

        self.dt                 = SIMULATION_DT
        self.T_sim              = T_sim
        self.time               = 0.0
        self.driver_type        = driver_type
        self.merge_scenario     = merge_scenario
        self.merge_trigger_time = merge_trigger_time
        self._noise_mode        = noise_mode
        self._fixed_lambda_long = fixed_lambda_long
        self._fixed_lambda_lat  = fixed_lambda_lat
        self._weight_overrides  = weight_overrides

        dp = DRIVER_PARAMS.get(driver_type, DRIVER_PARAMS['normal'])

        # ── Platoon vehicles — all start at v=0 ─────────────────────────────
        # Standstill spacing: bumper-to-bumper = PLATOON_STANDSTILL_DISTANCE
        self.platoon_vehicles: List[LongVehicle] = []
        for i in range(num_platoon_vehicles):
            x_i  = -i * (PLATOON_STANDSTILL_DISTANCE + PLATOON_VEHICLE_LENGTH)
            pv   = LongVehicle(
                initial_x=x_i,
                initial_y=PLATOON_LANE_Y,
                vehicle_id=f"Platoon_{i + 1}",
                initial_velocity=0.0,
            )
            pv.autonomous_mode = True
            pv.target_velocity = PLATOON_TARGET_VELOCITY
            pv.set_motion_model(use_kinematic=False, use_state_space=False,
                                use_hierarchical=True)
            self.platoon_vehicles.append(pv)

        # Keep a fixed snapshot for logging (platoon_manager.vehicles grows when ego joins)
        self._orig_platoon_vehicles: List[LongVehicle] = list(self.platoon_vehicles)
        self._ego_join_time: Optional[float] = None

        self.platoon_manager = LongPlatoonManager(self.platoon_vehicles,
                                                  use_state_space=True)
        self.platoon_manager.target_velocity = PLATOON_TARGET_VELOCITY

        # ── Ego initial position based on merge scenario ─────────────────────
        n_pv = num_platoon_vehicles

        if ego_initial_x is not None:
            _ego_x = float(ego_initial_x)
        else:
            # At cruise the platoon vehicles are spaced by:
            #   desired_gap = h*vx + d0  (bumper-to-bumper)
            #   centre-to-centre = desired_gap + L
            cruise_spacing = (PLATOON_TIME_GAP * PLATOON_TARGET_VELOCITY
                              + PLATOON_STANDSTILL_DISTANCE
                              + PLATOON_VEHICLE_LENGTH)

            # Desired relative x position of ego w.r.t. P1 AT trigger time.
            if merge_scenario == 'front':
                # Comfortably ahead of P1 (one full cruise gap ahead)
                desired_offset = cruise_spacing
            elif merge_scenario == 'middle':
                # Midpoint of the middle insertion gap
                #   n_pv=3 → gap 0 (between P1 and P2) → -0.5 * cruise_spacing
                #   n_pv=4 → gap 1 (between P2 and P3) → -1.5 * cruise_spacing
                gap_i = (n_pv - 2) // 2
                desired_offset = -(gap_i + 0.5) * cruise_spacing
            else:  # 'back'
                # Behind the last platoon vehicle by half a cruise gap
                desired_offset = -(n_pv - 0.5) * cruise_spacing

            # Ego free-road distance: IDM with MOBIL_IDM_A_MAX (Vehicle6D tracks this well).
            v_plat = float(PLATOON_TARGET_VELOCITY)
            v_ego  = max(v_plat + dp.get('velocity_offset', 0.0), 1.0)

            def _free_road_distance(v_target: float, T: float, a_max: float) -> float:
                """Integrate IDM free-road a = a_max*(1-(v/v0)^delta) from v=0."""
                v, x, dt_s = 0.0, 0.0, 0.1
                for _ in range(int(T / dt_s)):
                    a = a_max * max(1.0 - (v / v_target) ** MOBIL_IDM_DELTA, 0.0)
                    v  = min(v + a * dt_s, v_target)
                    x += v * dt_s
                return x

            d_ego = _free_road_distance(v_ego, merge_trigger_time, MOBIL_IDM_A_MAX)

            # Platoon leader (P1) uses the hierarchical model (throttle + gear dynamics),
            # which may travel significantly less than the simple IDM formula predicts.
            # Run a mini pre-simulation of P1 alone to get its ACTUAL position at
            # trigger time, then place ego so that ego.x = P1.x + desired_offset at
            # trigger time, i.e.: ego_initial = P1_actual_at_trigger + desired_offset - d_ego
            print(f"[Simulator] Pre-simulating P1 for {merge_trigger_time}s to calibrate ego x …")
            _p1_presim = LongVehicle(
                initial_x=0.0, initial_y=PLATOON_LANE_Y,
                vehicle_id='P1_presim', initial_velocity=0.0,
            )
            _p1_presim.autonomous_mode = True
            _p1_presim.target_velocity = PLATOON_TARGET_VELOCITY
            _p1_presim.set_motion_model(use_kinematic=False, use_state_space=False,
                                        use_hierarchical=True)
            _pm_presim = LongPlatoonManager([_p1_presim], use_state_space=True)
            _pm_presim.target_velocity = PLATOON_TARGET_VELOCITY
            _t_pre = 0.0
            _dt_pre = SIMULATION_DT
            while _t_pre < merge_trigger_time - 0.5 * _dt_pre:
                _pm_presim.update(_dt_pre, is_prediction_mode=False)
                _t_pre += _dt_pre
            p1_x_at_trigger = float(_p1_presim.state.x)
            print(f"[Simulator] P1 actual x at trigger: {p1_x_at_trigger:.1f}m  "
                  f"(d_ego_formula={d_ego:.1f}m)")

            _ego_x = p1_x_at_trigger + desired_offset - d_ego

        # ── Ego vehicle (Vehicle6D) — starts at v=0 ─────────────────────────
        self.ego = Vehicle6D(
            initial_x   = _ego_x,
            initial_vx  = 0.0,
            initial_y   = HUMAN_INITIAL_LANE_Y,   # adjacent lane
            initial_psi = 0.0,
            vehicle_id  = "Ego",
        )

        # ── Coordinator ──────────────────────────────────────────────────────
        self.coordinator = UnifiedCoordinator(
            self.ego,
            driver_type=driver_type,
            merge_scenario=merge_scenario,
            noise_mode=noise_mode,
            fixed_lambda_long=fixed_lambda_long,
            fixed_lambda_lat=fixed_lambda_lat,
            weight_overrides=weight_overrides,
        )

        # Platoon membership — set True once ego is inserted via add_vehicle()
        self.ego_in_platoon: bool = False

        # ── Data storage ─────────────────────────────────────────────────────
        self._init_data()

        print(f"\n{'=' * 60}")
        print(f"UnifiedSimulation: {num_platoon_vehicles} platoon vehicles + 1 ego")
        print(f"  scenario={merge_scenario}, driver_type={driver_type}")
        print(f"  ego x={self.ego.state.x:.1f}m, vx={self.ego.state.vx:.1f}m/s")
        print(f"  T_sim={T_sim}s, dt={SIMULATION_DT}s, trigger={merge_trigger_time}s")
        print(f"{'=' * 60}")

    # =========================================================================
    # Setup
    # =========================================================================

    def _init_data(self):
        """Initialise empty logging buffers."""
        n_pv = len(self.platoon_vehicles)
        self.data = {
            'time':           [],
            # Ego
            'ego_x':          [], 'ego_vx': [],
            'ego_y':          [], 'ego_vy': [],
            'ego_psi':        [], 'ego_psi_dot': [],
            'ego_ax':         [], 'ego_ay':  [],
            'ego_delta':      [],
            'u_long':         [], 'delta_lat': [],
            'long_lambda':    [], 'lat_lambda': [],
            'long_force':     [], 'lat_force':  [],
            'u1_long':        [], 'u2_long':    [],
            'u1_lat':         [], 'u2_lat':     [],
            'phase':          [],
            # Platoon
            'platoon_x':      [[] for _ in range(n_pv)],
            'platoon_vx':     [[] for _ in range(n_pv)],
            'platoon_ax':     [[] for _ in range(n_pv)],
            # Gaps
            'ego_gap_to_leader': [],
            'desired_gap':       [],
            # Leader/follower indices (index into _orig_platoon_vehicles, -1 = not set)
            'ego_leader_idx':    [],
            'ego_follower_idx':  [],
            # DSF ellipse longitudinal semi-axis a = max(|Δv|·TS, a_min) [m]
            'dsf_leader_a':   [],
            'dsf_follower_a': [],
        }

    def _add_ego_to_platoon(self):
        """Insert ego into platoon_manager.vehicles at the sorted position.

        Mirrors Longitudinal/simulation/simulator.py line 354:
            sim.platoon_manager.add_vehicle(sim.human_vehicle)

        Effect: the platoon vehicle behind ego now follows ego (not its old leader).
        Rajamani gap-keeping is applied relative to ego's position.
        Ego's a_desired is set by PlatoonManager reading ego.nash_acceleration each step.
        Ego's actual 6D physics remain in simulator.step() via update_dynamics_6d().
        """
        if self.ego_in_platoon:
            return
        self.platoon_manager.add_vehicle(self.ego.long_proxy)
        self.ego_in_platoon = True
        self._ego_join_time = self.time
        idx = self.platoon_manager.vehicles.index(self.ego.long_proxy)
        print(f"[Simulator] Ego inserted into platoon at index {idx} "
              f"(t={self.time:.1f}s, x={self.ego.state.x:.1f}m)")

    # =========================================================================
    # Single simulation step
    # =========================================================================

    def step(self):
        """Advance simulation by one SIMULATION_DT step."""
        dt = self.dt

        # 1. Coordinator: Nash solve (must run BEFORE platoon_manager so that
        #    ego.nash_acceleration is set before PlatoonManager reads it)
        if self.time >= self.merge_trigger_time:
            u_long, delta_lat = self.coordinator.step(
                self.time, dt, self.platoon_vehicles)
            # One-time insertion: add ego to platoon when MERGE phase starts
            if not self.ego_in_platoon and self.coordinator.phase.value in ('MERGE', 'FOLLOWING'):
                self._add_ego_to_platoon()
        else:
            # Before trigger: IDM free-road acceleration from rest (mirrors HumanDriver)
            dp       = DRIVER_PARAMS.get(self.driver_type, DRIVER_PARAMS['normal'])
            v_cruise = PLATOON_TARGET_VELOCITY + dp.get('velocity_offset', 0.0)
            vx       = max(self.ego.state.vx, 1e-3)   # avoid division by zero at v=0
            a_free   = MOBIL_IDM_A_MAX * (1.0 - (vx / v_cruise) ** MOBIL_IDM_DELTA)
            u_long   = float(np.clip(a_free, MAX_DECELERATION, MAX_ACCELERATION))
            delta_lat = 0.0

        # 2. Feed Nash longitudinal control to ego so PlatoonManager can read it.
        #    PlatoonManager.update() checks follower.nash_acceleration; if set, it
        #    uses that value and resets it to None (standard standalone pattern).
        self.ego.nash_acceleration = u_long

        # 3. Update platoon Rajamani (now includes ego once it has been added)
        self.platoon_manager.update(dt, is_prediction_mode=False)

        # 4. Update ego 6D dynamics (longitudinal + lateral in one RK4 step)
        self.ego.update_dynamics_6d(u_long, delta_lat, dt)

        # 5. Log
        self._log_step(u_long, delta_lat)

        # 6. Advance time
        self.time += dt

    # =========================================================================
    # Run
    # =========================================================================

    def run(self) -> Dict:
        """Run the full simulation and return the data dictionary."""
        print(f"\n Running unified simulation  ({self.T_sim:.0f}s)...")
        n_steps       = int(self.T_sim / self.dt)
        print_every   = max(1, n_steps // 20)
        t0            = time_module.time()

        for i in range(n_steps):
            self.step()
            if i % print_every == 0:
                pct = 100.0 * i / n_steps
                print(f"  {pct:5.1f}%  t={self.time:6.1f}s  "
                      f"phase={self.coordinator.phase.value}  "
                      f"ego=(x={self.ego.state.x:.1f}, y={self.ego.state.y:.2f}, "
                      f"vx={self.ego.state.vx:.1f})")

        elapsed = time_module.time() - t0
        print(f"\n Simulation complete  {elapsed:.1f}s wall, "
              f"{self.T_sim / elapsed:.1f}× real-time")

        # Finalise: convert lists → numpy arrays
        self._finalise_data()
        return self.data

    # =========================================================================
    # Logging
    # =========================================================================

    def _log_step(self, u_long: float, delta_lat: float):
        d   = self.data
        ego = self.ego
        co  = self.coordinator

        d['time'].append(self.time)

        # Ego state
        d['ego_x'].append(ego.state.x);    d['ego_vx'].append(ego.state.vx)
        d['ego_y'].append(ego.state.y);    d['ego_vy'].append(ego.state.vy)
        d['ego_psi'].append(ego.state.psi); d['ego_psi_dot'].append(ego.state.psi_dot)
        d['ego_ax'].append(ego.state.ax);   d['ego_ay'].append(ego.state.ay)
        d['ego_delta'].append(ego.state.delta)

        # Control outputs
        d['u_long'].append(u_long);       d['delta_lat'].append(delta_lat)
        d['phase'].append(co.phase.value)

        # Authority & force (last logged values from coordinator.data or defaults)
        co_data = co.data
        d['long_lambda'].append(co_data['long_lambda'][-1] if co_data['long_lambda'] else 0.0)
        d['lat_lambda'].append( co_data['lat_lambda'][-1]  if co_data['lat_lambda']  else 0.0)
        d['long_force'].append( co_data['long_force'][-1]  if co_data['long_force']  else 0.0)
        d['lat_force'].append(  co_data['lat_force'][-1]   if co_data['lat_force']   else 0.0)
        d['u1_long'].append(    co_data['u1_long'][-1]     if co_data['u1_long']     else 0.0)
        d['u2_long'].append(    co_data['u2_long'][-1]     if co_data['u2_long']     else 0.0)
        d['u1_lat'].append(     co_data['u1_lat'][-1]      if co_data['u1_lat']      else 0.0)
        d['u2_lat'].append(     co_data['u2_lat'][-1]      if co_data['u2_lat']      else 0.0)

        # Platoon positions (original vehicles only — list grows when ego joins)
        for i, pv in enumerate(self._orig_platoon_vehicles):
            d['platoon_x'][i].append(pv.state.x)
            d['platoon_vx'][i].append(pv.state.vx)
            d['platoon_ax'][i].append(pv.state.ax)

        # Gap from ego to nearest platoon leader
        leader = self.coordinator._find_leader(self.platoon_vehicles)
        if leader is not None:
            gap     = leader.state.x - ego.state.x - VEHICLE_LENGTH
            des_gap = PLATOON_STANDSTILL_DISTANCE + PLATOON_TIME_GAP * ego.state.vx
            d['ego_gap_to_leader'].append(gap)
            d['desired_gap'].append(des_gap)
        else:
            d['ego_gap_to_leader'].append(float('nan'))
            d['desired_gap'].append(float('nan'))

        # Leader/follower indices into _orig_platoon_vehicles (-1 = none)
        leader_idx = next(
            (i for i, pv in enumerate(self._orig_platoon_vehicles) if pv is leader), -1)
        d['ego_leader_idx'].append(leader_idx)
        follower_obj = getattr(self.coordinator, 'locked_follower', None)
        follower_idx = next(
            (i for i, pv in enumerate(self._orig_platoon_vehicles) if pv is follower_obj), -1)
        d['ego_follower_idx'].append(follower_idx)

        # DSF ellipse longitudinal semi-axis a = max(|Δv|·TS, a_min) [m]
        if leader is not None:
            dv_lead = abs(ego.state.vx - leader.state.vx)
            d['dsf_leader_a'].append(max(dv_lead * LAT_DSF_TS, LAT_DSF_A_MIN))
        else:
            d['dsf_leader_a'].append(float('nan'))
        if follower_obj is not None:
            dv_fol = abs(ego.state.vx - follower_obj.state.vx)
            d['dsf_follower_a'].append(max(dv_fol * LAT_DSF_TS, LAT_DSF_A_MIN))
        else:
            d['dsf_follower_a'].append(float('nan'))

    def _finalise_data(self):
        """Convert list entries to np.ndarray in place."""
        for key, val in self.data.items():
            if key in ('platoon_x', 'platoon_vx', 'platoon_ax'):
                self.data[key] = [np.array(lst) for lst in val]
            elif isinstance(val, list) and val and not isinstance(val[0], str):
                self.data[key] = np.array(val)
        self.data['ego_join_time'] = self._ego_join_time

    # =========================================================================
    # Convenience accessors
    # =========================================================================

    @property
    def phase(self) -> MergePhase:
        return self.coordinator.phase

    def print_status(self):
        """Print a one-line status summary."""
        ego = self.ego
        print(f"t={self.time:6.1f}s  phase={self.phase.value:<10}  "
              f"ego x={ego.state.x:8.1f}  y={ego.state.y:5.2f}  "
              f"vx={ego.state.vx:5.1f}  psi={np.degrees(ego.state.psi):6.2f}°  "
              f"lambda_long={self.coordinator._long_lambda:.2f}")


__all__ = ['UnifiedSimulation']
