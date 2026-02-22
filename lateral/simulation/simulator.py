"""
Lateral Control Simulator.
VERSION 2.3 - Simplified (Simulation Only)

Based on Li et al. 2019: R1 ≠ R2, same target but different trajectory shapes
"""

import numpy as np
from typing import Dict, Optional
import time as time_module
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (SIMULATION_DT, DEFAULT_SIMULATION_TIME, LANE_WIDTH, 
                    PLATOON_LANE_Y, HUMAN_INITIAL_LANE_Y, NASH_CONTROL_DT, NASH_NP, NASH_NU)
from vehicle import Vehicle
from control import HumanDriver, PlatoonManager, PlatoonParams, MOBILLaneChange
from nash_solver import (ConstrainedLateralNashSolver, LateralSafetyField, LateralSafetyFieldParams,
                         LateralAuthorityAllocator, SystemReferenceGenerator,
                         HumanReferenceGenerator)


class LateralSimulation:
    """
    Main simulation class - VERSION 2.3 (Simulation Only)
    """
    
    def __init__(self, dt: float = SIMULATION_DT, T_sim: float = DEFAULT_SIMULATION_TIME):
        self.dt = dt
        self.T_sim = T_sim
        self.time = 0.0
        
        # Human vehicle
        self.human_vehicle = Vehicle(
            initial_y=HUMAN_INITIAL_LANE_Y, 
            initial_psi=0.0,
            initial_x=0.0, 
            vehicle_id="Human", 
            longitudinal_velocity=20.0
        )
        
        # Platoon
        self.platoon_params = PlatoonParams(target_velocity=20.0, platoon_lane_y=PLATOON_LANE_Y)
        self.platoon_manager = PlatoonManager(self.platoon_params)
        
        # Human driver model (Stanley Controller)
        self.human_driver = HumanDriver(
            vehicle=self.human_vehicle, 
            target_lane_y=PLATOON_LANE_Y, 
            dt=dt
        )
        
        # Safety field (V2.0 - no lane centering)
        self.safety_field_params = LateralSafetyFieldParams(target_lane_y=PLATOON_LANE_Y)
        self.safety_field = LateralSafetyField(self.safety_field_params)
        
        # Authority allocator
        self.authority_allocator = LateralAuthorityAllocator()
        
        # Nash solver parameters (from config.py)
        self.Np = NASH_NP
        self.Nc = NASH_NU
        self.dt_nash = NASH_CONTROL_DT  # Nash solver dt (from config.py)
        
        # Reference generators (R1 and R2 - Li et al. 2019)
        # R1 = System reference (slow, safe - 5th order polynomial, T=6s)
        self.system_ref_generator = SystemReferenceGenerator(Np=self.Np, dt=self.dt_nash)
        
        # R2 = Human reference (faster, more direct - 3rd order polynomial)
        self.human_ref_generator = HumanReferenceGenerator(Np=self.Np, dt=self.dt_nash)
        
        # Use CONSTRAINED Nash solver WITH DPP (V4.0 Optimized)
        # With V1.0 calibrated weights for smooth trajectories
        from nash_solver.lateral_constrained_nash_solver import (
            ConstrainedLateralNashSolver, 
            ConstrainedLateralNashParams
        )
        
        nash_params = ConstrainedLateralNashParams(
            Np=self.Np, 
            Nu=self.Nc, 
            dt=self.dt_nash,
            driver_type='normal'
        )
        
        self.nash_solver = ConstrainedLateralNashSolver(
            vehicle=self.human_vehicle, 
            params=nash_params
        )
        
        # Flag to identify constrained solver
        self.use_constrained_solver = True
        
        # NOTE: Constraints handled inside CVXPY QP:
        # - Input bounds: delta_min ≤ δ ≤ delta_max
        # - Rate constraints: |Δδ| ≤ ddelta_max * dt
        
        # MOBIL lane change decision model (V2.4)
        self.mobil = MOBILLaneChange()
        self.mobil_approved = False  # Whether MOBIL has approved the lane change
        self.mobil_approval_time = None  # Time when MOBIL approved (for plotting)
        
        # Merge state
        self.merge_commanded = False
        self.merge_complete = False
        self.merge_trigger_time = 5.0
        
        # Data storage
        self.data = {
            'time': [], 'human_x': [], 'human_y': [], 'human_psi': [], 
            'human_vx': [], 'human_y_dot': [], 'human_psi_dot': [], 'human_ay': [],
            'delta_system': [], 'delta_human': [], 'delta_shared': [],
            'authority_ratio': [], 'field_force': [], 'phase': [],
            'platoon_positions': [], 'y_error': [], 'psi_error': [], 'nash_costs': [],
            'mobil_approved': []  # Track MOBIL approval status
        }
        
        print(f"🚗 Lateral Simulation V2.0 Initialized - dt={dt}s, T={T_sim}s")
    
    def setup_scenario(self, scenario_name: str, scenario_params: Dict):
        print(f"\n{'='*60}")
        print(f"🎯 Scenario: {scenario_name}")
        print(f"{'='*60}")
        
        self.reset()
        
        # Setup human vehicle
        self.human_vehicle.state.x = scenario_params.get('human_initial_x', 0.0)
        self.human_vehicle.state.y = scenario_params.get('human_initial_y', HUMAN_INITIAL_LANE_Y)
        self.human_vehicle.vx = scenario_params.get('human_velocity', 20.0)
        self.human_vehicle.reset_steering()
        
        # Setup driver
        driver_type = scenario_params.get('driver_type', 'normal')
        self.human_driver.set_driver_type(driver_type)
        self.human_driver.target_lane_y = scenario_params.get('target_lane_y', PLATOON_LANE_Y)
        
        # Setup MOBIL with driver personality
        self.mobil.set_politeness(driver_type)
        self.mobil_approved = False  # Reset for new scenario
        
        # Setup human reference generator (R2) with driver personality
        self.human_ref_generator.set_driver_type(driver_type)
        
        # Setup system reference generator (R1) with driver personality
        # CRITICAL: Without this, System T_lc stays at default (normal=6.8s)
        # causing Nash mismatch when driver_type != 'normal'
        self.system_ref_generator.set_driver_type(driver_type)
        
        # RE-INITIALIZE Nash solver with driver type
        # Use CONSTRAINED Nash solver WITH DPP (V4.0 with V1.0 weights)
        from nash_solver.lateral_constrained_nash_solver import (
            ConstrainedLateralNashSolver, 
            ConstrainedLateralNashParams
        )
        
        nash_params = ConstrainedLateralNashParams(
            Np=self.Np, 
            Nu=self.Nc, 
            dt=self.dt_nash,
            driver_type=driver_type
        )
        
        self.nash_solver = ConstrainedLateralNashSolver(
            vehicle=self.human_vehicle, 
            params=nash_params
        )
        print(f"🧠 Constrained Nash Solver initialized with driver type: {driver_type}")
        
        # Setup platoon
        platoon_config = scenario_params.get('platoon', {})
        self.platoon_manager.create_platoon(
            platoon_config.get('num_vehicles', 3),
            platoon_config.get('leader_x', 50.0),
            platoon_config.get('gap', None)
        )
        
        self.merge_trigger_time = scenario_params.get('merge_trigger_time', 5.0)
    
    def nash_control_step(self) -> Dict:
        """
        Execute one Nash control step.
        
        Li et al. 2019 Implementation:
        - R1 = System reference (5th order, T=8s, safe trajectory)
        - R2 = Human reference (3rd order, T=5-7s, driver's preferred trajectory)
        - Both target same goal (y=0, psi=0) but with different shapes
        - Q2 = λ * Q1 (authority allocation through cost weights)
        """
        current_state = self.human_vehicle.get_state_vector()
        target_y = self.human_driver.target_lane_y
        obstacles = self.platoon_manager.get_vehicles_as_obstacles()
        
        # 1. Compute safety field force (V2.0 - no lane centering!)
        field_force = self.safety_field.compute_risk_force(
            self.human_vehicle, obstacles, target_y
        )
        
        # 2. Compute authority ratio
        y_error = self.human_vehicle.state.y - target_y
        psi_error = self.human_vehicle.state.psi
        lambda_k = self.authority_allocator.compute_authority_ratio(
            field_force, y_error, psi_error
        )
        
        # 3. Update reference generator phases (sync with safety field)
        safety_phase = self.safety_field.get_phase_name()
        self.system_ref_generator.update_phase_from_safety_field(safety_phase)
        self.human_ref_generator.update_phase_from_safety_field(safety_phase)
        
        # Update times
        self.system_ref_generator.set_current_time(self.time)
        self.human_ref_generator.set_current_time(self.time)
        
        # 4. Generate reference trajectories
        # R1 = System reference (slow, safe - 5th order polynomial)
        R1_ref = self.system_ref_generator.generate_reference_trajectory(
            self.human_vehicle, target_y, obstacles
        )
        
        # R2 = Human reference (faster, more direct - 3rd order polynomial)
        R2_ref = self.human_ref_generator.generate_reference_trajectory(
            self.human_vehicle, target_y, obstacles
        )
        
        # 5. Solve Nash equilibrium for both players
        try:
            u1_opt, u2_opt = self.nash_solver.solve_nash_equilibrium(
                x0=current_state,
                R1_ref=R1_ref,
                R2_ref=R2_ref,
                lambda_k=lambda_k,
                field_force=field_force
            )
        except Exception as e:
            print(f"⚠️ Nash error: {e}")
            u1_opt = 0.0
            u2_opt = self.human_driver.get_human_steering_input(self.human_vehicle, target_y)
        
        # 6. Weighted combination (Shared Control!)
        # α determines how much authority the system has
        # Higher λ → Higher α → More system control
        alpha = lambda_k / (1.0 + lambda_k)
        u_shared = alpha * u1_opt + (1 - alpha) * u2_opt
        
        # NOTE: No external filtering or clipping on Nash variables!
        # All constraints on u1, u2 are handled INSIDE the Nash solver (CVXPY QP):
        # - Input bounds: delta_min ≤ δ ≤ delta_max
        # - Rate constraints: |Δδ| ≤ ddelta_max * dt
        
        return {
            'delta_system': u1_opt,
            'delta_human': u2_opt,
            'delta_shared': u_shared,
            'authority_ratio': lambda_k,
            'alpha': alpha,
            'field_force': field_force,
            'y_error': y_error,
            'psi_error': psi_error,
            'phase': safety_phase,
            'nash_costs': self.nash_solver.last_costs
        }
    
    def step(self):
        """Execute one simulation step."""
        self.time += self.dt
        self.safety_field.set_current_time(self.time)
        self.system_ref_generator.set_current_time(self.time)
        
        # Update platoon
        self.platoon_manager.update(self.dt)
        
        # Phase 1: Check if we want to merge and MOBIL approval
        if self.time >= self.merge_trigger_time and not self.merge_commanded:
            # Check MOBIL before commanding merge
            if not self.mobil_approved:
                self.mobil_approved = self._check_mobil_approval()
            
            if self.mobil_approved:
                self.command_merge()
        
        # Control
        if self.merge_commanded:
            control_result = self.nash_control_step()
            delta_shared = control_result['delta_shared']
        else:
            delta_shared = 0.0
            # Check MOBIL status during GAP_SEARCH phase
            phase_name = 'GAP_SEARCH' if (self.time >= self.merge_trigger_time and not self.mobil_approved) else 'CRUISE'
            control_result = {
                'delta_system': 0.0, 'delta_human': 0.0, 'delta_shared': 0.0,
                'authority_ratio': 0.1, 'alpha': 0.09, 'field_force': 0.0,
                'y_error': self.human_vehicle.state.y, 'psi_error': 0.0,
                'phase': phase_name, 'nash_costs': [0.0, 0.0]
            }
        
        # Update vehicle
        self.human_vehicle.update_dynamics(self.dt, delta_shared)
        
        # Store data
        self._store_data(control_result)
        
        # Check merge completion
        if self.merge_commanded and not self.merge_complete:
            self._check_merge_completion()
    
    def _check_mobil_approval(self) -> bool:
        """
        Check if MOBIL approves the lane change.
        Called during GAP_SEARCH phase.
        """
        # Get platoon vehicles as list of dicts
        platoon_vehicles = []
        for v in self.platoon_manager.vehicles:
            platoon_vehicles.append({
                'x': v.x,
                'v': self.platoon_manager.params.target_velocity
            })
        
        if not platoon_vehicles:
            return True  # No platoon - always approved
        
        # Determine merge position dynamically based on human position
        human_x = self.human_vehicle.state.x
        sorted_platoon = sorted(platoon_vehicles, key=lambda v: v['x'], reverse=True)
        leader_x = sorted_platoon[0]['x']
        last_x = sorted_platoon[-1]['x']
        
        if human_x > leader_x:
            merge_position = 'before'  # Human is ahead of platoon leader
        elif human_x < last_x:
            merge_position = 'after'   # Human is behind platoon tail
        else:
            merge_position = 'middle'  # Human is within platoon range
        
        # Check with MOBIL
        approved, details = self.mobil.check_platoon_merge(
            human_x=human_x,
            human_v=self.human_vehicle.vx,
            platoon_vehicles=platoon_vehicles,
            merge_position=merge_position
        )
        
        # Store MOBIL check time for logging frequency
        if not hasattr(self, '_last_mobil_log_time'):
            self._last_mobil_log_time = 0.0
        
        # Log every 1 second during GAP_SEARCH (not every timestep)
        if self.time - self._last_mobil_log_time >= 1.0:
            self._last_mobil_log_time = self.time
            gap_front = details.get('gap_to_new_leader')
            gap_rear = details.get('gap_from_new_follower')
            safety_ok = "✓" if details.get('safety_ok') else "✗"
            gaps_ok = "✓" if details.get('gaps_ok') else "✗"
            
            gap_front_str = f"{gap_front:.1f}m" if gap_front is not None else "N/A"
            gap_rear_str = f"{gap_rear:.1f}m" if gap_rear is not None else "N/A"
            
            status = "⏳ Waiting" if not approved else "✅ Ready"
            print(f"   🚦 MOBIL [{status}] t={self.time:.1f}s: "
                  f"Safety[{safety_ok}] Gaps[{gaps_ok}] "
                  f"(front={gap_front_str}, rear={gap_rear_str}, min=8.0m)")
        
        if approved:
            print(f"\n{'='*60}")
            print(f"✅ MOBIL APPROVED lane change at t={self.time:.1f}s")
            print(f"{'='*60}")
            print(f"   {self.mobil.get_status_string()}")
            # Store approval time for plotting
            self.mobil_approval_time = self.time
        
        return approved
    
    def command_merge(self):
        print(f"\n🚦 Merge commanded at t={self.time:.1f}s")
        self.merge_commanded = True
        self.safety_field.command_merge()
        self.platoon_manager.add_human_vehicle(self.human_vehicle)
    
    def _check_merge_completion(self):
        y_error = abs(self.human_vehicle.state.y - PLATOON_LANE_Y)
        psi_error = abs(self.human_vehicle.state.psi)
        phase = self.safety_field.get_current_phase()
        
        if phase.value == 'FOLLOWING' and y_error < 0.3 and psi_error < 0.05:
            self.merge_complete = True
            self.platoon_manager.complete_merge()
            print(f"\n✅ Merge complete at t={self.time:.1f}s")
    
    def _store_data(self, control_result: Dict):
        self.data['time'].append(self.time)
        self.data['human_x'].append(self.human_vehicle.state.x)
        self.data['human_y'].append(self.human_vehicle.state.y)
        self.data['human_psi'].append(self.human_vehicle.state.psi)
        self.data['human_vx'].append(self.human_vehicle.vx)
        self.data['human_y_dot'].append(self.human_vehicle.state.y_dot)
        self.data['human_psi_dot'].append(self.human_vehicle.state.psi_dot)
        self.data['human_ay'].append(self.human_vehicle.state.ay)
        self.data['delta_system'].append(control_result['delta_system'])
        self.data['delta_human'].append(control_result['delta_human'])
        self.data['delta_shared'].append(control_result['delta_shared'])
        self.data['authority_ratio'].append(control_result['authority_ratio'])
        self.data['field_force'].append(control_result['field_force'])
        self.data['phase'].append(control_result['phase'])
        self.data['y_error'].append(control_result['y_error'])
        self.data['psi_error'].append(control_result['psi_error'])
        self.data['nash_costs'].append(control_result['nash_costs'])
        self.data['platoon_positions'].append([(v.x, v.y) for v in self.platoon_manager.vehicles])
        self.data['mobil_approved'].append(self.mobil_approved)
    
    def run(self) -> Dict:
        print(f"\n🚀 Running simulation (T={self.T_sim}s)...")
        print(f"   Total steps: {int(self.T_sim / self.dt)} (dt={self.dt} seconds)")
        start_time = time_module.time()
        num_steps = int(self.T_sim / self.dt)
        
        # Print interval (every 5 seconds)
        print_interval = int(5.0 / self.dt)
        
        for i in range(num_steps):
            self.step()
            
            # Progress indicator (every 10%)
            if i % (num_steps // 10) == 0:
                progress = (i / num_steps) * 100
                print(f"👍 Progress: {progress:5.1f}% (t={self.time:6.1f}s)")
            
            # Detailed output every 5 seconds
            if i > 0 and i % print_interval == 0:
                self._print_status()
        
        end_time = time_module.time()
        exec_time = end_time - start_time
        speed_factor = self.T_sim / exec_time
        
        print(f"\n{'='*70}")
        print(f"✅ Simulation Complete!")
        print(f"   ⏱️ Execution time: {exec_time:.1f} seconds")
        print(f"   🚀 Speed factor: {speed_factor:.1f}x real time")
        print(f"{'='*70}")
        
        # Convert to numpy
        for key in self.data:
            if key != 'platoon_positions':
                self.data[key] = np.array(self.data[key])
        
        return self.data
    
    def _print_status(self):
        """Print detailed status like longitudinal controller."""
        state = self.human_vehicle.state
        phase = self.safety_field.get_phase_name()
        
        print(f"\n⏱️ Time: {self.time:.1f}s")
        
        # Platoon status
        for i, v in enumerate(self.platoon_manager.vehicles):
            role = "leader" if i == 0 else "follower"
            print(f"   Platoon_{i+1} ({role}): Pos=({v.x:.1f}, {v.y:.1f}), "
                  f"Speed={v.vx * 3.6:.1f} km/h")
        
        # Human vehicle status
        v_kmh = state.vx * 3.6
        print(f"   Human ({phase}): Pos=({state.x:.1f}, {state.y:.2f}), "
              f"Speed={v_kmh:.1f} km/h, ψ={np.degrees(state.psi):.2f}°")
        
        # Nash data if available
        if len(self.data['delta_system']) > 0:
            delta_sys = np.degrees(self.data['delta_system'][-1])
            delta_hum = np.degrees(self.data['delta_human'][-1])
            delta_shared = np.degrees(self.data['delta_shared'][-1])
            lambda_val = self.data['authority_ratio'][-1]
            field_force = self.data['field_force'][-1]
            
            print(f"   🎮 Nash: δ_shared={delta_shared:.3f}°, δ_sys={delta_sys:.3f}°, δ_human={delta_hum:.3f}°")
            print(f"   🛡️ Safety: Force={field_force:.1f}N, λ={lambda_val:.2f}, Phase={phase}")
            
            # Lateral error
            y_error = state.y - 0.0  # Target is y=0
            psi_error = np.degrees(state.psi)
            print(f"   📍 Errors: y_error={y_error:.2f}m, ψ_error={psi_error:.2f}°")
    
    def reset(self):
        self.time = 0.0
        self.merge_commanded = False
        self.merge_complete = False
        # NOTE: No delta_output_prev - no external filtering
        self.safety_field.reset()
        self.system_ref_generator.reset()
        self.human_ref_generator.reset()  # Reset R2 generator
        self.authority_allocator.reset()
        self.human_driver.reset()
        self.platoon_manager.reset()
        self.nash_solver.reset()
        for key in self.data:
            self.data[key] = []


__all__ = ['LateralSimulation']