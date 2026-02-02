#!/usr/bin/env python3
"""
File: main_with_nash.py
Description: Platoon joining simulation with Nash equilibrium control integration.
Combines the platoon simulation from main_without_nash with Nash equilibrium solver.
Updated with Low-Pass Filters for stability and Velocity Error Safety Check.
"""

import os
import sys
import time
import numpy as np
from typing import List, Dict

from nash_solver import longitudinal_constrained_nash_solver

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import platoon simulation modules
from config import setup_matplotlib, HEADLESS_MODE, RESULTS_DIR, SIMULATION_DT, DEFAULT_SIMULATION_TIME
from simulation.simulator import PlatoonSimulation, run_simulation
from visualization.animation import create_platoon_animation
from visualization.plots import create_comprehensive_plots, create_detailed_scenario_summary, create_nash_analysis_plots
from vehicle import Vehicle
from control.human_driver import HumanDriver
from control.platoon_control import PlatoonManager

# Import Nash solver modules
from nash_solver.longitudinal_authority_allocator import LongitudinalAuthorityAllocator
from nash_solver.longitudinal_safety_field import (
    EllipseLongitudinalSafetyField,
    EllipseLongitudinalParams,
    PlatoonContext
)
from nash_solver import longitudinal_nash_solver, system_reference_generator
from nash_solver.longitudinal_constrained_nash_solver import ConstrainedLongitudinalNashSolver

os.system('cls' if os.name == 'nt' else 'clear')

class LowPassFilter:
    """Simple First-Order Low Pass Filter for smoothing signals."""
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.last_val = None
        
    def filter(self, val):
        if self.last_val is None:
            self.last_val = val
            return val
        filtered = self.alpha * val + (1 - self.alpha) * self.last_val
        self.last_val = filtered
        return filtered

class PlatoonNashSimulation(PlatoonSimulation):
    """Extended platoon simulation with Nash equilibrium control"""
    
    def __init__(self, Constrained_Nash=True):
        super().__init__()
        
        # Initialize Nash control components
        self.safety_field = EllipseLongitudinalSafetyField()
        self.safety_field.params = EllipseLongitudinalParams()
        self.authority_allocator = LongitudinalAuthorityAllocator()
        
        self.Np = 20  # Prediction horizon
        self.Nu = 10  # Control horizon
        self.dt_nash = 0.1  # Time step
        
        self.system_ref_generator = system_reference_generator.SystemReferenceGenerator(
            Np=self.Np, dt=self.dt_nash
        )
        self.Constrained_Nash = Constrained_Nash
        # Initialize Nash solver
        try:
            if Constrained_Nash==True:
                self.nash_solver = longitudinal_constrained_nash_solver.ConstrainedLongitudinalNashSolver(
                    vehicle=self.human_vehicle, 
                    platoon_manager=self.platoon_manager, 
                    human_driver=self.human_driver, 
                    Np=self.Np, Nu=self.Nu, dt=self.dt_nash
                )
            else:
                self.nash_solver = longitudinal_nash_solver.LongitudinalNashSolver(
                    vehicle=self.human_vehicle, 
                    platoon_manager=self.platoon_manager, 
                    human_driver=self.human_driver, 
                    Np=self.Np, dt=self.dt_nash
                )
            self.nash_solver_available = True
        except Exception as e:
            print(f"⚠️ Nash solver initialization issue: {e}")
            self.nash_solver_available = False
        
        # Initialize Low Pass Filters
        self.force_filter = LowPassFilter(alpha=0.2)  # Smoothing for Field Force
        self.control_filter = LowPassFilter(alpha=0.05) # Smoothing for Final Control
        
        # Data storage
        self.nash_data = {
            'controller_inputs': [], 'human_inputs': [], 'shared_inputs': [],
            'authority_ratios': [], 'field_forces': [],
            'leader_forces': [], 'follower_forces': [],
            'cooperation_moments': 0, 'opposition_moments': 0
        }
        
        self.system_desired_acc_history = []
        self.human_desired_acc_history = []
        self.authority_history = []
        self.authority_safety_history = []  # Authority by safety
        self.authority_performance_history = []  # Authority by performance
        self.safety_force_history = []
        self.gap_error_history = []
        self.nash_cost_history = []
        
        print("🚗 Platoon Nash Simulation Initialized (With Filters)")

    def nash_control_step(self, human_vehicle: Vehicle, leader_vehicle: Vehicle = None) -> Dict:
        # Get current state
        current_state = np.array([human_vehicle.state.x, human_vehicle.state.vx])
        
        leader = self.safety_field._get_leader(human_vehicle, self.platoon_manager)
        follower = self.safety_field._get_follower(human_vehicle, self.platoon_manager)

        # --- 1. Calculate Gap Error & Velocity Error ---
        current_gap_error = 0.0
        velocity_error = 0.0
        desired_gap = 50.0
        
        # Calculate Velocity Error (Target - Current)
        # If result is negative (e.g. -10), it means we are 10 m/s FASTER than target (Overspeeding)
        target_vel = self.platoon_manager.target_velocity
        velocity_error = target_vel - human_vehicle.state.vx
        
        if leader:
            from control.platoon_control import rajamani
            _, desired_gap = rajamani(leader, human_vehicle)
            current_gap = leader.state.x - human_vehicle.state.x
            current_gap_error = current_gap - desired_gap # Positive = Behind
        elif follower:
            from control.platoon_control import rajamani
            _, desired_gap = rajamani(human_vehicle, follower)
            current_gap = human_vehicle.state.x - follower.state.x
            current_gap_error = desired_gap - current_gap # Positive = Behind
        
        # --- 2. Safety Field Calculation ---
        raw_field_force = self.safety_field.compute_risk_force_from_platoon(
            ego_vehicle=human_vehicle,
            platoon_manager=self.platoon_manager,
            desired_gap=desired_gap
        )
        
        # FILTER 1: Smooth the field force
        field_force = self.force_filter.filter(raw_field_force)
        
        breakdown = self.safety_field.get_force_breakdown_from_platoon(
            ego_vehicle=human_vehicle,
            platoon_manager=self.platoon_manager,
            desired_gap=desired_gap
        )

        # --- 3. Authority Allocation ---
        # Includes Risk, Performance (Gap Error), and Safety (Velocity Error) logic
        lambda_k = self.authority_allocator.compute_authority_ratio(
            field_force, 
            gap_error=current_gap_error,
            velocity_error=velocity_error
        )
        
        # --- 4. Generate Reference Trajectories ---
        # System Reference (R1) - With Kinematic Limits & Boosting
        accel_seq_sys, state_seq_sys = self.system_ref_generator.get_system_acceleration_and_state_sequence(self)
        R1_ref = state_seq_sys
        
        # Human Reference (R2) - With IDM
        accel_seq_human, state_seq_human = self.human_driver.get_human_acceleration_and_state_sequence(
            dt=self.dt_nash, Np=self.Np, vehicle=human_vehicle
        )
        R2_ref = state_seq_human
        
        # --- 5. Solve Nash Equilibrium ---
        u1_opt = 0.0
        u2_opt = 0.0
        
        if self.nash_solver_available:
            try:
                u1_opt, u2_opt = self.nash_solver.solve_nash_equilibrium(
                    x0=current_state, R1_ref=R1_ref, R2_ref=R2_ref,
                    lambda_k=lambda_k, field_force=field_force
                )
            except Exception as e:
                print(f"⚠️ Nash solver error: {e}")
                u1_opt = 0.0
                u2_opt = accel_seq_human[0] if len(accel_seq_human) > 0 else 0.0
        else:
             u2_opt = accel_seq_human[0] if len(accel_seq_human) > 0 else 0.0
        
        # --- 6. Shared Control ---
        alpha = lambda_k / (1.0 + lambda_k)
        u_raw = alpha * u1_opt + (1 - alpha) * u2_opt
        
        # FILTER 2: Smooth the final control input
        u_shared = u_raw#self.control_filter.filter(u_raw)
        
        # Apply hard constraints
        u_shared = self.safety_field.apply_hard_constraint(
            u_shared, self.human_vehicle, self.platoon_manager
        )

        # --- 7. Logging ---
        self.nash_data['controller_inputs'].append(u1_opt)
        self.nash_data['human_inputs'].append(u2_opt)
        self.nash_data['shared_inputs'].append(u_shared)
        self.nash_data['authority_ratios'].append(lambda_k)
        self.nash_data['field_forces'].append(field_force)
        
        leader_force = breakdown.get('leader', {}).get('total_force', 0.0) if breakdown else 0.0
        follower_force = breakdown.get('follower', {}).get('total_force', 0.0) if breakdown else 0.0
        self.nash_data['leader_forces'].append(leader_force)
        self.nash_data['follower_forces'].append(follower_force)
        
        self.system_desired_acc_history.append(u1_opt)
        self.human_desired_acc_history.append(u2_opt)
        self.authority_history.append(lambda_k)
        self.authority_safety_history.append(self.authority_allocator.last_lambda_safety)
        self.authority_performance_history.append(self.authority_allocator.last_lambda_performance)
        self.safety_force_history.append(field_force)
        self.gap_error_history.append(current_gap_error)

        if self.nash_solver_available and hasattr(self.nash_solver, 'last_costs'):
            self.nash_cost_history.append(self.nash_solver.last_costs)
        else:
            self.nash_cost_history.append([0.0, 0.0])
            
        if abs(u1_opt) > 0.01 and abs(u2_opt) > 0.01:
            if u1_opt * u2_opt > 0: self.nash_data['cooperation_moments'] += 1
            else: self.nash_data['opposition_moments'] += 1

        return {
            'controller_input': u1_opt,
            'human_input': u2_opt,
            'shared_input': u_shared,
            'authority_ratio': lambda_k,
            'field_force': field_force,
            'breakdown': breakdown
        }

    def update_with_nash(self):
        """Override update method to include Nash control"""
        if self.human_driver.merging or self.human_vehicle.joined_platoon:
            nash_result = self.nash_control_step(self.human_vehicle)
            shared_accel = nash_result['shared_input']
            self.human_vehicle.nash_acceleration = shared_accel
            self.human_vehicle.a_desired = shared_accel
            self.human_vehicle.state.ax = shared_accel
            
            # Periodic prints
            if int(self.time) % 2.5 == 0 and int(self.time) != int(self.time - self.dt):
                leader = self.safety_field._get_leader(self.human_vehicle, self.platoon_manager)
                follower = self.safety_field._get_follower(self.human_vehicle, self.platoon_manager)
                
                gap_leader = leader.state.x - self.human_vehicle.state.x if leader else 0.0
                gap_follower = self.human_vehicle.state.x - follower.state.x if follower else 0.0
                if self.safety_field.ignoring_follower:
                     print(f"   🚨 HUGE leader gap ({self.safety_field.leader_gap_error:.0f}m) - ignoring follower")
                
                print(f"🎯 Nash: u_sh={shared_accel:.2f} m/s², u_sys={nash_result['controller_input']:.2f}, u_human={nash_result['human_input']:.2f}")
                print(f"   📍 Gap_L={gap_leader:.1f}m, Gap_F={gap_follower:.1f}m, v={self.human_vehicle.state.vx*3.6:.1f}km/h")
                print(f"   🛡️ Risk_L={nash_result['breakdown'].get('leader', {}).get('total_force', 0.0):.1f}N, "
                      f"Risk_F={nash_result['breakdown'].get('follower', {}).get('total_force', 0.0):.1f}N, "
                      f"Total={nash_result['field_force']:.1f}N"
                      f", λ={nash_result['authority_ratio']:.2f}"
                      f", alpha={nash_result['authority_ratio']/(1+nash_result['authority_ratio']):.2f}")
                time.sleep(3)  # Small delay for readability

        super().update()

def run_scenario_with_nash(scenario_name: str, sim_params: Dict) -> PlatoonNashSimulation:
    """Run a single scenario with Nash control"""
    print(f"\n{'='*60}")
    print(f"🎯 {scenario_name} (with Nash Equilibrium Control)")
    print(f"{'='*60}")
    
    # Create Nash simulation
    sim = PlatoonNashSimulation()
    
    # Apply scenario parameters
    sim.human_vehicle.state.x = sim_params.get('initial_x', 0.0)
    sim.human_vehicle.state.y = sim_params.get('initial_y', -2.0)
    sim.human_driver.target_speed = sim_params.get('target_speed', 100.0) / 3.6
    join_trigger_time = sim_params.get('join_trigger_time', 20.0)
    
    print(f"🚗 Scenario settings:")
    print(f"   📍 Initial position: ({sim.human_vehicle.state.x:.1f}, {sim.human_vehicle.state.y:.1f})")
    print(f"   🎯 Target speed: {sim.human_driver.target_speed*3.6:.0f} km/h")
    print(f"   🧠 Nash control: ACTIVE (Bidirectional Safety)")
    
    # Run simulation
    max_time = sim.T_sim
    T = np.arange(0, max_time, sim.dt)
    max_iterations = len(T)
    human_joined = False
    
    # === TARGET POSITION LOCKING VARIABLES ===
    target_leader_id = None
    target_follower_id = None
    merge_position_locked = False
    
    print(f"⏱️ Simulation time: {max_time:.0f} seconds")
    print(f"   total steps: {max_iterations} (dt={sim.dt:.2f} seconds)")
    
    start_time = time.time()
    
    try:
        for iteration, t in enumerate(T):
            sim.safety_field.set_current_time(sim.time)
            # Print progress every 2000 iterations
            if iteration % 2000 == 0:
                progress = (iteration / max_iterations) * 100
                print(f"👍 Progress: {progress:5.1f}% (t={sim.time:6.1f}s)")
            
            # Update simulation with Nash control
            sim.update_with_nash()
            
            # Status report every 5 seconds
            if sim.time % 5.0 < sim.dt:
                sim.print_status()
                # time.sleep(5)  # Small delay for readability
                if human_joined and len(sim.nash_data['shared_inputs']) > 0:
                    # Safely get last forces
                    leader_force = sim.nash_data['leader_forces'][-1] if sim.nash_data['leader_forces'] else 0.0
                    follower_force = sim.nash_data['follower_forces'][-1] if sim.nash_data['follower_forces'] else 0.0
                    
                    print(f"🧠 Nash Data: lambda={sim.nash_data['authority_ratios'][-1]:.2f}, "
                          f"Shared={sim.nash_data['shared_inputs'][-1]:.2f}")
                    print(f"   🛡️ Risks: Total={sim.nash_data['field_forces'][-1]:.1f}N, "
                          f"Leader={leader_force:.1f}N, Follower={follower_force:.1f}N")
                    print(f"   🤝 Coop={sim.nash_data['cooperation_moments']}, "
                          f"Opp={sim.nash_data['opposition_moments']}")
                    
                
            # Trigger joining at the right time - WITH POSITION LOCKING
            if (sim.time >= join_trigger_time and 
                not human_joined and not sim.human_driver.merging):
                
                print(f"{'='*70}")
                print(f"🔒 MERGE DECISION POINT at t={sim.time:.1f}s")
                print(f"{'='*70}")
                
                # ============================================================
                # STEP 1: Determine WHERE the human wants to merge
                # ============================================================
                human_x = sim.human_vehicle.state.x
                human_v = sim.human_vehicle.state.vx
                
                print(f"📍 Human vehicle state:")
                print(f"   Position: x={human_x:.1f}m")
                print(f"   Velocity: v={human_v:.1f}m/s ({human_v*3.6:.1f} km/h)")
                
                # Get platoon vehicle positions sorted by x (front to back)
                platoon_positions = [(v.vehicle_id, v.state.x, v.state.vx) 
                                     for v in sim.platoon_vehicles]
                platoon_positions.sort(key=lambda p: p[1], reverse=True)  # Sort by x (descending)
                
                print(f"📋 Current platoon configuration:")
                for i, (vid, x, v) in enumerate(platoon_positions):
                    print(f"   {i+1}. {vid}: x={x:.1f}m, v={v:.1f}m/s")
                
                # ============================================================
                # STEP 2: Lock the target position based on current location
                # ============================================================
                # Find the closest vehicle ahead (leader) and behind (follower)
                leader_candidates = [(i, vid, x) for i, (vid, x, v) in enumerate(platoon_positions) if x > human_x]
                follower_candidates = [(i, vid, x) for i, (vid, x, v) in enumerate(platoon_positions) if x < human_x]
                
                # Find closest leader (vehicle ahead with smallest gap)
                if leader_candidates:
                    leader_candidates.sort(key=lambda t: t[2])  # Sort by x (ascending)
                    leader_idx, target_leader_id, leader_x = leader_candidates[0]
                else:
                    target_leader_id = None
                    leader_idx = -1
                
                # Find closest follower (vehicle behind with largest x)
                if follower_candidates:
                    follower_candidates.sort(key=lambda t: t[2], reverse=True)  # Sort by x (descending)
                    follower_idx, target_follower_id, follower_x = follower_candidates[0]
                else:
                    target_follower_id = None
                    follower_idx = -1
                
                # Determine merge position index
                if target_leader_id is None:
                    # Human is ahead of all vehicles - merge at front
                    merge_position_index = 0
                elif target_follower_id is None:
                    # Human is behind all vehicles - merge at back
                    merge_position_index = len(platoon_positions)
                else:
                    # Human is between leader and follower
                    merge_position_index = leader_idx + 1
                
                print(f"🎯 LOCKED MERGE POSITION:")
                print(f"   Target Leader:   {target_leader_id if target_leader_id else 'None (merge at front)'}")
                print(f"   Target Follower: {target_follower_id if target_follower_id else 'None (merge at back)'}")
                print(f"   Position Index:  {merge_position_index} (0=front, {len(platoon_positions)}=back)")
                
                # Calculate initial gaps
                if target_leader_id:
                    leader_x = next(x for vid, x, v in platoon_positions if vid == target_leader_id)
                    gap_to_leader = leader_x - human_x
                    print(f"   Gap to Leader:   {gap_to_leader:.1f}m")
                
                if target_follower_id:
                    follower_x = next(x for vid, x, v in platoon_positions if vid == target_follower_id)
                    gap_to_follower = human_x - follower_x
                    print(f"   Gap to Follower: {gap_to_follower:.1f}m")
                
                # ============================================================
                # STEP 3: Store locked positions in vehicle state
                # ============================================================
                sim.human_vehicle.target_leader_id = target_leader_id
                sim.human_vehicle.target_follower_id = target_follower_id
                sim.human_vehicle.merge_position_index = merge_position_index
                sim.human_vehicle.merge_trigger_time = sim.time
                
                # ============================================================
                # STEP 4: Activate Nash control
                # ============================================================
                print(f"✅ Activating Nash Equilibrium Control:")
                sim.human_driver.ignore_platoon_before_merge = False
                sim.human_driver.merging = True
                human_joined = True
                merge_position_locked = True
                
                sim.human_vehicle.joined_platoon = True
                sim.platoon_manager.add_vehicle(sim.human_vehicle)
                
                sim.human_vehicle.target_velocity = sim.platoon_manager.target_velocity
                sim.human_driver.target_speed = sim.platoon_manager.target_velocity
                
                print(f"   Nash game:       ACTIVE")
                print(f"   Target velocity: {sim.human_driver.target_speed * 3.6:.1f} km/h")
                print(f"   Position locked: YES")
                
                # Safety check
                min_gap = 10.0  # meters
                if target_leader_id and gap_to_leader < min_gap:
                    print(f"⚠️  WARNING: Gap to leader ({gap_to_leader:.1f}m) is below safe threshold ({min_gap}m)")
                    print(f"   Nash solver will work to prevent collision!")
                
                if target_follower_id and gap_to_follower < min_gap:
                    print(f"⚠️  WARNING: Gap to follower ({gap_to_follower:.1f}m) is below safe threshold ({min_gap}m)")
                    print(f"   Nash solver will work to prevent collision!")
                
                print(f"{'='*70}")
                
            if human_joined:  # Only after joining was triggered
                current_phase = sim.safety_field.get_current_phase()
                
                if current_phase == "FOLLOWING":
                    sim.human_driver.merging = False  # No longer actively merging
                elif current_phase == "MERGING":
                    sim.human_driver.merging = True   # Still merging (or returned to merging)
                print(f"    Current phase: {current_phase}") if sim.time % 5.0 < sim.dt else None
            
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Print Nash analysis
        print_nash_analysis(sim)
        
        # Create visualizations
        print(f"\n👍 Displaying results for {scenario_name}...")
        print(f"   👍“ Results will be saved to: {os.path.abspath(RESULTS_DIR)}")
        
        # 1. Create comprehensive plots (always works)
        try:
            print(f"\n👍 Creating comprehensive plots...")
            static_plots_fig = create_comprehensive_plots(sim, scenario_name)
            if static_plots_fig:
                print(f"✅ Comprehensive plots created and saved successfully")
            else:
                print(f"⚠️ Comprehensive plots returned None")
        except Exception as comp_error:
            print(f"❌ Failed to create comprehensive plots: {comp_error}")
            import traceback
            traceback.print_exc()
            static_plots_fig = None
        
        # 2. Create Nash analysis plots (9-grid layout)
        try:
            if hasattr(sim, 'authority_history') and len(sim.authority_history) > 0:
                print(f"\n👍Ž¨ Creating Nash analysis plots (9-grid)...")
                nash_fig = create_nash_analysis_plots(sim, scenario_name)
                if nash_fig:
                    print(f"✅ Nash analysis plots (9-grid) created and saved successfully")
                else:
                    print(f"⚠️ Nash analysis plots returned None")
            else:
                print(f"⚠️ No Nash control data - skipping Nash plots")
                nash_fig = None
        except Exception as nash_error:
            print(f"❌ Failed to create Nash analysis plots: {nash_error}")
            import traceback
            traceback.print_exc()
            nash_fig = None

        time.sleep(2)
        
        # Create animation
        try:
            import matplotlib.pyplot as plt
            plt.ioff()
            anim = create_platoon_animation(sim, f"{scenario_name} Animation (Nash)")
            if anim:
                sim._last_animation = anim
            plt.ion()
            if static_plots_fig:
                plt.figure(static_plots_fig.number)
                plt.draw()
                static_plots_fig.canvas.flush_events()
        except Exception as anim_error:
            print(f"⚠️ Animation skipped: {anim_error}")
        
        create_detailed_scenario_summary(sim, scenario_name, execution_time)
        
        # Final confirmation of saved files
        print(f"\n{'=' * 70}")
        print(f"👍“ SAVED FILES SUMMARY")
        print(f"{'=' * 70}")
        abs_results_dir = os.path.abspath(RESULTS_DIR)
        print(f"👍“‚ Directory: {abs_results_dir}")
        try:
            all_files = os.listdir(RESULTS_DIR)
            safe_scenario = scenario_name.replace(" ", "_").replace(":", "")
            saved_files = [f for f in all_files if safe_scenario in f]
            
            if saved_files:
                print(f"✅ Files created for this scenario ({len(saved_files)}):")
                for f in sorted(saved_files):
                    full_path = os.path.join(RESULTS_DIR, f)
                    size_kb = os.path.getsize(full_path) / 1024
                    print(f"   👍“„ {f}")
                    print(f"      👍’¾ Size: {size_kb:.1f} KB")
            else:
                print(f"⚠️ No files found matching scenario: {safe_scenario}")
                print(f" Total files in directory: {len(all_files)}")
                if all_files:
                    print(f"   Recent files:")
                    for f in sorted(all_files, key=lambda x: os.path.getmtime(os.path.join(RESULTS_DIR, x)), reverse=True)[:5]:
                        print(f"   • {f}")
        except Exception as list_error:
            print(f"❌ Could not list saved files: {list_error}")
            import traceback
            traceback.print_exc()
        print(f"{'=' * 70}")
        
        # Clean up figures to prevent memory issues when running multiple scenarios
        print(f"\n👍 Cleaning up matplotlib figures...")
        try:
            import matplotlib.pyplot as plt
            # Close specific figures if they exist
            if static_plots_fig is not None:
                plt.close(static_plots_fig)
            if 'nash_fig' in locals() and nash_fig is not None:
                plt.close(nash_fig)
            # Close all remaining figures
            plt.close('all')
            # Force garbage collection
            import gc
            gc.collect()
            print(f"✅ Figures cleaned up successfully")
        except Exception as cleanup_error:
            print(f"⚠️ Could not clean up figures: {cleanup_error}")
        
    except Exception as e:
        print(f"❌ Error in simulation: {e}")
        import traceback
        traceback.print_exc()
    
    return sim


def print_nash_analysis(sim: PlatoonNashSimulation):
    """Print Nash equilibrium analysis results - NOW WITH BIDIRECTIONAL INFO"""
    nash_data = sim.nash_data
    
    total_active = nash_data['cooperation_moments'] + nash_data['opposition_moments']
    if total_active > 0:
        coop_ratio = nash_data['cooperation_moments'] / total_active
        opp_ratio = nash_data['opposition_moments'] / total_active
    else:
        coop_ratio = opp_ratio = 0.0
    
    print(f"\n{'='*60}")
    print(f"🧠 Nash Equilibrium Analysis (Bidirectional)")
    print(f"{'='*60}")
    print(f"  • Cooperation moments: {nash_data['cooperation_moments']}")
    print(f"  • Opposition moments: {nash_data['opposition_moments']}")
    print(f"  • Cooperation ratio: {coop_ratio:.1%}")
    print(f"  • Opposition ratio: {opp_ratio:.1%}")
    
    if len(nash_data['authority_ratios']) > 0:
        avg_authority = np.mean(nash_data['authority_ratios'])
        max_authority = np.max(nash_data['authority_ratios'])
        print(f"  • Average authority ratio: {avg_authority:.2f}")
        print(f"  • Max authority ratio: {max_authority:.2f}")
    
    if len(nash_data['field_forces']) > 0:
        avg_total_force = np.mean(nash_data['field_forces'])
        max_total_force = np.max(nash_data['field_forces'])
        avg_leader_force = np.mean(nash_data['leader_forces'])
        max_leader_force = np.max(nash_data['leader_forces'])
        avg_follower_force = np.mean(nash_data['follower_forces'])
        max_follower_force = np.max(nash_data['follower_forces'])
        
        print(f"\n  🛡️ Safety Field Forces:")
        print(f"     Total:    Avg={avg_total_force:.2f}N, Max={max_total_force:.2f}N")
        print(f"     Leader:   Avg={avg_leader_force:.2f}N, Max={max_leader_force:.2f}N")
        print(f"     Follower: Avg={avg_follower_force:.2f}N, Max={max_follower_force:.2f}N")
        
        # Calculate contribution percentages
        if avg_total_force > 0:
            leader_contribution = (avg_leader_force / avg_total_force) * 100
            follower_contribution = (avg_follower_force / avg_total_force) * 100
            print(f"\n  👍 Risk Contribution:")
            print(f"     Leader:   {leader_contribution:.1f}%")
            print(f"     Follower: {follower_contribution:.1f}%")
    
    # === CONSTRAINT ACTIVITY SUMMARY ===
    if hasattr(sim, 'nash_solver') and sim.nash_solver_available:
        if hasattr(sim.nash_solver, 'get_constraint_summary'):
            print(sim.nash_solver.get_constraint_summary())
    
    print(f"{'='*60}")


def run_all_scenarios_with_nash():
    """Run all three scenarios with Nash equilibrium control"""
    print("👍š› Platoon Joining with Nash Equilibrium Control (Bidirectional)")
    print("=" * 60)
    
    scenarios = [
        {
            'name': 'Scenario 1: Join Before Platoon (Nash)',
            'params': {
                'initial_x': 100.0,
                'initial_y': -2.0,
                'target_speed': 100.0,
                'join_trigger_time': 25.0
            }
        },
        {
            'name': 'Scenario 2: Join Middle of Platoon (Nash)',
            'params': {
                'initial_x': -10.0,
                'initial_y': -2.0,
                'target_speed': 110.0,
                'join_trigger_time': 20.0
            }
        },
        {
            'name': 'Scenario 3: Join After Platoon (Nash)',
            'params': {
                'initial_x': -200.0,
                'initial_y': -2.0,
                'target_speed': 110.0,
                'join_trigger_time': 15.0
            }
        }
    ]
    
    results = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'👍Ž¬' * 20}")
        print(f"Starting {scenario['name']} [{i}/3]")
        print(f"{'👍Ž¬' * 20}")
        
        try:
            result = run_scenario_with_nash(scenario['name'], scenario['params'])
            results.append((scenario['name'], result))
            print(f"✅ {scenario['name']} completed!")
            
            if i < len(scenarios):
                print("\n⏸️ 3-second pause before next scenario...")
                time.sleep(3)
        except Exception as e:
            print(f"❌ Error in {scenario['name']}: {e}")
            results.append((scenario['name'], None))
    
    # Overall summary
    print(f"\n{'👍' * 25}")
    print("Nash Control Scenarios Summary (Bidirectional Safety)")
    print(f"{'👍' * 25}")
    
    successful = sum(1 for _, r in results if r is not None)
    print(f"👍 Completed: {successful}/3 scenarios")
    
    return results


if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("🚗 Platoon Simulation with Nash Equilibrium Control (Bidirectional)")
    print("=" * 60)
    print("1. Scenario 1 - Join Before Platoon (Nash)")
    print("2. Scenario 2 - Join Middle of Platoon (Nash)")
    print("3. Scenario 3 - Join After Platoon (Nash)")
    print("4. Run all scenarios with Nash control")
    print("5. Compare: Run original simulation (without Nash)")
    
    try:
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            run_scenario_with_nash("Join Before Platoon", {
                'initial_x': 100.0, 'initial_y': -2.0, 
                'target_speed': 100.0, 'join_trigger_time': 25.0
            })
        elif choice == "2":
            run_scenario_with_nash("Join Middle of Platoon", {
                'initial_x': -5.0, 'initial_y': -2.0,
                'target_speed': 110.0, 'join_trigger_time': 20.0
            })
        elif choice == "3":
            run_scenario_with_nash("Join After Platoon", {
                'initial_x': -200.0, 'initial_y': -2.0,
                'target_speed': 110.0, 'join_trigger_time': 15.0
            })
        elif choice == "4":
            run_all_scenarios_with_nash()
        elif choice == "5":
            print("🚗 Running original simulation without Nash...")
            simulation = run_simulation()
            print("\n👍 Original simulation completed!")
        else:
            print("❌ Invalid choice")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ User interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()