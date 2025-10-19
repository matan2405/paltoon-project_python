#!/usr/bin/env python3
"""
File: main_with_nash.py
Description: Platoon joining simulation with Nash equilibrium control integration.
Combines the platoon simulation from main_without_nash with Nash equilibrium solver from lateral application.
Updated to use bidirectional safety field with platoon integration.
"""

import os
import sys
import time
import numpy as np
from typing import List, Dict

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import platoon simulation modules
from config import setup_matplotlib, HEADLESS_MODE
from simulation.simulator import PlatoonSimulation, run_simulation
from visualization.animation import create_platoon_animation
from visualization.plots import create_comprehensive_plots, create_detailed_scenario_summary
from vehicle import Vehicle
from control.human_driver import HumanDriver
from control.platoon_control import PlatoonManager

# Import Nash solver modules
from nash_solver.longitudinal_authority_allocator import LongitudinalAuthorityAllocator
from nash_solver.longitudinal_safety_field import LongitudinalSafetyField
from nash_solver import longitudinal_nash_solver, system_reference_generator

os.system('cls' if os.name == 'nt' else 'clear')


class PlatoonNashSimulation(PlatoonSimulation):
    """Extended platoon simulation with Nash equilibrium control"""
    
    def __init__(self):
        # Initialize parent platoon simulation
        super().__init__()
        
        # Initialize Nash control components
        self.safety_field = LongitudinalSafetyField()
        self.authority_allocator = LongitudinalAuthorityAllocator()
        
        # Nash control parameters
        self.Np = 20  # Prediction horizon
        self.Nu = 10  # Control horizon
        self.dt_nash = 0.1  # Time step
        # Initialize system reference generator (for controller trajectory planning)
        self.system_ref_generator = system_reference_generator.SystemReferenceGenerator(
            Np=self.Np, dt=self.dt_nash
        )
        
        # Initialize Nash solver with human vehicle instance
        # Note: Nash solver needs a vehicle instance, not the class
        try:
            self.nash_solver = longitudinal_nash_solver.LongitudinalNashSolver(
                vehicle=self.human_vehicle, Np=self.Np, Nu=self.Nu, dt=self.dt_nash  # Pass the vehicle instance
            )
            self.nash_solver_available = True
        except Exception as e:
            print(f"⚠️  Nash solver initialization issue: {e}")
            print("    Using simplified Nash approach")
            self.nash_solver_available = False
        
        # Additional data storage for Nash analysis
        self.nash_data = {
            'controller_inputs': [],
            'human_inputs': [],
            'shared_inputs': [],
            'authority_ratios': [],
            'field_forces': [],
            'leader_forces': [],      # New: track leader risk
            'follower_forces': [],    # New: track follower risk
            'cooperation_moments': 0,
            'opposition_moments': 0
        }
        
        # Corrected weights and costs for Nash control (same as lateral)
        self.Q_output = np.diag([50.0, 200.0])  # Reasonable position/velocity weights
        self.R1 = 50.0   # Significant control cost (smooth control)
        self.R2 = 60.0   # Slightly higher for human (prefer less aggressive)
        
        print("🚗 Platoon Nash Simulation Initialized")
        print("🧠 Nash Equilibrium Control Active")
        print(f"   System Reference Generator: ✅")
        print(f"   Nash Solver: {'✅' if self.nash_solver_available else '⚠️ Simplified'}")
        print(f"   Safety Field: ✅ Bidirectional (Leader + Follower)")
        print("="*60)

    def nash_control_step(self, human_vehicle: Vehicle, leader_vehicle: Vehicle = None) -> Dict:
        """
        Execute one Nash equilibrium control step for the human vehicle.
        Now uses bidirectional safety field with platoon integration.
        """
        # Get current state from vehicle - [position, velocity]
        current_state = np.array([
            human_vehicle.state.x,   # position in x direction
            human_vehicle.state.vx   # velocity in x direction
        ])
        
        # Get leader and follower using the new platoon integration
        leader, follower = self.safety_field.get_leader_and_follower(
            human_vehicle, self.platoon_manager
        )
        
        # Calculate desired gap - prioritize leader, but use follower if no leader
        if leader is not None:
            from control.platoon_control import rajamani
            _, desired_gap = rajamani(leader, human_vehicle)
        elif follower is not None:
            # No leader but have follower - calculate gap based on follower
            from control.platoon_control import rajamani
            # Use Rajamani with reversed roles (follower becomes "leader" for calculation)
            _, desired_gap = rajamani(human_vehicle, follower)
            print(f"   ℹ️  No leader - using follower-based gap: {desired_gap:.1f}m")
        else:
            # No leader and no follower - use default
            desired_gap = 50.0  # large default gap
            print(f"   ℹ️  No leader or follower - using default gap: {desired_gap:.1f}m")
        
        # 1. Safety field evaluation - NOW BIDIRECTIONAL!
        # Compute risk from both leader and follower
        field_force = self.safety_field.compute_risk_force_from_platoon(
            ego_vehicle=human_vehicle,
            platoon_manager=self.platoon_manager,
            desired_gap=desired_gap
        )
        
        # Get detailed breakdown for analysis
        breakdown = self.safety_field.get_force_breakdown_from_platoon(
            ego_vehicle=human_vehicle,
            platoon_manager=self.platoon_manager,
            desired_gap=desired_gap
        )
        print(f"   🛡️  Field Force Breakdown: "
              f"Leader={breakdown['leader']['total_force']:.1f}N, "
              f"Headway={breakdown['leader']['headway_force']:.1f}N, "
              f"GapError={breakdown['leader']['gap_error_force']:.1f}N, "
              f"RelVel={breakdown['leader']['rel_vel_force']:.1f}N, "
              f"TTC={breakdown['leader']['ttc']:.2f}, "
              f"HeadwayTime={breakdown['leader']['headway']:.2f}, "
              f"GapErrorVal={breakdown['leader']['gap_error']:.2f}, "
              f"RelVelVal={breakdown['leader']['relative_vel']:.2f}")
        # 2. Authority allocation
        lambda_k = self.authority_allocator.compute_authority_ratio(field_force)
        
        # Print authority split
        alpha = lambda_k / (1.0 + lambda_k)
        print(f"   🎛️  Authority: λ={lambda_k:.2f}, α={alpha:.2%} system, {(1-alpha):.2%} human")
        print(f"      Risk: {field_force:.1f}N → {'SYSTEM TAKES CONTROL' if alpha > 0.5 else 'Human dominant'}")
        
        # 3. Plan reference trajectories using existing functions
        # Controller reference: system reference generator returns [acceleration_sequence, state_sequence]
        # state_sequence is [Np, 2] with columns [position, velocity]
        accel_seq_controller, state_seq_controller = self.system_ref_generator.get_system_acceleration_and_state_sequence(self)
        
        # Human reference: human driver model returns [acceleration_sequence, state_sequence]
        # state_sequence is [Np, 2] with columns [position, velocity]
        accel_seq_human, state_seq_human = self.human_driver.get_human_acceleration_and_state_sequence(
            dt=self.dt_nash, Np=self.Np, vehicle=human_vehicle
        )
        
        # Get human desired acceleration
        if not self.human_driver.merging and not human_vehicle.joined_platoon:
            human_desired_accel = self.human_driver.update(self, self.platoon_vehicles)
        else:
            human_desired_accel = accel_seq_human[0] if len(accel_seq_human) > 0 else 0.0
        
        # 4. Format REFERENCE TRAJECTORIES
        R1_ref_trajectory = np.zeros((self.Np, 2)) # Controller reference: [desired_position, desired_velocity]
        R2_ref_trajectory = np.zeros((self.Np, 2)) # Human reference: [planned_position, planned_velocity]
        
        for i in range(self.Np):
            # Controller reference: use the system's planned trajectory
            R1_ref_trajectory[i, 0] = state_seq_controller[i, 0]  # desired position
            R1_ref_trajectory[i, 1] = state_seq_controller[i, 1]  # desired velocity
            
            # Human reference: use human's planned trajectory
            R2_ref_trajectory[i, 0] = state_seq_human[i, 0]  # planned position
            R2_ref_trajectory[i, 1] = state_seq_human[i, 1]  # planned velocity
        
        # Flatten for Nash solver: (Np * 2, 1) = (40, 1) format
        R1_ref = R1_ref_trajectory.reshape(self.Np * 2, 1)
        R2_ref = R2_ref_trajectory.reshape(self.Np * 2, 1)
        
        # 5. Solve Nash equilibrium
        try:
            u1_opt, u2_opt = self.nash_solver.solve_nash_equilibrium(
                current_state, R1_ref, R2_ref, lambda_k
            )
                    
        except Exception as e:
            print(f"⚠️  Nash solver error: {e}, using fallback control")
            u1_opt = 0.0
            u2_opt = human_desired_accel
        
        # 6. Calculate shared control
        alpha = lambda_k / (1.0 + lambda_k)
        u_shared = alpha * u1_opt + (1 - alpha) * u2_opt
        print(f"   🤝 Shared Control: u_shared={u_shared:.2f} m/s²")
        
        # Store data - NOW INCLUDING BIDIRECTIONAL BREAKDOWN
        self.nash_data['controller_inputs'].append(u1_opt)
        self.nash_data['human_inputs'].append(u2_opt)
        self.nash_data['shared_inputs'].append(u_shared)
        self.nash_data['authority_ratios'].append(lambda_k)
        self.nash_data['field_forces'].append(field_force)
        self.nash_data['leader_forces'].append(breakdown['leader']['total_force'])
        self.nash_data['follower_forces'].append(breakdown['follower']['total_force'])
        
        # Analyze cooperation vs opposition
        if abs(u1_opt) > 0.01 and abs(u2_opt) > 0.01:
            if u1_opt * u2_opt > 0:  # Same direction -> cooperation
                self.nash_data['cooperation_moments'] += 1
            else:  # Opposing directions -> opposition
                self.nash_data['opposition_moments'] += 1
        
        return {
            'controller_input': u1_opt,
            'human_input': u2_opt,
            'shared_input': u_shared,
            'authority_ratio': lambda_k,
            'field_force': field_force,
            'leader_force': breakdown['leader']['total_force'],
            'follower_force': breakdown['follower']['total_force'],
            'breakdown': breakdown  # Full breakdown for detailed analysis
        }
    
    def update_with_nash(self):
        """Override update method to include Nash control"""
        # Apply Nash control to human vehicle if it's in merging phase or joined
        if self.human_driver.merging or self.human_vehicle.joined_platoon:
            nash_result = self.nash_control_step(self.human_vehicle)
            
            # Apply shared control input as acceleration
            shared_accel = nash_result['shared_input']
            u_sys = nash_result['controller_input']
            u_human = nash_result['human_input']
            
            # Store Nash acceleration for platoon manager to use
            self.human_vehicle.nash_acceleration = shared_accel
            self.human_vehicle.a_desired = shared_accel
            self.human_vehicle.state.ax = shared_accel
            
            # Debug info every second - NOW WITH BIDIRECTIONAL INFO
            if int(self.time) != int(self.time - self.dt):
                gap_to_leader = "N/A"
                gap_to_follower = "N/A"
                
                # Get leader and follower info
                leader, follower = self.safety_field.get_leader_and_follower(
                    self.human_vehicle, self.platoon_manager
                )
                
                if leader is not None:
                    gap_to_leader = leader.state.x - self.human_vehicle.state.x
                    
                if follower is not None:
                    gap_to_follower = self.human_vehicle.state.x - follower.state.x
                
                print(f"   🎯 Nash: u_sh={shared_accel:.2f} m/s², "
                      f"u_sys={u_sys:.2f}, u_human={u_human:.2f}")
                print(f"      📍 Gap_L={gap_to_leader if isinstance(gap_to_leader, str) else f'{gap_to_leader:.1f}m'}, "
                      f"Gap_F={gap_to_follower if isinstance(gap_to_follower, str) else f'{gap_to_follower:.1f}m'}, "
                      f"v={self.human_vehicle.state.vx*3.6:.1f}km/h")
                print(f"      🛡️ Risk_L={nash_result['leader_force']:.1f}N, "
                      f"Risk_F={nash_result['follower_force']:.1f}N, "
                      f"Total={nash_result['field_force']:.1f}N")
        
        # Call parent update - this will use nash_acceleration in platoon manager
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
    
    print(f"⏱️ Simulation time: {max_time:.0f} seconds")
    print(f"   total steps: {max_iterations} (dt={sim.dt:.2f} seconds)")
    
    start_time = time.time()
    
    try:
        for iteration, t in enumerate(T):
            # Print progress every 2000 iterations
            if iteration % 2000 == 0:
                progress = (iteration / max_iterations) * 100
                print(f"📈 Progress: {progress:5.1f}% (t={sim.time:6.1f}s)")
            
            # Update simulation with Nash control
            sim.update_with_nash()
            
            # Status report every 5 seconds
            if sim.time % 5.0 < sim.dt:
                sim.print_status()
                time.sleep(3)  # Small delay for readability
                if human_joined and len(sim.nash_data['shared_inputs']) > 0:
                    leader_force = sim.nash_data['leader_forces'][-1]
                    follower_force = sim.nash_data['follower_forces'][-1]
                    print(f"🧠 Nash Data: λ={sim.nash_data['authority_ratios'][-1]:.2f}, "
                          f"Shared={sim.nash_data['shared_inputs'][-1]:.2f}")
                    print(f"   🛡️ Risks: Leader={leader_force:.1f}N, "
                          f"Follower={follower_force:.1f}N, "
                          f"Total={sim.nash_data['field_forces'][-1]:.1f}N")
                    print(f"   🤝 Coop={sim.nash_data['cooperation_moments']}, "
                          f"Opp={sim.nash_data['opposition_moments']}")
                
            # Trigger joining at the right time
            if (sim.time >= join_trigger_time and 
                not human_joined and not sim.human_driver.merging):
                
                print(f"\n🚨 Activating joining with Nash control at t={sim.time:.1f}s")
                print(f"📍 Human vehicle position: x={sim.human_vehicle.state.x:.1f}m")
                print(f"📍 platoon positions: {[f'{v.state.x:.1f}m' for v in sim.platoon_vehicles]}")
                sim.human_driver.merging = True
                human_joined = True
                sim.human_vehicle.joined_platoon = True
                sim.platoon_manager.add_vehicle(sim.human_vehicle)
                sim.human_vehicle.target_velocity = sim.platoon_manager.target_velocity
                sim.human_driver.target_speed = sim.human_driver.target_speed + 30.0 / 3.6  # Slightly higher to facilitate merging
                # sim.human_driver.delta_IDM += 1.0  # More aggressive gap closing
                print(f"joining platoon at index {sim.platoon_manager.get_new_vehicle_index(sim.human_vehicle)}")
                print(f"✅ Human vehicle started joining with Nash equilibrium control with target velocity {sim.human_driver.target_speed * 3.6:.1f} km/h")
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Print Nash analysis
        print_nash_analysis(sim)
        
        # Create visualizations
        print(f"\n📊 Displaying results for {scenario_name}...")
        static_plots_fig = create_comprehensive_plots(sim, scenario_name)
        
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
            print(f"\n  📊 Risk Contribution:")
            print(f"     Leader:   {leader_contribution:.1f}%")
            print(f"     Follower: {follower_contribution:.1f}%")
    
    print(f"{'='*60}")


def run_all_scenarios_with_nash():
    """Run all three scenarios with Nash equilibrium control"""
    print("🚛 Platoon Joining with Nash Equilibrium Control (Bidirectional)")
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
                'target_speed': 70.0,
                'join_trigger_time': 20.0
            }
        },
        {
            'name': 'Scenario 3: Join After Platoon (Nash)',
            'params': {
                'initial_x': -100.0,
                'initial_y': -2.0,
                'target_speed': 70.0,
                'join_trigger_time': 15.0
            }
        }
    ]
    
    results = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'🎬' * 20}")
        print(f"Starting {scenario['name']} [{i}/3]")
        print(f"{'🎬' * 20}")
        
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
    print(f"\n{'🏁' * 25}")
    print("Nash Control Scenarios Summary (Bidirectional Safety)")
    print(f"{'🏁' * 25}")
    
    successful = sum(1 for _, r in results if r is not None)
    print(f"📊 Completed: {successful}/3 scenarios")
    
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
                'initial_x': -10.0, 'initial_y': -2.0,
                'target_speed': 70.0, 'join_trigger_time': 20.0
            })
        elif choice == "3":
            run_scenario_with_nash("Join After Platoon", {
                'initial_x': -300.0, 'initial_y': -2.0,
                'target_speed': 110.0, 'join_trigger_time': 15.0
            })
        elif choice == "4":
            run_all_scenarios_with_nash()
        elif choice == "5":
            print("🚗 Running original simulation without Nash...")
            simulation = run_simulation()
            print("\n📊 Original simulation completed!")
        else:
            print("❌ Invalid choice")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ User interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()