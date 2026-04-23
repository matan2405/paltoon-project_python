"""
Main script for Lateral Control Simulation with Nash Equilibrium.

Research linkage:
- Li-style shared-control flow with distinct system/human references.
- Pustilnik-style embedded authority (implemented in solver modules).
- Scenario runner and reporting entry-point for reproducible experiments.

Based on Li et al. 2019:
- R1 = System reference (5th order polynomial, safe trajectory)
- R2 = Human reference (3rd order polynomial, driver-style trajectory)
- Nash equilibrium for shared control
- Safety Field for authority allocation
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time as time_module
from datetime import datetime
from typing import Dict, Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (RESULTS_DIR, SIMULATION_DT, LANE_WIDTH, PLATOON_LANE_Y,
                    HUMAN_INITIAL_LANE_Y, DEFAULT_SIMULATION_TIME, NOMINAL_VELOCITY, DRIVER_PARAMS)
from simulation import LateralSimulation
from visualization import (create_comprehensive_plots, create_trajectory_plot,
                          create_nash_analysis_plots, print_simulation_summary)
from metrics.comfort import evaluate_lane_change_comfort, print_comfort_report


def print_banner():
    """Print welcome banner."""
    print("\n" + "="*70)
    print("🚗 Lateral Control Simulation with Nash Equilibrium")
    print("="*70)


def print_menu():
    """Print main menu."""
    print("1. Scenario 1 - Join Before Platoon (Nash)")
    print("2. Scenario 2 - Join Middle of Platoon (Nash)")
    print("3. Scenario 3 - Join After Platoon (Nash)")
    print("4. Run all scenarios with Nash control")
    print("5. Run single scenario with driver type selection")
    print("6. Exit")
    print()


def get_scenario_params(scenario_name: str, driver_type: str = 'normal') -> Dict:
    """Get scenario parameters.

    human_initial_x is computed so that at merge_trigger_time the human vehicle
    arrives at the desired relative position in the platoon regardless of driver speed.

    Platoon: leader=60m, gap=25m → P2=35m, P3=10m, all at v_platoon=NOMINAL_VELOCITY.
    Desired relative positions at merge start:
      join_before : 20 m ahead of leader
      join_middle : midway between P1 and P2
      join_after  : 20 m behind P3
    """
    platoon_config = {'num_vehicles': 3, 'leader_x': 60.0, 'gap': 25.0}

    v_platoon = NOMINAL_VELOCITY
    v_human   = NOMINAL_VELOCITY + DRIVER_PARAMS.get(driver_type, DRIVER_PARAMS['normal'])['velocity_offset']

    leader_x0, p2_x0, p3_x0 = 60.0, 35.0, 10.0

    if scenario_name == 'join_before':
        merge_trigger_time = 3.0
        target_x = (leader_x0 + v_platoon * merge_trigger_time) + platoon_config['gap']*3
    elif scenario_name == 'join_middle':
        merge_trigger_time = 3.0
        p1_at_merge = leader_x0 + v_platoon * merge_trigger_time
        p2_at_merge = p2_x0    + v_platoon * merge_trigger_time + DRIVER_PARAMS.get(driver_type, DRIVER_PARAMS['normal'])['x_error_offset']
        target_x = (p1_at_merge + p2_at_merge) / 2.0
    elif scenario_name == 'join_after':
        merge_trigger_time = 5.0
        target_x = (p3_x0 + v_platoon * merge_trigger_time) - platoon_config['gap']*3
    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    human_initial_x = target_x - v_human * merge_trigger_time

    return {
        'human_initial_x': round(human_initial_x, 1),
        'human_initial_y': HUMAN_INITIAL_LANE_Y,
        'target_lane_y': PLATOON_LANE_Y,
        'driver_type': driver_type,
        'platoon': platoon_config,
        'merge_trigger_time': merge_trigger_time
    }


def select_driver_type() -> str:
    """Let user select driver type."""
    print("\nSelect driver type:")
    print("1. Cautious")
    print("2. Normal")
    print("3. Aggressive")
    
    while True:
        try:
            choice = input("Enter choice (1-3): ").strip()
            if choice == '1':
                return 'cautious'
            elif choice == '2':
                return 'normal'
            elif choice == '3':
                return 'aggressive'
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except EOFError:
            return 'normal'


def run_scenario(scenario_name: str, driver_type: str = 'normal', 
                T_sim: float = DEFAULT_SIMULATION_TIME) -> Optional[Dict]:
    """
    Run a single scenario with detailed output matching longitudinal format.
    """
    display_name = scenario_name.replace('_', ' ').title()
    
    print(f"\n{'='*70}")
    print(f"🎯 {display_name} (with Nash Equilibrium Control)")
    print(f"{'='*70}")
    
    # Get parameters
    params = get_scenario_params(scenario_name, driver_type)
    
    # Initialize simulation with correct driver_type from the start
    sim = LateralSimulation(dt=SIMULATION_DT, T_sim=T_sim, driver_type=driver_type)
    sim.setup_scenario(f"{scenario_name}_{driver_type}", params)
    
    # Print initialization info
    print(f"🚗 Lateral Simulation Initialized")
    print(f"🚗 Scenario settings:")
    print(f"   📍 Initial position: ({params['human_initial_x']:.1f}, {params['human_initial_y']:.1f})")
    print(f"   🎯 Target lane: y = {params['target_lane_y']:.1f}m")
    human_vx = NOMINAL_VELOCITY + DRIVER_PARAMS.get(driver_type, DRIVER_PARAMS['normal'])['velocity_offset']
    print(f"   🚀 Target speed: {human_vx * 3.6:.0f} km/h")
    print(f"   🧑‍✈️ Driver type: {driver_type}")
    print(f"   🧠 Nash control: ACTIVE")
    print(f"⏱️ Simulation time: {T_sim} seconds")
    print(f"   total steps: {int(T_sim / SIMULATION_DT)} (dt={SIMULATION_DT} seconds)")
    
    # Run simulation
    exec_start = time_module.time()
    data = run_simulation_with_output(sim, T_sim)
    exec_time = time_module.time() - exec_start
    
    if data is None:
        print("❌ Simulation failed!")
        return None
    
    # Print Nash analysis with constraint summary
    print_nash_analysis(data, nash_solver=sim.nash_solver)
    
    # Print final summary
    print_final_summary(data, display_name, T_sim, exec_time)
    
    # Create and display plots (pass MOBIL approval time for vertical line)
    full_name = f"{scenario_name}_{driver_type}"
    mobil_time = getattr(sim, 'mobil_approval_time', None)
    create_and_display_plots(data, full_name, display_name, mobil_approval_time=mobil_time)
    
    # Ask for animation
    create_animation_if_requested(data, full_name)
    
    # Final message
    print(f"\n{'='*70}")
    print(f"✅ Simulation {display_name} completed!")
    print(f"⏱️ Execution time: {exec_time:.2f} seconds")
    print(f"⚡ Speed: {T_sim/exec_time:.1f}x real time")
    print(f"{'='*70}")
    
    return data


def run_simulation_with_output(sim: LateralSimulation, T_sim: float) -> Optional[Dict]:
    """Run simulation with detailed periodic output."""
    print(f"\n🚀 Running simulation (T={T_sim}s)...")
    print(f"👍 Progress:   0.0% (t=   0.0s)\n")
    
    num_steps = int(T_sim / sim.dt)
    print_interval = int(5.0 / sim.dt)  # Every 5 seconds
    progress_interval = num_steps // 3  # 33%, 66%
    
    merge_announced = False
    phase_transitions = []
    last_phase = None
    
    for i in range(num_steps):
        sim.step()
        current_time = sim.time
        
        # Get current phase
        current_phase = sim.safety_field.get_phase_name()
        
        # Detect phase transitions
        if last_phase is not None and current_phase != last_phase:
            phase_transitions.append((current_time, last_phase, current_phase))
            print(f"✅ Phase transition: {last_phase} → {current_phase} at t={current_time:.1f}s")
            
            # Print additional info on transition
            y_error = abs(sim.human_vehicle.state.y - 0.0)
            psi_error = abs(np.degrees(sim.human_vehicle.state.psi))
            print(f"   📏 y_error: {y_error:.2f}m, ψ_error: {psi_error:.2f}°")
        
        last_phase = current_phase
        
        # Announce merge start
        if not merge_announced and sim.merge_commanded:
            merge_announced = True
            print(f"\n👍 Activating lane change with Nash control at t={current_time:.1f}s")
            print(f"👍 Human vehicle position: x={sim.human_vehicle.state.x:.1f}m, y={sim.human_vehicle.state.y:.1f}m")
            platoon_pos = [f"{v.x:.1f}m" for v in sim.platoon_manager.vehicles]
            print(f"👍 Platoon positions: {platoon_pos}")
            print(f"✅ Human vehicle started lane change with Nash equilibrium control")
            print(f"    Current phase: {current_phase}")
        
        # Detailed output every 5 seconds
        if i > 0 and i % print_interval == 0:
            print_detailed_status(sim, current_time, current_phase)
        
        # Progress indicator
        if i > 0 and i % progress_interval == 0:
            progress = (i / num_steps) * 100
            print(f"👍 Progress: {progress:5.1f}% (t={current_time:6.1f}s)")
    
    # Convert data to numpy arrays
    for key in sim.data:
        if key != 'platoon_positions':
            sim.data[key] = np.array(sim.data[key])
    
    return sim.data


def print_detailed_status(sim: LateralSimulation, current_time: float, phase: str):
    """Print detailed status like longitudinal controller."""
    state = sim.human_vehicle.state
    
    print(f"\n⏱️Time: {current_time:.1f}s")
    
    # Platoon vehicles
    for i, v in enumerate(sim.platoon_manager.vehicles):
        role = "leader" if i == 0 else "follower"
        print(f"Platoon_{i+1} (state-space, {role}): "
              f"Pos=({v.x:.1f}, {v.y:.1f}), Speed={v.vx * 3.6:.1f} km/h")
    
    # Human vehicle
    status = "lane changing" if phase in ['LANE_CHANGE', 'LANE_CHANGE_EXEC'] else phase.lower()
    print(f"Human (state-space, {status}): "
          f"Pos=({state.x:.1f}, {state.y:.2f}), Speed={state.vx * 3.6:.1f} km/h, "
          f"ψ={np.degrees(state.psi):.2f}°")
    
    # Nash data
    if len(sim.data['delta_system']) > 0:
        delta_sys = np.degrees(sim.data['delta_system'][-1])
        delta_hum = np.degrees(sim.data['delta_human'][-1])
        delta_shared = np.degrees(sim.data['delta_shared'][-1])
        lambda_val = sim.data['authority_ratio'][-1]
        field_force = sim.data['field_force'][-1]
        y_error = sim.data['y_error'][-1]
        psi_error = np.degrees(sim.data['psi_error'][-1])
        
        # Cooperation detection
        same_sign = np.sign(delta_sys) == np.sign(delta_hum)
        coop_status = "Coop" if same_sign else "Opp"
        
        print(f"🧠 Nash Data: lambda={lambda_val:.2f}, Shared={delta_shared:.3f}°")
        print(f"   🛡️ Safety Field: {field_force:.1f}N, {coop_status}")
        print(f"    Current phase: {phase}")
        print(f"🎯 Nash: δ_sh={delta_shared:.3f}°, δ_sys={delta_sys:.3f}°, δ_human={delta_hum:.3f}°")
        print(f"   📍 y_err={y_error:.2f}m, ψ_err={psi_error:.2f}°")
        print(f"   🛡️ Force={field_force:.1f}N, λ={lambda_val:.2f}")


def print_nash_analysis(data: Dict, nash_solver=None):
    """Print Nash equilibrium analysis summary."""
    print(f"\n{'='*70}")
    print(f"🧠 Nash Equilibrium Analysis")
    print(f"{'='*70}")
    
    delta_sys = data['delta_system']
    delta_hum = data['delta_human']
    
    # Cooperation/Opposition analysis
    same_sign = np.sign(delta_sys) == np.sign(delta_hum)
    coop_count = np.sum(same_sign)
    opp_count = len(delta_sys) - coop_count
    
    total = len(delta_sys)
    coop_ratio = coop_count / total * 100 if total > 0 else 0
    opp_ratio = opp_count / total * 100 if total > 0 else 0
    
    print(f"  • Cooperation moments: {coop_count}")
    print(f"  • Opposition moments: {opp_count}")
    print(f"  • Cooperation ratio: {coop_ratio:.1f}%")
    print(f"  • Opposition ratio: {opp_ratio:.1f}%")
    
    # Authority ratio stats
    lambda_vals = data['authority_ratio']
    print(f"  • Average authority ratio: {np.mean(lambda_vals):.2f}")
    print(f"  • Max authority ratio: {np.max(lambda_vals):.2f}")
    
    # Safety field stats
    field_forces = data['field_force']
    print(f"\n  🛡️ Safety Field Forces:")
    print(f"     Total:    Avg={np.mean(np.abs(field_forces)):.2f}N, Max={np.max(np.abs(field_forces)):.2f}N")
    
    # Steering stats
    print(f"\n  🎮 Steering Statistics:")
    print(f"     System:  Avg={np.mean(np.abs(np.degrees(delta_sys))):.4f}°, Max={np.max(np.abs(np.degrees(delta_sys))):.4f}°")
    print(f"     Human:   Avg={np.mean(np.abs(np.degrees(delta_hum))):.4f}°, Max={np.max(np.abs(np.degrees(delta_hum))):.4f}°")
    print(f"     Shared:  Avg={np.mean(np.abs(np.degrees(data['delta_shared']))):.4f}°, Max={np.max(np.abs(np.degrees(data['delta_shared']))):.4f}°")
    
    # Constraint summary (like longitudinal)
    if nash_solver is not None and hasattr(nash_solver, 'get_constraint_summary'):
        print(nash_solver.get_constraint_summary())
    
    print(f"{'='*70}")


def print_final_summary(data: Dict, display_name: str, T_sim: float, exec_time: float):
    """Print final simulation summary."""
    print(f"\n📊 === {display_name} Lateral Control Simulation Results ===\n")
    
    print(f"🎯 Simulation Performance:")
    print(f"   ⏱️ Total simulation time: {T_sim:.1f} seconds")
    print(f"   ⚡ Execution time: {exec_time:.2f} seconds")
    print(f"   🚀 Speed factor: {T_sim/exec_time:.1f}x real time")
    
    # Final state
    final_y = data['human_y'][-1]
    final_psi = np.degrees(data['human_psi'][-1])
    final_x = data['human_x'][-1]
    
    # Get final velocity
    if 'human_vx' in data and len(data['human_vx']) > 0:
        final_v = data['human_vx'][-1] * 3.6  # Convert m/s to km/h
    else:
        final_v = 72.0  # Default
    
    print(f"\n🏎️ Human Vehicle Final Status:")
    print(f"   📍 Final position: ({final_x:.1f}, {final_y:.3f})")
    print(f"   🏃 Final speed: {final_v:.1f} km/h")
    print(f"   🧭 Final heading: {final_psi:.3f}°")
    
    # Convergence check
    y_error = abs(final_y - 0.0)
    psi_error = abs(final_psi)
    converged = y_error < 0.1 and psi_error < 1.0
    
    print(f"   ✅ Successfully merged: {'Yes' if converged else 'No'}")
    print(f"   📏 Final y error: {y_error:.3f}m")
    print(f"   📐 Final ψ error: {psi_error:.3f}°")
    
    # Comfort metrics — ISO 2631-1 + Nash Authority Disruption Index
    print(f"\nComfort Metrics (ISO 2631-1 + Nash ADI):")
    comfort_report = evaluate_lane_change_comfort(data)
    print_comfort_report(comfort_report)
    
    print(f"\n{'='*70}")


def create_and_display_plots(data: Dict, full_name: str, display_name: str, 
                             mobil_approval_time: float = None):
    """Create plots and display them."""
    print(f"\n👍 Displaying results for {display_name}...")
    print(f"   👍 Results will be saved to: {os.path.abspath(RESULTS_DIR)}")
    
    print(f"\n👍 Creating comprehensive plots...")
    create_comprehensive_plots(data, full_name, mobil_approval_time=mobil_approval_time)
    print(f"✅ Comprehensive plots created and saved successfully")
    
    print(f"\n👍 Creating trajectory plot...")
    create_trajectory_plot(data, full_name)
    print(f"✅ Trajectory plot created and saved successfully")
    
    print(f"\n👍 Creating Nash analysis plots (9-grid)...")
    create_nash_analysis_plots(data, full_name, mobil_approval_time=mobil_approval_time)
    print(f"✅ Nash analysis plots (9-grid) created and saved successfully")

    # Try to display interactively
    try:
        print(f"📊 Displaying interactive plots...")
        plt.show(block=False)
        plt.pause(1.0)
        print(f"✅ Interactive plots displayed successfully!")
    except Exception as e:
        print(f"   Note: Could not display interactively: {e}")


def create_animation_if_requested(data: Dict, full_name: str):
    """Create and display animation, then ask if user wants to save."""
    try:
        print(f"\n🎬 Creating animation...")
        from visualization.animation import create_lane_change_animation
        
        # Create animation with the new API (matches longitudinal)
        anim = create_lane_change_animation(data, title=full_name)
        
        if anim:
            print(f"✅ Animation completed successfully")
        else:
            print(f"⚠️ Animation creation returned None")
            
    except EOFError:
        print(f"ℹ️ Animation skipped (non-interactive mode)")
    except Exception as e:
        print(f"⚠️ Could not create animation: {e}")


def run_all_scenarios(T_sim: float = DEFAULT_SIMULATION_TIME) -> Dict[str, Dict]:
    """Run all 9 scenarios."""
    scenarios = ['join_before', 'join_middle', 'join_after']
    driver_types = ['cautious', 'normal', 'aggressive']
    
    all_results = {}
    
    print("\n" + "="*70)
    print("🚀 LATERAL CONTROL - FULL TEST SUITE (9 Scenarios)")
    print(f"   Simulation time: {T_sim}s per scenario")
    print("="*70)
    
    total_start = time_module.time()
    
    for scenario in scenarios:
        for driver in driver_types:
            key = f"{scenario}_{driver}"
            print(f"\n{'='*70}")
            print(f"Running: {key}")
            print(f"{'='*70}")
            
            try:
                params = get_scenario_params(scenario, driver)
                sim = LateralSimulation(dt=SIMULATION_DT, T_sim=T_sim)
                sim.setup_scenario(key, params)
                data = run_simulation_with_output(sim, T_sim)
                all_results[key] = data
                
                # Quick summary
                if data is not None:
                    final_y_err = abs(data['human_y'][-1])
                    max_ay = np.max(np.abs(data['human_ay']))
                    print(f"\n   ✅ {key}: y_err={final_y_err:.3f}m, max_ay={max_ay:.3f}m/s²")
                
            except Exception as e:
                print(f"❌ Error in {key}: {e}")
                import traceback
                traceback.print_exc()
                all_results[key] = None
    
    total_time = time_module.time() - total_start
    
    # Print summary table
    print_summary_table(all_results)
    
    print(f"\n✅ All simulations complete!")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Average per scenario: {total_time/9:.1f}s")
    
    return all_results


def print_summary_table(results: Dict[str, Dict]):
    """Print detailed summary table of all results."""
    # Symbol helpers (same fallback logic as comfort.py)
    _SYM = {'Good': '\u2705', 'Acceptable': '\U0001f7e1', 'Poor': '\u274c'}

    def _s(verdict):
        return _SYM.get(verdict, '?')

    def _safe(text):
        try:
            print(text)
        except UnicodeEncodeError:
            import sys
            print(text.encode(sys.stdout.encoding or 'utf-8', errors='replace')
                      .decode(sys.stdout.encoding or 'utf-8'))

    W = 145
    _safe("\n" + "="*W)
    _safe("SUMMARY TABLE - All Scenarios")
    _safe("="*W)

    # Header: convergence | 6 comfort metrics (emoji+value) | overall
    _safe(
        f"{'Scenario':<26} {'y_err':>7} {'psi':>6}  "
        f"{'Dur[s]':^9} {'ay_pk':^9} {'rms_ay':^9} {'Jerk':^9} {'Vy':^9} {'ADI':^9}  "
        f"{'%':>5}  {'Label':<14} {'OK?':>4}"
    )
    _safe("-"*W)

    for name, data in results.items():
        if data is None:
            _safe(f"{name:<26}  FAILED")
            continue

        final_y_err   = abs(data['human_y'][-1])
        final_psi_err = abs(np.degrees(data['human_psi'][-1]))
        converged     = "\u2714" if final_y_err < 0.1 and final_psi_err < 1.0 else "\u2718"

        cr = evaluate_lane_change_comfort(data)
        m  = cr['metrics']
        sc = cr['scores']

        # Each cell: emoji + numeric value
        col_dur  = f"{_s(sc['lc_duration'][1])}{m['lc_duration_s']:>5.1f}s"
        col_pay  = f"{_s(sc['peak_ay'][1])}{m['peak_ay_ms2']:>5.3f}"
        col_rms  = f"{_s(sc['rms_ay'][1])}{m['rms_ay_ms2']:>5.3f}"
        col_jerk = f"{_s(sc['peak_jerk'][1])}{m['peak_jerk_ms3']:>5.2f}"
        col_vy   = f"{_s(sc['peak_vy'][1])}{m['peak_vy_ms']:>5.3f}"
        col_adi  = f"{_s(sc['adi'][1])}{m['adi']:>6.4f}"

        _safe(
            f"{name:<26} {final_y_err:>7.4f} {final_psi_err:>6.3f}  "
            f"{col_dur:^9} {col_pay:^9} {col_rms:^9} {col_jerk:^9} {col_vy:^9} {col_adi:^9}  "
            f"{cr['overall_score']:>5.1f}  {cr['overall_label']:<14} {converged:>4}"
        )

    _safe("="*W)
    _safe(f"  Metrics: Dur=LC duration[s]  ay_pk=peak |ay| [m/s\u00b2]  "
          f"rms_ay=RMS |ay| [m/s\u00b2]  Jerk=peak jerk [m/s\u00b3]  "
          f"Vy=peak lat.vel [m/s]  ADI=authority disruption index")


def main():
    """Main function with interactive menu."""
    print_banner()
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    while True:
        print_menu()
        
        try:
            choice = input("Enter choice (1-6): ").strip()
        except EOFError:
            print("\nExiting...")
            break
        
        if choice == '1':
            driver_type = select_driver_type()
            run_scenario('join_before', driver_type, T_sim=DEFAULT_SIMULATION_TIME)
            
        elif choice == '2':
            driver_type = select_driver_type()
            run_scenario('join_middle', driver_type, T_sim=DEFAULT_SIMULATION_TIME)
            
        elif choice == '3':
            driver_type = select_driver_type()
            run_scenario('join_after', driver_type, T_sim=DEFAULT_SIMULATION_TIME)
            
        elif choice == '4':
            run_all_scenarios(T_sim=DEFAULT_SIMULATION_TIME)
            
        elif choice == '5':
            print("\nSelect scenario:")
            print("1. Join Before Platoon")
            print("2. Join Middle of Platoon")
            print("3. Join After Platoon")
            
            try:
                scenario_choice = input("Enter choice (1-3): ").strip()
                scenarios = {'1': 'join_before', '2': 'join_middle', '3': 'join_after'}
                if scenario_choice in scenarios:
                    driver_type = select_driver_type()
                    
                    try:
                        T_sim = float(input(f"Enter simulation time (seconds, default={DEFAULT_SIMULATION_TIME}): ").strip() or str(DEFAULT_SIMULATION_TIME))
                    except ValueError:
                        T_sim = DEFAULT_SIMULATION_TIME
                    
                    run_scenario(scenarios[scenario_choice], driver_type, T_sim=T_sim)
                else:
                    print("Invalid scenario choice.")
            except EOFError:
                pass
            
        elif choice == '6':
            print("\n👋 Goodbye!")
            break
            
        else:
            print("Invalid choice. Please enter 1-6.")
        
        print("\n" + "-"*70)
        input("Press Enter to continue...")


if __name__ == "__main__":
    main()
