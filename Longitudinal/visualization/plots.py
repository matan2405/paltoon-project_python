"""
Static plotting module for creating comprehensive analysis plots.
Contains functions for creating detailed simulation analysis visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import HEADLESS_MODE, RESULTS_DIR


def create_comprehensive_plots(simulation, scenario_name="Simulation", show_plots=True):
    """Create comprehensive 9-plot analysis for any simulation"""
    try:
        import matplotlib.pyplot as plt
        
        print(f"[PALETTE] Creating comprehensive plots for: {scenario_name}")
        print(f"   📊 Data points: {len(simulation.time_history)}")
        print(f"   🚗 Vehicles: {len(simulation.all_vehicles)}")
        
        # Create figure with proper cleanup
        fig = plt.figure(figsize=(16, 12))
        print(f"   ✅ Figure created: {fig.number}")
        
        # Plot 1: Vehicle positions
        plt.subplot(3, 3, 1)
        for i, vehicle in enumerate(simulation.all_vehicles):
            x_positions = []
            valid_times = []
            for j, pos in enumerate(simulation.position_history):
                if i < len(pos) and not np.isnan(pos[i][0]):
                    x_positions.append(pos[i][0])
                    valid_times.append(simulation.time_history[j])
            if x_positions:
                plt.plot(valid_times, x_positions, label=vehicle.vehicle_id, linewidth=2)
        plt.xlabel('Time [s]')
        plt.ylabel('Position [m]')
        plt.title('Vehicle Positions')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Vehicle velocities
        plt.subplot(3, 3, 2)
        for i, vehicle in enumerate(simulation.all_vehicles):
            velocities = []
            valid_times = []
            for j, vel in enumerate(simulation.velocity_history):
                if i < len(vel) and not np.isnan(vel[i]):
                    velocities.append(vel[i] / 3.6)  # Convert back to m/s for plotting
                    valid_times.append(simulation.time_history[j])
            if velocities:
                plt.plot(valid_times, velocities, label=vehicle.vehicle_id, linewidth=2)
        # Add target velocity line
        target_line = [simulation.platoon_manager.target_velocity] * len(simulation.time_history)
        plt.plot(simulation.time_history, target_line, 'k--', label='Target (Platoon)', linewidth=2)
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.title('Vehicle Velocities')
        plt.legend()
        plt.grid(True)
        
        # Plot 3: Inter-vehicle gaps (platoon only)
        plt.subplot(3, 3, 3)
        if simulation.gap_history:
            max_gaps = max(len(gap_set) for gap_set in simulation.gap_history) if simulation.gap_history else 0
            for i in range(max_gaps):
                gaps = []
                desired_gaps = []
                valid_times = []
                for j, (gap_set, desired_set) in enumerate(zip(simulation.gap_history, simulation.desired_gap_history)):
                    if i < len(gap_set) and i < len(desired_set):
                        gaps.append(gap_set[i])
                        desired_gaps.append(desired_set[i])
                        valid_times.append(simulation.time_history[j])
                if gaps:
                    plt.plot(valid_times, gaps, label=f'Gap {i+1}', linewidth=2)
                    plt.plot(valid_times, desired_gaps, '--', label=f'Desired {i+1}', linewidth=2)
        plt.xlabel('Time [s]')
        plt.ylabel('Gap [m]')
        plt.title('Inter-vehicle Gaps')
        plt.legend()
        plt.grid(True)
        
        # Plot 4: Gap errors
        plt.subplot(3, 3, 4)
        if simulation.gap_history:
            max_gaps = max(len(gap_set) for gap_set in simulation.gap_history) if simulation.gap_history else 0
            for i in range(max_gaps):
                gap_errors = []
                valid_times = []
                for j, (gap_set, desired_set) in enumerate(zip(simulation.gap_history, simulation.desired_gap_history)):
                    if i < len(gap_set) and i < len(desired_set):
                        gap_errors.append(gap_set[i] - desired_set[i])
                        valid_times.append(simulation.time_history[j])
                if gap_errors:
                    plt.plot(valid_times, gap_errors, label=f'Error {i+1}', linewidth=2)
        plt.xlabel('Time [s]')
        plt.ylabel('Gap Error [m]')
        plt.title('Gap Tracking Errors')
        plt.legend()
        plt.grid(True)
        
        # Plot 5: Vehicle trajectories (top view)
        plt.subplot(3, 3, 5)
        for i, vehicle in enumerate(simulation.all_vehicles):
            x_positions = []
            y_positions = []
            for pos in simulation.position_history:
                if i < len(pos) and not np.isnan(pos[i][0]) and not np.isnan(pos[i][1]):
                    x_positions.append(pos[i][0])
                    y_positions.append(pos[i][1])
            if x_positions:
                plt.plot(x_positions, y_positions, 'o-', label=vehicle.vehicle_id, markersize=1, linewidth=2)
        
        # Draw road lanes
        if simulation.position_history:
            all_x = [pos[i][0] for pos in simulation.position_history for i in range(len(pos)) if not np.isnan(pos[i][0])]
            if all_x:
                x_range = [min(all_x), max(all_x)]
                plt.plot(x_range, [0, 0], 'k-', alpha=0.3, linewidth=3, label='Right Lane')
                plt.plot(x_range, [-2, -2], 'k-', alpha=0.3, linewidth=3, label='Left Lane')
                plt.plot(x_range, [-1, -1], 'y--', alpha=0.5, linewidth=2, label='Lane Divider')
        
        plt.xlabel('X Position [m]')
        plt.ylabel('Y Position [m]')
        plt.title('Vehicle Trajectories (Top View)')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        # Plot 6: String stability analysis
        plt.subplot(3, 3, 6)
        if len(simulation.velocity_history) > 0:
            max_platoon_vehicles = 4  # Original platoon size
            for i in range(max_platoon_vehicles - 1):
                vel_leader = []
                vel_follower = []
                valid_times = []
                
                for j, vel_data in enumerate(simulation.velocity_history):
                    if (i < len(vel_data) and (i+1) < len(vel_data) and 
                        not np.isnan(vel_data[i]) and not np.isnan(vel_data[i+1])):
                        vel_leader.append(vel_data[i] / 3.6)  # Convert to m/s
                        vel_follower.append(vel_data[i+1] / 3.6)  # Convert to m/s
                        valid_times.append(simulation.time_history[j])
                
                if vel_leader and vel_follower:
                    vel_diff = [vl - vf for vl, vf in zip(vel_leader, vel_follower)]
                    plt.plot(valid_times, vel_diff, label=f'ΔV {i+1}-{i+2}', linewidth=2)
        
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity Difference [m/s]')
        plt.title('String Stability Analysis')
        plt.legend()
        plt.grid(True)
        
        # Plot 7: Accelerations
        plt.subplot(3, 3, 7)
        for i, vehicle in enumerate(simulation.all_vehicles):
            accelerations = []
            valid_times = []
            for j, acc in enumerate(simulation.acceleration_history):
                if i < len(acc) and not np.isnan(acc[i]):
                    accelerations.append(acc[i])
                    valid_times.append(simulation.time_history[j])
            if accelerations:
                plt.plot(valid_times, accelerations, label=vehicle.vehicle_id, linewidth=2)
        plt.xlabel('Time [s]')
        plt.ylabel('Acceleration [m/s^2]')
        plt.title('Vehicle Accelerations')
        plt.legend()
        plt.grid(True)
        
        # Plot 8: Human vehicle lane change detail
        plt.subplot(3, 3, 8)
        human_idx = simulation.vehicle_indices.get("Human", -1)
        if human_idx >= 0:
            y_positions = []
            valid_times = []
            for j, pos in enumerate(simulation.position_history):
                if human_idx < len(pos) and not np.isnan(pos[human_idx][1]):
                    y_positions.append(pos[human_idx][1])
                    valid_times.append(simulation.time_history[j])
            if y_positions:
                plt.plot(valid_times, y_positions, 'r-', linewidth=3, label='Human Y-position')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3, label='Right Lane Center')
        plt.axhline(y=-2, color='k', linestyle='-', alpha=0.3, label='Left Lane Center')
        plt.axvline(x=20.0, color='g', linestyle='--', alpha=0.7, label='Merge Start')
        plt.xlabel('Time [s]')
        plt.ylabel('Lateral Position [m]')
        plt.title('Human Vehicle Lane Change')
        plt.legend()
        plt.grid(True)
        
        # Plot 9: Vehicle count over time
        plt.subplot(3, 3, 9)
        platoon_count = []
        for t in simulation.time_history:
            if t < 20.0:
                platoon_count.append(4)  # Original platoon size
            else:
                platoon_count.append(5)  # After human joins
        plt.plot(simulation.time_history, platoon_count, 'b-', linewidth=3, label='Platoon Size')
        plt.axvline(x=20.0, color='g', linestyle='--', alpha=0.7, label='Human Joins')
        plt.xlabel('Time [s]')
        plt.ylabel('Number of Vehicles')
        plt.title('Platoon Size Over Time')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.suptitle(f'{scenario_name} - Comprehensive Analysis', fontsize=16, y=0.98)
        
        # Always save plot to results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_scenario_name = scenario_name.replace(" ", "_").replace(":", "")
        filename = f'{safe_scenario_name}_results_{timestamp}.png'
        
        # Ensure results directory exists
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
            print(f"📁 Created results directory: {RESULTS_DIR}")
        
        filepath = os.path.join(RESULTS_DIR, filename)
        
        try:
            # Save with explicit figure reference
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            abs_path = os.path.abspath(filepath)
            file_size = os.path.getsize(filepath) / 1024  # KB
            print(f"✅ Comprehensive plot saved successfully!")
            print(f"   📄 File: {filename}")
            print(f"   📁 Path: {abs_path}")
            print(f"   💾 Size: {file_size:.1f} KB")
        except Exception as save_error:
            print(f"❌ Could not save plot: {save_error}")
            import traceback
            traceback.print_exc()
        
        # Display plots
        if show_plots:
            try:
                if HEADLESS_MODE:
                    print("📊 Plots saved in headless mode")
                    try:
                        import subprocess
                        import platform
                        if platform.system() == 'Windows':
                            os.startfile(filename)
                            print("📊 Opening plots with system image viewer")
                        else:
                            subprocess.run(['xdg-open', filename], check=False)
                            print("📊 Opening plots with system image viewer")
                    except:
                        print("📊 Plots saved - you can view them manually")
                else:
                    print("📊 Displaying interactive plots...")
                    plt.ion()
                    plt.show(block=False)
                    plt.draw()
                    fig.canvas.flush_events()
                    print("✅ Interactive plots displayed successfully!")
                    time.sleep(2)
            except Exception as show_error:
                print(f"⚠️ Could not display plots interactively: {show_error}")
        
        return fig if 'fig' in locals() else None
        
    except Exception as plot_error:
        print(f"Could not create comprehensive plots: {plot_error}")
        import traceback
        traceback.print_exc()
        # Clean up figure if it was created
        if 'fig' in locals():
            try:
                plt.close(fig)
            except:
                pass
        return None


def create_detailed_scenario_summary(simulation, scenario_name, execution_time):
    """Create comprehensive unified summary for each scenario"""
    print(f"\n{'=' * 70}")
    print(f"✅ Simulation {scenario_name} completed!")
    print(f"⏱️ Execution time: {execution_time:.2f} seconds")
    print(f"[ZAP] Speed: {simulation.time_history[-1]/execution_time:.1f}x real time")
    print(f"{'=' * 70}")
    
    print(f"\n📊 === {scenario_name} Platoon Control Simulation Results ===")
    
    # Simulation performance summary  
    print(f"\n🎯 Simulation Performance:")
    print(f"   ⏱️ Total simulation time: {simulation.time_history[-1]:.1f} seconds")
    print(f"   [ZAP] Execution time: {execution_time:.2f} seconds")
    print(f"   🚀 Speed factor: {simulation.time_history[-1]/execution_time:.1f}x real time")
    
    # platoon composition
    print(f"\n[TRUCK] platoon Composition:")
    print(f"   📊 Total platoon vehicles: {len(simulation.platoon_manager.vehicles)}")
    print(f"   🎯 Target velocity: {simulation.platoon_manager.target_velocity*3.6:.1f} km/h")
    
    # Human vehicle status
    print(f"\n🏎️ Human Vehicle Status:")
    print(f"   📍 Final position: ({simulation.human_vehicle.state.x:.1f}, {simulation.human_vehicle.state.y:.1f})")
    print(f"   🏃 Final speed: {simulation.human_vehicle.state.vx*3.6:.1f} km/h")
    print(f"   ✅ Successfully joined platoon: {'Yes' if simulation.human_vehicle.joined_platoon else 'No'}")
    
    # Final positions
    final_positions = []
    for vehicle in simulation.platoon_manager.vehicles:
        final_positions.append(f"{vehicle.state.x:.1f}m")
    print(f"\n📍 Final platoon Positions: {final_positions}")
    
    # Final gaps between vehicles
    final_gaps = []
    if len(simulation.platoon_manager.vehicles) > 1:
        for i in range(len(simulation.platoon_manager.vehicles) - 1):
            leader = simulation.platoon_manager.vehicles[i]
            follower = simulation.platoon_manager.vehicles[i + 1]
            gap = leader.state.x - follower.state.x
            final_gaps.append(f"{gap:.1f}m")
    print(f"📏 Final inter-vehicle gaps: {final_gaps}")
    
    # Calculate and display desired gaps
    final_desired_gaps = []
    if len(simulation.platoon_manager.vehicles) > 1:
        h = 1.5  # Desired time headway (same as in simulation)
        for i in range(len(simulation.platoon_manager.vehicles) - 1):
            leader = simulation.platoon_manager.vehicles[i]
            follower = simulation.platoon_manager.vehicles[i + 1]
            desired_gap = leader.L + h * follower.state.vx  # Default desired gap formula
            final_desired_gaps.append(f"{desired_gap:.1f}m")
    print(f"📏 Desired inter-vehicle gaps: {final_desired_gaps}")
    
    # Calculate and display average gap
    if final_gaps:
        avg_gap = np.mean([float(gap.replace('m', '')) for gap in final_gaps])
        print(f"📊 Average platoon spacing: {avg_gap:.1f} meters")
    
    # Calculate and display average desired gap
    if final_desired_gaps:
        avg_desired_gap = np.mean([float(gap.replace('m', '')) for gap in final_desired_gaps])
        print(f"📊 Average desired spacing: {avg_desired_gap:.1f} meters")
    
    print(f"{'=' * 70}")
    print(f"🎯 Scenario {scenario_name} Summary Complete")
    print(f"{'=' * 70}")


def create_nash_analysis_plots(simulation, scenario_name="Nash Equilibrium Analysis", show_plots=True):
    """Create comprehensive 9-plot Nash equilibrium analysis for longitudinal control"""
    try:
        print(f"[PALETTE] Creating enhanced 9-grid Nash analysis plots for: {scenario_name}")
        
        # Check if Nash data exists
        if not hasattr(simulation, 'authority_history') or len(simulation.authority_history) == 0:
            print(f"⚠️ No Nash control data available")
            return None
        
        # Extract data
        time = np.array(simulation.time_history)
        system_acc_attr = 'system_desired_acc_history' if hasattr(simulation, 'system_desired_acc_history') else 'system_control_history'
        human_acc_attr = 'human_desired_acc_history' if hasattr(simulation, 'human_desired_acc_history') else 'human_control_history'
        
        min_length = min(len(time), 
                        len(simulation.authority_history),
                        len(getattr(simulation, system_acc_attr, [])),
                        len(getattr(simulation, human_acc_attr, [])))
        
        if min_length < 2:
            print(f"⚠️ Insufficient data points ({min_length})")
            return None
        
        # Truncate arrays
        time = time[:min_length]
        authority = np.array(simulation.authority_history[:min_length])
        u_system = np.array(getattr(simulation, system_acc_attr)[:min_length])
        u_human = np.array(getattr(simulation, human_acc_attr)[:min_length])
        
        # Calculate shared control
        alpha = authority / (1.0 + authority)
        u_shared = alpha * u_system + (1.0 - alpha) * u_human
        
        # Get additional data
        safety_forces = simulation.safety_force_history[:min_length] if hasattr(simulation, 'safety_force_history') else []
        gap_errors = simulation.gap_error_history[:min(len(simulation.gap_error_history), min_length)] if hasattr(simulation, 'gap_error_history') else []
        
        # Velocity data
        velocity_history = []
        if hasattr(simulation, 'human_vehicle'):
            for i in range(min_length):
                if i < len(simulation.velocity_history):
                    vel_data = simulation.velocity_history[i]
                    # Get human vehicle velocity (last one if it exists)
                    if len(vel_data) > len(simulation.platoon_manager.vehicles):
                        velocity_history.append(vel_data[-1])
                    elif len(vel_data) > 0:
                        velocity_history.append(vel_data[0])
                    else:
                        velocity_history.append(0)
        
        # Create figure
        fig = plt.figure(figsize=(20, 14))
        fig.suptitle(f'{scenario_name} - Nash Equilibrium Analysis', fontsize=16, fontweight='bold')
        
        # ========== Row 1 ==========
        # Plot 1: Longitudinal Acceleration Components
        plt.subplot(3, 3, 1)
        plt.plot(time, u_shared, 'r-', linewidth=2.5, label='Shared (Applied)', alpha=0.8)
        plt.plot(time, u_system, 'g--', linewidth=1.5, label='System Desired', alpha=0.7)
        plt.plot(time, u_human, 'b--', linewidth=1.5, label='Human Desired', alpha=0.7)
        plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        plt.axhline(y=2.0, color='gray', linestyle='--', linewidth=1, alpha=0.4)
        plt.axhline(y=-3.0, color='gray', linestyle='--', linewidth=1, alpha=0.4)
        plt.xlabel('Time [s]', fontsize=10)
        plt.ylabel('Acceleration [m/s^2]', fontsize=10)
        plt.title('Longitudinal Acceleration Components', fontsize=11, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=9)
        
        # Plot 2: Authority Ratio lambda(t) - LOG SCALE
        plt.subplot(3, 3, 2)
        plt.semilogy(time, authority, 'm-', linewidth=2.5, label='Final Authority')
        
        # Plot Safety and Performance authority if available
        if hasattr(simulation, 'authority_safety_history') and len(simulation.authority_safety_history) >= min_length:
            auth_safety = np.array(simulation.authority_safety_history[:min_length])
            plt.semilogy(time, auth_safety, 'r--', linewidth=1.5, alpha=0.7, label='Safety Authority')
        if hasattr(simulation, 'authority_performance_history') and len(simulation.authority_performance_history) >= min_length:
            auth_perf = np.array(simulation.authority_performance_history[:min_length])
            plt.semilogy(time, auth_perf, 'g--', linewidth=1.5, alpha=0.7, label='Performance Authority')
        
        plt.axhline(y=1.0, color='k', linestyle=':', linewidth=1.5, alpha=0.7, label='Equal Authority')
        plt.fill_between(time, authority, 1.0, where=(authority >= 1.0), 
                       color='blue', alpha=0.2, label='System Dominant')
        plt.fill_between(time, authority, 1.0, where=(authority < 1.0), 
                       color='green', alpha=0.2, label='Human Dominant')
        plt.xlabel('Time [s]', fontsize=10)
        plt.ylabel('Authority Ratio lambda(t)', fontsize=10)
        plt.title('Authority Ratio (Log Scale)', fontsize=11, fontweight='bold')
        plt.grid(True, alpha=0.3, which='both')
        plt.legend(loc='best', fontsize=8)
        
        # Plot 3: Longitudinal Safety Field Force
        plt.subplot(3, 3, 3)
        if len(safety_forces) > 0:
            plt.plot(time[:len(safety_forces)], safety_forces, 'm-', linewidth=2.5, label='Safety Force')
            plt.axhline(y=300, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Warning Level')
            plt.axhline(y=600, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='Danger Level')
            plt.fill_between(time[:len(safety_forces)], 0, 300, color='orange', alpha=0.15)
            plt.fill_between(time[:len(safety_forces)], 300, 600, color='red', alpha=0.15)
            plt.xlabel('Time [s]', fontsize=10)
            plt.ylabel('Field Force Magnitude [N]', fontsize=10)
            plt.title('Longitudinal Safety Field Force', fontsize=11, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best', fontsize=9)
        else:
            plt.text(0.5, 0.5, 'No safety force data', ha='center', va='center', transform=plt.gca().transAxes)
        
        # ========== Row 2 ==========
        # Plot 4: Control Authority Distribution
        plt.subplot(3, 3, 4)
        plt.fill_between(time, 0, alpha, color='blue', alpha=0.6, label='System Authority')
        plt.fill_between(time, alpha, 1.0, color='green', alpha=0.6, label='Human Authority')
        plt.axhline(y=0.5, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Equal (50%)')
        plt.xlabel('Time [s]', fontsize=10)
        plt.ylabel('Authority Fraction', fontsize=10)
        plt.title('Control Authority Distribution', fontsize=11, fontweight='bold')
        plt.ylim([0, 1])
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=9)
        
        # Plot 5: Velocity Profile
        plt.subplot(3, 3, 5)
        if len(velocity_history) > 0:
            target_velocity = simulation.platoon_manager.target_speed * 3.6 if hasattr(simulation.platoon_manager, 'target_speed') else 120.0
            plt.plot(time[:len(velocity_history)], velocity_history, 'b-', linewidth=2.5, label='Human Vehicle Velocity')
            plt.axhline(y=target_velocity, color='k', linestyle='--', linewidth=1.5, alpha=0.7, label='Target Velocity')
            plt.xlabel('Time [s]', fontsize=10)
            plt.ylabel('Velocity [km/h]', fontsize=10)
            plt.title('Velocity Profile Under Nash Control', fontsize=11, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best', fontsize=9)
        else:
            plt.text(0.5, 0.5, 'No velocity data', ha='center', va='center', transform=plt.gca().transAxes)
        
        # Plot 6: Gap Error vs Authority Correlation
        plt.subplot(3, 3, 6)
        if len(gap_errors) > 0:
            auth_for_gap = authority[:len(gap_errors)]
            scatter = plt.scatter(gap_errors, auth_for_gap, c=time[:len(gap_errors)], 
                                cmap='viridis', s=20, alpha=0.6)
            plt.colorbar(scatter, label='Time [s]')
            plt.xlabel('Gap Error [m]', fontsize=10)
            plt.ylabel('Authority Ratio lambda(t)', fontsize=10)
            plt.title('Gap Error vs Authority Correlation', fontsize=11, fontweight='bold')
            plt.yscale('log')
            plt.axhline(y=1.0, color='r', linestyle='--', linewidth=1, alpha=0.5)
            plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No gap data', ha='center', va='center', transform=plt.gca().transAxes)
        
        # ========== Row 3 ==========
        # Plot 7: Nash Equilibrium Cost Functions (placeholder)
        plt.subplot(3, 3, 7)
        plt.text(0.5, 0.5, 'Nash Equilibrium\nCost Functions\n(Future Implementation)', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=11)
        plt.title('Nash Equilibrium Cost Functions', fontsize=11, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Plot 8: Control Effort Analysis
        plt.subplot(3, 3, 8)
        dt = time[1] - time[0] if len(time) > 1 else 0.1
        cumulative_effort = np.cumsum(np.abs(u_shared) * dt)
        plt.plot(time, cumulative_effort, 'orange', linewidth=2.5, label='Cumulative Control Effort')
        plt.xlabel('Time [s]', fontsize=10)
        plt.ylabel('Cumulative ∫|a| dt [m/s]', fontsize=10)
        plt.title('Control Effort Analysis', fontsize=11, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=9)
        
        # Plot 9: Nash Performance Metrics
        plt.subplot(3, 3, 9)
        metrics = []
        values = []
        
        max_lambda = np.max(authority)
        metrics.append('Max lambda')
        values.append(max_lambda)
        
        min_lambda = np.min(authority)
        metrics.append('Min lambda')
        values.append(min_lambda)
        
        if len(safety_forces) > 0:
            max_force = np.max(safety_forces)
            metrics.append('Max Force')
            values.append(max_force)
        
        if len(gap_errors) > 0:
            max_gap_err = np.max(np.abs(gap_errors))
            metrics.append('Max Gap Err')
            values.append(max_gap_err)
        
        colors = ['magenta', 'blue', 'red', 'orange'][:len(metrics)]
        bars = plt.bar(metrics, values, color=colors, alpha=0.7)
        for bar, val in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        plt.ylabel('Metric Value', fontsize=10)
        plt.title('Nash Performance Metrics', fontsize=11, fontweight='bold')
        plt.xticks(rotation=15, ha='right', fontsize=9)
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = scenario_name.replace(" ", "_").replace(":", "")
        filename = f"{safe_name}_nash_analysis_{timestamp}.png"
        
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        
        filepath = os.path.join(RESULTS_DIR, filename)
        
        try:
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            
            abs_path = os.path.abspath(filepath)
            file_size = os.path.getsize(filepath) / 1024
            print(f"✅ Enhanced Nash 9-grid plot saved!")
            print(f"   📄 File: {filename}")
            print(f"   📁 Path: {abs_path}")
            print(f"   💾 Size: {file_size:.1f} KB")
            
            if not HEADLESS_MODE and show_plots:
                plt.show(block=False)
                plt.pause(0.1)
            
            return fig
            
        except Exception as save_error:
            print(f"❌ Could not save Nash plot: {save_error}")
            import traceback
            traceback.print_exc()
            return None
        
    except Exception as e:
        print(f"❌ Failed to create enhanced Nash plots: {e}")
        import traceback
        traceback.print_exc()
        if 'fig' in locals():
            try:
                plt.close(fig)
            except:
                pass
        return None


def create_hierarchical_control_plots(simulation, scenario_name="Simulation", show_plots=True):
    """Create plots showing lower-level controller metrics (throttle, brake, RPM, gear).
    
    Only meaningful when hierarchical control is active (use_hierarchical=True).
    Shows how the lower-level controller translates desired acceleration into
    actuator commands and how the powertrain responds.
    """
    try:
        import matplotlib.pyplot as plt

        # Check if hierarchical data exists
        if (not hasattr(simulation, 'throttle_history') or
                len(simulation.throttle_history) == 0):
            print("⚠️ No hierarchical controller data to plot")
            return None

        # Identify which vehicles have hierarchical data
        hierarchical_indices = []
        for i, vehicle in enumerate(simulation.all_vehicles):
            if vehicle.use_hierarchical_model:
                hierarchical_indices.append((i, vehicle.vehicle_id))

        if not hierarchical_indices:
            print("⚠️ No vehicles using hierarchical control")
            return None

        print(f"[PALETTE] Creating hierarchical control plots for: {scenario_name}")
        print(f"   ⚙️ Hierarchical vehicles: {[vid for _, vid in hierarchical_indices]}")

        times = np.array(simulation.time_history)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # --- Plot 1: Throttle ---
        ax = axes[0, 0]
        for idx, vid in hierarchical_indices:
            throttle_vals = []
            valid_times = []
            for j, th_row in enumerate(simulation.throttle_history):
                if idx < len(th_row) and not np.isnan(th_row[idx]):
                    throttle_vals.append(th_row[idx])
                    valid_times.append(times[j])
            if throttle_vals:
                ax.plot(valid_times, throttle_vals, label=vid, linewidth=1.2)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Throttle [0-1]')
        ax.set_title('Throttle Input')
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(True)

        # --- Plot 2: Brake ---
        ax = axes[0, 1]
        for idx, vid in hierarchical_indices:
            brake_vals = []
            valid_times = []
            for j, br_row in enumerate(simulation.brake_history):
                if idx < len(br_row) and not np.isnan(br_row[idx]):
                    brake_vals.append(br_row[idx])
                    valid_times.append(times[j])
            if brake_vals:
                ax.plot(valid_times, brake_vals, label=vid, linewidth=1.2)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Brake [0-1]')
        ax.set_title('Brake Input')
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(True)

        # --- Plot 3: Engine RPM ---
        ax = axes[1, 0]
        for idx, vid in hierarchical_indices:
            rpm_vals = []
            valid_times = []
            for j, rpm_row in enumerate(simulation.rpm_history):
                if idx < len(rpm_row) and not np.isnan(rpm_row[idx]):
                    rpm_vals.append(rpm_row[idx])
                    valid_times.append(times[j])
            if rpm_vals:
                ax.plot(valid_times, rpm_vals, label=vid, linewidth=1.2)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Engine RPM')
        ax.set_title('Engine Speed')
        ax.legend()
        ax.grid(True)

        # --- Plot 4: Gear ---
        ax = axes[1, 1]
        for idx, vid in hierarchical_indices:
            gear_vals = []
            valid_times = []
            for j, g_row in enumerate(simulation.gear_history):
                if idx < len(g_row) and not np.isnan(g_row[idx]):
                    gear_vals.append(g_row[idx])
                    valid_times.append(times[j])
            if gear_vals:
                ax.step(valid_times, gear_vals, label=vid, linewidth=1.2, where='post')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Gear')
        ax.set_title('Transmission Gear')
        ax.set_ylim(0.5, 7.5)
        ax.set_yticks(range(1, 7))
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        fig.suptitle(f'{scenario_name} - Hierarchical Controller Metrics', fontsize=14, y=1.01)

        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = scenario_name.replace(" ", "_").replace(":", "")
        filename = f'{safe_name}_hierarchical_ctrl_{timestamp}.png'
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        filepath = os.path.join(RESULTS_DIR, filename)

        try:
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            abs_path = os.path.abspath(filepath)
            file_size = os.path.getsize(filepath) / 1024
            print(f"✅ Hierarchical control plot saved!")
            print(f"   📄 File: {filename}")
            print(f"   📁 Path: {abs_path}")
            print(f"   💾 Size: {file_size:.1f} KB")
        except Exception as save_error:
            print(f"❌ Could not save hierarchical plot: {save_error}")

        if show_plots:
            try:
                if not HEADLESS_MODE:
                    plt.ion()
                    plt.show(block=False)
                    plt.draw()
                    fig.canvas.flush_events()
                    time.sleep(1)
            except Exception:
                pass

        return fig

    except Exception as e:
        print(f"❌ Failed to create hierarchical control plots: {e}")
        import traceback
        traceback.print_exc()
        if 'fig' in locals():
            try:
                plt.close(fig)
            except:
                pass
        return None


__all__ = [
    'create_comprehensive_plots', 
    'create_detailed_scenario_summary',
    'create_nash_analysis_plots',
    'create_hierarchical_control_plots'
]