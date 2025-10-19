"""
Static plotting module for creating comprehensive analysis plots.
Contains functions for creating detailed simulation analysis visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import HEADLESS_MODE, RESULTS_DIR


def create_comprehensive_plots(simulation, scenario_name="Simulation"):
    """Create comprehensive 9-plot analysis for any simulation"""
    try:
        import matplotlib.pyplot as plt
        
        # Create figure with proper cleanup
        fig = plt.figure(figsize=(16, 12))
        
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
        plt.ylabel('Acceleration [m/s²]')
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
        filename = os.path.join(RESULTS_DIR, f'{scenario_name.replace(" ", "_").replace(":", "")}_results.png')
        try:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            abs_path = os.path.abspath(filename)
            print(f"📁 Results saved to {filename}")
            print(f"📁 Full path: {abs_path}")
            
            # Keep a global reference to prevent garbage collection
            global _static_plots_figure
            _static_plots_figure = fig
        except Exception as save_error:
            print(f"⚠️ Could not save plot: {save_error}")
        
        # Display plots
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
                import time
                time.sleep(2)
        except Exception as show_error:
            print(f"⚠️ Could not display plots interactively: {show_error}")
        
        return fig if 'fig' in locals() else None
        
    except Exception as plot_error:
        print(f"Could not create comprehensive plots: {plot_error}")
        return None


def create_detailed_scenario_summary(simulation, scenario_name, execution_time):
    """Create comprehensive unified summary for each scenario"""
    print(f"\n{'=' * 70}")
    print(f"✅ Simulation {scenario_name} completed!")
    print(f"⏱️ Execution time: {execution_time:.2f} seconds")
    print(f"⚡ Speed: {simulation.time_history[-1]/execution_time:.1f}x real time")
    print(f"{'=' * 70}")
    
    print(f"\n📊 === {scenario_name} Platoon Control Simulation Results ===")
    
    # Simulation performance summary  
    print(f"\n🎯 Simulation Performance:")
    print(f"   ⏱️ Total simulation time: {simulation.time_history[-1]:.1f} seconds")
    print(f"   ⚡ Execution time: {execution_time:.2f} seconds")
    print(f"   🚀 Speed factor: {simulation.time_history[-1]/execution_time:.1f}x real time")
    
    # platoon composition
    print(f"\n🚛 platoon Composition:")
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


__all__ = [
    'create_comprehensive_plots', 
    'create_detailed_scenario_summary'
]