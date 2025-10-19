#!/usr/bin/env python3
"""
File: visualization.py
Description: Contains functions for static and animated plotting of simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
import os
from datetime import datetime

def _create_output_directory():
    """Create output directory for saving plots"""
    output_dir = "convoy_simulation_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def _sanitize_filename(title: str) -> str:
    """Sanitize title for use as filename"""
    # Remove or replace problematic characters
    sanitized = title.replace(' ', '_').replace(':', '').replace('-', '_')
    sanitized = ''.join(c for c in sanitized if c.isalnum() or c in ('_', '.'))
    return sanitized

def _save_plot(fig, title: str, plot_type: str = "static"):
    """Save plot to file with timestamp"""
    try:
        output_dir = _create_output_directory()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{_sanitize_filename(title)}_{plot_type}_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        
        fig.savefig(filepath, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"📊 Plot saved: {filepath}")
        
    except Exception as e:
        print(f"⚠️ Failed to save plot: {e}")

def visualize_results(results: Dict, title: str = "Convoy Merging Results"):
    """Create comprehensive visualization with automatic saving"""
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        time = results['time']
        states = results['states']
        x_positions = results['x_positions']
        
        # 1. Trajectory Plot
        ax = axes[0, 0]
        ax.plot(x_positions, states[:, 2], 'b-', linewidth=3, label='Vehicle Path')
        ax.plot(x_positions[0], states[0, 2], 'go', markersize=10, label='Start')
        ax.plot(x_positions[-1], states[-1, 2], 'ro', markersize=10, label='End')
        
        # Plot planned trajectories (only the first one with a label)
        if 'controller_trajectories' in results and results['controller_trajectories']:
            for i, traj_data in enumerate(results['controller_trajectories']):
                if traj_data.get('trajectory') is not None and traj_data['trajectory'].shape[1] >= 1:
                    x_start = traj_data['x_start']
                    y_traj = traj_data['trajectory'][:, 0]
                    x_traj = x_start + np.arange(len(y_traj)) * 20.0 * 0.1
                    ax.plot(x_traj, y_traj, 'g--', alpha=0.6, linewidth=1.5, label='System Plan' if i == 0 else '')

        if 'human_trajectories' in results and results['human_trajectories']:
            for i, traj_data in enumerate(results['human_trajectories']):
                if traj_data.get('trajectory') is not None and traj_data['trajectory'].shape[1] >= 1:
                    x_start = traj_data['x_start']
                    y_traj = traj_data['trajectory'][:, 0]
                    x_traj = x_start + np.arange(len(y_traj)) * 20.0 * 0.1
                    ax.plot(x_traj, y_traj, 'm--', alpha=0.6, linewidth=1.5, label='Human Intent' if i == 0 else '')
        
        # CORRECTED: Plot initial positions of obstacles with a single label
        if 'obstacles' in results:
            has_obstacle_label = False
            for obs in results['obstacles']:
                if 'pos' in obs:
                    label = ''
                    if not has_obstacle_label:
                        label = 'Obstacle (Start)'
                        has_obstacle_label = True
                    ax.scatter(obs['pos'][0], obs['pos'][1], c='red', s=200, marker='s', label=label, alpha=0.6)

        ax.axhline(y=4.0, color='k', linestyle='--', alpha=0.5, label='Road Boundary')
        ax.axhline(y=-4.0, color='k', linestyle='--', alpha=0.5)
        ax.axhline(y=0.0, color='y', linestyle=':', alpha=0.7, label='Target Lane')
        
        ax.set_xlabel('Longitudinal Position [m]')
        ax.set_ylabel('Lateral Position [m]')
        ax.set_title('Vehicle Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Control inputs
        ax = axes[0, 1]
        controls = results['controls']
        ax.plot(time, np.degrees(controls['controller']), 'b-', label='Controller', linewidth=2)
        ax.plot(time, np.degrees(controls['human']), 'g-', label='Human', linewidth=2)
        ax.plot(time, np.degrees(controls['shared']), 'r--', label='Shared', linewidth=2)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Steering Angle [degrees]')
        ax.set_title('Nash Equilibrium Control Inputs')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Authority allocation
        ax = axes[0, 2]
        ax.plot(time, results['authority_ratios'], 'purple', linewidth=3, label='Authority Ratio λ(k)')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Equal Authority')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Authority Ratio λ(k)')
        ax.set_title('Dynamic Authority Allocation')
        ax.set_yscale('log') # Log scale helps visualize changes better
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        
        # 4. Safety field force
        ax = axes[1, 0]
        field_magnitudes = np.linalg.norm(results['field_forces'], axis=1)
        ax.plot(time, field_magnitudes, 'm-', linewidth=2)
        ax.axhline(y=100, color='orange', linestyle='--', alpha=0.7, label='Warning Level')
        ax.axhline(y=300, color='red', linestyle='--', alpha=0.7, label='Danger Level')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Field Force Magnitude [N]')
        ax.set_title('Safety Field Force')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Lateral acceleration
        ax = axes[1, 1]
        ax.plot(time, results['lateral_accelerations'], 'orange', linewidth=2)
        ax.axhline(y=2.5, color='red', linestyle='--', alpha=0.7, label='Comfort Limit')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Lateral Acceleration [m/s²]')
        ax.set_title('Comfort Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Performance summary
        ax = axes[1, 2]
        metrics = results['summary']
        labels = ['Max Lat Dev', 'Max Authority', 'Max Lat Acc', 'Violations']
        values = [metrics['max_lateral_deviation'], metrics['max_authority_ratio'], metrics['max_lateral_acceleration'], metrics['comfort_violations']]
        bars = ax.bar(labels, values, color=['blue', 'purple', 'orange', 'red'])
        ax.bar_label(bars, fmt='%.2f') # Add values on top of bars
        ax.set_ylabel('Values')
        ax.set_title('Performance Summary')
        ax.tick_params(axis='x', rotation=30)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 🔥 Save the plot automatically
        _save_plot(fig, title, "static")
        
        plt.show()
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")

def create_animated_visualization(results: Dict, title: str = "Convoy Merging Animation"):
    """Create animated visualization of convoy merging scenario with automatic saving"""
    try:
        from matplotlib.animation import FuncAnimation
        
        fig, ax = plt.subplots(figsize=(16, 8))
        time = results['time']
        x_positions = results['x_positions']
        states = results['states']
        initial_obstacles = results['obstacles']

        def animate(frame):
            ax.clear()
            
            ego_x = x_positions[frame]
            ax.set_xlim(ego_x - 40, ego_x + 80)
            ax.set_ylim(-6, 6)
            ax.set_title(f'{title} | Time: {time[frame]:.1f}s | Authority λ: {results["authority_ratios"][frame]:.2f}')
            ax.set_xlabel('Longitudinal Position [m]')
            ax.set_ylabel('Lateral Position [m]')
            ax.grid(True, alpha=0.3)
            
            ax.axhline(y=4.0, color='k', linestyle='-'); ax.axhline(y=-4.0, color='k', linestyle='-')
            ax.axhline(y=0, color='gray', linestyle=':', alpha=0.7)

            ax.plot(x_positions[:frame+1], states[:frame+1, 2], 'b--', alpha=0.5)

            ego_y, ego_phi = states[frame, 2], states[frame, 3]
            ego_patch = plt.Rectangle((ego_x - 2.25, ego_y - 1.0), 4.5, 2.0, angle=np.degrees(ego_phi), rotation_point='center', color='blue', label='Ego Vehicle')
            ax.add_patch(ego_patch)

            for i, obs in enumerate(initial_obstacles):
                current_x = obs['pos'][0] + obs.get('vel', [0])[0] * time[frame]
                current_y = obs['pos'][1]
                obs_patch = plt.Rectangle((current_x - 2.25, current_y - 1.0), 4.5, 2.0, color='red', alpha=0.8, label='Obstacle' if i == 0 else "")
                ax.add_patch(obs_patch)

            ax.legend(loc='upper right')

        # To prevent animation from stopping, it needs to be assigned to a variable
        anim = FuncAnimation(fig, animate, frames=len(time), interval=100, repeat=False)
        plt.tight_layout()
        
        # 🔥 Save the animation as GIF
        _save_animation(anim, title)
        
        plt.show()
        return anim

    except ImportError:
        print("⚠️ Animation requires matplotlib.animation to be installed.")
    except Exception as e:
        print(f"❌ Animation failed: {e}")

def _save_animation(anim, title: str):
    """Save animation as GIF file"""
    try:
        output_dir = _create_output_directory()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{_sanitize_filename(title)}_animation_{timestamp}.gif"
        filepath = os.path.join(output_dir, filename)
        
        # Save as GIF (requires pillow or imagemagick)
        anim.save(filepath, writer='pillow', fps=10, dpi=150)
        print(f"🎬 Animation saved: {filepath}")
        
    except ImportError:
        print("⚠️ Animation saving requires 'pillow' package. Install with: pip install pillow")
    except Exception as e:
        print(f"⚠️ Failed to save animation: {e}")

def visualize_integrated_results(results: Dict, title: str = "Integrated Convoy Control Results"):
    """Create comprehensive visualization for integrated lateral and longitudinal control with automatic saving"""
    try:
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        time = results['time']
        states = results['states']
        x_positions = results['x_positions']
        velocities = results['velocities']
        
        # 1. Trajectory Plot (top-left)
        ax = axes[0, 0]
        ax.plot(x_positions, states[:, 2], 'b-', linewidth=3, label='Vehicle Path')
        ax.plot(x_positions[0], states[0, 2], 'go', markersize=10, label='Start')
        ax.plot(x_positions[-1], states[-1, 2], 'ro', markersize=10, label='End')
        
        # Plot obstacles
        if 'obstacles' in results:
            for i, obs in enumerate(results['obstacles']):
                if 'pos' in obs:
                    label = 'Convoy Vehicle' if i == 0 else ''
                    ax.scatter(obs['pos'][0], obs['pos'][1], c='red', s=200, marker='s', 
                             label=label, alpha=0.6)
        
        ax.axhline(y=0.0, color='y', linestyle=':', alpha=0.7, label='Target Lane')
        ax.set_xlabel('Longitudinal Position [m]')
        ax.set_ylabel('Lateral Position [m]')
        ax.set_title('Vehicle Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Velocity Profile (top-center)
        ax = axes[0, 1]
        ax.plot(time, velocities, 'purple', linewidth=3, label='Actual Velocity')
        ax.axhline(y=20.0, color='gray', linestyle='--', alpha=0.7, label='Initial Velocity')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Velocity [m/s]')
        ax.set_title('Longitudinal Velocity Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Authority Allocation Comparison (top-right)
        ax = axes[0, 2]
        ax.plot(time, results['lateral_authority_ratios'], 'blue', linewidth=2, label='Lateral Authority λ_lat')
        ax.plot(time, results['longitudinal_authority_ratios'], 'red', linewidth=2, label='Longitudinal Authority λ_long')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Equal Authority')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Authority Ratio')
        ax.set_title('Dynamic Authority Allocation')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        
        # 4. Lateral Controls (middle-left)
        ax = axes[1, 0]
        lat_controls = results['lateral_controls']
        ax.plot(time, np.degrees(lat_controls['controller']), 'b-', label='System', linewidth=2)
        ax.plot(time, np.degrees(lat_controls['human']), 'g-', label='Human', linewidth=2)
        ax.plot(time, np.degrees(lat_controls['shared']), 'r--', label='Shared', linewidth=2)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Steering Angle [degrees]')
        ax.set_title('Lateral Control Inputs')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Longitudinal Controls (middle-center)
        ax = axes[1, 1]
        long_controls = results['longitudinal_controls']
        ax.plot(time, long_controls['controller'], 'b-', label='System', linewidth=2)
        ax.plot(time, long_controls['human'], 'g-', label='Human', linewidth=2)
        ax.plot(time, long_controls['shared'], 'r--', label='Shared', linewidth=2)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Acceleration [m/s²]')
        ax.set_title('Longitudinal Control Inputs')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Safety Field Forces (middle-right)
        ax = axes[1, 2]
        lat_force_mag = np.linalg.norm(results['lateral_field_forces'], axis=1)
        long_force_mag = np.abs(results['longitudinal_field_forces'])
        ax.plot(time, lat_force_mag, 'blue', linewidth=2, label='Lateral Force')
        ax.plot(time, long_force_mag, 'red', linewidth=2, label='Longitudinal Force')
        ax.axhline(y=100, color='orange', linestyle='--', alpha=0.7, label='Warning Level')
        ax.axhline(y=300, color='red', linestyle='--', alpha=0.7, label='Danger Level')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Field Force Magnitude [N]')
        ax.set_title('Safety Field Forces')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 7. Acceleration Profile (bottom-left)
        ax = axes[2, 0]
        ax.plot(time, results['lateral_accelerations'], 'blue', linewidth=2, label='Lateral Acceleration')
        ax.plot(time, results['longitudinal_accelerations'], 'red', linewidth=2, label='Longitudinal Acceleration')
        ax.axhline(y=2.5, color='orange', linestyle='--', alpha=0.7, label='Lateral Comfort Limit')
        ax.axhline(y=-4.0, color='orange', linestyle='--', alpha=0.7, label='Longitudinal Comfort Limit')
        ax.axhline(y=4.0, color='orange', linestyle='--', alpha=0.7)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Acceleration [m/s²]')
        ax.set_title('Acceleration Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 8. Control Authority Timeline (bottom-center)
        ax = axes[2, 1]
        # Create stacked area plot showing authority distribution
        lat_system_authority = np.array(results['lateral_authority_ratios']) / (1 + np.array(results['lateral_authority_ratios']))
        long_system_authority = np.array(results['longitudinal_authority_ratios']) / (1 + np.array(results['longitudinal_authority_ratios']))
        
        ax.fill_between(time, 0, lat_system_authority, alpha=0.6, color='blue', label='Lateral System Authority')
        ax.fill_between(time, 0, long_system_authority, alpha=0.6, color='red', label='Longitudinal System Authority')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Equal Authority')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('System Authority Fraction')
        ax.set_title('Control Authority Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 9. Performance Summary (bottom-right)
        ax = axes[2, 2]
        metrics = results['summary']
        labels = ['Max Lat Dev', 'Max Lat Auth', 'Max Long Auth', 'Velocity Δ', 'Comfort Viol']
        values = [
            metrics['max_lateral_deviation'], 
            metrics['max_lateral_authority_ratio'], 
            metrics['max_longitudinal_authority_ratio'],
            abs(metrics['velocity_change']),
            metrics['comfort_violations']
        ]
        colors = ['blue', 'purple', 'red', 'green', 'orange']
        bars = ax.bar(labels, values, color=colors)
        ax.bar_label(bars, fmt='%.2f')
        ax.set_ylabel('Values')
        ax.set_title('Performance Summary')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 🔥 Save the integrated plot automatically
        _save_plot(fig, title, "integrated")
        
        plt.show()
        
    except Exception as e:
        print(f"❌ Integrated visualization failed: {e}")

def save_summary_report(all_results: Dict, filename: str = None):
    """Save a comprehensive text summary of all simulation results"""
    try:
        output_dir = _create_output_directory()
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"convoy_simulation_summary_{timestamp}.txt"
        
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("CONVOY SIMULATION COMPREHENSIVE REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for scenario_name, scenario_results in all_results.items():
                f.write(f"\nSCENARIO: {scenario_name}\n")
                f.write("-" * 40 + "\n")
                
                for driver_type, results in scenario_results.items():
                    f.write(f"\nDriver Type: {driver_type}\n")
                    summary = results['summary']
                    
                    f.write(f"  Max Lateral Deviation: {summary['max_lateral_deviation']:.3f} m\n")
                    f.write(f"  Max Authority Ratio: {summary['max_authority_ratio']:.3f}\n")
                    f.write(f"  Max Lateral Acceleration: {summary['max_lateral_acceleration']:.3f} m/s²\n")
                    f.write(f"  Comfort Violations: {summary['comfort_violations']}\n")
                    
                    # Add integrated metrics if available
                    if 'max_longitudinal_authority_ratio' in summary:
                        f.write(f"  Max Longitudinal Authority Ratio: {summary['max_longitudinal_authority_ratio']:.3f}\n")
                        f.write(f"  Velocity Change: {summary['velocity_change']:.3f} m/s\n")
                        f.write(f"  Cooperation Time (Lateral): {summary['cooperation_time_lateral']:.1f} s\n")
                        f.write(f"  Cooperation Time (Longitudinal): {summary['cooperation_time_longitudinal']:.1f} s\n")
                    
                    f.write("\n")
        
        print(f"📝 Summary report saved: {filepath}")
        
    except Exception as e:
        print(f"⚠️ Failed to save summary report: {e}")

