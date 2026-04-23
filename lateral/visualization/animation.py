"""
Animation module for Lateral Control V2.3
Creates animated visualizations of lane change maneuvers with platoon.
Based on the proven longitudinal animation structure.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import os
import time
import gc
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import HEADLESS_MODE, RESULTS_DIR, LANE_WIDTH, VEHICLE_COLORS
from vehicle.components import VehicleParameters

_vp = VehicleParameters()


def create_lane_change_animation(sim_data: dict, title: str = "Lane Change Animation"):
    """
    Create animated visualization of lane change maneuver.
    Based on the proven longitudinal animation structure.
    
    Args:
        sim_data: Dictionary containing simulation results
        title: Title for the animation
    
    Returns:
        Animation object or None if failed
    """
    try:
        # Check if we're in headless mode
        global HEADLESS_MODE
        
        if HEADLESS_MODE:
            print("🎬 Creating animation in headless mode (will save as GIF file)")
        else:
            print("🎬 Creating interactive animation")
        
        # Force garbage collection before creating animation
        gc.collect()
        
        # Create figure with vertical layout - all graphs stacked vertically
        plt.ioff()  # Turn off interactive mode temporarily
        
        # Create a separate figure for animation to avoid conflicts with static plots
        animation_fig_num = plt.gcf().number + 100
        fig = plt.figure(num=animation_fig_num, figsize=(14, 12))
        
        # Create vertical layout: 4 graphs stacked vertically
        gs = fig.add_gridspec(4, 1, height_ratios=[1.2, 0.8, 0.8, 0.25], hspace=0.3)
        
        ax1 = fig.add_subplot(gs[0])  # Top: Bird's eye view (lane change)
        ax2 = fig.add_subplot(gs[1])  # Second: Lateral position y
        ax3 = fig.add_subplot(gs[2])  # Third: Steering inputs
        ax4 = fig.add_subplot(gs[3])  # Bottom: Status bar (narrow)
        
        # Extract data
        time_history = np.array(sim_data['time'])
        human_x = np.array(sim_data['human_x'])
        human_y = np.array(sim_data['human_y'])
        human_psi = np.array(sim_data['human_psi'])
        phases = sim_data['phase']
        
        # Get steering data if available
        delta_system = np.array(sim_data.get('delta_system', np.zeros_like(time_history)))
        delta_human = np.array(sim_data.get('delta_human', np.zeros_like(time_history)))
        delta_shared = np.array(sim_data.get('delta_shared', np.zeros_like(time_history)))
        
        # Get platoon positions if available
        platoon_positions = sim_data.get('platoon_positions', None)
        
        # Phase colors
        phase_colors = {
            'CRUISE': 'gray',
            'GAP_SEARCH': 'yellow', 
            'LANE_CHANGE': 'orange',
            'LANE_KEEPING': 'lightgreen',
            'FOLLOWING': 'darkgreen'
        }
        
        def animate(frame):
            # Clear axes safely
            try:
                for ax in [ax1, ax2, ax3, ax4]:
                    ax.clear()
                    ax.set_rasterized(True)  # Improve performance
            except Exception as e:
                print(f"Warning: Could not clear axes: {e}")
                return []
            
            try:
                current_time = time_history[frame]
            except IndexError:
                return []
            
            current_x = human_x[frame]
            current_y = human_y[frame]
            current_psi = human_psi[frame]
            current_phase = phases[frame] if frame < len(phases) else phases[-1]
            
            # ===== TOP: Bird's Eye View =====
            ax1.set_title(f'{title} | Time: {current_time:.1f}s | Phase: {current_phase}')
            ax1.set_xlabel('Longitudinal Position x [m]')
            ax1.set_ylabel('Lateral Position y [m]')
            ax1.grid(True, alpha=0.3)
            
            # Set view window following the vehicle
            ax1.set_xlim(current_x - 50, current_x + 100)
            ax1.set_ylim(-2, 6)
            
            # Draw road lanes
            ax1.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Target Lane (y=0)')
            ax1.axhline(y=LANE_WIDTH, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Initial Lane')
            ax1.fill_between([current_x-100, current_x+150], -0.5, 0.5, alpha=0.1, color='green')
            ax1.fill_between([current_x-100, current_x+150], LANE_WIDTH-0.5, LANE_WIDTH+0.5, alpha=0.1, color='blue')
            
            # Draw platoon vehicles if available
            if platoon_positions is not None and frame < len(platoon_positions):
                platoon_pos = platoon_positions[frame]
                for i, pos in enumerate(platoon_pos):
                    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                        px, py = pos[0], pos[1]
                    else:
                        continue
                    
                    # Platoon vehicle rectangle
                    vehicle_patch = plt.Rectangle((px - _vp.length/2, py - _vp.width/2),
                                                 _vp.length, _vp.width,
                                                 color='blue', alpha=0.7)
                    ax1.add_patch(vehicle_patch)
                    ax1.text(px, py, f'P{i+1}', ha='center', va='center', 
                            fontsize=8, fontweight='bold', color='white')
            
            # Draw human vehicle
            color = phase_colors.get(current_phase, 'red')
            
            # Vehicle rectangle (rotated by psi)
            vehicle_length = _vp.length
            vehicle_width = _vp.width
            
            # Simple rectangle without rotation for stability
            human_patch = plt.Rectangle((current_x - vehicle_length/2, current_y - vehicle_width/2),
                                        vehicle_length, vehicle_width,
                                        color=color, alpha=0.9)
            ax1.add_patch(human_patch)
            ax1.text(current_x, current_y, 'H', ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')
            
            # Draw trajectory trail
            trail_start = max(0, frame - 500)
            ax1.plot(human_x[trail_start:frame+1], human_y[trail_start:frame+1],
                    'b-', alpha=0.4, linewidth=1.5, label='Trajectory')
            
            # Show heading angle
            arrow_length = 8
            ax1.arrow(current_x, current_y, 
                     arrow_length * np.cos(current_psi), arrow_length * np.sin(current_psi),
                     head_width=0.5, head_length=0.3, fc='red', ec='red', alpha=0.7)
            
            ax1.legend(loc='upper right', fontsize=8)
            
            # ===== SECOND: Lateral Position =====
            ax2.set_xlim(0, time_history[-1])
            ax2.set_ylim(-0.5, LANE_WIDTH + 0.5)
            ax2.set_xlabel('Time [s]')
            ax2.set_ylabel('Lateral Position y [m]')
            ax2.set_title('Lateral Position Over Time')
            ax2.grid(True, alpha=0.3)
            
            # Plot y history up to current frame
            ax2.plot(time_history[:frame+1], human_y[:frame+1], 'b-', linewidth=2, label='y position')
            ax2.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Target')
            ax2.axhline(y=LANE_WIDTH, color='gray', linestyle=':', alpha=0.3, label='Initial')
            
            # Mark current point
            ax2.plot(current_time, current_y, 'ro', markersize=8)
            ax2.text(current_time + 1, current_y, f'{current_y:.2f}m', fontsize=9)
            
            ax2.legend(loc='upper right', fontsize=8)
            
            # ===== THIRD: Steering Inputs =====
            ax3.set_xlim(0, time_history[-1])
            
            # Find y limits for steering
            all_steering = np.concatenate([delta_system[:frame+1], delta_human[:frame+1], delta_shared[:frame+1]])
            if len(all_steering) > 0 and np.max(np.abs(all_steering)) > 0:
                y_max = max(0.15, np.max(np.abs(all_steering)) * 1.2)
            else:
                y_max = 0.15
            ax3.set_ylim(-y_max, y_max)
            
            ax3.set_xlabel('Time [s]')
            ax3.set_ylabel('Steering Angle [rad]')
            ax3.set_title('Steering Inputs')
            ax3.grid(True, alpha=0.3)
            
            # Plot steering histories
            ax3.plot(time_history[:frame+1], delta_system[:frame+1], 'r-', 
                    linewidth=1.5, alpha=0.7, label='System')
            ax3.plot(time_history[:frame+1], delta_human[:frame+1], 'b-',
                    linewidth=1.5, alpha=0.7, label='Human')
            ax3.plot(time_history[:frame+1], delta_shared[:frame+1], 'g-',
                    linewidth=2, label='Shared')
            
            # Mark current points
            ax3.plot(current_time, delta_shared[frame], 'go', markersize=6)
            
            ax3.legend(loc='upper right', fontsize=8, ncol=3)
            
            # ===== FOURTH: Status Bar =====
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            
            # Create status text
            y_error = current_y - 0  # Target is y=0
            psi_deg = np.degrees(current_psi)
            
            status_text = (f"Time: {current_time:.1f}s  |  "
                          f"Phase: {current_phase}  |  "
                          f"y: {current_y:.2f}m  |  "
                          f"ψ: {psi_deg:.2f}°  |  "
                          f"Error: {y_error:.2f}m")
            
            # Color based on phase
            bg_color = phase_colors.get(current_phase, 'lightgray')
            ax4.text(0.5, 0.5, status_text, transform=ax4.transAxes,
                    ha='center', va='center', fontfamily='monospace', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=bg_color, alpha=0.8))
            
            return []
        
        # Create animation with robust error handling
        try:
            frame_step = max(1, len(time_history) // 150)
            frames_to_use = range(0, len(time_history), frame_step)
            
            print(f"🎬 Creating animation with {len(list(frames_to_use))} frames...")
            
            # Recreate frames_to_use since range is consumed
            frames_to_use = range(0, len(time_history), frame_step)
            
            anim = FuncAnimation(fig, animate, frames=frames_to_use,
                               interval=200, repeat=False, blit=False)
            
            plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08)
            
            # Display animation
            try:
                if HEADLESS_MODE:
                    print("🎬 Animation in headless mode (will save as GIF file)")
                else:
                    print("🎬 Displaying interactive animation...")
                    plt.ion()
                    plt.show(block=False)
                    plt.draw()
                    fig.canvas.flush_events()
                    print("✅ Interactive animation displayed successfully!")
                    time.sleep(2)
            except Exception as show_error:
                print(f"⚠️ Could not display animation: {show_error}")
            
            # Ask user if they want to save the animation
            try:
                if not sys.stdin.isatty():
                    save_choice = 'n'
                else:
                    save_choice = input("\n💾 Do you want to save the animation as GIF? (y/n): ").strip().lower()
                
                if save_choice in ['y', 'yes', '1']:
                    anim_filename = os.path.join(RESULTS_DIR, f"{title.replace(' ', '_')}_animation.gif")
                    print(f"💾 Saving animation as {anim_filename}...")
                    anim.save(anim_filename, writer='pillow', fps=8, dpi=80)
                    abs_path = os.path.abspath(anim_filename)
                    print(f"✅ Animation saved successfully!")
                    print(f"📁 Full path: {abs_path}")
                else:
                    print("ℹ️ Animation not saved (user choice)")
                    
            except KeyboardInterrupt:
                print("\n⚠️ Animation save cancelled by user")
            except Exception as save_error:
                print(f"⚠️ Could not save animation: {save_error}")
            
            print("✅ Animation created successfully")
            return anim
            
        except Exception as anim_error:
            print(f"❌ Animation creation failed: {anim_error}")
            return None
        finally:
            # Safe cleanup
            try:
                if 'fig' in locals():
                    plt.figure(fig.number)
                    plt.close(fig)
            except Exception as cleanup_error:
                print(f"⚠️ Animation cleanup warning: {cleanup_error}")
    
    except ImportError:
        print("⚠️ Animation requires matplotlib.animation to be installed.")
        return None
    except Exception as e:
        print(f"❌ Animation setup failed: {e}")
        return None


__all__ = ['create_lane_change_animation']
