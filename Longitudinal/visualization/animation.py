"""
Animation module for creating animated visualizations of platoon simulations.
Contains functions for creating real-time animated plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import time
import gc
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import HEADLESS_MODE, RESULTS_DIR, VEHICLE_COLORS


def create_platoon_animation(simulation, title: str = "Platoon Simulation Animation"):
    """Create animated visualization of platoon simulation - with improved stability"""
    try:
        # Import here to ensure proper backend is set
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        
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
        animation_fig_num = plt.gcf().number + 100  # Use a high figure number to avoid conflicts
        fig = plt.figure(num=animation_fig_num, figsize=(14, 12))  # Specific figure number for animation
        
        # Create vertical layout: 4 graphs stacked vertically
        gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 0.25], hspace=0.3)
        
        ax1 = fig.add_subplot(gs[0])  # Top: Spatial view
        ax2 = fig.add_subplot(gs[1])  # Second: Velocities  
        ax3 = fig.add_subplot(gs[2])  # Third: Inter-vehicle gaps
        ax4 = fig.add_subplot(gs[3])  # Bottom: Status bar (narrow)
        
        time_history = simulation.time_history
        position_history = simulation.position_history
        velocity_history = simulation.velocity_history
        gap_history = simulation.gap_history
        desired_gap_history = simulation.desired_gap_history
        
        # Vehicle colors - different color for each vehicle
        colors = VEHICLE_COLORS
        
        def animate(frame):
            # Clear axes safely with better error handling
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
            
            # Top subplot: Spatial representation of vehicles
            ax1.set_title(f'{title} | Time: {current_time:.1f}s')
            ax1.set_xlabel('Longitudinal Position [m]')
            ax1.set_ylabel('Lane')
            ax1.grid(True, alpha=0.3)
            
            # Get current positions and find plot range
            current_positions = position_history[frame]
            all_x_positions = [pos[0] for pos in current_positions]
            min_x, max_x = min(all_x_positions), max(all_x_positions)
            ax1.set_xlim(min_x - 20, max_x + 50)
            ax1.set_ylim(-3, 3)
            
            # Draw road lanes
            ax1.axhline(y=0, color='yellow', linestyle='-', linewidth=3, alpha=0.8, label='Lane Center')
            ax1.axhline(y=1.5, color='black', linestyle='-', linewidth=2, alpha=0.5)
            ax1.axhline(y=-1.5, color='black', linestyle='-', linewidth=2, alpha=0.5)
            ax1.axhline(y=-2, color='yellow', linestyle='-', linewidth=3, alpha=0.8, label='Left Lane')
            
            # Draw vehicles
            for i, (vehicle, pos) in enumerate(zip(simulation.all_vehicles, current_positions)):
                x_pos, y_pos = pos
                color = colors[i % len(colors)]
                
                # Vehicle rectangle
                vehicle_patch = plt.Rectangle((x_pos - 2, y_pos - 0.4), 4, 0.8, 
                                            color=color, alpha=0.8, 
                                            label=vehicle.vehicle_id if frame == 0 else "")
                ax1.add_patch(vehicle_patch)
                
                # Vehicle label inside the rectangle
                ax1.text(x_pos, y_pos, vehicle.vehicle_id, 
                        ha='center', va='center', fontsize=8, fontweight='bold', color='white')
                
                # Show vehicle speed
                speed_kmh = velocity_history[frame][i]
                ax1.text(x_pos, y_pos + 0.8, f'{speed_kmh:.0f} km/h', 
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            # Show inter-vehicle gaps based on gap_history (proper platoon gaps)
            gap_displayed = False
            
            # First, show platoon inter-vehicle gaps from gap_history
            if gap_history and len(gap_history) > frame and len(gap_history[frame]) > 0:
                current_gaps = gap_history[frame]
                
                # Find ALL vehicles that are part of the platoon (including human after joining)
                platoon_positions = []
                
                for i, vehicle in enumerate(simulation.all_vehicles):
                    if i < len(current_positions):
                        # Include vehicle if it's a platoon vehicle OR if it's human and joined platoon
                        if (vehicle.vehicle_id.startswith("Platoon") or 
                            (vehicle.vehicle_id == "Human" and 
                             hasattr(vehicle, 'joined_platoon') and vehicle.joined_platoon)):
                            platoon_positions.append((current_positions[i], i, vehicle))
                
                # Sort ALL platoon vehicles (including human) by x-position (front to back)
                platoon_positions.sort(key=lambda x: x[0][0], reverse=True)
                
                # Display gaps between consecutive vehicles in the platoon
                for gap_idx, gap_value in enumerate(current_gaps):
                    if gap_idx < len(platoon_positions) - 1:
                        leader_pos, leader_idx, leader_vehicle = platoon_positions[gap_idx]
                        follower_pos, follower_idx, follower_vehicle = platoon_positions[gap_idx + 1]
                        
                        # Position text at midpoint between vehicles
                        mid_x = (leader_pos[0] + follower_pos[0]) / 2
                        mid_y = min(leader_pos[1], follower_pos[1]) - 1.5
                        
                        # Use different colors based on vehicle types
                        if leader_vehicle.vehicle_id == "Human" or follower_vehicle.vehicle_id == "Human":
                            # Gap involving human vehicle - use orange color
                            color = 'darkorange'
                            bgcolor = 'lightyellow'
                        else:
                            # Pure platoon gap - use blue color
                            color = 'darkblue'
                            bgcolor = 'lightcyan'
                        
                        # Display the gap value from gap_history
                        ax1.text(mid_x, mid_y, f'{gap_value:.1f}m', 
                                ha='center', va='top', fontsize=8, color=color, fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor=bgcolor, alpha=0.9))
                
                gap_displayed = True
            
            ax1.legend(loc='upper right')
            
            # Second subplot: Vehicle velocities over time
            ax2.set_xlim(0, time_history[-1])
            ax2.set_ylim(0, max([max(v) for v in velocity_history]) * 1.1)
            ax2.set_xlabel('Time [s]')
            ax2.set_ylabel('Velocity [km/h]')
            ax2.set_title('Vehicle Velocities')
            ax2.grid(True, alpha=0.3)
            
            # Plot velocity histories up to current frame
            current_range = slice(0, frame + 1)
            time_current = time_history[current_range]
            
            for i, vehicle in enumerate(simulation.all_vehicles):
                color = colors[i % len(colors)]
                velocities = [vel[i] for vel in velocity_history[current_range]]
                ax2.plot(time_current, velocities, color=color, linewidth=2, 
                        label=vehicle.vehicle_id)
                
                # Mark current point
                if len(velocities) > 0:
                    ax2.plot(current_time, velocities[-1], 'o', color=color, markersize=6)
            
            # Add target velocity line for platoon
            if hasattr(simulation.platoon_manager, 'target_velocity'):
                target_kmh = simulation.platoon_manager.target_velocity * 3.6
                ax2.axhline(y=target_kmh, color='gray', linestyle='--', alpha=0.7, 
                           label=f'Target: {target_kmh:.0f} km/h')
            
            ax2.legend(loc='upper right')
            
            # Third subplot: Inter-vehicle gaps
            ax3.clear()
            ax3.set_title('Inter-vehicle Gaps')
            ax3.set_xlabel('Time [s]')
            ax3.set_ylabel('Gap [m]')
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(0, time_history[-1])
            
            # Plot gaps between platoon vehicles
            if gap_history and len(gap_history) > frame:
                max_gaps = max(len(gap_set) for gap_set in gap_history) if gap_history else 0
                gap_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
                
                for i in range(max_gaps):
                    gaps = []
                    desired_gaps = []
                    valid_times = []
                    
                    for j in range(frame + 1):
                        if (j < len(gap_history) and j < len(desired_gap_history) and
                            i < len(gap_history[j]) and i < len(desired_gap_history[j])):
                            gaps.append(gap_history[j][i])
                            desired_gaps.append(desired_gap_history[j][i])
                            valid_times.append(time_history[j])
                    
                    if gaps and valid_times:
                        color = gap_colors[i % len(gap_colors)]
                        ax3.plot(valid_times, gaps, color=color, linewidth=2, 
                               label=f'Gap {i+1}', alpha=0.8)
                        ax3.plot(valid_times, desired_gaps, color=color, linestyle='--', 
                               linewidth=2, alpha=0.6, label=f'Desired {i+1}')
                        if gaps:
                            ax3.plot(current_time, gaps[-1], 'o', color=color, markersize=6)
            
            # Set appropriate y-limits for better visualization
            if gap_history and len(gap_history) > 0:
                all_gaps = [gap for gap_set in gap_history[:frame+1] for gap in gap_set if isinstance(gap, (int, float))]
                if all_gaps:
                    y_min = max(0, min(all_gaps) - 5)
                    y_max = max(all_gaps) + 10
                    ax3.set_ylim(y_min, y_max)
                else:
                    ax3.set_ylim(0, 60)
            else:
                ax3.set_ylim(0, 60)
            
            ax3.legend(loc='upper right', ncol=2)
            
            # Fourth subplot: Status bar
            ax4.clear()
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            
            # Create status text
            status_text = f"Time: {current_time:.1f}s  |  Platoon Vehicles: {len(simulation.platoon_vehicles)}  |  Target Speed: {simulation.platoon_manager.target_velocity * 3.6:.0f} km/h"
            
            ax4.text(0.5, 0.5, status_text, transform=ax4.transAxes, 
                    ha='center', va='center', fontfamily='monospace', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

        # Create animation with robust error handling
        try:
            frame_step = max(1, len(time_history) // 150)
            frames_to_use = range(0, len(time_history), frame_step)
            
            print(f"🎬 Creating animation with {len(frames_to_use)} frames...")
            
            anim = FuncAnimation(fig, animate, frames=frames_to_use, 
                               interval=200, repeat=False, blit=False)
            
            plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08)
            
            # Display animation
            try:
                if HEADLESS_MODE:
                    print("🎬 Creating animation in headless mode (will save as GIF file)")
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
                import sys
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


__all__ = ['create_platoon_animation']