import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
"""
This script reads simulation data from CSV files and plots various input and output parameters.
It visualizes the steering angle, longitudinal force, lateral position, yaw angle, velocities, and accelerations.
It also plots the vehicle trajectory in the X-Z plane and provides summary statistics.
"""
os.system('cls' if os.name == 'nt' else 'clear')  # Clear console for better readability
def plot_simulation_data_Player():
    """
    Plot all input and output parameters from simulation CSV files
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # Change to the script's directory
    try:
        # Load input data
        input_data = pd.read_csv('Inputs_Player.csv')
        print("Input data loaded successfully")
        print(f"Input columns: {list(input_data.columns)}")
    except FileNotFoundError:
        print("Error: Inputs.csv not found!")
        return
    
    try:
        # Load output data
        output_data = pd.read_csv('Outputs_Player.csv')
        print("Output data loaded successfully")
        print(f"Output columns: {list(output_data.columns)}")
    except FileNotFoundError:
        print("Error: Outputs.csv not found!")
        return
    
    # Create figure for input parameters
    fig_input, axes_input = plt.subplots(2, 1, figsize=(12, 8))
    fig_input.suptitle('Input Parameters vs Time', fontsize=16, fontweight='bold')
    
    # Plot Lambda (steering angle)
    axes_input[0].plot(input_data['Time'], input_data['Lambda'], 'b-', linewidth=2)
    axes_input[0].set_title('Steering Angle (Lambda)')
    axes_input[0].set_xlabel('Time (s)')
    axes_input[0].set_ylabel('Angle (rad)')
    axes_input[0].grid(True, alpha=0.3)
    
    # Plot Fx (longitudinal force)
    axes_input[1].plot(input_data['Time'], input_data['Fx'], 'r-', linewidth=2)
    axes_input[1].set_title('Longitudinal Force (Fx)')
    axes_input[1].set_xlabel('Time (s)')
    axes_input[1].set_ylabel('Force (N)')
    axes_input[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Create figure for output parameters (excluding X and Z positions)
    fig_output, axes_output = plt.subplots(3, 2, figsize=(15, 12))
    fig_output.suptitle('Output Parameters vs Time', fontsize=16, fontweight='bold')
    
    # Plot Y (lateral position)
    axes_output[0,0].plot(output_data['Time'], output_data['Y'], 'g-', linewidth=2)
    axes_output[0,0].set_title('Lateral Position (Y)')
    axes_output[0,0].set_xlabel('Time (s)')
    axes_output[0,0].set_ylabel('Position (m)')
    axes_output[0,0].grid(True, alpha=0.3)
    
    # Plot YDot (lateral velocity)
    axes_output[0,1].plot(output_data['Time'], output_data['YDot'], 'orange', linewidth=2)
    axes_output[0,1].set_title('Lateral Velocity (YDot)')
    axes_output[0,1].set_xlabel('Time (s)')
    axes_output[0,1].set_ylabel('Velocity (m/s)')
    axes_output[0,1].grid(True, alpha=0.3)
    
    # Plot Psi (yaw angle) in degrees
    axes_output[1,0].plot(output_data['Time'], np.degrees(output_data['Psi']), 'purple', linewidth=2)
    axes_output[1,0].set_title('Yaw Angle (Psi)')
    axes_output[1,0].set_xlabel('Time (s)')
    axes_output[1,0].set_ylabel('Angle (degrees)')
    axes_output[1,0].grid(True, alpha=0.3)
    
    # Plot PsiDot (yaw rate) in degrees/s
    axes_output[1,1].plot(output_data['Time'], np.degrees(output_data['PsiDot']), 'brown', linewidth=2)
    axes_output[1,1].set_title('Yaw Rate (PsiDot)')
    axes_output[1,1].set_xlabel('Time (s)')
    axes_output[1,1].set_ylabel('Rate (degrees/s)')
    axes_output[1,1].grid(True, alpha=0.3)
    
    # Plot Vx (longitudinal velocity) in km/h
    axes_output[2,0].plot(output_data['Time'], output_data['Vx'] * 3.6, 'cyan', linewidth=2)
    axes_output[2,0].set_title('Longitudinal Velocity (Vx)')
    axes_output[2,0].set_xlabel('Time (s)')
    axes_output[2,0].set_ylabel('Speed (km/h)')
    axes_output[2,0].grid(True, alpha=0.3)
    
    # Plot Ax (longitudinal acceleration)
    axes_output[2,1].plot(output_data['Time'], output_data['Ax'], 'magenta', linewidth=2)
    axes_output[2,1].set_title('Longitudinal Acceleration (Ax)')
    axes_output[2,1].set_xlabel('Time (s)')
    axes_output[2,1].set_ylabel('Acceleration (m/s²)')
    axes_output[2,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Create trajectory plot (X-Z movement in space)
    if 'PositionX' in output_data.columns and 'PositionZ' in output_data.columns:
        fig_trajectory, ax_trajectory = plt.subplots(1, 1, figsize=(12, 10))
        
        # Highlight region where 3 < X 
        ax_trajectory.axvspan(3, 8, color="#5F5F5F", alpha=0.5)
        ax_trajectory.axvspan(-8, -5, color="#5F5F5F", alpha=0.5)
        
        # Plot trajectory
        trajectory = ax_trajectory.plot(output_data['PositionX'], output_data['PositionZ'], 
                                      'b-', linewidth=3, label='Vehicle Trajectory')
        
        # Mark start and end points
        ax_trajectory.plot(output_data['PositionX'].iloc[0], output_data['PositionZ'].iloc[0], 
                          'go', markersize=10, label='Start')
        ax_trajectory.plot(output_data['PositionX'].iloc[-1], output_data['PositionZ'].iloc[-1], 
                          'ro', markersize=10, label='End')
        
        # Add arrows to show direction
        n_arrows = 10
        arrow_indices = np.linspace(0, len(output_data)-2, n_arrows, dtype=int)
        for i in arrow_indices:
            dx = output_data['PositionX'].iloc[i+1] - output_data['PositionX'].iloc[i]
            dz = output_data['PositionZ'].iloc[i+1] - output_data['PositionZ'].iloc[i]
            ax_trajectory.arrow(output_data['PositionX'].iloc[i], output_data['PositionZ'].iloc[i], 
                              dx, dz, head_width=0.3, head_length=0.3, 
                              fc='red', ec='red', alpha=0.7)
        
        # Set X limits
        ax_trajectory.set_xlim([-8, 8])

        # Draw lane center lines at x = -4, -2, 0, 2 (gray, dashed)
        for x in [-4, -2, 0, 2]:
            ax_trajectory.axvline(x, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='Lane Center' if x == 0 else None)

        # Draw margin lines at every odd number between -8 and 8 (black, solid)
        for x in range(-5, 4, 2):  # -7, -5, ..., 7
            ax_trajectory.axvline(x, color='black', linestyle='-', linewidth=1, alpha=0.5, label='Margin' if x == -7 else None)

        ax_trajectory.set_xlabel('X Position (m)', fontsize=12)
        ax_trajectory.set_ylabel('Z Position (m)', fontsize=12)
        ax_trajectory.set_title('Vehicle Trajectory in X-Z Plane', fontsize=14, fontweight='bold')
        ax_trajectory.grid(True, alpha=0.3)
        ax_trajectory.legend()
        #ax_trajectory.axis('equal')  # Equal aspect ratio for realistic trajectory
        
        plt.tight_layout()
        plt.show()
        
    else:
        print("Warning: PositionX and PositionZ columns not found in output data")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SIMULATION SUMMARY STATISTICS")
    print("="*50)
    
    print(f"\nInput Parameters:")
    print(f"  Time range: {input_data['Time'].min():.2f} - {input_data['Time'].max():.2f} seconds")
    print(f"  Lambda range: {input_data['Lambda'].min():.4f} - {input_data['Lambda'].max():.4f} rad")
    print(f"  Lambda range: {np.degrees(input_data['Lambda'].min()):.2f} - {np.degrees(input_data['Lambda'].max()):.2f} degrees")
    print(f"  Fx range: {input_data['Fx'].min():.1f} - {input_data['Fx'].max():.1f} N")
    
    print(f"\nOutput Parameters:")
    print(f"  Max lateral displacement: {max(abs(output_data['Y'])):.3f} m")
    print(f"  Max yaw angle: {np.degrees(max(abs(output_data['Psi']))):.2f} degrees")
    print(f"  Max speed: {output_data['Vx'].max() * 3.6:.2f} km/h")
    print(f"  Max acceleration: {output_data['Ax'].max():.2f} m/s²")
    print(f"  Min acceleration: {output_data['Ax'].min():.2f} m/s²")
    
    if 'PositionX' in output_data.columns and 'PositionZ' in output_data.columns:
        total_distance = 0
        for i in range(1, len(output_data)):
            dx = output_data['PositionX'].iloc[i] - output_data['PositionX'].iloc[i-1]
            dz = output_data['PositionZ'].iloc[i] - output_data['PositionZ'].iloc[i-1]
            total_distance += np.sqrt(dx**2 + dz**2)
        
        print(f"  Total distance traveled: {total_distance:.2f} m")
        print(f"  Final position: X={output_data['PositionX'].iloc[-1]:.2f}m, Z={output_data['PositionZ'].iloc[-1]:.2f}m")

def plot_simulation_data_Platoon():
    """
    Plot all input and output parameters from simulation CSV files
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # Change to the script's directory
    try:
        # Load input data
        input_data0 = pd.read_csv('Inputs_0.csv')
        input_data1 = pd.read_csv('Inputs_1.csv')
        input_data2 = pd.read_csv('Inputs_2.csv')
        print("Input data loaded successfully")
        #print(f"Input columns: {list(input_data.columns)}")
    except FileNotFoundError:
        print("Error: Inputs.csv not found!")
        return
    
    try:
        # Load output data
        output_data0 = pd.read_csv('Outputs_0.csv')
        output_data1 = pd.read_csv('Outputs_1.csv')
        output_data2 = pd.read_csv('Outputs_2.csv')
        platoon_data = pd.read_csv('platoon_data.csv')
        print("Output data loaded successfully")
        #print(f"Output columns: {list(output_data.columns)}")
    except FileNotFoundError:
        print("Error: Outputs.csv not found!")
        return
    
    # Create figure for input parameters
    fig_input, axes_input = plt.subplots(2, 1, figsize=(12, 8))
    fig_input.suptitle('Input Parameters vs Time', fontsize=16, fontweight='bold')
    
    # Plot Lambda (steering angle)
    axes_input[0].plot(input_data0['Time'], input_data0['Lambda'], 'r-', linewidth=2)
    axes_input[0].plot(input_data1['Time'], input_data1['Lambda'], 'g--', linewidth=2)
    axes_input[0].plot(input_data2['Time'], input_data2['Lambda'], 'b-*', linewidth=2)
    axes_input[0].set_title('Steering Angle (Lambda)')
    axes_input[0].set_xlabel('Time (s)')
    axes_input[0].set_ylabel('Angle (rad)')
    axes_input[0].grid(True, alpha=0.3)
    axes_input[0].legend(['Vehicle Laeder', 'Vehicle follower 1', 'Vehicle follower 2'], loc='upper right')
    
    # Plot Fx (longitudinal force)
    axes_input[1].plot(input_data0['Time'], input_data0['Fx'], 'r-', linewidth=2)
    axes_input[1].plot(input_data1['Time'], input_data1['Fx'], 'g--', linewidth=2)
    axes_input[1].plot(input_data2['Time'], input_data2['Fx'], 'b-*', linewidth=2)
    axes_input[1].set_title('Longitudinal Force (Fx)')
    axes_input[1].set_xlabel('Time (s)')
    axes_input[1].set_ylabel('Force (N)')
    axes_input[1].grid(True, alpha=0.3)
    axes_input[1].legend(['Vehicle Laeder', 'Vehicle follower 1', 'Vehicle follower 2'], loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    # Create figure for output parameters (excluding X and Z positions)
    fig_output, axes_output = plt.subplots(3, 2, figsize=(15, 12))
    fig_output.suptitle('Output Parameters vs Time', fontsize=16, fontweight='bold')
    
   # Plot 1: Vehicle Positions (PositionZ - longitudinal)
    axes_output[0,0].plot(output_data0['Time'], output_data0['PositionZ'], 'orange', linewidth=2)
    axes_output[0,0].plot(output_data1['Time'], output_data1['PositionZ'], 'purple', linewidth=2)
    axes_output[0,0].plot(output_data2['Time'], output_data2['PositionZ'], 'brown', linewidth=2)
    axes_output[0,0].set_title('Vehicle Longitudinal Positions (Z)')
    axes_output[0,0].set_xlabel('Time (s)')
    axes_output[0,0].set_ylabel('Position Z (m)')
    axes_output[0,0].grid(True, alpha=0.3)
    axes_output[0,0].legend(['Vehicle Laeder', 'Vehicle follower 1', 'Vehicle follower 2'], loc='upper right')
    
   # Plot 2: Vehicle Velocities (Vx) in km/h
    axes_output[0,1].plot(output_data0['Time'], output_data0['Vx'] * 3.6, 'orange', linewidth=2)
    axes_output[0,1].plot(output_data1['Time'], output_data1['Vx'] * 3.6, 'purple', linewidth=2)
    axes_output[0,1].plot(output_data2['Time'], output_data2['Vx'] * 3.6, 'brown', linewidth=2)
    axes_output[0,1].plot(platoon_data['Time'], platoon_data['TargetVelocity'] * 3.6, 'black', linewidth=2, linestyle='--')
    axes_output[0,1].set_title('Longitudinal Velocity (Vx)')
    axes_output[0,1].set_xlabel('Time (s)')
    axes_output[0,1].set_ylabel('Speed (km/h)')
    axes_output[0,1].grid(True, alpha=0.3)
    axes_output[0,1].legend(['Vehicle Laeder', 'Vehicle follower 1', 'Vehicle follower 2'], loc='upper right')
    
    # Plot 3: Vehicle Accelerations (Ax)
    axes_output[1,0].plot(output_data0['Time'], output_data0['Ax'], 'orange', linewidth=2)
    axes_output[1,0].plot(output_data1['Time'], output_data1['Ax'], 'purple', linewidth=2)
    axes_output[1,0].plot(output_data2['Time'], output_data2['Ax'], 'brown', linewidth=2)
    axes_output[1,0].set_title('Longitudinal Acceleration (Ax)')
    axes_output[1,0].set_xlabel('Time (s)')
    axes_output[1,0].set_ylabel('Acceleration (m/s²)')
    axes_output[1,0].grid(True, alpha=0.3)
    axes_output[1,0].legend(['Vehicle Laeder', 'Vehicle follower 1', 'Vehicle follower 2'], loc='upper right')
    
    # Plot 4: Inter-vehicle gaps
    gap_1_2 = output_data0['PositionZ'] - output_data1['PositionZ']  # Gap between leader and follower 1
    gap_2_3 = output_data1['PositionZ'] - output_data2['PositionZ']  # Gap between follower 1 and 2
    axes_output[1,1].plot(output_data0['Time'], gap_1_2, 'b', linewidth=2, label='Actual Gap 1-2')
    axes_output[1,1].plot(output_data1['Time'], gap_2_3, 'r', linewidth=2, label='Actual Gap 2-3')
    axes_output[1,1].plot(output_data2['Time'], platoon_data['actual_gap1'], 'orange', linewidth=2, label='Actual Gap 1-2 (Platoon Data)')
    axes_output[1,1].plot(output_data2['Time'], platoon_data['actual_gap2'], 'purple', linewidth=2, label='Actual Gap 2-3 (Platoon Data)')
    axes_output[1,1].plot(output_data0['Time'], platoon_data['des_gap1'], 'b--', linewidth=2, label='Desired Gap 1-2')
    axes_output[1,1].plot(output_data1['Time'], platoon_data['des_gap2'], 'r--', linewidth=2, label='Desired Gap 2-3')
    axes_output[1,1].set_title('Inter-Vehicle Gaps')
    axes_output[1,1].set_xlabel('Time (s)')
    axes_output[1,1].set_ylabel('Gap (m)')
    axes_output[1,1].grid(True, alpha=0.3)
    axes_output[1,1].legend(loc='upper right')
    
    
    # Plot 5: Gap errors
    gap_error1 = np.array(platoon_data['actual_gap1']) - np.array(platoon_data['des_gap1'])
    gap_error2 = np.array(platoon_data['actual_gap2']) - np.array(platoon_data['des_gap2'])
    axes_output[2,0].plot(output_data1['Time'], gap_error1, 'purple', linewidth=2,label=f'Error {1}-{2}')
    axes_output[2,0].plot(output_data2['Time'], gap_error2, 'brown', linewidth=2, label=f'Error {2}-{3}')
    axes_output[2,0].set_title('Longitudinal Acceleration (Ax)')
    axes_output[2,0].set_xlabel('Time (s)')
    axes_output[2,0].set_ylabel('Gap Error [m]')
    axes_output[2,0].grid(True, alpha=0.3)
    axes_output[2,0].legend(loc='upper right')
    
    # Plot 6: Platoon performance metrics
    vel_diff1 = np.array(output_data0['Vx']) - np.array(output_data1['Vx'])
    vel_diff2 = np.array(output_data1['Vx']) - np.array(output_data2['Vx'])
    axes_output[2,1].plot(output_data0['Time'], vel_diff1, 'orange', linewidth=2, label='Velocity Diff 1-2')
    axes_output[2,1].plot(output_data1['Time'], vel_diff2, 'purple', linewidth=2, label='Velocity Diff 2-3')
    axes_output[2,1].set_title('String Stability Analysis')
    axes_output[2,1].set_xlabel('Time (s)')
    axes_output[2,1].set_ylabel('Velocity Difference (m/s)')
    axes_output[2,1].grid(True, alpha=0.3)
    axes_output[2,1].legend(loc='upper right') 
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("PLATOON SIMULATION ANALYSIS RESULTS")
    print("="*60)
    
    print(f"Simulation Duration: {output_data0['Time'].iloc[-1]:.2f} seconds")
    print(f"Number of Data Points: {len(output_data0)}")
    print(f"Sampling Rate: {1/(output_data0['Time'].iloc[1] - output_data0['Time'].iloc[0]):.1f} Hz")
    
    print(f"\nVEHICLE FINAL STATES:")
    print(f"{'Vehicle':<15} {'Position Z (m)':<15} {'Velocity (km/h)':<15} {'Acceleration (m/s²)':<15}")
    print("-" * 60)
    
    vehicles_data = [output_data0, output_data1, output_data2]
    labels = ['Vehicle Leader', 'Vehicle Follower 1', 'Vehicle Follower 2']
    for i, (data, label) in enumerate(zip(vehicles_data, labels)):
        final_pos = data['PositionZ'].iloc[-1]
        final_vel = data['Vx'].iloc[-1] * 3.6
        final_acc = data['Ax'].iloc[-1]
        print(f"{label:<15} {final_pos:<15.2f} {final_vel:<15.2f} {final_acc:<15.2f}")
    
    print(f"\nGAP ANALYSIS:")
    final_gap_1_2 = output_data0['PositionZ'].iloc[-1] - output_data1['PositionZ'].iloc[-1]
    final_gap_2_3 = output_data1['PositionZ'].iloc[-1] - output_data2['PositionZ'].iloc[-1]
    print(f"Final Gap 1-2: {final_gap_1_2:.2f} m")
    print(f"Final Gap 2-3: {final_gap_2_3:.2f} m")
    
    # Gap statistics
    avg_gap_1_2 = np.mean(gap_1_2)
    avg_gap_2_3 = np.mean(gap_2_3)
    std_gap_1_2 = np.std(gap_1_2)
    std_gap_2_3 = np.std(gap_2_3)
    print(f"Average Gap 1-2: {avg_gap_1_2:.2f} ± {std_gap_1_2:.2f} m")
    print(f"Average Gap 2-3: {avg_gap_2_3:.2f} ± {std_gap_2_3:.2f} m")
    
    if platoon_data is not None:
        print(f"\nCONTROL PERFORMANCE:")
        gap_error1 = platoon_data['actual_gap1'] - platoon_data['des_gap1']
        gap_error2 = platoon_data['actual_gap2'] - platoon_data['des_gap2']
        gap_error1_act =platoon_data['actual_gap1']-(output_data0['PositionZ'] - output_data1['PositionZ'])
        gap_error2_act =platoon_data['actual_gap2']-(output_data1['PositionZ'] - output_data2['PositionZ'])
        print(f"Average Gap Error 1-2: {np.mean(gap_error1_act):.2f} m")
        print(f"Average Gap Error 2-3: {np.mean(gap_error2_act):.2f} m")
        rms_error1 = np.sqrt(np.mean(gap_error1**2))
        rms_error2 = np.sqrt(np.mean(gap_error2**2))
        mae_error1 = np.mean(np.abs(gap_error1))
        mae_error2 = np.mean(np.abs(gap_error2))
        
        print(f"Gap 1-2 RMS Error: {rms_error1:.3f} m")
        print(f"Gap 2-3 RMS Error: {rms_error2:.3f} m")
        print(f"Gap 1-2 MAE: {mae_error1:.3f} m")
        print(f"Gap 2-3 MAE: {mae_error2:.3f} m")
        
        # Acceleration analysis
        print(f"\nACCELERATION ANALYSIS:")
        max_acc_leader = np.max(output_data0['Ax'])
        max_acc_follower1 = np.max(output_data1['Ax'])
        max_acc_follower2 = np.max(output_data2['Ax'])
        min_acc_leader = np.min(output_data0['Ax'])
        min_acc_follower1 = np.min(output_data1['Ax'])
        min_acc_follower2 = np.min(output_data2['Ax'])
        
        print(f"Maximum Accelerations:")
        print(f"  Leader: {max_acc_leader:.2f} m/s²")
        print(f"  Follower 1: {max_acc_follower1:.2f} m/s²")
        print(f"  Follower 2: {max_acc_follower2:.2f} m/s²")
        print(f"  Platoon Max: {max(max_acc_leader, max_acc_follower1, max_acc_follower2):.2f} m/s²")
        
        print(f"Maximum Decelerations:")
        print(f"  Leader: {min_acc_leader:.2f} m/s²")
        print(f"  Follower 1: {min_acc_follower1:.2f} m/s²")
        print(f"  Follower 2: {min_acc_follower2:.2f} m/s²")
        print(f"  Platoon Min: {min(min_acc_leader, min_acc_follower1, min_acc_follower2):.2f} m/s²")
        
        # Check if accelerations exceed limits (updated to 11.0 m/s²)
        acc_limit = 2.5  # Updated max acceleration from PlatoonManager
        exceeded_positive = max(max_acc_leader, max_acc_follower1, max_acc_follower2) > acc_limit
        exceeded_negative = min(min_acc_leader, min_acc_follower1, min_acc_follower2) < -acc_limit
        
        if exceeded_positive:
            print(f"⚠️  WARNING: Maximum acceleration ({max(max_acc_leader, max_acc_follower1, max_acc_follower2):.2f} m/s²) exceeds limit ({acc_limit} m/s²)")
        else:
            print(f"✅ Acceleration within limits (max: {max(max_acc_leader, max_acc_follower1, max_acc_follower2):.2f} m/s² ≤ {acc_limit} m/s²)")
            
        if exceeded_negative:
            print(f"⚠️  WARNING: Maximum deceleration ({min(min_acc_leader, min_acc_follower1, min_acc_follower2):.2f} m/s²) exceeds limit ({-acc_limit} m/s²)")
        else:
            print(f"✅ Deceleration within limits (min: {min(min_acc_leader, min_acc_follower1, min_acc_follower2):.2f} m/s² ≥ {-acc_limit} m/s²)")
        
        print(f"\nTarget Velocity: {platoon_data['TargetVelocity'].iloc[-1] * 3.6:.2f} km/h")
        
        # Time to reach target velocity analysis
        target_velocity_ms = platoon_data['TargetVelocity'].iloc[-1]
        target_threshold = target_velocity_ms * 0.95  # 95% of target velocity
        
        print(f"\nTIME TO TARGET VELOCITY ANALYSIS:")
        vehicles_data = [output_data0, output_data1, output_data2]
        labels = ['Leader', 'Follower 1', 'Follower 2']
        
        for i, (data, label) in enumerate(zip(vehicles_data, labels)):
            # Find when vehicle reaches 95% of target velocity
            target_reached_indices = np.where(data['Vx'] >= target_threshold)[0]
            if len(target_reached_indices) > 0:
                time_to_target = data['Time'].iloc[target_reached_indices[0]]
                print(f"  {label}: Reached 95% target velocity at {time_to_target:.2f}s")
            else:
                final_velocity_kmh = data['Vx'].iloc[-1] * 3.6
                print(f"  {label}: Did not reach target (final: {final_velocity_kmh:.1f} km/h)")
        
        # Calculate how well the high acceleration helps
        print(f"\nACCELERATION EFFECTIVENESS:")
        print(f"  Max acceleration limit: {acc_limit} m/s²")
        print(f"  Theoretical 0-50km/h time: {(platoon_data['TargetVelocity'].iloc[-1] / acc_limit):.2f}s")
        avg_acc_leader = np.mean(output_data0['Ax'][output_data0['Ax'] > 0])  # Average positive acceleration
        if not np.isnan(avg_acc_leader):
            print(f"  Leader average acceleration: {avg_acc_leader:.2f} m/s²")
            print(f"  Acceleration utilization: {(avg_acc_leader/acc_limit)*100:.1f}%")
    
    print("="*60)

if __name__ == "__main__":
    #plot_simulation_data_Player()
    plot_simulation_data_Platoon()

