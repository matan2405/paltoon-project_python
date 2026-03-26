"""Visualization module for lateral simulation outputs and Nash diagnostics."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
from datetime import datetime
from typing import Dict
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR, LANE_WIDTH, PLATOON_LANE_Y, HUMAN_INITIAL_LANE_Y


def create_comprehensive_plots(sim_data: Dict, scenario_name: str = "Simulation", 
                               mobil_approval_time: float = None) -> plt.Figure:
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle(f"Lateral Control Analysis: {scenario_name}", fontsize=14, fontweight='bold')
    
    time = sim_data['time']
    
    # Helper function to add MOBIL approval line to any axis
    def add_mobil_line(ax):
        if mobil_approval_time is not None:
            ax.axvline(x=mobil_approval_time, color='cyan', linestyle='--', 
                      linewidth=2, alpha=0.8, label='MOBIL OK')
    
    ax1 = axes[0, 0]
    ax1.plot(time, sim_data['human_y'], 'b-', linewidth=2, label='y')
    ax1.axhline(y=PLATOON_LANE_Y, color='g', linestyle='--', label='Target')
    ax1.axhline(y=HUMAN_INITIAL_LANE_Y, color='r', linestyle=':', alpha=0.5, label='Initial')
    add_mobil_line(ax1)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('y [m]')
    ax1.set_title('Lateral Position')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.plot(time, np.degrees(sim_data['human_psi']), 'b-', linewidth=2)
    ax2.axhline(y=0, color='g', linestyle='--')
    add_mobil_line(ax2)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('psi [deg]')
    ax2.set_title('Heading Angle')
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[0, 2]
    ax3.plot(time, sim_data['human_y_dot_road'], 'b-', linewidth=2, label='road-frame (Ẏ_world)')
    ax3.plot(time, sim_data['human_y_dot'], 'b--', linewidth=1, alpha=0.5, label='vy_body')
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    add_mobil_line(ax3)
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('dy/dt [m/s]')
    ax3.set_title('Lateral Velocity (road-frame)')
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 0]
    ax4.plot(time, np.degrees(sim_data['delta_system']), 'r-', alpha=0.7, label='System')
    ax4.plot(time, np.degrees(sim_data['delta_human']), 'b-', alpha=0.7, label='Human')
    ax4.plot(time, np.degrees(sim_data['delta_shared']), 'g-', linewidth=2, label='Shared')
    add_mobil_line(ax4)
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('delta [deg]')
    ax4.set_title('Steering Inputs')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    ax5 = axes[1, 1]
    lambda_k   = np.array(sim_data['authority_ratio'])
    lam_safety = np.array(sim_data.get('lambda_safety', lambda_k))
    lam_perf   = np.array(sim_data.get('lambda_performance', np.ones_like(lambda_k) * 0.1))

    ax5.fill_between(time, 1.0, np.maximum(lambda_k, 1.0),
                     color='steelblue', alpha=0.25, label='System Dominant')
    ax5.fill_between(time, np.minimum(lambda_k, 1.0), 1.0,
                     color='lightgreen', alpha=0.35, label='Human Dominant')
    ax5.semilogy(time, lam_safety, 'r--',          linewidth=1.5, alpha=0.85, label='Safety Authority')
    ax5.semilogy(time, lam_perf,   'g--',          linewidth=1.5, alpha=0.85, label='Performance Authority')
    ax5.semilogy(time, lambda_k,   color='magenta', linewidth=2,               label='Final Authority')
    ax5.axhline(y=1.0, color='k', linestyle=':', linewidth=1.5, label='Equal Authority')
    add_mobil_line(ax5)
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('Authority Ratio λ (log scale)')
    ax5.set_title('Authority Ratio (Log Scale)')
    ax5.legend(fontsize=7, loc='upper right')
    ax5.grid(True, alpha=0.3)
    
    ax6 = axes[1, 2]
    ax6.plot(time, sim_data['field_force'], 'r-', linewidth=2)
    ax6.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    add_mobil_line(ax6)
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('Force [N]')
    ax6.set_title('Safety Field Force')
    ax6.grid(True, alpha=0.3)
    
    ax7 = axes[2, 0]
    ax7.plot(time, sim_data['y_error'], 'b-', linewidth=2)
    ax7.axhline(y=0, color='g', linestyle='--')
    add_mobil_line(ax7)
    ax7.set_xlabel('Time [s]')
    ax7.set_ylabel('Error [m]')
    ax7.set_title('Lateral Error')
    ax7.grid(True, alpha=0.3)
    
    ax8 = axes[2, 1]
    ax8.plot(time, sim_data['human_ay'], 'b-', linewidth=2)
    ax8.axhline(y=2.5, color='r', linestyle='--', alpha=0.5)
    ax8.axhline(y=-2.5, color='r', linestyle='--', alpha=0.5)
    add_mobil_line(ax8)
    ax8.set_xlabel('Time [s]')
    ax8.set_ylabel('ay [m/s^2]')
    ax8.set_title('Lateral Acceleration')
    ax8.grid(True, alpha=0.3)
    
    ax9 = axes[2, 2]
    phases = sim_data['phase']
    phase_names = ['CRUISE', 'GAP_SEARCH', 'LANE_CHANGE', 'LANE_KEEPING', 'FOLLOWING']
    phase_colors = ['gray', 'yellow', 'orange', 'lightgreen', 'green']
    
    for i in range(len(time) - 1):
        phase = phases[i]
        idx = phase_names.index(phase) if phase in phase_names else 0
        ax9.axvspan(time[i], time[i+1], alpha=0.5, color=phase_colors[idx])
    
    # Add MOBIL approval marker to phase plot
    if mobil_approval_time is not None:
        ax9.axvline(x=mobil_approval_time, color='cyan', linestyle='--', 
                   linewidth=2, alpha=0.8)
        ax9.text(mobil_approval_time + 0.5, 0.5, 'MOBIL\nOK', fontsize=8,
                color='cyan', fontweight='bold', va='center')
    
    legend_elements = [Patch(facecolor=c, alpha=0.5, label=n) for c, n in zip(phase_colors, phase_names)]
    if mobil_approval_time is not None:
        legend_elements.append(plt.Line2D([0], [0], color='cyan', linestyle='--', linewidth=2, label='MOBIL OK'))
    ax9.legend(handles=legend_elements, loc='upper right', fontsize=7)
    ax9.set_xlabel('Time [s]')
    ax9.set_title('Control Phase')
    ax9.set_yticks([])
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{scenario_name.replace(' ', '_')}_results_{timestamp}.png"
    filepath = os.path.join(RESULTS_DIR, filename)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"📊 Saved: {filepath}")
    
    return fig


def create_trajectory_plot(sim_data: Dict, scenario_name: str = "Simulation") -> plt.Figure:
    """Road-relative trajectory: lateral position y vs longitudinal distance x (body frame).

    Uses body-frame coordinates (state.x, state.y) which measure displacement from
    the road center-line — directly relevant for lane-change analysis.
    World-frame (X_world, Y_world) grows without bound when ψ ≠ 0 and is not
    suitable for road-relative visualization.
    """
    human_x = np.array(sim_data['human_x'])
    human_y = np.array(sim_data['human_y'])
    phases = sim_data['phase']

    fig, ax = plt.subplots(figsize=(16, 6))

    phase_colors = {'CRUISE': 'gray', 'GAP_SEARCH': 'yellow', 'LANE_CHANGE': 'orange',
                    'LANE_KEEPING': 'lightgreen', 'FOLLOWING': 'green'}

    for i in range(len(human_x) - 1):
        color = phase_colors.get(phases[i], 'gray')
        ax.plot(human_x[i:i+2], human_y[i:i+2], color=color, linewidth=3)

    ax.axhline(y=PLATOON_LANE_Y, color='g', linestyle='--', alpha=0.6, label='Target Lane')
    ax.axhline(y=HUMAN_INITIAL_LANE_Y, color='r', linestyle=':', alpha=0.4, label='Initial Lane')
    ax.plot(human_x[0], human_y[0], 'go', markersize=15, label='Start')
    ax.plot(human_x[-1], human_y[-1], 'r*', markersize=20, label='End')

    ax.set_xlabel('Longitudinal distance x [m]')
    ax.set_ylabel('Lateral position y [m]')
    ax.set_title(f"Road-Relative Trajectory (Body Frame): {scenario_name}")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Zoom Y-axis to show lane change detail
    y_min = min(np.min(human_y), PLATOON_LANE_Y - 1) - 0.5
    y_max = max(np.max(human_y), HUMAN_INITIAL_LANE_Y + 1) + 0.5
    ax.set_ylim(y_min, y_max)

    lc_indices = np.where((human_y > PLATOON_LANE_Y + 0.1) & (human_y < HUMAN_INITIAL_LANE_Y - 0.1))[0]
    if len(lc_indices) > 10:
        dx = human_x[lc_indices[-1]] - human_x[lc_indices[0]]
        dy = human_y[lc_indices[0]] - human_y[lc_indices[-1]]
        angle_deg = np.degrees(np.arctan2(dy, dx))
        ax.text(0.02, 0.98, f"Lane change angle: {angle_deg:.1f}°\nDistance: {dx:.0f}m",
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{scenario_name.replace(' ', '_')}_trajectory_{timestamp}.png"
    filepath = os.path.join(RESULTS_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Saved: {filepath}")

    return fig


def create_nash_analysis_plots(sim_data: Dict, scenario_name: str = "Simulation",
                               mobil_approval_time: float = None) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Nash Analysis: {scenario_name}", fontsize=14, fontweight='bold')
    
    time = sim_data['time']
    
    # Helper function to add MOBIL approval line
    def add_mobil_line(ax):
        if mobil_approval_time is not None:
            ax.axvline(x=mobil_approval_time, color='cyan', linestyle='--', 
                      linewidth=2, alpha=0.8, label='MOBIL OK')
    
    ax1 = axes[0, 0]
    ax1.plot(time, np.degrees(sim_data['delta_system']), 'r-', alpha=0.7, label='System')
    ax1.plot(time, np.degrees(sim_data['delta_human']), 'b-', alpha=0.7, label='Human')
    ax1.plot(time, np.degrees(sim_data['delta_shared']), 'g-', linewidth=2, label='Shared')
    add_mobil_line(ax1)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('delta [deg]')
    ax1.set_title('Control Inputs')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    lambda_final = np.array(sim_data['authority_ratio'])
    lambda_safety = np.array(sim_data.get('lambda_safety', lambda_final))
    lambda_perf   = np.array(sim_data.get('lambda_performance', np.ones_like(lambda_final) * 0.1))

    # Shaded regions
    ax2.fill_between(time, 1.0, np.maximum(lambda_final, 1.0),
                     color='steelblue', alpha=0.25, label='System Dominant')
    ax2.fill_between(time, np.minimum(lambda_final, 1.0), 1.0,
                     color='lightgreen', alpha=0.35, label='Human Dominant')

    # Component lines
    ax2.semilogy(time, lambda_safety, 'r--',      linewidth=1.5, alpha=0.85, label='Safety Authority')
    ax2.semilogy(time, lambda_perf,   'g--',      linewidth=1.5, alpha=0.85, label='Performance Authority')
    ax2.semilogy(time, lambda_final,  color='magenta', linewidth=2, label='Final Authority')
    ax2.axhline(y=1.0, color='k', linestyle=':', linewidth=1.5, label='Equal Authority')

    add_mobil_line(ax2)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Authority Ratio λ (log scale)')
    ax2.set_title('Authority Ratio (Log Scale)')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    lambda_k = sim_data['authority_ratio']
    alpha = lambda_k / (1 + lambda_k)
    ax3.fill_between(time, 0, alpha * 100, color='red', alpha=0.5, label='System')
    ax3.fill_between(time, alpha * 100, 100, color='blue', alpha=0.5, label='Human')
    add_mobil_line(ax3)
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Authority [%]')
    ax3.set_title('Authority Split')
    ax3.set_ylim([0, 100])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    ax4.axis('off')
    y_err = sim_data['y_error']
    psi_err = np.degrees(sim_data['psi_error'])
    ay = sim_data['human_ay']
    
    # Add MOBIL info to summary
    mobil_info = f"\n    MOBIL approved: {mobil_approval_time:.1f}s" if mobil_approval_time else ""
    
    summary = f"""
    Performance Summary
    ===================
    
    Max |y_error|:   {np.max(np.abs(y_err)):.3f} m
    Final |y_error|: {np.abs(y_err[-1]):.3f} m
    Max |psi_error|: {np.max(np.abs(psi_err)):.2f} deg
    Max |ay|:        {np.max(np.abs(ay)):.2f} m/s^2
    Avg lambda:      {np.mean(lambda_k):.2f}{mobil_info}
    """
    ax4.text(0.1, 0.9, summary, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{scenario_name.replace(' ', '_')}_nash_{timestamp}.png"
    filepath = os.path.join(RESULTS_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"📊 Saved: {filepath}")
    
    return fig



def print_simulation_summary(sim_data: Dict, scenario_name: str = "Simulation"):
    time = sim_data['time']
    y_err = sim_data['y_error']
    psi_err = np.degrees(sim_data['psi_error'])
    ay = sim_data['human_ay']
    lambda_k = sim_data['authority_ratio']
    
    print(f"\n{'='*60}")
    print(f"📊 Summary: {scenario_name}")
    print(f"{'='*60}")
    print(f"  Max |y_error|:   {np.max(np.abs(y_err)):.3f} m")
    print(f"  Final |y_error|: {np.abs(y_err[-1]):.3f} m")
    print(f"  Max |psi_error|: {np.max(np.abs(psi_err)):.2f} deg")
    print(f"  Max |ay|:        {np.max(np.abs(ay)):.2f} m/s^2")
    print(f"  Avg lambda:      {np.mean(lambda_k):.2f}")
    print(f"  Total time:      {time[-1]:.1f} s")
    print(f"{'='*60}\n")


__all__ = ['create_comprehensive_plots', 'create_trajectory_plot',
           'create_nash_analysis_plots', 'print_simulation_summary']
