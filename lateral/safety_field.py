#!/usr/bin/env python3
"""
File: safety_field.py
Description: Contains the enhanced driving safety field model for risk assessment.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class SafetyFieldParams:
    """Safety field parameters from Li et al. 2019"""
    mo: float = 1864.0     # Obstacle mass [kg]
    Ri: float = 1.0        # Road influence factor
    ts: float = 2.0        # Time margin [s]
    s: float = 3.0         # Safety radius [m] - INCREASED for better avoidance
    eta_q: float = 200.0   # Road boundary scaling - INCREASED
    B: float = 8.0         # Road width [m]
    k1: float = 0.01       # Kinetic field scaling - INCREASED
    k2: float = 0.01       # Force field scaling - INCREASED
    DRi: float = 0.8       # Driving risk factor - INCREASED

class EnhancedSafetyField:
    """Improved safety field with proper obstacle avoidance"""
    
    def __init__(self, params: SafetyFieldParams):
        self.params = params
        
    def compute_field_force(self, host_pos: np.ndarray, host_vel: float, 
                           obstacles: List[Dict]) -> np.ndarray:
        """Compute total safety field force with PROPER obstacle avoidance"""
        
        total_force = np.zeros(2)
        
        # Process each obstacle
        for i, obs in enumerate(obstacles):
            if 'pos' not in obs or len(obs['pos']) < 2:
                continue
                
            obs_pos = np.array(obs['pos'][:2])
            obs_vel = obs.get('vel', [0.0])[0] if 'vel' in obs else 0.0
            
            # Distance vector
            r_vec = host_pos[:2] - obs_pos
            r_mag = np.linalg.norm(r_vec)
            
            if r_mag < 0.1:  # Too close
                continue
                
            # IMPROVED: Elliptic potential with proper scaling
            vel_diff = abs(host_vel - obs_vel)
            a = max(vel_diff * self.params.ts, self.params.s)  # Longitudinal
            b = self.params.s  # Lateral
            
            # Elliptic distance
            r_elliptic = np.sqrt((r_vec[0]/a)**2 + (r_vec[1]/b)**2)
            
            if r_elliptic < 0.01:
                continue
                
            # FIXED: Enhanced obstacle potential with direction preference
            Mo = self.params.mo
            Ro = self.params.Ri
            
            # Potential magnitude with distance decay - REDUCED for smoother forces
            potential_mag = (Mo * Ro) / (r_elliptic + 2.0)**2  # INCREASED stabilizer from 1.0 to 2.0
            
            # Direction vector (normalized)
            r_normalized = r_vec / r_mag
            
            # ENHANCED: Add lateral bias to encourage side avoidance
            lateral_bias = np.array([0, np.sign(r_vec[1]) * 0.5])  # Encourage side movement
            
            # Total obstacle force
            obs_force = potential_mag * (r_normalized + lateral_bias)
            
            # Distance weighting (closer = stronger) - FIXED for convoy scenario
            # Use exponential decay for better long-range influence - REDUCED for challenging scenario
            weight = np.exp(-r_mag / 15.0)  # Faster decay - influence up to ~45m for more reasonable forces
            obs_force *= weight
            
            total_force += obs_force
            
        # Road boundary forces
        y_pos = host_pos[1]
        road_width = self.params.B / 2
        
        # Road boundary potential - Equation (16) corrected
        if abs(y_pos) > road_width * 0.7:  # Close to boundary
            boundary_dist = road_width - abs(y_pos)
            if boundary_dist > 0.1:
                road_force_mag = self.params.eta_q * np.log(boundary_dist)
                road_direction = -np.sign(y_pos)  # Push toward center
                total_force[1] += road_force_mag * road_direction
                
        # ENHANCED: Velocity-dependent scaling - Equation (21)
        theta_i = np.arctan2(total_force[1], total_force[0])
        vel_factor = np.exp(self.params.k2 * host_vel * np.cos(theta_i))
        behavior_factor = 1 + self.params.DRi
        
        final_force = total_force * vel_factor * behavior_factor
        
        # Force magnitude limiting for stability - REDUCED for smoother behavior
        force_mag = np.linalg.norm(final_force)
        if force_mag > 500.0:  # REDUCED from 1000 to 500 for smoother response
            final_force = final_force * (500.0 / force_mag)
            
        return final_force
