#!/usr/bin/env python3
"""
File: human_model.py
Description: Enhanced human driver model with realistic and independent behavior
"""

import numpy as np
from typing import List, Tuple, Dict, Optional

class EnhancedHumanModel:
    """Enhanced human driver model with independent decision making"""
    
    def __init__(self, Np: int = 15):
        self.Np = Np
        self.dt = 0.1
        self.reaction_time = 0.5
        
        # Driver personality traits (not just aggressiveness)
        self.risk_tolerance = 0.5      # 0=very cautious, 1=risk-taking
        self.cooperation_tendency = 0.7 # 0=selfish, 1=cooperative
        self.comfort_priority = 0.6     # 0=efficiency focused, 1=comfort focused
        self.confidence_level = 0.8     # 0=uncertain, 1=very confident
        
        # Personal preferences (independent of controller)
        self.preferred_lateral_speed = 0.03  # m/s lateral movement preference
        self.personal_safety_margin = 6.0    # meters
        self.lane_keeping_preference = 0.8   # tendency to stay in current lane
        
        print(f"🧠 Enhanced Human Model initialized with independent decision making")
        
    def set_driver_personality(self, driver_type: str):
        """Set personality traits based on driver type"""
        if driver_type == 'conservative':
            self.risk_tolerance = 0.2
            self.cooperation_tendency = 0.9
            self.comfort_priority = 0.9
            self.confidence_level = 0.6
            self.preferred_lateral_speed = 0.02
            self.personal_safety_margin = 8.0
            self.lane_keeping_preference = 0.9
            
        elif driver_type == 'aggressive':
            self.risk_tolerance = 0.8
            self.cooperation_tendency = 0.4
            self.comfort_priority = 0.3
            self.confidence_level = 0.9
            self.preferred_lateral_speed = 0.05
            self.personal_safety_margin = 4.0
            self.lane_keeping_preference = 0.5
            
        else:  # normal
            self.risk_tolerance = 0.5
            self.cooperation_tendency = 0.7
            self.comfort_priority = 0.6
            self.confidence_level = 0.8
            self.preferred_lateral_speed = 0.03
            self.personal_safety_margin = 6.0
            self.lane_keeping_preference = 0.7
            
        print(f"👤 Driver personality set: {driver_type}")
        print(f"   Risk tolerance: {self.risk_tolerance:.2f}")
        print(f"   Cooperation tendency: {self.cooperation_tendency:.2f}")
        
    def estimate_human_trajectory(self, current_pos: Tuple[float, float],
                                 current_state: np.ndarray, human_input: float,
                                 obstacles: List[Dict], target_lane: float = 0.0,
                                 velocity: float = 20.0) -> np.ndarray:
        """Estimate human intended trajectory with independent decision making"""
        
        x_current, y_current = current_pos
        trajectory = np.zeros((self.Np, 2))
        
        # STEP 1: Personal situation assessment (independent of controller)
        personal_threat = self._assess_personal_threat_level(current_pos, obstacles, velocity)
        comfort_level = self._assess_current_comfort(current_state, y_current)
        lane_change_urgency = self._calculate_lane_change_urgency(y_current, target_lane, obstacles)
        
        # STEP 2: Personal decision making (not reactive to controller)
        personal_target = self._calculate_personal_target_lane(y_current, target_lane, personal_threat)
        movement_strategy = self._select_movement_strategy(personal_threat, comfort_level, lane_change_urgency)
        
        # STEP 3: Generate trajectory based on independent logic
        for i in range(self.Np):
            t = i * self.dt
            
            # Base trajectory calculation (independent)
            y_ref, phi_ref = self._calculate_trajectory_point(
                i, t, y_current, personal_target, movement_strategy, 
                current_pos, obstacles, velocity, human_input
            )
            
            trajectory[i, :] = [y_ref, phi_ref]
        
        return trajectory
    
    def _assess_personal_threat_level(self, pos: Tuple[float, float], 
                                    obstacles: List[Dict], velocity: float) -> float:
        """Assess threat level from human driver's perspective"""
        x_pos, y_pos = pos
        threat_level = 0.0
        preview_distance = velocity * 3.0  # Driver's preview distance
        
        for obs in obstacles:
            if 'pos' in obs and len(obs['pos']) >= 2:
                obs_x, obs_y = obs['pos'][0], obs['pos'][1]
                
                # Distance-based threat
                longitudinal_dist = obs_x - x_pos
                lateral_dist = abs(obs_y - y_pos)
                
                if 0 < longitudinal_dist < preview_distance:
                    # Calculate personal threat based on driver's safety margin
                    if lateral_dist < self.personal_safety_margin:
                        threat_intensity = 1.0 - (lateral_dist / self.personal_safety_margin)
                        distance_factor = 1.0 - (longitudinal_dist / preview_distance)
                        threat_level += threat_intensity * distance_factor * (1.0 - self.risk_tolerance)
        
        return min(threat_level, 1.0)
    
    def _assess_current_comfort(self, current_state: np.ndarray, y_current: float) -> float:
        """Assess current comfort level"""
        # Based on lateral velocity and position
        lateral_velocity = abs(current_state[0]) if len(current_state) > 0 else 0.0
        
        # Comfort decreases with high lateral velocity and extreme positions
        velocity_discomfort = min(lateral_velocity / 2.0, 1.0)  # Normalize
        position_discomfort = min(abs(y_current) / 4.0, 1.0)   # Normalize to road width
        
        comfort_level = 1.0 - (velocity_discomfort + position_discomfort) / 2.0
        return max(comfort_level, 0.0)
    
    def _calculate_lane_change_urgency(self, y_current: float, target_lane: float, 
                                     obstacles: List[Dict]) -> float:
        """Calculate urgency of lane change based on personal assessment"""
        lane_difference = abs(target_lane - y_current)
        
        # Base urgency from lane difference
        urgency = lane_difference / 3.5  # Normalize by lane width
        
        # Modify based on obstacles in current path
        obstacle_pressure = 0.0
        for obs in obstacles:
            if 'pos' in obs and len(obs['pos']) >= 2:
                obs_y = obs['pos'][1]
                if abs(obs_y - y_current) < 2.0:  # Obstacle in current lane area
                    obstacle_pressure += 0.3
        
        urgency += obstacle_pressure
        urgency *= (1.0 - self.lane_keeping_preference)  # Personality factor
        
        return min(urgency, 1.0)
    
    def _calculate_personal_target_lane(self, y_current: float, target_lane: float, 
                                      threat_level: float) -> float:
        """Calculate where the human personally wants to go"""
        if threat_level > 0.7:
            # High threat: prioritize safety over target
            safe_direction = 1.0 if target_lane > y_current else -1.0
            personal_target = y_current + safe_direction * 2.0  # Move to safer area
        elif threat_level > 0.3:
            # Moderate threat: compromise between safety and target
            safety_factor = 0.7
            personal_target = y_current + (target_lane - y_current) * safety_factor
        else:
            # Low threat: move toward target with personal preferences
            movement_factor = 1.0 - self.lane_keeping_preference
            personal_target = y_current + (target_lane - y_current) * movement_factor
        
        # Clamp to road boundaries
        return np.clip(personal_target, -3.5, 3.5)
    
    def _select_movement_strategy(self, threat_level: float, comfort_level: float, 
                                urgency: float) -> Dict:
        """Select movement strategy based on personal assessment"""
        strategy = {
            'aggression': 0.5,
            'caution': 0.5,
            'cooperation': self.cooperation_tendency,
            'max_lateral_rate': self.preferred_lateral_speed
        }
        
        if threat_level > 0.6:
            # High threat: more cooperative and cautious
            strategy['cooperation'] = min(self.cooperation_tendency + 0.3, 1.0)
            strategy['caution'] = min(self.risk_tolerance + 0.4, 1.0)
            strategy['aggression'] = max(self.risk_tolerance - 0.3, 0.1)
            
        elif comfort_level < 0.3:
            # Low comfort: prioritize comfort
            strategy['max_lateral_rate'] *= 0.5  # Smoother movements
            strategy['caution'] += 0.2
            
        elif urgency > 0.7:
            # High urgency: more decisive
            strategy['aggression'] = min(self.confidence_level, 1.0)
            strategy['max_lateral_rate'] *= 1.5
        
        return strategy
    
    def _calculate_trajectory_point(self, i: int, t: float, y_current: float,
                                  personal_target: float, strategy: Dict,
                                  current_pos: Tuple[float, float], obstacles: List[Dict],
                                  velocity: float, human_input: float) -> Tuple[float, float]:
        """Calculate single trajectory point based on independent human logic"""
        
        # Base movement toward personal target
        progress = min(t / 5.0, 1.0)  # 5-second planning horizon
        base_y = y_current + (personal_target - y_current) * progress
        
        # Add human input influence (driver's active input)
        human_influence = human_input * velocity * t * strategy['aggression']
        y_ref = base_y + human_influence
        
        # Local obstacle avoidance (independent of controller)
        avoidance_offset = self._calculate_local_avoidance(
            current_pos, obstacles, t, velocity, strategy
        )
        y_ref += avoidance_offset
        
        # Add personality-based variation (not always predictable)
        if strategy['aggression'] > 0.7:
            # Aggressive drivers have more variation
            noise_amplitude = 0.1 * np.sin(t * 2.0) * strategy['aggression']
            y_ref += noise_amplitude
        
        # Clamp to road boundaries
        y_ref = np.clip(y_ref, -3.5, 3.5)
        
        # Calculate corresponding phi (yaw angle)
        if i == 0:
            phi_ref = human_input * strategy['aggression'] * 0.3
        else:
            # Base phi on trajectory curvature
            dy = min(abs(y_ref - y_current), strategy['max_lateral_rate'] * self.dt)
            phi_ref = np.arctan2(dy, velocity * self.dt) * strategy['aggression']
            if y_ref < y_current:
                phi_ref *= -1
        
        phi_ref = np.clip(phi_ref, np.deg2rad(-25), np.deg2rad(25))
        
        return y_ref, phi_ref
    
    def _calculate_local_avoidance(self, current_pos: Tuple[float, float],
                                 obstacles: List[Dict], t: float, velocity: float,
                                 strategy: Dict) -> float:
        """Calculate local obstacle avoidance (independent logic)"""
        x_pos, y_pos = current_pos
        future_x = x_pos + velocity * t
        avoidance_offset = 0.0
        
        for obs in obstacles:
            if 'pos' in obs and len(obs['pos']) >= 2:
                obs_x, obs_y = obs['pos'][0], obs['pos'][1]
                
                # Check if obstacle will be close in future
                future_distance = abs(future_x - obs_x)
                lateral_distance = abs(y_pos - obs_y)
                
                if future_distance < self.personal_safety_margin and lateral_distance < 3.0:
                    # Calculate avoidance direction based on personal preference
                    avoid_direction = np.sign(y_pos - obs_y)
                    if avoid_direction == 0:
                        # Default avoidance direction based on personality
                        avoid_direction = 1.0 if strategy['aggression'] > 0.5 else -1.0
                    
                    avoidance_intensity = (self.personal_safety_margin - future_distance) / self.personal_safety_margin
                    avoidance_offset += avoid_direction * avoidance_intensity * 0.5 * strategy['caution']
        
        return avoidance_offset
    
    def update_cooperation_level(self, controller_success: float, situation_safety: float):
        """Update cooperation tendency based on controller performance"""
        if controller_success > 0.8 and situation_safety > 0.7:
            # Controller is doing well and situation is safe - be more cooperative
            self.cooperation_tendency = min(self.cooperation_tendency + 0.1, 1.0)
        elif controller_success < 0.4 or situation_safety < 0.3:
            # Controller struggling or dangerous situation - be less cooperative
            self.cooperation_tendency = max(self.cooperation_tendency - 0.1, 0.2)
        
        print(f"🤝 Cooperation level updated to: {self.cooperation_tendency:.2f}")
    
    def _find_nearest_threat(self, pos: Tuple[float, float], obstacles: List[Dict], 
                           preview_distance: float) -> Optional[Dict]:
        """Find nearest threatening obstacle (enhanced version)"""
        x_pos, y_pos = pos
        nearest_threat = None
        min_threat_score = float('inf')
        
        for obs in obstacles:
            if 'pos' in obs and len(obs['pos']) >= 2:
                obs_x, obs_y = obs['pos'][0], obs['pos'][1]
                
                longitudinal_distance = obs_x - x_pos
                lateral_distance = abs(obs_y - y_pos)
                
                if 0 < longitudinal_distance < preview_distance and lateral_distance < 4.0:
                    # Threat score considers both distance and personal safety margin
                    threat_score = (longitudinal_distance / self.personal_safety_margin + 
                                  lateral_distance / 2.0) * (1.0 - self.risk_tolerance)
                    
                    if threat_score < min_threat_score:
                        min_threat_score = threat_score
                        nearest_threat = obs
        
        return nearest_threat