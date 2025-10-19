#!/usr/bin/env python3
"""
File: trajectory_planner.py
Description: Implements the State Lattice trajectory planner with A* search.
"""

import numpy as np
import math
import heapq
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

@dataclass
class Node:
    """A* search node for state lattice planner"""
    x: float
    y: float
    yaw: float
    cost: float
    h_cost: float
    parent: Optional['Node'] = None
    primitive_params: Optional[dict] = None

    @property
    def f_cost(self):
        return self.cost + self.h_cost

    def __lt__(self, other):
        return self.f_cost < other.f_cost

@dataclass
class Obstacle:
    """Simple circular obstacle for state lattice planner"""
    x: float
    y: float
    radius: float

class MotionModel:
    """Motion model compatible with Li et al. 2019 system"""
    
    @staticmethod
    def generate_trajectory(s, km, kf, k0=0.0):
        """Generate trajectory using simplified kinematic model"""
        if s <= 0:
            return [0.0], [0.0], [0.0]
            
        num_points = max(int(float(s) * 10), 5)
        t = np.linspace(0, 1, num_points)
        
        curvature = float(k0) * (1 - t)**2 + float(km) * 2 * t * (1 - t) + float(kf) * t**2
        
        dt = 1.0 / (num_points - 1) if num_points > 1 else 1.0
        
        x, y, yaw = [0.0], [0.0], [0.0]
        for i in range(1, num_points):
            dx = (float(s) / num_points) * math.cos(yaw[-1])
            dy = (float(s) / num_points) * math.sin(yaw[-1])
            dyaw = float(curvature[i]) * (float(s) / num_points)
            
            x.append(x[-1] + dx)
            y.append(y[-1] + dy)
            yaw.append(yaw[-1] + dyaw)
        
        return x, y, yaw

class StateLatticeTrajectoryPlanner:
    """
    State Lattice Trajectory Planner with A* search
    Replaces cubic spline approach with kinematically feasible motion primitives
    """
    
    def __init__(self, Np: int = 15):
        self.Np = Np
        self.dt = 0.1
        self.max_lateral_acceleration = 3.0
        self.max_yaw_rate = 0.8
        self.max_position_deviation = 3.0
        self.look_ahead_distance_factor = 2.0
        
        self.position_tolerance = 5.0
        self.yaw_tolerance = np.deg2rad(45.0)
        
        self.primitives = self._create_motion_primitives()
        self.motion_model = MotionModel()
        
        print(f"🎯 State Lattice Trajectory Planner integrated")
        print(f"📊 Prediction horizon: Np={self.Np}")
        print(f"🚗 Loaded {len(self.primitives)} motion primitives")
        
    def _create_motion_primitives(self) -> List[dict]:
        """Create improved motion primitives for realistic trajectory planning"""
        primitives = []
        distances = [3.0, 5.0, 8.0, 12.0]
        lateral_offsets = [-1.5, -0.8, -0.3, 0.0, 0.3, 0.8, 1.5]
        yaw_changes = np.deg2rad([-15, -8, -4, -2, 0, 2, 4, 8, 15])
        
        for distance in distances:
            for lat_offset in lateral_offsets:
                for yaw_change in yaw_changes:
                    if abs(lat_offset) > 0.1 and abs(yaw_change) < np.deg2rad(1): continue
                    if abs(lat_offset) > 1.0 and abs(yaw_change) < np.deg2rad(3): continue
                    
                    s = math.sqrt(float(distance)**2 + float(lat_offset)**2)
                    if s < 0.1: continue
                        
                    km = (2 * float(yaw_change)) / s if s > 0 else 0.0
                    if abs(km) > 0.3: continue
                    
                    kf = -km * 0.5
                    cost = s * (1.0 + 0.2 * abs(lat_offset) + 0.3 * abs(yaw_change) + 0.1 * abs(km))
                    
                    primitives.append({'dx': float(distance), 'dy': float(lat_offset), 'dyaw': float(yaw_change), 's': float(s), 'km': float(km), 'kf': float(kf), 'cost': float(cost)})
        
        lane_width = 3.5
        lane_distances = [25.0, 35.0, 45.0]
        for distance in lane_distances:
            for direction in [-1, 1]:
                dy = direction * lane_width
                s = math.sqrt(float(distance)**2 + float(dy)**2)
                km = (1.5 * dy) / (distance * 0.6)
                kf = -km * 0.3
                if abs(km) <= 0.15:
                    primitives.append({'dx': float(distance), 'dy': float(dy), 'dyaw': float(np.deg2rad(direction * 3)), 's': float(s), 'km': float(km), 'kf': float(kf), 'cost': float(s * 1.2)})

        for direction in [-1, 1]:
            primitives.append({'dx': 8.0, 'dy': float(direction * 2.5), 'dyaw': float(np.deg2rad(direction * 10)), 's': 8.5, 'km': float(direction * 0.2), 'kf': float(-direction * 0.15), 'cost': 15.0})
        
        return primitives

    def _convert_obstacles_to_lattice_format(self, obstacles: List[dict]) -> List[Obstacle]:
        """Convert Li et al. obstacle format to state lattice format"""
        lattice_obstacles = []
        for obs in obstacles:
            if 'pos' in obs and len(obs['pos']) >= 2:
                x, y = obs['pos'][0], obs['pos'][1]
                radius = 2.5
                lattice_obstacles.append(Obstacle(x=x, y=y, radius=radius))
        return lattice_obstacles

    def _heuristic(self, current_x, current_y, goal_x, goal_y) -> float:
        """Heuristic function for A*"""
        return math.sqrt((goal_x - current_x)**2 + (goal_y - current_y)**2)

    def _is_goal_reached(self, x, y, yaw, goal_x, goal_y, goal_yaw) -> bool:
        """Check if current state is within goal tolerance"""
        distance = math.sqrt((x - goal_x)**2 + (y - goal_y)**2)
        angle_diff = abs(yaw - goal_yaw)
        angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
        return (distance < self.position_tolerance and angle_diff < self.yaw_tolerance)

    def _is_collision_free(self, start_node: Node, primitive: dict, obstacles: List[Obstacle]) -> bool:
        """Check for collisions along the trajectory of a motion primitive"""
        if not obstacles: return True
        try:
            traj_x, traj_y, _ = self.motion_model.generate_trajectory(float(primitive['s']), float(primitive['km']), float(primitive['kf']), k0=0.0)
        except Exception:
            num_steps = max(int(float(primitive['s']) * 5), 3)
            traj_x = np.linspace(0, float(primitive['dx']), num_steps)
            traj_y = np.linspace(0, float(primitive['dy']), num_steps)

        cos_yaw = math.cos(start_node.yaw)
        sin_yaw = math.sin(start_node.yaw)
        step_size = max(1, len(traj_x) // 5)
        for i in range(0, len(traj_x), step_size):
            lx, ly = traj_x[i], traj_y[i]
            gx = start_node.x + float(lx) * cos_yaw - float(ly) * sin_yaw
            gy = start_node.y + float(lx) * sin_yaw + float(ly) * cos_yaw
            for obs in obstacles:
                if math.sqrt((gx - obs.x)**2 + (gy - obs.y)**2) < obs.radius + 0.5:
                    return False
        return True

    def _get_neighbors(self, current_node: Node, obstacles: List[Obstacle]) -> List[Node]:
        """Generate neighboring nodes using motion primitives"""
        neighbors = []
        step_size = max(1, len(self.primitives) // 50)
        for i in range(0, len(self.primitives), step_size):
            primitive = self.primitives[i]
            dx_global = (primitive['dx'] * math.cos(current_node.yaw) - primitive['dy'] * math.sin(current_node.yaw))
            dy_global = (primitive['dx'] * math.sin(current_node.yaw) + primitive['dy'] * math.cos(current_node.yaw))
            new_x = current_node.x + dx_global
            new_y = current_node.y + dy_global
            new_yaw = current_node.yaw + primitive['dyaw']
            new_yaw = math.atan2(math.sin(new_yaw), math.cos(new_yaw))

            if self._is_collision_free(current_node, primitive, obstacles):
                neighbors.append(Node(x=new_x, y=new_y, yaw=new_yaw, cost=current_node.cost + primitive['cost'], h_cost=0.0, parent=current_node, primitive_params=primitive))
        return neighbors

    def _plan_path_with_astar(self, start_pos: Tuple, goal_pos: Tuple, obstacles: List[Obstacle], debug: bool = False) -> List[dict]:
        """Plan path using A* with state lattice"""
        start_x, start_y, start_yaw = start_pos
        goal_x, goal_y, goal_yaw = goal_pos

        start_node = Node(x=float(start_x), y=float(start_y), yaw=float(start_yaw), cost=0.0, h_cost=0.0)
        start_node.h_cost = self._heuristic(start_x, start_y, goal_x, goal_y)

        open_list = [start_node]
        closed_set = {}
        
        num_obstacles = len(obstacles)
        if num_obstacles > 4:
            max_iterations, discretization, angle_discretization, goal_tolerance = 2000, 1.0, 0.3, 3.0
        else:
            max_iterations, discretization, angle_discretization, goal_tolerance = 500, 0.5, 0.2, 2.0
            
        iteration = 0
        best_node = start_node
        best_distance = self._heuristic(start_x, start_y, goal_x, goal_y)
        
        while open_list and iteration < max_iterations:
            iteration += 1
            current = heapq.heappop(open_list)
            
            current_dist = self._heuristic(current.x, current.y, goal_x, goal_y)
            if current_dist < best_distance:
                best_distance, best_node = current_dist, current
            
            key = (round(float(current.x) / discretization) * discretization, round(float(current.y) / discretization) * discretization, round(float(current.yaw) / angle_discretization) * angle_discretization)
            
            if key in closed_set and closed_set[key] <= current.f_cost: continue
            closed_set[key] = current.f_cost

            if self._is_goal_reached_tolerant(current.x, current.y, current.yaw, goal_x, goal_y, goal_yaw, goal_tolerance):
                if debug: print(f"✅ A* found path in {iteration} iterations!")
                return self._reconstruct_path(current)

            neighbors = self._get_neighbors(current, obstacles)
            for neighbor in neighbors:
                neighbor.h_cost = self._heuristic(neighbor.x, neighbor.y, goal_x, goal_y)
                heapq.heappush(open_list, neighbor)

        if debug:
            print(f"A* failed after {iteration} iterations. Final open_list size: {len(open_list)}")
            print(f"Closest distance to goal: {best_distance:.2f}m")
        
        if best_distance < 5.0:
            if debug: print(f"🔄 Using best partial path (distance: {best_distance:.2f}m)")
            return self._reconstruct_path(best_node)
        
        return []

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi: angle -= 2 * math.pi
        while angle < -math.pi: angle += 2 * math.pi
        return angle

    def _is_goal_reached_tolerant(self, x: float, y: float, yaw: float, goal_x: float, goal_y: float, goal_yaw: float, tolerance: float = 2.0) -> bool:
        """More tolerant goal checking for challenging scenarios"""
        position_error = math.sqrt((x - goal_x)**2 + (y - goal_y)**2)
        angle_error = abs(self._normalize_angle(yaw - goal_yaw))
        return position_error < tolerance and angle_error < math.pi/3

    def _reconstruct_path(self, goal_node: Node) -> List[dict]:
        """Reconstruct the path from goal node to start"""
        path_segments, current = [], goal_node
        while current.parent is not None:
            parent, primitive = current.parent, current.primitive_params
            try:
                traj_x_local, traj_y_local, traj_yaw_local = self.motion_model.generate_trajectory(primitive['s'], primitive['km'], primitive['kf'])
                cos_yaw, sin_yaw = math.cos(parent.yaw), math.sin(parent.yaw)
                traj_x_global = [parent.x + x * cos_yaw - y * sin_yaw for x, y in zip(traj_x_local, traj_y_local)]
                traj_y_global = [parent.y + x * sin_yaw + y * cos_yaw for x, y in zip(traj_x_local, traj_y_local)]
                traj_yaw_global = [parent.yaw + yaw for yaw in traj_yaw_local]
                path_segments.append({'x': traj_x_global, 'y': traj_y_global, 'yaw': traj_yaw_global, 'start_node': parent})
            except Exception:
                path_segments.append({'x': [parent.x, current.x], 'y': [parent.y, current.y], 'yaw': [parent.yaw, current.yaw], 'start_node': parent})
            current = parent
        path_segments.reverse()
        return path_segments

    def _extract_reference_trajectory(self, path_segments: List[dict], horizon: int) -> np.ndarray:
        """Extract reference trajectory in Li et al. format"""
        horizon = int(horizon)
        if not path_segments: return np.zeros((horizon, 2))
            
        all_x, all_y, all_yaw = [], [], []
        for segment in path_segments:
            all_x.extend(segment['x']); all_y.extend(segment['y']); all_yaw.extend(segment['yaw'])
        
        if len(all_x) == 0: return np.zeros((horizon, 2))
        
        reference_matrix = np.zeros((horizon, 2))
        for i in range(horizon):
            progress = min(float(i) / max(float(horizon) - 1.0, 1.0), 1.0)
            idx = min(int(progress * (len(all_x) - 1)), len(all_x) - 1)
            reference_matrix[i, 0] = float(all_y[idx])
            reference_matrix[i, 1] = float(all_yaw[idx])
        return reference_matrix

    def _find_safe_lateral_target(self, goal_x: float, current_y: float, obstacles: List[dict], target_lane: float = 0.0) -> float:
        """Find safe lateral target position moving toward target lane"""
        target_y = current_y + (target_lane - current_y) * 0.1
        for obs in obstacles:
            if 'pos' in obs and len(obs['pos']) >= 2:
                obs_x, obs_y = obs['pos'][0], obs['pos'][1]
                if abs(goal_x - obs_x) < 15.0 and abs(target_y - obs_y) < 3.0:
                    target_y = obs_y - 4.0 if obs_y > target_y else obs_y + 4.0
        return np.clip(target_y, -3.5, 3.5)

    def plan_trajectory(self, current_pos: Tuple[float, float], obstacles: List[Dict], target_lane: float = 0.0, velocity: float = 20.0) -> np.ndarray:
        """Plan collision-free trajectory using State Lattice + A*"""
        x_current, y_current = float(current_pos[0]), float(current_pos[1])
        if abs(y_current) > 4.5:
            print(f"Emergency: Extreme position y={y_current:.2f}m, using emergency path")
            return self._generate_emergency_safe_path(current_pos, velocity, self.Np)
        
        try:
            nearby_obstacles = [obs for obs in obstacles if 'pos' in obs and abs(obs['pos'][0] - x_current) < 40.0 and abs(obs['pos'][1] - y_current) < 8.0]
            if not nearby_obstacles:
                return self._generate_improved_fallback_path(current_pos, obstacles, velocity, self.Np, target_lane)
            
            goal_distance = min(float(velocity) * self.dt * float(self.Np) * 0.8, 25.0)
            goal_x = x_current + goal_distance
            goal_y = self._find_safe_lateral_target(goal_x, y_current, obstacles, target_lane)
            
            start_pos, goal_pos = (x_current, y_current, 0.0), (goal_x, goal_y, 0.0)
            path_segments = self._plan_path_with_astar(start_pos, goal_pos, self._convert_obstacles_to_lattice_format(obstacles), debug=True)
            
            if path_segments:
                reference_matrix = self._extract_reference_trajectory(path_segments, self.Np)
                if self.validate_trajectory(reference_matrix, velocity):
                    return reference_matrix
        except Exception as e:
            if "A*" not in str(e): print(f"State lattice planning error: {e}")
        
        return self._generate_improved_fallback_path(current_pos, obstacles, velocity, self.Np, target_lane)

    def _generate_improved_fallback_path(self, current_pos: Tuple[float, float], obstacles: List[dict], velocity: float, horizon: int, target_lane: float = 0.0) -> np.ndarray:
        """Generate an improved fallback path"""
        x_start, y_start = current_pos
        horizon, reference_matrix = int(horizon), np.zeros((horizon, 2))
        max_lat_change, max_phi_change = 0.02, 0.005
        convergence_factor = min(0.08, 0.02 + abs(target_lane - y_start) * 0.01)
        current_y, current_phi = y_start, 0.0
        
        for i in range(horizon):
            dy_step = np.clip((target_lane - current_y) * convergence_factor, -max_lat_change, max_lat_change)
            y_ref = current_y + dy_step
            
            x_pred = x_start + float(i) * float(velocity) * self.dt
            for obs in obstacles:
                if 'pos' in obs and len(obs['pos']) >= 2:
                    obs_x, obs_y = obs['pos'][0], obs['pos'][1]
                    if abs(x_pred - obs_x) < 8.0 and abs(y_ref - obs_y) < 2.5:
                        y_ref += 0.3 * np.sign(y_start - obs_y)
            
            y_ref = np.clip(y_ref, -4.0, 4.0)
            
            if i == 0: phi_ref = current_phi
            else:
                slope = np.clip((y_ref - reference_matrix[i-1, 0]) / (float(velocity) * self.dt), -0.005, 0.005)
                phi_ref = current_phi + np.clip(np.arctan(slope) - current_phi, -max_phi_change, max_phi_change)
            
            phi_ref = np.clip(phi_ref, np.deg2rad(-20), np.deg2rad(20))
            reference_matrix[i, :] = [y_ref, phi_ref]
            current_y, current_phi = y_ref, phi_ref
        
        return reference_matrix

    def _generate_emergency_safe_path(self, current_pos: Tuple[float, float], velocity: float, horizon: int) -> np.ndarray:
        """Emergency ultra-conservative path"""
        x_start, y_start = current_pos
        horizon, reference_matrix = int(horizon), np.zeros((horizon, 2))
        for i in range(horizon):
            progress = float(i) / max(float(horizon) - 1.0, 1.0)
            reference_matrix[i, :] = [y_start * (1.0 - progress * 0.03), 0.0]
        return reference_matrix

    def validate_trajectory(self, trajectory: np.ndarray, velocity: float) -> bool:
        """Improved trajectory validation"""
        if trajectory.shape[0] < 2: return False
        max_check_points = min(3, int(len(trajectory)) - 1)
        for i in range(max_check_points):
            y_curr, phi_curr = float(trajectory[i][0]), float(trajectory[i][1])
            y_next, phi_next = float(trajectory[i+1][0]), float(trajectory[i+1][1])
            if abs(y_curr) > 5.0 or abs(y_next) > 5.0: return False
            
            lat_accel = abs(((y_next - y_curr) / self.dt) / self.dt)
            if lat_accel > self.max_lateral_acceleration * (1.0 + float(velocity) / 40.0): return False
            
            dphi = phi_next - phi_curr
            while dphi > np.pi: dphi -= 2*np.pi
            while dphi < -np.pi: dphi += 2*np.pi
            if abs(dphi) / self.dt > self.max_yaw_rate * (1.0 + float(velocity) / 50.0): return False
            
            if abs(phi_curr) > np.deg2rad(45) or abs(phi_next) > np.deg2rad(45): return False
        return True
