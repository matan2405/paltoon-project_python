#!/usr/bin/env python3
"""
File: main.py
Description: Main execution file to run the convoy merging simulation with enhanced human model.
"""

import numpy as np
import os
from typing import List, Dict

# Import all the system components from their respective files
from vehicle_model import VehicleParams, VehicleModel
from safety_field import SafetyFieldParams, EnhancedSafetyField
from authority_allocator import ImprovedAuthorityAllocator
from nash_solver import EnhancedNashSolver
from trajectory_planner import StateLatticeTrajectoryPlanner
from human_model import EnhancedHumanModel
from visualization import visualize_results, create_animated_visualization
import time
# Clear console for clean output
os.system('cls' if os.name == 'nt' else 'clear')

class ConvoyMergingSystem:
    """Main system that integrates all components for convoy merging simulation"""
    
    def __init__(self, vehicle_params: VehicleParams = None):
        self.params = vehicle_params or VehicleParams()
        
        # Initialize all subsystems
        self.vehicle = VehicleModel(self.params, velocity=20.0)
        self.safety_field = EnhancedSafetyField(SafetyFieldParams())
        self.authority_allocator = ImprovedAuthorityAllocator()
        self.nash_solver = EnhancedNashSolver(self.vehicle)
        self.trajectory_planner = StateLatticeTrajectoryPlanner(Np=self.nash_solver.Np)
        self.human_model = EnhancedHumanModel(Np=self.nash_solver.Np)
        
        print("🚗 Convoy Merging DMPC System Initialized (Enhanced Human Model)")
        print("🧠 Independent Human Decision Making - No More Always-Opposing Behavior")
        print("Coordinate System: X-axis = longitudinal (forward), Y-axis = lateral (lane position)")
        print("Lane positions: Right lane (y=3.5m), Left lane (y=0.0m)")
        print("Both ego vehicle and convoy move forward, convoy in parallel lane")
    
    def _update_obstacle_positions(self, initial_obstacles: List[Dict], current_time: float) -> List[Dict]:
        """Update dynamic obstacle positions based on their velocity from their initial positions"""
        updated_obstacles = []
        for obs in initial_obstacles:
            # Important: Work on a copy to not modify the original list
            updated_obs = {key: val for key, val in obs.items()}
            
            initial_pos = np.array(updated_obs.get('pos', [0.0, 0.0]))
            velocity = updated_obs.get('vel', [0.0])[0]
            
            # Calculate new position based on INITIAL position and current time
            new_pos = initial_pos.copy()
            new_pos[0] += velocity * current_time
            updated_obs['current_pos'] = new_pos.tolist() # Use a new key for clarity
            updated_obstacles.append(updated_obs)
        return updated_obstacles
        
    def step(self, current_state: np.ndarray, human_input: float,
             dynamic_obstacles: List[Dict], x_position: float, target_lane: float = 0.0) -> Dict:
        """Execute one full control and simulation step"""
        time_step_start = time.time()
        current_pos = (x_position, current_state[2])
        velocity = 20.0  # Assume constant velocity for now
        
        # Create a list of obstacle positions for the current step
        current_obstacle_positions = [{'pos': obs['current_pos'], 'vel': obs['vel']} for obs in dynamic_obstacles]

        # 1. Evaluate driving risk based on current positions
        field_force = self.safety_field.compute_field_force(
            np.array([current_pos[0], current_pos[1]]), velocity, current_obstacle_positions)
        
        # 2. Allocate control authority
        lambda_k = self.authority_allocator.compute_authority_ratio(field_force)

        self.vehicle.lambda_k = lambda_k

        # 3. Plan reference trajectories
        R1_ref = self.trajectory_planner.plan_trajectory(current_pos, current_obstacle_positions, target_lane, velocity)
        R2_ref = self.human_model.estimate_human_trajectory(current_pos, current_state, human_input, current_obstacle_positions, target_lane, velocity)
        
          # DEBUG: בדיקת מסלולי הייחוס לפני Nash
        print(f"\n🔍 DEBUG Step {x_position:.0f}m:")
        print(f"  R1_ref (Controller) first 3 points: {R1_ref[:3, :].flatten()}")
        print(f"  R2_ref (Human) first 3 points: {R2_ref[:3, :].flatten()}")
        
        # בדיקה האם המסלולים תמיד מנוגדים
        r1_avg = np.mean(R1_ref[:5, 0])  # ממוצע 5 הנקודות הראשונות
        r2_avg = np.mean(R2_ref[:5, 0])
        print(f"  R1 avg lateral: {r1_avg:.4f}, R2 avg lateral: {r2_avg:.4f}")
        
        if abs(r1_avg) > 0.001 and abs(r2_avg) > 0.001:
            if np.sign(r1_avg) != np.sign(r2_avg):
                print(f"  ⚠️ WARNING: Reference trajectories are opposing!")
            else:
                print(f"  ✅ Reference trajectories in same direction")
        
        # 4. Solve for Nash equilibrium control inputs
        u1_opt, u2_opt = self.nash_solver.solve_nash_equilibrium(current_state, R1_ref, R2_ref, lambda_k)
        
        # DEBUG: בדיקת תוצאות Nash
        print(f"  Nash results: u1_opt={u1_opt:.4f}, u2_opt={u2_opt:.4f}")
        print(f"  Sum: {u1_opt + u2_opt:.4f}, Product: {u1_opt * u2_opt:.6f}")
        if abs(u1_opt + u2_opt) < 0.001 and abs(u1_opt) > 0.001:
            print(f"  🚨 PROBLEM: Nash solutions are perfectly opposing!")
            
        # הוסף delay קטן לבדיקה
        if x_position < 50:  # רק בתחילה
            time.sleep(0.5)
        
        # 4. Solve for Nash equilibrium control inputs
        # u1_opt, u2_opt = self.nash_solver.solve_nash_equilibrium(current_state, R1_ref, R2_ref, lambda_k)
        
        # 5. Calculate the shared control input
        alpha = lambda_k / (1.0 + lambda_k)
        u_shared = alpha * u1_opt + (1 - alpha) * u2_opt
        u_shared = np.clip(u_shared, min(u1_opt, u2_opt), max(u1_opt, u2_opt))

        # 6. Simulate the vehicle's next state using the individual optimal inputs
        next_state = self.vehicle.step(current_state, u1_opt, u2_opt)
        
        # 7. Calculate comfort metrics
        lateral_acc = abs((next_state[0] - current_state[0]) / self.params.dt)
        time_step_end = time.time()

        print(f"  [Step Time: {time_step_end - time_step_start:.3f}s]")
        return {
            'next_state': next_state,
            'controller_input': u1_opt,
            'human_input_estimated': u2_opt,  
            'shared_input': u_shared,
            'authority_ratio': lambda_k,
            'field_force': field_force,
            'lateral_acceleration': lateral_acc,
            'controller_trajectory': R1_ref,
            'human_trajectory': R2_ref,
        }
    
    def simulate_scenario(self, initial_state: np.ndarray, obstacles: List[Dict],
                         human_behavior: str, T: float, target_lane: float) -> Dict:
        """Run a full simulation for a given scenario with enhanced human model"""
        # Set human driver personality at start of simulation
        self.human_model.set_driver_personality(human_behavior)
        
        dt = self.params.dt
        steps = int(T / dt)
        
        # Data storage arrays
        states = np.zeros((steps, 4))
        x_positions = np.zeros(steps)
        controls = {'controller': np.zeros(steps), 'human': np.zeros(steps), 'shared': np.zeros(steps)}
        authority_ratios = np.zeros(steps)
        field_forces = np.zeros((steps, 2))
        lateral_accelerations = np.zeros(steps)
        controller_trajectories, human_trajectories = [], []

        # Initial conditions
        states[0] = initial_state
        human_inputs = self._generate_human_inputs(steps, human_behavior)
        
        print(f"\n🚀 Starting {T}s simulation: '{human_behavior}' behavior with enhanced independent model...")
        
        # Main simulation loop
        for i in range(1, steps):
            t = i * dt  # Correct time calculation for current step
            # Ego vehicle moves forward at 20 m/s
            x_positions[i] = x_positions[i-1] + 20.0 * dt
            
            # Get dynamic obstacle positions for the current time
            # Convoy vehicles also move forward at their velocity (18 m/s)
            dynamic_obstacles = self._update_obstacle_positions(obstacles, t)
            
            result = self.step(states[i-1], human_inputs[i-1], dynamic_obstacles, x_positions[i-1], target_lane)
            
            # Store results
            states[i] = result['next_state']
            controls['controller'][i] = result['controller_input']
            controls['human'][i] = result['human_input_estimated']
            controls['shared'][i] = result['shared_input']
            authority_ratios[i] = result['authority_ratio']
            field_forces[i] = result['field_force']
            lateral_accelerations[i] = result['lateral_acceleration']
            
            if i % 10 == 0: # Store trajectories periodically
                controller_trajectories.append({'x_start': x_positions[i-1], 'trajectory': result['controller_trajectory']})
                human_trajectories.append({'x_start': x_positions[i-1], 'trajectory': result['human_trajectory']})

            if i > 0 and i % (steps // 5) == 0:
                print(f"  -> Progress: {i/steps*100:.0f}%")

        summary = self._calculate_summary_metrics(states, authority_ratios, lateral_accelerations)
        
        # Monitor cooperation levels and human behavior (NEW!)
        cooperation_analysis = self._monitor_cooperation_levels(states, controls, field_forces, summary)
        
        # For visualization, we need the initial obstacle positions
        # The animation function will handle their movement
        return {
            'time': np.arange(steps) * dt, 'states': states, 'x_positions': x_positions,
            'controls': controls, 'authority_ratios': authority_ratios, 'field_forces': field_forces,
            'lateral_accelerations': lateral_accelerations, 'summary': summary,
            'controller_trajectories': controller_trajectories, 'human_trajectories': human_trajectories,
            'obstacles': obstacles, # Pass initial obstacle data to visualization
            'cooperation_analysis': cooperation_analysis  # NEW: Enhanced behavior analysis
        }

    def _generate_human_inputs(self, steps: int, behavior: str) -> np.ndarray:
        """Generate more realistic and diverse human steering inputs based on behavior profile"""
        base_inputs = np.zeros(steps)
        
        if behavior == 'aggressive':
            # Aggressive: More activity, less smooth, more confident
            for i in range(steps):
                t = i / steps
                # High initial activity that moderates over time
                intensity = 0.04 * np.exp(-2*t) + 0.01
                base_inputs[i] = np.sin(4*np.pi*t) * intensity
                # Add random noise for unpredictability
                base_inputs[i] += np.random.normal(0, 0.005)
                # Occasional strong corrections
                if np.random.random() < 0.05:  # 5% chance
                    base_inputs[i] += np.random.choice([-0.02, 0.02])
            
        elif behavior == 'conservative':
            # Conservative: Minimal, smooth, very predictable activity
            for i in range(steps):
                t = i / steps
                # Only gentle activity in middle phase
                if 0.2 < t < 0.4:
                    phase_progress = (t - 0.2) / 0.2
                    base_inputs[i] = 0.01 * np.sin(np.pi * phase_progress)
                # Very small random adjustments
                base_inputs[i] += np.random.normal(0, 0.002)
                
            # Additional smoothing for conservative behavior
            base_inputs = np.convolve(base_inputs, np.ones(7)/7, mode='same')
            
        else:  # normal
            # Normal: Balanced between activity and calmness
            for i in range(steps):
                t = i / steps
                # Main activity period
                if 0.15 < t < 0.35:
                    phase = (t - 0.15) / 0.2
                    base_inputs[i] = 0.025 * np.sin(np.pi * phase)
                # Secondary smaller correction later
                elif 0.6 < t < 0.8:
                    correction_intensity = 0.01 * np.sin(np.pi * (t - 0.6) / 0.2)
                    base_inputs[i] = correction_intensity
                # Small random adjustments throughout
                base_inputs[i] += np.random.normal(0, 0.003)
            
            # Light smoothing for normal behavior
            base_inputs = np.convolve(base_inputs, np.ones(3)/3, mode='same')
        
        print(f"  🎮 Generated {behavior} human inputs - Activity level: {np.mean(np.abs(base_inputs)):.4f}")
        return base_inputs

    def _calculate_summary_metrics(self, states, authority_ratios, lateral_accelerations) -> Dict:
        """Calculate key performance indicators from the simulation results"""
        return {
            'max_lateral_deviation': np.max(np.abs(states[:, 2])),
            'max_authority_ratio': np.max(authority_ratios),
            'max_lateral_acceleration': np.max(lateral_accelerations),
            'comfort_violations': np.sum(lateral_accelerations > self.params.max_lat_acc)
        }

    def _monitor_cooperation_levels(self, states, controls, field_forces, summary) -> Dict:
        """Monitor and analyze human-controller cooperation patterns"""
        controls_controller = controls['controller']
        controls_human = controls['human']
        
        # Analyze cooperation vs opposition patterns
        cooperation_moments = 0
        opposition_moments = 0
        neutral_moments = 0
        
        for i in range(len(controls_controller)):
            controller_val = controls_controller[i]
            human_val = controls_human[i]
            
            # Skip very small values (noise)
            if abs(controller_val) < 0.005 and abs(human_val) < 0.005:
                neutral_moments += 1
                continue
                
            # Check if they're working in same direction (cooperation)
            if controller_val * human_val > 0:
                cooperation_moments += 1
            else:
                opposition_moments += 1
        
        total_active_moments = cooperation_moments + opposition_moments
        if total_active_moments > 0:
            cooperation_ratio = cooperation_moments / total_active_moments
            opposition_ratio = opposition_moments / total_active_moments
        else:
            cooperation_ratio = 0.0
            opposition_ratio = 0.0
        
        # Calculate human and controller activity levels
        human_activity = np.sum(np.abs(controls_human) > 0.01) / len(controls_human)
        controller_activity = np.sum(np.abs(controls_controller) > 0.01) / len(controls_controller)
        
        # Calculate controller success score
        max_deviation = summary['max_lateral_deviation']
        comfort_violations = summary['comfort_violations']
        controller_success = 1.0 - min(max_deviation / 4.0, 1.0)
        controller_success -= comfort_violations * 0.1
        controller_success = max(controller_success, 0.0)
        
        # Assess situation safety
        avg_field_force = np.mean(np.linalg.norm(field_forces, axis=1))
        situation_safety = 1.0 - min(avg_field_force / 500.0, 1.0)
        
        # Update human model cooperation level based on performance
        self.human_model.update_cooperation_level(controller_success, situation_safety)
        
        return {
            'cooperation_ratio': cooperation_ratio,
            'opposition_ratio': opposition_ratio,
            'neutral_ratio': neutral_moments / len(controls_controller),
            'human_activity': human_activity,
            'controller_activity': controller_activity,
            'controller_success': controller_success,
            'situation_safety': situation_safety,
            'final_cooperation_tendency': self.human_model.cooperation_tendency,
            'total_active_moments': total_active_moments
        }

def run_convoy_scenarios():
    """Run 3 convoy joining scenarios with different driver types"""
    system = ConvoyMergingSystem()
    
    print("="*80)
    print("🚗 Parallel Lane Convoy Merging Scenarios - Enhanced Human Model")
    print("="*80)
    print("Ego vehicle in right lane (y=3.5m), Convoy in left lane (y=0.0m)")
    print("Both moving forward, ego vehicle wants to merge left into convoy lane")
    print("🧠 NEW: Independent human decision making - realistic cooperation/opposition")
    print("="*80)
    
    # Convoy vehicles in parallel lane (y=0.0) moving at same speed as ego vehicle
    # Scenarios differ by convoy positioning relative to ego vehicle
    
    scenarios = [
        {
            'name': 'Join Before Convoy',
            'description': 'Ego vehicle merges into target lane before convoy reaches that position',
            'obstacles': [
                # Convoy is behind ego vehicle's future position
                {'pos': [0.0, 0.0], 'vel': [18.0], 'type': 'convoy_vehicle_1'},
                {'pos': [-40.0, 0.0], 'vel': [18.0], 'type': 'convoy_vehicle_2'},
                {'pos': [-80.0, 0.0], 'vel': [18.0], 'type': 'convoy_vehicle_3'},
                {'pos': [-120.0, 0.0], 'vel': [18.0], 'type': 'convoy_vehicle_4'}
            ],
            'initial_state': np.array([0.0, 0.0, 3.5, 0.0]),  # Start in right lane
            'target_lane': 0.0,  # Merge into left lane
            'simulation_time': 120.0
        },
        {
            'name': 'Join Middle of Convoy',
            'description': 'Ego vehicle attempts to merge into gap within convoy',
            'obstacles': [
                # Convoy vehicles positioned to create a gap near ego vehicle
                {'pos': [20.0, 0.0], 'vel': [18.0], 'type': 'convoy_front'},
                {'pos': [-20.0, 0.0], 'vel': [18.0], 'type': 'convoy_rear'}
            ],
            'initial_state': np.array([0.0, 0.0, 3.5, 0.0]),
            'target_lane': 0.0,
            'simulation_time': 120.0
        },
        {
            'name': 'Join After Convoy',
            'description': 'Ego vehicle waits for convoy to pass then merges',
            'obstacles': [
                # Convoy is ahead of ego vehicle
                {'pos': [80.0, 0.0], 'vel': [18.0], 'type': 'convoy_vehicle_1'},
                {'pos': [95.0, 0.0], 'vel': [18.0], 'type': 'convoy_vehicle_2'},
                {'pos': [110.0, 0.0], 'vel': [18.0], 'type': 'convoy_vehicle_3'}
            ],
            'initial_state': np.array([0.0, 0.0, 3.5, 0.0]),
            'target_lane': 0.0,
            'simulation_time': 120.0
        }
    ]
    
    # Different driver types
    driver_types = ['conservative', 'normal', 'aggressive']
    driver_names = ['Conservative', 'Normal', 'Aggressive']
    
    all_results = {}
    
    for scenario_idx, scenario in enumerate(scenarios):
        print(f"\n{'='*60}")
        print(f"Scenario {scenario_idx + 1}: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"{'='*60}")
        
        scenario_results = {}
        
        for driver_idx, (driver_type, driver_name) in enumerate(zip(driver_types, driver_names)):
            print(f"\n--- {driver_name} Driver ({driver_type}) ---")
            
            results = system.simulate_scenario(
                scenario['initial_state'],
                scenario['obstacles'],
                driver_type,
                scenario['simulation_time'],
                scenario['target_lane']
            )
            
            scenario_results[driver_type] = results
            
            # Enhanced results display
            cooperation_analysis = results['cooperation_analysis']
            print(f"\n📊 Enhanced Results for {driver_name} Driver:")
            print(f"  • Max lateral deviation: {results['summary']['max_lateral_deviation']:.2f}m")
            print(f"  • Max authority ratio: {results['summary']['max_authority_ratio']:.2f}")
            print(f"  • Max lateral acceleration: {results['summary']['max_lateral_acceleration']:.2f}m/s²")
            print(f"  • Comfort violations: {results['summary']['comfort_violations']}")
            print(f"  • Controller success: {cooperation_analysis['controller_success']:.2f}")
            print(f"  • Final cooperation level: {cooperation_analysis['final_cooperation_tendency']:.2f}")
            print(f"  • Situation safety: {cooperation_analysis['situation_safety']:.2f}")
            
            # Analyze enhanced route behavior
            _analyze_enhanced_route_behavior(results, driver_name, scenario['name'])
        
        all_results[scenario['name']] = scenario_results
        
        # Compare drivers in this scenario with enhanced analysis
        _compare_drivers_enhanced(scenario_results, scenario['name'])
        
        # Create visualizations for each scenario
        print(f"\n🎯 Creating visualizations for {scenario['name']}...")
        for driver_type, results in scenario_results.items():
            driver_name = {'conservative': 'Conservative', 'normal': 'Normal', 'aggressive': 'Aggressive'}[driver_type]
            title = f"{scenario['name']} - {driver_name} Driver"
            
            # Create static visualization
            visualize_results(results, title)
            
            # Create animated visualization
            # anim = create_animated_visualization(results, f"{title} Animation")
    
    # Overall summary with enhanced insights
    _generate_enhanced_overall_summary(all_results)
    
    return all_results

def _analyze_enhanced_route_behavior(results, driver_name, scenario_name):
    """Analyze enhanced route behavior for specific driver and scenario"""
    states = results['states']
    authority_ratios = results['authority_ratios']
    cooperation_analysis = results['cooperation_analysis']
    
    # Calculate additional performance metrics
    lane_changes = np.sum(np.abs(np.diff(states[:, 2])) > 0.1)
    avg_authority = np.mean(authority_ratios[authority_ratios > 0])
    time_in_cooperation = np.sum(authority_ratios > 0.1) * results['time'][1]
    
    print(f"  🔍 Enhanced Route Analysis:")
    print(f"    - Number of lane changes: {lane_changes}")
    print(f"    - Time in cooperation: {time_in_cooperation:.1f}s")
    print(f"    - Average authority: {avg_authority:.2f}")
    print(f"    - Cooperation moments: {cooperation_analysis['cooperation_ratio']:.1%}")
    print(f"    - Opposition moments: {cooperation_analysis['opposition_ratio']:.1%}")
    print(f"    - Human activity level: {cooperation_analysis['human_activity']:.1%}")
    print(f"    - Controller activity level: {cooperation_analysis['controller_activity']:.1%}")
    
    # Route quality assessment with cooperation analysis
    deviation = results['summary']['max_lateral_deviation']
    coop_ratio = cooperation_analysis['cooperation_ratio']
    
    if deviation < 1.0 and coop_ratio > 0.6:
        route_quality = "Excellent (Good Cooperation)"
    elif deviation < 2.0 and coop_ratio > 0.4:
        route_quality = "Good (Balanced Interaction)"
    elif deviation < 3.0:
        route_quality = "Average (Independent Behavior)"
    else:
        route_quality = "Needs Improvement (Conflicting Behavior)"
    
    print(f"    - Enhanced route quality: {route_quality}")
    
    # Behavioral pattern analysis
    if cooperation_analysis['opposition_ratio'] > 0.8:
        print(f"    ⚠️  WARNING: Excessive opposition detected!")
    elif cooperation_analysis['cooperation_ratio'] > 0.7:
        print(f"    ✅ Excellent human-controller harmony")
    elif cooperation_analysis['opposition_ratio'] < 0.3:
        print(f"    🤝 High cooperation tendency")
    else:
        print(f"    ⚖️  Balanced independent decision making")

def _compare_drivers_enhanced(scenario_results, scenario_name):
    """Compare different driver types in given scenario with enhanced metrics"""
    print(f"\n📊 Enhanced Driver Comparison in {scenario_name}:")
    print("-" * 70)
    
    for driver_type, results in scenario_results.items():
        driver_name = {'conservative': 'Conservative', 'normal': 'Normal', 'aggressive': 'Aggressive'}[driver_type]
        summary = results['summary']
        coop_analysis = results['cooperation_analysis']
        
        print(f"{driver_name:12}: "
              f"deviation={summary['max_lateral_deviation']:.2f}m, "
              f"authority={summary['max_authority_ratio']:.2f}, "
              f"cooperation={coop_analysis['cooperation_ratio']:.1%}, "
              f"opposition={coop_analysis['opposition_ratio']:.1%}")

def _generate_enhanced_overall_summary(all_results):
    """Generate enhanced overall summary with cooperation analysis"""
    print(f"\n{'='*80}")
    print("📋 Enhanced Overall Summary - Human-Controller Interaction Analysis")
    print(f"{'='*80}")
    
    cooperation_insights = {
        'Join Before Convoy': {
            'conservative': 'Expected: High cooperation (70-80%) - Safe and predictable',
            'normal': 'Expected: Balanced interaction (50-60%) - Situational cooperation',
            'aggressive': 'Expected: Independent behavior (30-40%) - Task-focused'
        },
        'Join Middle of Convoy': {
            'conservative': 'Expected: Very high cooperation (80-90%) - Safety priority',
            'normal': 'Expected: Adaptive cooperation (60-70%) - Context dependent',
            'aggressive': 'Expected: Strategic cooperation (40-50%) - Goal oriented'
        },
        'Join After Convoy': {
            'conservative': 'Expected: Moderate cooperation (60-70%) - Comfort focused',
            'normal': 'Expected: Situational cooperation (50-60%) - Flexible approach',
            'aggressive': 'Expected: Low cooperation (20-30%) - Independent execution'
        }
    }
    
    for scenario, drivers in cooperation_insights.items():
        print(f"\n🎯 {scenario} - Expected Cooperation Patterns:")
        for driver_type, expectation in drivers.items():
            driver_name = {'conservative': 'Conservative Driver', 'normal': 'Normal Driver', 'aggressive': 'Aggressive Driver'}[driver_type]
            print(f"  • {driver_name}: {expectation}")
    
    # Performance recommendations based on cooperation patterns
    print(f"\n💡 Performance Optimization Recommendations:")
    print(f"  • If cooperation ratio > 80%: Consider increasing controller confidence")
    print(f"  • If opposition ratio > 70%: Review human model personality tuning")
    print(f"  • If neutral ratio > 60%: Increase system responsiveness")
    print(f"  • Optimal balance: 40-60% cooperation, 30-50% opposition, <20% neutral")

def run_standard_test():
    """Run standard test with enhanced convoy scenarios"""
    return run_convoy_scenarios()


if __name__ == "__main__":
    run_standard_test()